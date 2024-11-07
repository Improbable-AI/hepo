# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from .base.vec_task import VecTask


class FrankaCabinet(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 23
        num_acts = 9

        self.cfg["env"]["numObservations"] = 23
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.frankas = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            cabinet_pose = cabinet_start_pose
            cabinet_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cabinet_pose.p.y += self.start_position_noise * dy
            cabinet_pose.p.z += self.start_position_noise * dz
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
                drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.cabinets.append(cabinet_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1
        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

    def compute_reward(self, actions):
        global HUMAN_REWARD_FUNCS
        if self.use_human_design_reward in [
                *[f"ppo{d+1}" for d in range(6)],
                *[f"hepo{d+1}" for d in range(6)],
            ]:
            # print(f"Use human reward: {self.use_human_design_reward} ({HUMAN_REWARD_FUNCS[self.use_human_design_reward]})")
            # import ipdb; ipdb.set_trace()
            self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.consecutive_successes[:] = HUMAN_REWARD_FUNCS[self.use_human_design_reward](
                    self.reset_buf, self.progress_buf, self.reset_goal_buf, self.successes, self.consecutive_successes, self.actions, self.cabinet_dof_pos,
                    self.franka_grasp_pos, self.drawer_grasp_pos, self.franka_grasp_rot, self.drawer_grasp_rot,
                    self.franka_lfinger_pos, self.franka_rfinger_pos,
                    self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
                    self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
                    self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
            )
        else:                        
            self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.consecutive_successes[:] = compute_franka_reward(
                self.reset_buf, self.progress_buf, self.reset_goal_buf, self.successes, self.consecutive_successes, self.actions, self.cabinet_dof_pos,
                self.franka_grasp_pos, self.drawer_grasp_pos, self.franka_grasp_rot, self.drawer_grasp_rot,
                self.franka_lfinger_pos, self.franka_rfinger_pos,
                self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
                self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
                self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
            )
        self.success_buf[:] = self.reset_goal_buf[:] / (self.max_episode_length - 1)

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # reset props
        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.reset_goal_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, reset_goal_buf, successes, consecutive_successes, actions, cabinet_dof_pos,
    franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if left finger is above the drawer handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                       torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                     torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # how far the cabinet has been opened out
    open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty

    # bonus for opening drawer properly
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # prevent bad style in opening drawer
    rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)

    return rewards, reset_buf, goal_resets, consecutive_successes

@torch.jit.script
def compute_ppo1_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
 
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # TODO: Write your reward function here
    # stage 1: reaching reward for drawer
    # stage 2: pulling drawer out
    # in stage 1, we align the drawer inward axis and gripper forward axis, by being as negative as possible. 
    # so we negate the axis error, because -1 * -1 = +1, reward is maximized 
    # we minimize also the l2 error of the franka grasp pos and drawer grasp pos
    # in stage 2, we maximize the cabinet dof pos 
    # franka_drawer_pos_error = torch.sum(torch.abs(franka_grasp_pos - drawer_grasp_pos), dim=1)

    franka_drawer_pos_error = torch.linalg.norm(franka_grasp_pos - drawer_grasp_pos, dim=1)**2

    # franka_drawer_alignment_error = torch.bmm(gripper_forward_axis.unsqueeze(1), drawer_inward_axis.unsqueeze(2))
    stage1_reward = -franka_drawer_pos_error

    stage2_reward = 1000 * cabinet_dof_pos[:, 3]**2
    rewards = 100 * successes.float() + stage1_reward + stage2_reward

    return rewards, reset_buf, goal_resets, consecutive_successes

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(
        torch.clamp(
            torch.norm(
                mul[:, 0:3],
                p=2, dim=-1), max=1.0)
    )


@torch.jit.script
def compute_ppo2_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).  (n_envs, 4)
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
 
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # TODO: Write your reward function here
    rewards = successes

    # don't make the action to be too large, 
    action_penalty = - torch.norm(actions, dim = -1)

    # make the gripper axis be aligned to the door handle axis, and I want this to be maximized
    gripper_alignment = - quat_diff_rad(drawer_grasp_rot, franka_grasp_rot)  

    # make the gripper to be close to door handle, I want this to be minimized 

    gripper_distance = franka_grasp_pos - drawer_grasp_pos 
    gripper_distance = - torch.norm(gripper_distance, dim = -1)

    # dense reward for opening the drawer, I want this to be maximized 

    opened_reward = cabinet_dof_pos[..., 3]

    # i won't do things related to fingers for now 

    rewards += action_penalty_scale * action_penalty + rot_reward_scale * gripper_alignment + dist_reward_scale * gripper_distance + open_reward_scale * opened_reward 

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_ppo3_reward(
    reset_buf, progress_buf, reset_goal_buf, successes, consecutive_successes, actions, cabinet_dof_pos,
    franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: Write your reward function here


    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)

    shaping = torch.zeros_like(successes)
    drawer_out = cabinet_dof_pos[:, 3] # Cabinet out
    # Penalty for distance between gripper end effector and drawer - it should be close (reward expressed as a gaussian)
    # import ipdb; ipdb.set_trace()
    # dist = torch.pow(franka_grasp_pos - cabinet_dof_pos[:, :3], 2).mean(axis=1)
    dist_reward = torch.exp(-torch.pow(franka_grasp_pos - drawer_grasp_pos, 2).mean(dim=1)) # Higher reward for closer to drawer handle

    # print(franka_grasp_pos.shape)


    rewards = successes + drawer_out * open_reward_scale + dist_reward * dist_reward_scale

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_ppo4_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
 
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # TODO: Write your reward function here

    dist_reward  = -dist_reward_scale * torch.linalg.norm(franka_grasp_pos - drawer_grasp_pos, dim = 1)
    rot_reward = -rot_reward_scale * torch.linalg.norm(franka_grasp_rot - drawer_grasp_rot, dim = 1)                                
    around_handle_reward  = -around_handle_reward_scale * (torch.linalg.norm(gripper_forward_axis - drawer_inward_axis, dim = 1) + 
                                    torch.linalg.norm(gripper_up_axis - drawer_up_axis, dim = 1))
    open_reward = successes * open_reward_scale
    finger_dist_reward = -finger_dist_reward_scale * torch.linalg.norm(franka_lfinger_pos -franka_rfinger_pos, dim = 1)
    action_penalty = -action_penalty_scale * (actions**2).mean(dim = 1)
    rewards = dist_reward + rot_reward + around_handle_reward + open_reward + finger_dist_reward + action_penalty
    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_ppo5_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
 
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # TODO: Write your reward function here
    rewards = successes

    # Ensure that the gripper is pointing towards the drawer
    # Have a bonus for the gripper forward axis and the drawer inward axis being close
    dotprod = (gripper_forward_axis * drawer_inward_axis).sum(dim=-1)
    rewards += dotprod # Coef here?

    # Bonus for getting closer to the drawer
    approach_bonus = 1 / (1 + torch.linalg.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1))
    rewards += approach_bonus

    # Get reward for pulling out the drawer
    cabinet_dist = cabinet_dof_pos[:, 3]
    rewards += cabinet_dist

    # Penalize when the gripper closes far away
    action_penalty = ((torch.linalg.norm(franka_rfinger_pos-franka_lfinger_pos, dim=-1) < 0.001) & (torch.linalg.norm(franka_grasp_pos - drawer_grasp_pos) > 0.01)).float()
    rewards -= action_penalty

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_ppo6_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # print(drawer_grasp_pos.shape, franka_grasp_pos.shape)
    # TODO: Write your reward function here
    franka_dist_to_drawer = abs(drawer_grasp_pos[:, 0] - franka_grasp_pos[:, 0]) + abs(drawer_grasp_pos[:, 1] - franka_grasp_pos[:, 1]) + abs(drawer_grasp_pos[:, 2] - franka_grasp_pos[:, 2])
    
    rewards = -franka_dist_to_drawer + 10*(cabinet_dof_pos[:, 3] - 0.39)  # successes

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo1_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks.
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment.
    actions,              # Tensor: The actions taken by the agent in each environment.
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper.
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point.
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper.
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point.
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger.
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger.
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper.
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer.
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper.
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer.

    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: Write your reward function here
    # import IPython
    # IPython.embed()

    # Compute distance from franka gripper to the handle
    dist_hand_handle = ((franka_grasp_pos - drawer_grasp_pos)**2).mean(dim=1)#.sum()

    cabinet_open_dof = cabinet_dof_pos[:,3]

    alignment_gripper_franka = torch.logical_and(franka_lfinger_pos[:,2] > drawer_grasp_pos[:,2]+0.01,franka_rfinger_pos[:,2] < drawer_grasp_pos[:,2]-0.01)
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    action_penalty_scale = torch.norm(actions, dim=1)
    # Modify this too
    scale_success = 20
    scale_open_joint = 10
    scale_dist_hand = -3
    scale_alignment = 0.1
    scale_action_penalty = 0
    rewards = scale_dist_hand* dist_hand_handle + scale_open_joint*cabinet_open_dof + successes*scale_success + scale_alignment * alignment_gripper_franka + scale_action_penalty*action_penalty_scale

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo2_reward(
    reset_buf, progress_buf, reset_goal_buf, successes, consecutive_successes, actions, cabinet_dof_pos,
    franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: Write your reward function here
    dist_reward = -1*torch.norm((franka_grasp_pos - drawer_grasp_pos), dim=-1)
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)

    rewards = successes + dist_reward + 10*cabinet_dof_pos[:, 3]

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo3_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (n_envs,).
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment. (n_envs,).
    actions,              # Tensor: The actions taken by the agent in each environment. (n_envs, action_dim).
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper (n_envs, 3).
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point (n_envs, 3).
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper (n_envs, 4).
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point (n_envs, 4)..
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger (n_envs, 3).
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger (n_envs, 3).
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper (n_envs, 3).
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer (n_envs, 3).
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper (n_envs, 3).
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer (n_envs, 3).

    num_envs,

    # You may find these scale helpful
    dist_reward_scale, 
    rot_reward_scale, 
    around_handle_reward_scale, 
    open_reward_scale,
    finger_dist_reward_scale, 
    action_penalty_scale, 
    distX_offset, 

    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
 
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # TODO: Write your reward function here
    mse_pos = (torch.sum((franka_grasp_pos-drawer_grasp_pos)**2, -1))**0.5
    mse_rot = (torch.sum((franka_grasp_rot-drawer_grasp_rot)**2, -1))**0.5
    mse_up_axis = (torch.sum((gripper_up_axis-drawer_up_axis)**2, -1))**0.5

    mse_finger_dist = (torch.sum((franka_lfinger_pos-franka_rfinger_pos)**2, -1))**0.5

    # breakpoint()
    #mse_finger_dist
    
    rewards = successes + 0.1*mse_pos + 0.01*mse_rot + 100*(mse_pos-0.1)*mse_finger_dist

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo4_reward(
    reset_buf, progress_buf, reset_goal_buf, successes, consecutive_successes, actions, cabinet_dof_pos,
    franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: Write your reward function here
    #import ipdb; ipdb.set_trace()
    dv = drawer_grasp_pos - ((franka_lfinger_pos + franka_rfinger_pos)/2)
    distance_rewards = -1* torch.norm(dv, dim = 1)
    av = gripper_forward_axis - drawer_inward_axis
    axis_rewards = -1*torch.norm(av, dim = 1)

    rewards = distance_rewards + axis_rewards
    

    # reset if drawer is open or max length reached
    #print(successes)
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)

    # rewards = successes

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo5_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks. (num_envs,)
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment.
    actions,              # Tensor: The actions taken by the agent in each environment. (num_envs, action_dim)
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper.(num_envs, 3)
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point. (num_envs, 3)
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper. (num_envs, 4)
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point. (num_envs, 4)
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger.
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger.
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper.
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer.
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper.
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer.

    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: Write your reward function here
    # print('cabinet',cabinet_dof_pos[0, 3], 'norm',torch.norm(franka_grasp_pos-drawer_grasp_pos,p=2,dim=1)[0], 'succ',successes[0])
    rewards = 5*cabinet_dof_pos[:, 3] - torch.norm(franka_grasp_pos-drawer_grasp_pos,p=2,dim=1) + 10*successes

    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)
    
    # Modify this too
    # rewards = successes

    return rewards, reset_buf, goal_resets, consecutive_successes


@torch.jit.script
def compute_hepo6_reward(
    reset_buf, progress_buf, reset_goal_buf, 

    successes,            # Tensor: Buffer to track successful completion of tasks.
    consecutive_successes,# Tensor: Buffer to track consecutive successes for each environment.
    actions,              # Tensor: The actions taken by the agent in each environment.
    cabinet_dof_pos,      # Tensor: Degrees of freedom positions for the cabinet (including drawer).
    franka_grasp_pos,     # Tensor: Position of the Franka robot's gripper.
    drawer_grasp_pos,     # Tensor: Position of the drawer's grasp point.
    franka_grasp_rot,     # Tensor: Rotation of the Franka robot's gripper.
    drawer_grasp_rot,     # Tensor: Rotation of the drawer's grasp point.
    franka_lfinger_pos,   # Tensor: Position of the Franka robot's left finger.
    franka_rfinger_pos,   # Tensor: Position of the Franka robot's right finger.
    gripper_forward_axis, # Tensor: Forward axis direction vector of the gripper.
    drawer_inward_axis,   # Tensor: Inward axis direction vector of the drawer.
    gripper_up_axis,      # Tensor: Up axis direction vector of the gripper.
    drawer_up_axis,       # Tensor: Up axis direction vector of the drawer.

    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # TODO: Write your reward function here
    aux_rewards = ((franka_grasp_pos - drawer_grasp_pos)**2).mean(dim=-1) #+ \
        # ((franka_lfinger_pos - drawer_grasp_pos)**2).mean(dim=-1) + ((franka_rfinger_pos - drawer_grasp_pos)**2).mean(dim=-1)     

    # quaternion_distance(franka_grasp_rot, franka_grasp_rot)
    # reset if drawer is open or max length reached
    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    # goal_resets = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes)

    # Modify this too
    # print(aux_rewards)
    rewards = successes + aux_rewards

    return rewards, reset_buf, goal_resets, consecutive_successes

@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos


HUMAN_REWARD_FUNCS = {
    "ppo1": compute_ppo1_reward,
    "ppo2": compute_ppo2_reward,
    "ppo3": compute_ppo3_reward,
    "ppo4": compute_ppo4_reward,
    "ppo5": compute_ppo5_reward,
    "ppo6": compute_ppo6_reward,

    "hepo1": compute_hepo1_reward,
    "hepo2": compute_hepo2_reward,
    "hepo3": compute_hepo3_reward,
    "hepo4": compute_hepo4_reward,
    "hepo5": compute_hepo5_reward,
    "hepo6": compute_hepo6_reward,
}