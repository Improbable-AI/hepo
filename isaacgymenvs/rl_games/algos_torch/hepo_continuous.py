import copy
import os
import wandb

from rl_games.common import vecenv
from rl_games.common import a2c_common

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time
import gym

from torch import optim
from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
 
from time import sleep

from rl_games.common import common_losses

WANDB_MODE = os.environ.get("WANDB_MODE", "online")
if WANDB_MODE == "offline":
    try:
        import wandb_osh
        from wandb_osh.hooks import TriggerWandbSyncHook
        trigger_sync = TriggerWandbSyncHook()
    except:
        print("Wandb is set as offline, but wandb-osh is not installed. Logs won't be synced online.")
        trigger_sync = lambda: None # Dummy trigger_sync function
else:
    trigger_sync = lambda: None # Dummy trigger_sync function


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class ContinuousHEPOAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        self.use_switch = params.get('use_switch', False)
        self.last_lr = float(self.last_lr)
        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.model_list_all = ['hepo', 'ref']
        self.current_type = 'hepo'
        self.replace_list = ['old_logp_actions', 'actions', 'advantages', 'advantages_mixed', 'advantages_ref']
        self.model_list = [self.current_type] if self.use_switch else self.model_list_all
        if self.has_central_value:
            self.construct_central_value_net()
        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        
        self.states = None
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        
        self.initialize_eipo_train()
        self.initialize_eipo_dataset()
        
    def initialize_eipo_train(self):
        self.model_dict = {}
        self.optimizer_dict = {}
        self.value_mean_std_dict = {}
        self.value_mean_std_int_dict = {}
        self.last_lr_dict = {}
        self.entropy_coef_dict = {}
        self.scheduler_dict = {}
        for model_type in self.model_list_all:
            self.model_dict[model_type] = self.construct_a2c_model()
            self.optimizer_dict[model_type] = optim.Adam(self.model_dict[model_type].parameters(),
                                        self.last_lr, eps=1e-08, weight_decay=self.weight_decay)
            if self.normalize_advantage and self.normalize_rms_advantage:
                self.advantage_mean_std_dict = {'hepo': self.advantage_mean_std, 
                                                'ref': copy.deepcopy(self.advantage_mean_std)}
            if self.normalize_value:
                if self.has_central_value:
                    self.value_mean_std_dict[model_type] = self.central_value_net.model.value_mean_std
                    self.value_mean_std_int_dict[model_type] = self.central_value_net.model.value_mean_std_int
                else:
                    self.value_mean_std_dict[model_type] = self.model_dict[model_type].value_mean_std
                    self.value_mean_std_int_dict[model_type] = self.model_dict[model_type].value_mean_std_int  
            self.last_lr_dict[model_type] = self.last_lr
            self.entropy_coef_dict[model_type] = self.entropy_coef
            self.scheduler_dict[model_type] = copy.deepcopy(self.scheduler)

        self.model = self.model_dict['hepo']
        self.optimizer = self.optimizer_dict['hepo']
        self.init_rnn_from_model(self.model)

    def initialize_eipo_dataset(self):
        self.dataset_dict = {}
        for model_type in self.model_list_all:
            dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, 
                                               self.is_rnn, self.ppo_device, self.seq_length)
            self.dataset_dict[model_type] = dataset
        self.game_rewards = {'hepo': copy.deepcopy(self.game_rewards),
                            'ref': copy.deepcopy(self.game_rewards)}
        self.game_rewards_int = {'hepo': copy.deepcopy(self.game_rewards_int),
                            'ref': copy.deepcopy(self.game_rewards_int)}
        self.game_success = {'hepo': copy.deepcopy(self.game_success),
                            'ref': copy.deepcopy(self.game_success)}
        self.game_shaped_rewards = {'hepo': copy.deepcopy(self.game_shaped_rewards),
                            'ref': copy.deepcopy(self.game_shaped_rewards)}

    def construct_a2c_model(self):
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : self.obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'use_int': self.use_int
        }
        
        model = self.network.build(build_config)
        model.to(self.ppo_device)        
        return model
    
    def construct_central_value_net(self):
        from rl_games.algos_torch import central_value
        cv_config = {
            'state_shape' : self.state_shape, 
            'value_size' : self.value_size,
            'ppo_device' : self.ppo_device, 
            'num_agents' : self.num_agents, 
            'horizon_length' : self.horizon_length,
            'num_actors' : self.num_actors, 
            'num_actions' : self.actions_num, 
            'seq_length' : self.seq_length,
            'normalize_value' : self.normalize_value,
            'network' : self.central_value_config['network'],
            'config' : self.central_value_config, 
            'writter' : self.writer,
            'max_epochs' : self.max_epochs,
            'multi_gpu' : self.multi_gpu,
            'zero_rnn_on_done' : self.zero_rnn_on_done
        }
        self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer = {'hepo': copy.deepcopy(self.experience_buffer),
                                  'ref': copy.deepcopy(self.experience_buffer)}

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            for model_type in self.model_list_all:
                self.experience_buffer[model_type].update_data('obses', n, self.obs['obs'])
                self.experience_buffer[model_type].update_data('dones', n, self.dones)

                for k in update_list:
                    self.experience_buffer[model_type].update_data(k, n, res_dict[model_type][k]) 
                if self.has_central_value:
                    self.experience_buffer[model_type].update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            if self.use_switch:
                adopted_actions = res_dict[self.current_type]['actions']
            else:
                half_bsz = len(self.obs['obs']) // 2
                adopted_actions = torch.cat([res_dict['hepo']['actions'][:half_bsz], 
                                             res_dict['ref']['actions'][half_bsz:]])
            self.obs, rewards, success, self.dones, infos = self.env_step(adopted_actions)
            rewards_ext, rewards_int = rewards['ext'], rewards['int']
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            for model_type in self.model_list_all:
                shaped_rewards = self.rewards_shaper(rewards_ext)
                if self.value_bootstrap and 'time_outs' in infos:
                    shaped_rewards += self.gamma * res_dict[model_type]['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

                self.experience_buffer[model_type].update_data('rewards', n, shaped_rewards)
                if self.use_int:
                    shaped_rewards_int = self.rewards_shaper(rewards_int)
                    if self.value_bootstrap and 'time_outs' in infos:
                        shaped_rewards_int += self.gamma * res_dict[model_type]['values_int'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

                    self.experience_buffer[model_type].update_data('rewards_int', n, shaped_rewards_int)
                
            self.current_rewards += rewards_ext
            self.current_rewards_int += rewards_int
            self.current_success += success
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
            if not self.use_switch:
                half_bsz = len(self.dones) // 2
                env_done_indices_dict = {'hepo': [], 'ref': []}
                for i in env_done_indices:
                    if i < half_bsz:
                        env_done_indices_dict['hepo'].append(i)
                    else:
                        env_done_indices_dict['ref'].append(i)
                for model_type in self.model_list_all:
                    env_done_indices_dict[model_type] = torch.LongTensor(env_done_indices_dict[model_type])
                    env_done_indices_dict[model_type] = env_done_indices_dict[model_type].view(len(env_done_indices_dict[model_type]), 1)
            else:
                env_done_indices_dict = {'hepo': env_done_indices, 'ref': env_done_indices}
            for model_type in self.model_list_all:
                self.game_rewards[model_type].update(self.current_rewards[env_done_indices_dict[model_type]])
                self.game_rewards_int[model_type].update(self.current_rewards_int[env_done_indices_dict[model_type]])
                self.game_success[model_type].update(self.current_success[env_done_indices_dict[model_type]])
                self.game_shaped_rewards[model_type].update(self.current_shaped_rewards[env_done_indices_dict[model_type]])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_rewards_int = self.current_rewards_int * not_dones.unsqueeze(1)
            self.current_success = self.current_success * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer['hepo'].tensor_dict['dones'].float()
        batch_dict = {}
        for model_type in self.model_list_all:
            batch_dict[model_type] = {}
            mb_returns, mb_returns_int = self.compute_returns(fdones, last_values[model_type], 
                                                              mb_fdones, 
                                                              self.experience_buffer[model_type])
            batch_dict[model_type] = self.experience_buffer[model_type].get_transformed_list(
                swap_and_flatten01, self.tensor_list)
            batch_dict[model_type]['returns'] = swap_and_flatten01(mb_returns)
            batch_dict[model_type]['returns_int'] = swap_and_flatten01(mb_returns_int)
            batch_dict[model_type]['played_frames'] = self.batch_size
            batch_dict[model_type]['step_time'] = step_time

        return batch_dict

    def reset_gradients(self):
        for model_type in self.model_list_all:
            model = self.model_dict[model_type]
            optimizer = self.optimizer_dict[model_type]
            
            if self.multi_gpu:
                optimizer.zero_grad()
            else:
                for param in model.parameters():
                    param.grad = None

    def trancate_gradients_and_step(self):
        for model_type in self.model_list_all:
            model = self.model_dict[model_type]
            optimizer = self.optimizer_dict[model_type]
            if self.multi_gpu:
                # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                all_grads_list = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_grads_list.append(param.grad.view(-1))

                all_grads = torch.cat(all_grads_list)
                dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                offset = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.copy_(
                            all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                        )
                        offset += param.numel()

            if self.truncate_grads:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)

            self.scaler.step(optimizer)
        self.scaler.update()

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, 
                    a_losses, c_losses, entropies, kls, last_lr, lr_mul, 
                    frame, scaled_time, scaled_play_time, curr_frames, **kwargs):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        time_record = {'performance/step_inference_rl_update_fps': curr_frames / scaled_time,
                        'performance/step_inference_fps': curr_frames / scaled_play_time, 
                        'performance/step_fps': curr_frames / step_time,
                        'performance/rl_update_time': update_time,
                        'performance/step_inference_time': play_time,
                        'performance/step_time': step_time}
        wandb.log(time_record, step=epoch_num)
        # self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        # self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        # self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        # self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        # self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        # self.writer.add_scalar('performance/step_time', step_time, frame)
        # self.writer.add_scalar('info/epochs', epoch_num, frame)
        loss_record = {}
        for model_type in self.model_list_all:
            loss_record.update({f'losses/{model_type}_a_loss': torch_ext.mean_list(a_losses[model_type]).item(),
            f'losses/{model_type}_c_loss': torch_ext.mean_list(c_losses[model_type]).item(),
            f'losses/{model_type}_entropy': torch_ext.mean_list(entropies[model_type]).item(),
            f'info/{model_type}_e_clip': self.e_clip * lr_mul[model_type],
            f'info/{model_type}_last_lr': last_lr[model_type] * lr_mul[model_type],
            f'info/{model_type}_lr_mul': lr_mul[model_type],
            # to be removed
            f'info/{model_type}kl': torch_ext.mean_list(kls[model_type]).item(),
            })
        wandb.log(loss_record, step=epoch_num)
        trigger_sync()
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        for model_type in self.model_list_all:
            self.model_dict[model_type].eval()
            if self.normalize_rms_advantage:
                self.advantage_mean_std_dict[model_type].eval()
                self.advantage_mean_std_int_dict[model_type].eval()

    def set_train(self):
        for model_type in self.model_list_all:
            self.model_dict[model_type].train()
            if self.normalize_rms_advantage:
                self.advantage_mean_std_dict[model_type].train()
                self.advantage_mean_std_int_dict[model_type].train()
    
    def update_lr(self, model_type):
        optimizer = self.optimizer_dict[model_type]
        lr = self.last_lr_dict[model_type]

        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        with torch.no_grad():
            res_dict = {}
            input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
            }
            for model_type in self.model_list_all:
                self.model_dict[model_type].eval()
                res_dict[model_type] = \
                    self.model_dict[model_type](input_dict)
            
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                for model_type in self.model_list_all:
                    res_dict[model_type]['values'] = value
            return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
                value_int = None
                values_dict = {'ext': value, 'int': value_int}
            else:
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
                }
                res_dict = {}
                for model_type in self.model_list_all:
                    self.model_dict[model_type].eval()
                    res_dict[model_type] = \
                        self.model_dict[model_type](input_dict)
                values_dict = {}
                for model_type in self.model_list_all:
                    values_dict[model_type] = {'ext': res_dict[model_type]['values'],
                                               'int': res_dict[model_type]['values_int']}
            return values_dict

    def process_kls(self, kl):
        av_kl = kl.clone()
        if self.multi_gpu:
            dist.all_reduce(av_kl, op=dist.ReduceOp.SUM)
            av_kl /= self.world_size
        av_kl = av_kl
        return av_kl
                            
    def schedule(self, av_kl, model_type):
        self.last_lr_dict[model_type], self.entropy_coef_dict[model_type] = \
            self.scheduler_dict[model_type].update(self.last_lr_dict[model_type], 
                                    self.entropy_coef_dict[model_type], 
                                self.epoch_num, 0, av_kl.item())
        self.update_lr(model_type)
    
    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict[self.current_type].get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict[self.current_type].pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = {'hepo': [], 'ref': []}
        c_losses = {'hepo': [], 'ref': []}
        b_losses = {'hepo': [], 'ref': []}
        entropies = {'hepo': [], 'ref': []}
        kls = {'hepo': [], 'ref': []}

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = {'hepo': [], 'ref': []}
            for i in range(len(self.dataset_dict[self.current_type])):
                batch_input = {model_type: self.dataset_dict[model_type][i] \
                                   for model_type in self.model_list_all}
                losses_dict, stat_info_dict = self.train_actor_critic(batch_input)
                for model_type in self.model_list_all:
                    a_losses[model_type].append(losses_dict[model_type]['a_loss'])
                    c_losses[model_type].append(losses_dict[model_type]['c_loss'])
                    ep_kls[model_type].append(stat_info_dict[model_type]['kl'])
                    entropies[model_type].append(stat_info_dict[model_type]['entropy'])
                    if self.bounds_loss_coef is not None:
                        b_losses[model_type].append(losses_dict[model_type]['b_loss'])

                    self.dataset_dict[model_type].update_mu_sigma(
                        stat_info_dict[model_type]['cmu'], 
                        stat_info_dict[model_type]['csigma'])
                    if self.schedule_type == 'legacy':
                        self.schedule(self.process_kls(stat_info_dict[model_type]['kl']), model_type)
            for model_type in self.model_list_all:
                av_kl = torch_ext.mean_list(ep_kls[model_type])   
                av_kl = self.process_kls(av_kl)   
                if self.schedule_type == 'standard':
                    self.schedule(av_kl, model_type)
                kls[model_type].append(av_kl)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                for model_type in self.model_list_all:
                    self.model_dict[model_type].running_mean_std.eval()

        update_time_end = time.time()
        step_time = batch_dict['hepo']['step_time']
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        
        times = {'step_time': step_time, 'play_time': play_time, 
                 'update_time': update_time, 'sum_time': total_time}
        losses = {}
        stat_info = {}
        losses['a_losses'] = {model_type: a_losses[model_type] for model_type in self.model_list_all}
        losses['b_losses'] = {model_type: b_losses[model_type] for model_type in self.model_list_all}
        losses['c_losses'] = {model_type: c_losses[model_type] for model_type in self.model_list_all} 
        stat_info['entropies'] = {model_type: entropies[model_type] \
                                  for model_type in self.model_list_all}
        stat_info['kls'] = {model_type: kls[model_type] \
                            for model_type in self.model_list_all}
        stat_info['last_lr'] = {model_type: stat_info_dict[model_type]['last_lr'] \
                                for model_type in self.model_list_all}
        stat_info['lr_mul'] = {model_type: stat_info_dict[model_type]['lr_mul'] \
                               for model_type in self.model_list_all}
        max_eps_leng = self.vec_env.env.max_episode_length
        eps_leng = None
        if self.game_lengths.current_size > 0:
            eps_leng = self.game_lengths.get_mean()
            max_eps_leng = self.vec_env.env.max_episode_length
        self.lgrgn_mtpr.update_alpha_values(eps_leng, max_eps_leng)
        if self.use_switch:
            self.current_type = self.lgrgn_mtpr.rollout_policy
        return times, losses, stat_info

    def process_batch_dict(self, batch_dict, model_type):
        obses = batch_dict['obses']
        rewards = batch_dict['rewards']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        rewards_int = batch_dict.get('rewards_int', None)
        returns_int = batch_dict.get('returns_int', None)
        values_int = batch_dict.get('values_int', None)
        advantages_int = None

        advantages = returns - values
        if returns_int is not None and values_int is not None:
            advantages_int = returns_int - values_int

        if self.normalize_value:
            self.value_mean_std_dict[model_type].train()
            values = self.value_mean_std_dict[model_type](values)
            returns = self.value_mean_std_dict[model_type](returns)
            self.value_mean_std_dict[model_type].eval()
            if returns_int is not None and values_int is not None:
                self.value_mean_std_int_dict[model_type].train()
                values_int = self.value_mean_std_int_dict[model_type](values_int)
                returns_int = self.value_mean_std_int_dict[model_type](returns_int)
                self.value_mean_std_int_dict[model_type].eval()

        advantages = torch.sum(advantages, axis=1)
        if advantages_int is not None:
            advantages_int = torch.sum(advantages_int, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std_dict[model_type](advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std_dict[model_type](advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            if advantages_int is not None:
                if self.is_rnn:
                    if self.normalize_rms_advantage:
                        advantages_int = self.advantage_mean_std_int_dict[model_type](advantages_int, mask=rnn_masks)
                    else:
                        advantages_int = torch_ext.normalization_with_masks(advantages_int, rnn_masks)
                else:
                    if self.normalize_rms_advantage:
                        advantages_int = self.advantage_mean_std_int_dict[model_type](advantages_int)
                    else:
                        advantages_int = (advantages_int - advantages_int.mean()) / (advantages_int.std() + 1e-8)

        if advantages_int is not None:
            advantages_mixed, advantages_ref = self.lgrgn_mtpr.compute_advantages(advantages, advantages_int, 
                        rewards, model_type=model_type)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        if values_int is not None:
            dataset_dict['old_values_int'] = values_int
        if advantages_int is not None:
            dataset_dict['advantages_mixed'] = advantages_mixed
            dataset_dict['advantages_ref'] = advantages_ref
        if returns_int is not None:
            if rewards_int.sum().item() == 0:
                returns_int = torch.zeros(returns_int.shape).to(self.ppo_device)
            dataset_dict['returns_int'] = returns_int
        return dataset_dict

    def generate_value_masks(self):
        value_mask_dict = {}
        if self.use_switch:
            for model_type in self.model_list_all:
                if model_type == self.current_type:
                    value_mask_dict[model_type] = torch.ones(
                        self.experience_buffer[model_type].tensor_dict['values'].shape).to(\
                            self.ppo_device)
                else:
                    value_mask_dict[model_type] = torch.zeros(
                        self.experience_buffer[model_type].tensor_dict['values'].shape).to(\
                            self.ppo_device)
        else:
            half_bsz = len(self.obs['obs']) // 2
            for model_type in self.model_list_all:
                value_mask_dict[model_type] = torch.ones(
                        self.experience_buffer[model_type].tensor_dict['values'].shape).to(\
                            self.ppo_device) * 2
            value_mask_dict['hepo'][:, half_bsz:] = 0.
            value_mask_dict['ref'][:, :half_bsz] = 0.

        for model_type in self.model_list_all:
            value_mask_dict[model_type] = swap_and_flatten01(value_mask_dict[model_type])
        return value_mask_dict
    
    def prepare_dataset(self, batch_dicts):
        dataset_dicts = {}
        sample_mask = self.generate_value_masks()
        for model_type, batch_dict in batch_dicts.items():
            dataset_dicts[model_type] = self.process_batch_dict(batch_dict, model_type)
            dataset_dicts[model_type]['value_masks'] = sample_mask[model_type]
        if self.use_switch:
            for model_type, batch_dict in batch_dicts.items():
                if model_type != self.current_type:
                    for key in self.replace_list:
                        dataset_dicts[model_type][key] = \
                            dataset_dicts[self.current_type][key].clone()                
        else:
            for key in self.replace_list:
                half_bsz = len(dataset_dicts['ref'][key]) // 2
                dataset_dicts['ref'][key][:half_bsz] = \
                    dataset_dicts['hepo'][key][:half_bsz]
                dataset_dicts['hepo'][key][half_bsz:] = \
                    dataset_dicts['ref'][key][half_bsz:]
        for model_type in self.model_list_all:    
            self.dataset_dict[model_type].update_values_dict(dataset_dicts[model_type])

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_losses(self, input_dict, model_type):
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        model = self.model_dict[model_type]
        last_lr = self.last_lr_dict[model_type]
        entropy_coef = self.entropy_coef_dict[model_type]

        value_preds_batch = input_dict['old_values']
        value_masks = input_dict['value_masks']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages_mixed'] if model_type == 'hepo' \
            else input_dict['advantages_ref']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                if 'values_int' in res_dict:
                    c_loss = c_loss + common_losses.critic_loss(model, input_dict['old_values_int'], res_dict['values_int'], 
                                                            curr_e_clip, input_dict['returns_int'], self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * entropy_coef + b_loss * self.bounds_loss_coef
        
        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      
        losses = {'a_loss': a_loss, 'b_loss': b_loss, 'c_loss': c_loss, 'total_loss': loss}
        stat_info = {'entropy': entropy, 'kl': kl_dist, 
                     'last_lr': last_lr, 'lr_mul': lr_mul, 
                     'cmu': mu.detach(), 'csigma': sigma.detach()}
        return losses, stat_info

    def train_actor_critic(self, input_dict):
        self.reset_gradients()
        losses_dict = {}
        stat_info_dict = {}
        loss = 0
        for model_type in self.model_list_all:
            losses, stat_info = self.calc_losses(input_dict[model_type], model_type)
            losses_dict[model_type] = losses
            stat_info_dict[model_type] = stat_info
            # print(model_type, losses['total_loss'].item())
            loss = loss + losses['total_loss']
        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()
        return losses_dict, stat_info_dict

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        if not self.use_swith:
            for model_type in self.model_list_all:
                self.game_rewards[model_type].clear()
                self.game_rewards_int[model_type].clear()
                self.game_success[model_type].clear()
                self.game_shaped_rewards[model_type].clear()
        else:
            self.game_rewards[self.current_type].clear()
            self.game_rewards_int[self.current_type].clear()
            self.game_success[self.current_type].clear()
            self.game_shaped_rewards[self.current_type].clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()
    
    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            times, losses, stat_info = self.train_epoch()
            total_time += times['sum_time']
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            for model_type in self.model_list_all:
                self.dataset_dict[model_type].update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * times['sum_time']
                scaled_play_time = self.num_agents * times['play_time']
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, times['step_time'], scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)
                self.write_stats(total_time=total_time, epoch_num=epoch_num, 
                                 frame=frame, curr_frames=curr_frames,
                                scaled_time=scaled_time, scaled_play_time=scaled_play_time, 
                                **times, **losses, **stat_info)

                reward_record = {}
                for model_type in self.model_list_all:
                    if len(losses['b_losses'][model_type]) > 0:
                        wandb.log({f'losses/{model_type}_bounds_loss': torch_ext.mean_list(losses['b_losses'][model_type]).item()}, 
                                    step=epoch_num)
                        # self.writer.add_scalar(f'losses/{model_type}_bounds_loss', 
                        #                        torch_ext.mean_list(losses['b_losses'][model_type]).item(), frame)
                    if self.has_soft_aug:
                        wandb.log({f'losses/{model_type}_aug_loss': np.mean(aug_losses)}, 
                                    step=epoch_num)
                        # self.writer.add_scalar(f'losses/{model_type}_aug_loss', np.mean(aug_losses), frame)
                    
                    if self.game_rewards[model_type].current_size > 0:
                        mean_rewards = self.game_rewards[model_type].get_mean()
                        mean_rewards_int = self.game_rewards_int[model_type].get_mean()
                        mean_success = self.game_success[model_type].get_mean()
                        mean_shaped_rewards = self.game_shaped_rewards[model_type].get_mean()
                        if model_type == self.current_type:
                            self.mean_rewards = mean_rewards[0]
                            reward_record['episode_lengths/step'] = self.game_lengths.get_mean()
                            # reward_record['eipo_info/complete_rate'] = self.lgrgn_mtpr.complete_rate
                        if model_type in self.lgrgn_mtpr.alpha_grad:
                            reward_record.update({f'eipo_info/{model_type}_alpha_value': self.lgrgn_mtpr.alpha[model_type].item(), 
                                        f'eipo_info/{model_type}_alpha_gradient': self.lgrgn_mtpr.alpha_grad[model_type].item()})
                       
                        for i in range(self.value_size):
                            rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                            reward_record.update({f'{rewards_name}/{model_type}_ext': mean_rewards[i],
                                    f'{rewards_name}/{model_type}_int': mean_rewards_int[i],
                                    f'{rewards_name}/{model_type}_metric': mean_success[i], 
                                    f'shaped_{rewards_name}/{model_type}': mean_shaped_rewards[i]})

                wandb.log(reward_record, step=epoch_num)
                trigger_sync()
                
                if self.has_self_play_config:
                    self.self_play_manager.update(self)

                checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(self.mean_rewards)

                if self.save_freq > 0:
                    if epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if self.mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', self.mean_rewards)
                    self.last_mean_rewards = self.mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))

                    if 'score_to_win' in self.config:
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Maximum reward achieved. Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards[self.current_type].current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards[self.current_type].current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
