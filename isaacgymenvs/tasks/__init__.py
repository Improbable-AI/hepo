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

from .ant import Ant
from .anymal import Anymal
from .anymal_terrain import AnymalTerrain
from .ball_balance import BallBalance
from .cartpole import Cartpole 
from .factory.factory_task_gears import FactoryTaskGears
from .factory.factory_task_insertion import FactoryTaskInsertion
from .factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
from .factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
from .factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
from .franka_cabinet import FrankaCabinet
from .franka_cube_stack import FrankaCubeStack
from .humanoid import Humanoid
from .humanoid_amp import HumanoidAMP
from .ingenuity import Ingenuity
from .quadcopter import Quadcopter
from .shadow_hand import ShadowHand
from .allegro_hand import AllegroHand
from .dextreme.allegro_hand_dextreme import AllegroHandDextremeManualDR, AllegroHandDextremeADR
from .trifinger import Trifinger

from .shadow_hand_spin import ShadowHandSpin
from .shadow_hand_upside_down import ShadowHandUpsideDown
from .shadow_hand_block_stack import ShadowHandBlockStack
from .shadow_hand_bottle_cap import ShadowHandBottleCap
from .shadow_hand_catch_abreast import ShadowHandCatchAbreast
from .shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from .shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from .shadow_hand_door_close_inward import ShadowHandDoorCloseInward
from .shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
from .shadow_hand_door_open_inward import ShadowHandDoorOpenInward
from .shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
from .shadow_hand_grasp_and_place import ShadowHandGraspAndPlace
from .shadow_hand_kettle import ShadowHandKettle
from .shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from .shadow_hand_over import ShadowHandOver
from .shadow_hand_pen import ShadowHandPen
from .shadow_hand_push_block import ShadowHandPushBlock
from .shadow_hand_re_orientation import ShadowHandReOrientation
from .shadow_hand_scissors import ShadowHandScissors
from .shadow_hand_swing_cup import ShadowHandSwingCup
from .shadow_hand_switch import ShadowHandSwitch
from .shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarm

from .anymal_orig import Anymal_orig
from .ingenuity_orig import Ingenuity_orig
from .quadcopter_orig import Quadcopter_orig
from .franka_cabinet_orig import FrankaCabinet_orig
from .franka_cube_stack_orig import FrankaCubeStack_orig
from .shadow_hand_spin_orig import ShadowHandSpin_orig
from .shadow_hand_upside_down_orig import ShadowHandUpsideDown_orig
from .shadow_hand_block_stack_orig import ShadowHandBlockStack_orig
from .shadow_hand_bottle_cap_orig import ShadowHandBottleCap_orig
from .shadow_hand_catch_abreast_orig import ShadowHandCatchAbreast_orig
from .shadow_hand_catch_over2underarm_orig import ShadowHandCatchOver2Underarm_orig
from .shadow_hand_catch_underarm_orig import ShadowHandCatchUnderarm_orig
from .shadow_hand_door_close_inward_orig import ShadowHandDoorCloseInward_orig
from .shadow_hand_door_close_outward_orig import ShadowHandDoorCloseOutward_orig
from .shadow_hand_door_open_inward_orig import ShadowHandDoorOpenInward_orig
from .shadow_hand_door_open_outward_orig import ShadowHandDoorOpenOutward_orig
from .shadow_hand_grasp_and_place_orig import ShadowHandGraspAndPlace_orig
from .shadow_hand_kettle_orig import ShadowHandKettle_orig
from .shadow_hand_lift_underarm_orig import ShadowHandLiftUnderarm_orig
from .shadow_hand_over_orig import ShadowHandOver_orig
from .shadow_hand_pen_orig import ShadowHandPen_orig
from .shadow_hand_push_block_orig import ShadowHandPushBlock_orig
from .shadow_hand_re_orientation_orig import ShadowHandReOrientation_orig
from .shadow_hand_scissors_orig import ShadowHandScissors_orig
from .shadow_hand_swing_cup_orig import ShadowHandSwingCup_orig
from .shadow_hand_switch_orig import ShadowHandSwitch_orig
from .shadow_hand_two_catch_underarm_orig import ShadowHandTwoCatchUnderarm_orig

from .allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
from .allegro_kuka.allegro_kuka_regrasping import AllegroKukaRegrasping
from .allegro_kuka.allegro_kuka_throw import AllegroKukaThrow
from .allegro_kuka.allegro_kuka_two_arms_regrasping import AllegroKukaTwoArmsRegrasping
from .allegro_kuka.allegro_kuka_two_arms_reorientation import AllegroKukaTwoArmsReorientation

from .industreal.industreal_task_pegs_insert import IndustRealTaskPegsInsert
from .industreal.industreal_task_gears_insert import IndustRealTaskGearsInsert


def resolve_allegro_kuka(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaReorientation,
        throw=AllegroKukaThrow,
        regrasping=AllegroKukaRegrasping,
    )

    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)

def resolve_allegro_kuka_two_arms(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaTwoArmsReorientation,
        regrasping=AllegroKukaTwoArmsRegrasping,
    )

    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)


# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "AllegroKuka": resolve_allegro_kuka,
    "AllegroKukaTwoArms": resolve_allegro_kuka_two_arms,
    "AllegroHandManualDR": AllegroHandDextremeManualDR,
    "AllegroHandADR": AllegroHandDextremeADR,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FactoryTaskGears": FactoryTaskGears,
    "FactoryTaskInsertion": FactoryTaskInsertion,
    "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
    "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
    "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
    "IndustRealTaskPegsInsert": IndustRealTaskPegsInsert,
    "IndustRealTaskGearsInsert": IndustRealTaskGearsInsert,
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "Trifinger": Trifinger,
    "ShadowHand": ShadowHand,
    "ShadowHandSpin": ShadowHandSpin, 
    "ShadowHandUpsideDown": ShadowHandUpsideDown,
    "ShadowHandBlockStack": ShadowHandBlockStack,
    "ShadowHandBottleCap": ShadowHandBottleCap,
    "ShadowHandCatchAbreast": ShadowHandCatchAbreast,
    "ShadowHandCatchOver2Underarm": ShadowHandCatchOver2Underarm,
    "ShadowHandCatchUnderarm": ShadowHandCatchUnderarm,
    "ShadowHandDoorCloseInward": ShadowHandDoorCloseInward,
    "ShadowHandDoorCloseOutward": ShadowHandDoorCloseOutward,
    "ShadowHandDoorOpenInward": ShadowHandDoorOpenInward,
    "ShadowHandDoorOpenOutward": ShadowHandDoorOpenOutward,
    "ShadowHandGraspAndPlace": ShadowHandGraspAndPlace,
    "ShadowHandKettle": ShadowHandKettle,
    "ShadowHandLiftUnderarm": ShadowHandLiftUnderarm,
    "ShadowHandOver": ShadowHandOver,
    "ShadowHandPen": ShadowHandPen,
    "ShadowHandPushBlock": ShadowHandPushBlock,
    "ShadowHandReOrientation": ShadowHandReOrientation,
    "ShadowHandScissors": ShadowHandScissors,
    "ShadowHandSwingCup": ShadowHandSwingCup,
    "ShadowHandSwitch": ShadowHandSwitch,
    "ShadowHandTwoCatchUnderarm": ShadowHandTwoCatchUnderarm,
    "FrankaCabinet_orig": FrankaCabinet_orig,
    "FrankaCubeStack_orig": FrankaCubeStack_orig,
    "Anymal_orig": Anymal_orig,
    "Ingenuity_orig": Ingenuity_orig,
    "Quadcopter_orig": Quadcopter_orig,
    "ShadowHandSpin_orig": ShadowHandSpin_orig, 
    "ShadowHandUpsideDown_orig": ShadowHandUpsideDown_orig,
    "ShadowHandBlockStack_orig": ShadowHandBlockStack_orig,
    "ShadowHandBottleCa_orig": ShadowHandBottleCap_orig,
    "ShadowHandCatchAbreast_orig": ShadowHandCatchAbreast_orig,
    "ShadowHandCatchOver2Underarm_orig": ShadowHandCatchOver2Underarm_orig,
    "ShadowHandCatchUnderarm_orig": ShadowHandCatchUnderarm_orig,
    "ShadowHandDoorCloseInward_orig": ShadowHandDoorCloseInward_orig,
    "ShadowHandDoorCloseOutward_orig": ShadowHandDoorCloseOutward_orig,
    "ShadowHandDoorOpenInward_orig": ShadowHandDoorOpenInward_orig,
    "ShadowHandDoorOpenOutward_orig": ShadowHandDoorOpenOutward_orig,
    "ShadowHandGraspAndPlace_orig": ShadowHandGraspAndPlace_orig,
    "ShadowHandKettle_orig": ShadowHandKettle_orig,
    "ShadowHandLiftUnderarm_orig": ShadowHandLiftUnderarm_orig,
    "ShadowHandOver_orig": ShadowHandOver_orig,
    "ShadowHandPen_orig": ShadowHandPen_orig,
    "ShadowHandPushBlock_orig": ShadowHandPushBlock_orig,
    "ShadowHandReOrientation_orig": ShadowHandReOrientation_orig,
    "ShadowHandScissors_orig": ShadowHandScissors_orig,
    "ShadowHandSwingCup_orig": ShadowHandSwingCup_orig,
    "ShadowHandSwitch_orig": ShadowHandSwitch_orig,
    "ShadowHandTwoCatchUnderarm_orig": ShadowHandTwoCatchUnderarm_orig,
}