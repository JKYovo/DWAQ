"""Original still-task behavior with stronger zero leg pose and waist pose.

The 102500 reference used -0.02 baseline plus -0.08 extra on four leg joints.
This independent scratch task uses -0.13 extra for -0.15 total at zero, and
-0.10 extra on the three waist joints for -0.30 total at zero. All original
walking rewards, stillness/phase preprocessing, physical config and PPO stay
inherited from elf3_dwaq_still; no previous model is a training initializer.
"""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.envs.elf3.elf3_still_config import Elf3DwaqStillAgentCfg, Elf3DwaqStillEnvCfg
from legged_lab.envs.elf3.still_pose_rewards import STAND_POSE_JOINT_NAMES, stand_leg_pose_l1
from legged_lab.envs.elf3.still_v2_contract import pose_settings
from legged_lab.envs.elf3.still_v2_rewards import STAND_WAIST_JOINT_NAMES, stand_waist_pose_l1


@configclass
class Elf3DwaqStillV2EnvCfg(Elf3DwaqStillEnvCfg):
    still_v2_pose: dict = pose_settings()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16384
        settings = self.still_v2_pose
        self.reward.stand_leg_pose = RewardTermCfg(
            func=stand_leg_pose_l1, weight=settings["leg_incremental_weight"], params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(STAND_POSE_JOINT_NAMES), preserve_order=True),
                "command_epsilon": settings["command_epsilon"],
            })
        self.reward.stand_waist_pose = RewardTermCfg(
            func=stand_waist_pose_l1, weight=settings["waist_incremental_weight"], params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(STAND_WAIST_JOINT_NAMES), preserve_order=True),
                "command_epsilon": settings["command_epsilon"],
            })


@configclass
class Elf3DwaqStillV2AgentCfg(Elf3DwaqStillAgentCfg):
    experiment_name: str = "elf3_dwaq_still_v2"
    wandb_project: str = "elf3_dwaq_still_v2"
    runner_class_name: str = "Elf3DwaqStillV2Runner"
