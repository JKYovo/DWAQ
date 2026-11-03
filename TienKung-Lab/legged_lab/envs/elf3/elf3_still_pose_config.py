"""Keep the original still task; strengthen default leg pose only at zero command."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.envs.elf3.elf3_still_config import Elf3DwaqStillAgentCfg, Elf3DwaqStillEnvCfg
from legged_lab.envs.elf3.still_pose_contract import pose_settings
from legged_lab.envs.elf3.still_pose_rewards import STAND_POSE_JOINT_NAMES, stand_leg_pose_l1


@configclass
class Elf3DwaqStillPoseEnvCfg(Elf3DwaqStillEnvCfg):
    zero_command_pose: dict = pose_settings()

    def __post_init__(self):
        super().__post_init__()
        self.reward.stand_leg_pose = RewardTermCfg(
            func=stand_leg_pose_l1,
            # The inherited -0.02 remains: -0.02 + -0.08 = -0.10, not -0.12.
            weight=self.zero_command_pose["incremental_weight"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(STAND_POSE_JOINT_NAMES),
                                                preserve_order=True),
                    "command_epsilon": self.zero_command_pose["command_epsilon"]},
        )


@configclass
class Elf3DwaqStillPoseAgentCfg(Elf3DwaqStillAgentCfg):
    experiment_name: str = "elf3_dwaq_still_pose_v1"
    wandb_project: str = "elf3_dwaq_still_pose_v1"
    runner_class_name: str = "Elf3DwaqStillPoseRunner"
