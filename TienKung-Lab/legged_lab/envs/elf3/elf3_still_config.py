"""Original feedback DWAQ walking with only a zero-command stillness behavior."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackAgentCfg, Elf3DwaqFeedbackEnvCfg
from legged_lab.envs.elf3.standing_rewards import feet_swing_height_when_moving, gait_phase_contact_when_moving
from legged_lab.envs.elf3.still_contract import still_settings
from legged_lab.envs.elf3.stillness import stand_joint_motion


@configclass
class Elf3DwaqStillEnvCfg(Elf3DwaqFeedbackEnvCfg):
    stillness: dict = still_settings()

    def __post_init__(self):
        super().__post_init__()
        self.reward.gait_phase_contact.func = gait_phase_contact_when_moving
        self.reward.gait_phase_contact.params["command_epsilon"] = self.stillness["command_epsilon"]
        self.reward.feet_swing_height.func = feet_swing_height_when_moving
        self.reward.feet_swing_height.params["command_epsilon"] = self.stillness["command_epsilon"]
        self.reward.stand_joint_motion = RewardTermCfg(
            func=stand_joint_motion,
            weight=self.stillness["joint_motion_weight"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
                    "command_epsilon": self.stillness["command_epsilon"]},
        )


@configclass
class Elf3DwaqStillAgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_still"
    wandb_project: str = "elf3_dwaq_still"
    runner_class_name: str = "Elf3DwaqStillRunner"
