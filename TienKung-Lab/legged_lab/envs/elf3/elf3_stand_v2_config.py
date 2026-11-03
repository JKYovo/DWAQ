"""Opt-in walking task with immediate zero-command arm-stillness objectives."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackAgentCfg, Elf3DwaqFeedbackEnvCfg
from legged_lab.envs.elf3.quiet_contract import COMMAND_EPSILON, quiet_settings
from legged_lab.envs.elf3.quiet_rewards import upper_body_stillness
from legged_lab.envs.elf3.standing_rewards import feet_swing_height_when_moving, gait_phase_contact_when_moving


@configclass
class Elf3DwaqStandV2EnvCfg(Elf3DwaqFeedbackEnvCfg):
    quiet_standing: dict = quiet_settings()

    def __post_init__(self):
        super().__post_init__()
        self.reward.gait_phase_contact.func = gait_phase_contact_when_moving
        self.reward.gait_phase_contact.params["command_epsilon"] = COMMAND_EPSILON
        self.reward.feet_swing_height.func = feet_swing_height_when_moving
        self.reward.feet_swing_height.params["command_epsilon"] = COMMAND_EPSILON
        self.reward.upper_body_stillness = RewardTermCfg(
            func=upper_body_stillness, weight=0.3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES[15:]), preserve_order=True),
                    "velocity_std": self.quiet_standing["physical"]["velocity_std"]},
        )


@configclass
class Elf3DwaqStandV2AgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_stand_v2"
    wandb_project: str = "elf3_dwaq_stand_v2"
    runner_class_name: str = "Elf3DwaqStandV2Runner"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "Elf3DwaqStandPPO"
