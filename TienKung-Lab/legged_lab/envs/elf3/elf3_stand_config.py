"""Preserve feedback-v1 walking, disabling only two gait rewards at zero command."""
from isaaclab.utils import configclass

from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackAgentCfg, Elf3DwaqFeedbackEnvCfg
from legged_lab.envs.elf3.standing_contract import COMMAND_EPSILON, MODE, VERSION
from legged_lab.envs.elf3.standing_rewards import (
    feet_swing_height_when_moving,
    gait_phase_contact_when_moving,
)


@configclass
class Elf3DwaqStandEnvCfg(Elf3DwaqFeedbackEnvCfg):
    standing_experiment: dict = {
        "version": VERSION,
        "mode": MODE,
        "command_epsilon": COMMAND_EPSILON,
    }

    def __post_init__(self):
        super().__post_init__()
        # Elf3DwaqEnvCfg already deep-copies the reward tree per instance.
        self.reward.gait_phase_contact.func = gait_phase_contact_when_moving
        self.reward.gait_phase_contact.params["command_epsilon"] = COMMAND_EPSILON
        self.reward.feet_swing_height.func = feet_swing_height_when_moving
        self.reward.feet_swing_height.params["command_epsilon"] = COMMAND_EPSILON


@configclass
class Elf3DwaqStandAgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_stand_v1"
    wandb_project: str = "elf3_dwaq_stand_v1"
    runner_class_name: str = "Elf3DwaqStandRunner"
