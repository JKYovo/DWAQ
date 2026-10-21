"""Opt-in feedback experiment; the registered 0918 robot/rewards are unchanged."""
from isaaclab.utils import configclass

from legged_lab.envs.elf3.elf3_dwaq_config import (
    Elf3DwaqEnvCfg,
    Elf3DwaqUpperBodySymmetryPoseAgentCfg,
)


@configclass
class Elf3DwaqFeedbackEnvCfg(Elf3DwaqEnvCfg):
    velocity_feedback: dict = {
        "mode": "lpf20",
        "cutoff_hz": 20.0,
        "model_version": "joint_velocity_tustin_v1",
        "latest_critic_contacts": True,
    }


@configclass
class Elf3DwaqConsistentEnvCfg(Elf3DwaqFeedbackEnvCfg):
    velocity_feedback: dict = {
        "mode": "ideal",
        "cutoff_hz": 20.0,
        "model_version": "joint_velocity_tustin_v1",
        "latest_critic_contacts": True,
    }


@configclass
class Elf3DwaqFeedbackAgentCfg(Elf3DwaqUpperBodySymmetryPoseAgentCfg):
    experiment_name: str = "elf3_dwaq_feedback_v1"
    wandb_project: str = "elf3_dwaq_feedback_v1"
    runner_class_name: str = "Elf3DwaqFeedbackRunner"
    max_iterations: int = 100001
    save_interval: int = 500

    def __post_init__(self):
        super().__post_init__()
        self.policy.context_mode = "mean"


@configclass
class Elf3DwaqConsistentAgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_consistent_v1"
    wandb_project: str = "elf3_dwaq_consistent_v1"
