"""Recommended ELF3 feedback task with a replacement zero-command objective.

This derives directly from feedback_v1, NOT the leg-only still_pose continuation
or an older stand_v2 experiment. Moving rewards and learning settings are intact.
"""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackAgentCfg, Elf3DwaqFeedbackEnvCfg
from legged_lab.envs.elf3.feedback_still_v2_contract import standing_settings
from legged_lab.envs.elf3.feedback_still_v2_rewards import (
    full_body_default_pose_l1, joint_deviation_l1_when_moving,
    stand_base_stillness, stand_joint_motion, torso_upright,
)
from legged_lab.envs.elf3.standing_rewards import feet_swing_height_when_moving, gait_phase_contact_when_moving


@configclass
class Elf3DwaqFeedbackStillV2EnvCfg(Elf3DwaqFeedbackEnvCfg):
    strict_standing_v2: dict = standing_settings()

    def __post_init__(self):
        super().__post_init__()
        settings = self.strict_standing_v2
        epsilon = settings["command_epsilon"]
        for name, function in (("gait_phase_contact", gait_phase_contact_when_moving),
                               ("feet_swing_height", feet_swing_height_when_moving)):
            term = getattr(self.reward, name)
            term.func = function
            term.params["command_epsilon"] = epsilon
        # No overlapping baseline group pose costs at zero command. The exact
        # same L1 groups, coefficients and targets remain active while moving.
        for name in ("joint_deviation_hip", "joint_deviation_ankle", "joint_deviation_arms", "joint_deviation_legs"):
            term = getattr(self.reward, name)
            term.func = joint_deviation_l1_when_moving
            term.params["command_epsilon"] = epsilon
        joints = lambda: SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)
        self.reward.zero_full_body_pose = RewardTermCfg(
            func=full_body_default_pose_l1, weight=settings["full_body_pose_weight"],
            params={"asset_cfg": joints(), "command_epsilon": epsilon})
        self.reward.zero_torso_upright = RewardTermCfg(
            func=torso_upright, weight=settings["torso_upright_weight"],
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"], preserve_order=True),
                    "command_epsilon": epsilon})
        self.reward.zero_joint_motion = RewardTermCfg(
            func=stand_joint_motion, weight=settings["joint_motion_weight"],
            params={"asset_cfg": joints(), "command_epsilon": epsilon})
        self.reward.zero_base_stillness = RewardTermCfg(
            func=stand_base_stillness, weight=settings["base_stillness_weight"],
            params={"asset_cfg": SceneEntityCfg("robot"), "command_epsilon": epsilon,
                    "linear_std": settings["base_linear_std"], "angular_std": settings["base_angular_std"]})


@configclass
class Elf3DwaqFeedbackStillV2AgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_feedback_still_v2"
    wandb_project: str = "elf3_dwaq_feedback_still_v2"
    runner_class_name: str = "Elf3DwaqFeedbackStillV2Runner"
