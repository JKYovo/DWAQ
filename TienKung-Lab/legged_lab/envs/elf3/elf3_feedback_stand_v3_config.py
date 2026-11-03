"""Feedback-v1 walking, with strict-zero pose/motion costs and no periodic drive."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackAgentCfg, Elf3DwaqFeedbackEnvCfg
from legged_lab.envs.elf3.feedback_stand_v3_contract import stand_settings
from legged_lab.envs.elf3.feedback_stand_v3_rewards import zero_command_default_pose_l1, zero_command_joint_motion_l1
from legged_lab.envs.elf3.standing_rewards import gait_phase_contact_when_moving, feet_swing_height_when_moving


@configclass
class Elf3DwaqFeedbackStandV3EnvCfg(Elf3DwaqFeedbackEnvCfg):
    strict_standing_v3: dict = stand_settings()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16384
        # Keep the parent's reward at every nonzero command, including pure yaw.
        self.reward.gait_phase_contact.func = gait_phase_contact_when_moving
        self.reward.feet_swing_height.func = feet_swing_height_when_moving
        for name in ("gait_phase_contact", "feet_swing_height"):
            getattr(self.reward, name).params["command_epsilon"] = self.strict_standing_v3["command_epsilon"]
        self.reward.zero_default_pose_l1 = RewardTermCfg(
            func=zero_command_default_pose_l1,
            weight=self.strict_standing_v3["default_pose_weight"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
                    "command_epsilon": self.strict_standing_v3["command_epsilon"]},
        )
        self.reward.zero_joint_motion = RewardTermCfg(
            func=zero_command_joint_motion_l1,
            weight=self.strict_standing_v3["joint_motion_weight"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
                    "command_epsilon": self.strict_standing_v3["command_epsilon"]},
        )


@configclass
class Elf3DwaqFeedbackStandV3AgentCfg(Elf3DwaqFeedbackAgentCfg):
    experiment_name: str = "elf3_dwaq_feedback_stand_v3"
    wandb_project: str = "elf3_dwaq_feedback_stand_v3"
    runner_class_name: str = "Elf3DwaqFeedbackStandV3Runner"
