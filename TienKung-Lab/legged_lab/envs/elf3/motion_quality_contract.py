"""Reward-only successor to ankle-cost v1; old experiments remain immutable."""
from copy import deepcopy
import hashlib
from pathlib import Path


VERSION = "elf3_dwaq_still_v2_motion_quality_v1"
SETTINGS = {
    "version": VERSION,
    "parent_task": "elf3_dwaq_still_v2_ankle_cost_v1",
    "action_rate_previous_weight": -0.01,
    "action_rate_weight": -0.012,
    "action_rate_scope": "original callback; sum of squared consecutive actions over all 29 joints",
    "knee_weight": -0.05,
    "knee_threshold_deg": 50.0,
    "knee_formula": "sum_feet support * relu(actual_knee_rad - radians(50)) squared",
    "knee_joint_names": ["l_knee_y_joint", "r_knee_y_joint"],
    "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
    "support_force_threshold_n": 1.0,
    "support_definition": "current corresponding foot positive world Fz > 1 N; not clock phase",
    "command_epsilon": 1e-6,
    "knee_activation": "finite command max(abs(vx,vy,wz)) > 1e-6; existing zero-command rewards unchanged",
    "sampling": "current control-end state at 50 Hz; no new timer or contact-history updates",
    "units": "knee cost rad^2, not deg^2; RewardManager applies original control dt",
    "excluded_soft_step_rewards": ["near_ground_descent", "support_knee_pose"],
    "base_height_reward": False,
    "new_actor_inputs_or_filters": False,
    "unchanged": ["all_other_parent_rewards", "ankle_torque_l2", "stance_foot_speed_cost",
                  "asset", "PD", "armature", "action_scale", "20Hz_velocity_feedback", "commands",
                  "terrain", "domain_randomization", "noise", "CENet", "PPO", "learning_rate"],
}


def motion_quality_contract():
    directory = Path(__file__).resolve().parent
    names = ("motion_quality.py", "motion_quality_contract.py", "motion_quality_rewards.py")
    return {**deepcopy(SETTINGS), "sources_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
