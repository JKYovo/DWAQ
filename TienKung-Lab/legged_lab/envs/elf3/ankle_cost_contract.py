"""Versioned two-reward refinement, separate from the original still-v2 task."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


VERSION = "elf3_dwaq_still_v2_ankle_cost_v1"
PARENT_TASK = "elf3_dwaq_still_v2"
ANKLE_JOINT_NAMES = ["l_ankle_y_joint", "l_ankle_x_joint", "r_ankle_y_joint", "r_ankle_x_joint"]
FOOT_BODY_NAMES = ["l_ankle_x_link", "r_ankle_x_link"]
SETTINGS = {
    "version": VERSION,
    "parent_task": PARENT_TASK,
    "ankle_torque_weight": -0.0005,
    "ankle_joint_names": ANKLE_JOINT_NAMES,
    "torque_channel": "Isaac Lab implicit-actuator applied_torque estimate; not hardware measured torque",
    "torque_formula": "sum over four ankles of applied_torque squared, at every command",
    "stance_speed_weight": -1.0,
    "foot_body_names": FOOT_BODY_NAMES,
    "stance_threshold": 0.55,
    "phase_sigma": 0.015,
    "speed_scale": 200.0,
    "command_epsilon": 1e-6,
    "speed_formula": "sum I_stance_smooth(leg_phase) * (1-exp(-200*mean_substep_speed_norm_squared))",
    "speed_sampling": "mean of four actual world-frame 3D foot-link linear speed norms at 200Hz; no filter",
    "phase": "retain parent 0.8s clock and left/right half-cycle offset; use parent updated end-of-step phase",
    "speed_activation": "finite current command max(abs(vx,vy,wz))>1e-6; no actual-contact gate",
    "zero_command_periodic_speed_cost": False,
    "new_actor_inputs": False,
    "new_action_or_PD_filter": False,
    "unchanged": ["asset", "PD", "armature", "action_scale", "20Hz_velocity_feedback", "commands",
                  "terrain", "domain_randomization", "noise", "all_parent_rewards", "CENet", "PPO", "learning_rate"],
}


def reward_stage_contract():
    root = Path(__file__).resolve().parent
    names = ("ankle_cost_contract.py", "ankle_cost_rewards.py", "ankle_cost.py")
    return {**deepcopy(SETTINGS),
            "sources_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def validate_reward_stage_contract(value):
    expected = reward_stage_contract()
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Ankle-cost checkpoint reward version, parameters or source fingerprint changed")
    return expected
