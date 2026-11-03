"""Isolated feedback-v1 standing recipe with zero-only periodic gating."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


VERSION = "elf3_dwaq_feedback_stand_v3_zero_phase_v1"
PARENT_TASK = "elf3_dwaq_feedback_v1"


def stand_settings():
    return {
        "version": VERSION,
        "parent_task": PARENT_TASK,
        "command_epsilon": 1e-6,
        "default_pose_weight": -0.1,
        "joint_motion_weight": -0.5,
        "pose_target": "asset.default_joint_pos",
        "baseline_pose_preserved": True,
        "actor_inputs": "zero_command_fixed_phase_only",
        "fixed_phase": [0., 0., 1., 1.],
        "phase_scope": "current_and_all_history_frames",
        "zero_command_periodic_rewards": False,
        "additional_ppo_losses": False,
    }


def stand_experiment_contract(settings=None):
    expected = stand_settings()
    settings = expected if settings is None else settings
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported zero-phase v3 settings; earlier v3 recipes are incompatible")
    directory = Path(__file__).resolve().parent
    names = ("feedback_stand_v3_contract.py", "feedback_stand_v3_rewards.py",
             "feedback_stand_v3_env.py", "feedback_stand_v3_runner.py", "elf3_feedback_stand_v3_config.py",
             "feedback_stand_v3_inputs.py", "standing_rewards.py")
    return {
        **deepcopy(expected),
        "activation": "finite current max(abs(vx,vy,wz)) <= 1e-6; no settled/contact/timer gate",
        "added_reward": "-0.1 * sum(abs(actual_q - native_asset_default_q)) over 29 named joints",
        "added_motion_cost": "-0.5 * mean(abs(actual_raw_joint_velocity)) over 29 named joints, strict zero command only",
        "baseline": "all feedback-v1 rewards and coefficients retained except zero-command contact/swing gating; raw gait clock/history retained",
        "no_phase_preprocessing": False,
        "periodic_gating": "gait_phase_contact and feet_swing_height are zero at strict zero command; original functions/weights at nonzero command",
        "initialization": "from_scratch; no failed composite-v3 or older checkpoint migration",
        "deployment": "deterministic 500 inputs / 29 actions; same zero-only phase canonicalization embedded in export; no action filter or lock",
        "sources_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names},
    }
