"""Minimal, opt-in zero-command standing contract for the proven DWAQ baseline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


VERSION = "elf3_dwaq_still_v1"


def still_settings():
    return {
        "version": VERSION,
        "command_epsilon": 1e-6,
        "fixed_phase": [0., 0., 1., -1.],
        "phase_scope": "current_and_all_history_if_current_zero",
        "joint_motion_weight": -0.5,
    }


def still_experiment_contract(settings):
    expected = still_settings()
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported minimal stillness settings; version objective changes explicitly")
    directory = Path(__file__).resolve().parent
    names = ("still_contract.py", "stillness.py", "still_env.py", "still_runner.py", "elf3_still_config.py")
    return {
        **deepcopy(expected),
        "activation": "current zero vx/vy/wz only; no support, posture, actual-speed, timer or history gate",
        "motion_cost": "mean absolute raw joint velocity, all 29 joints; linear and unsaturated",
        "zero_command_gait_rewards_disabled": True,
        "baseline": "feedback-v1 DWAQ, original PPO/VAE and global mirror/pose auxiliaries",
        "additional_ppo_losses": False,
        "critic_and_environment_clock": "unchanged; mask applies to actor inputs only",
        "deployment": "phase preprocessing inside exported model; unchanged raw 500-input/29-action interface",
        "sources_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names},
    }
