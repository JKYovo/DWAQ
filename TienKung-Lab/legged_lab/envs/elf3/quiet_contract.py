"""Versioned identity for zero-command upper-body stillness, not stand-only walking."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


VERSION = "elf3_stand_v2_command_only_v2"
COMMAND_EPSILON = 1.0e-6


def quiet_settings():
    return {
        "version": VERSION,
        "mode": "zero_command_upper_body_stillness",
        "command_epsilon": COMMAND_EPSILON,
        "activation": {"mode": "zero_command_only", "history_requirement": "none"},
        "auxiliary": {"phase_invariance_coeff": 1.0, "arm_pose_coeff": 0.1,
                      "phase_offsets": [math.pi / 2, math.pi, 3 * math.pi / 2]},
        "physical": {"velocity_std": 0.05},
    }


def quiet_experiment_contract(settings):
    expected = quiet_settings()
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported stand-v2 settings; create an explicit new version for changed objectives")
    directory = Path(__file__).resolve().parent
    sources = {name: directory / name for name in (
        "quiet_contract.py", "quiet_rewards.py", "quiet_env.py", "quiet_learning.py", "quiet_runner.py",
        "elf3_stand_v2_config.py", "standing_rewards.py", "standing_contract.py",
    )}
    lab = directory.parents[1]
    rsl = lab.parent / "rsl_rl/rsl_rl"
    sources["base_rewards"] = lab / "mdp/rewards.py"
    sources["base_ppo"] = rsl / "algorithms/dwaq_ppo.py"
    sources["base_runner"] = rsl / "runners/dwaq_on_policy_runner.py"
    return {
        **deepcopy(expected),
        "supports_walking": True,
        "zero_command_gait_rewards_disabled": True,
        "arm_action_ids": list(range(15, 29)),
        "phase_augmentation": "same additive phase offset for current frame, both legs and all five history frames; auxiliary only",
        "aux_mask_alignment": "pre-action current-zero-command mask stored with obs_t; no support, motion, posture, timer or history-command gate",
        "deployment": "unchanged cyclic phase and deterministic mean-context policy; no clamp, lock or new filter",
        "sources_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
    }
