"""Versioned sampling-only repair; immutable old reward stages stay available."""
from copy import deepcopy
import hashlib
from pathlib import Path


VERSION = "elf3_height_force_v2_smooth_force_peak_v1"
SETTINGS = {
    "version": VERSION,
    "sampling": "every 5ms physics substep, maximum over all 4 substeps per 20ms control step",
    "metric": "max_substep(sqrt(Fz_left**2 + Fz_right**2)); simultaneous two-foot norm retained",
    "feet_force_weight": -.005, "threshold_n": 400., "excess_cap_n": 400.,
    "dt": "RewardManager applies control_dt exactly once; no extra factor of four",
    "reset": "clear per-environment caches on reset; clear all caches at each control-step start",
    "threshold_boundary": "exactly threshold has zero excess (fixes inherited strict-comparison corner case)",
    "action_smooth_weight": -.001, "dof_acc_weight": -3e-7,
    "unchanged": ["asset", "PD", "armature", "20Hz_feedback", "actor", "CENet", "PPO",
                  "height", "commands", "terrain", "randomization", "other_reward_weights"],
}


def peak_contract():
    directory = Path(__file__).resolve().parent
    names = ("foot_force_peak.py", "foot_force_peak_rewards.py", "foot_force_peak_contract.py")
    return {**deepcopy(SETTINGS), "sources_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
