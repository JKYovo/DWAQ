"""Explicit 400 -> 500 N reward-only promotion; old stages remain immutable."""
from copy import deepcopy
import hashlib
from pathlib import Path


VERSION = "elf3_force_peak_threshold500_v1"
SETTINGS = {
    "version": VERSION,
    "previous_threshold_n": 400., "threshold_n": 500.,
    "feet_force_weight": -.005, "excess_cap_n": 400.,
    "only_environment_change": "reward.feet_force.params.threshold:400->500",
    "sampling": "unchanged max of simultaneous two-foot Fz norm over four physics substeps",
    "action_smooth_weight": -.001, "dof_acc_weight": -3e-7,
    "action_rate_weight": -.012, "ankle_torque_weight": -.0005,
    "unchanged": ["height", "standing", "ankle_costs", "asset", "PD", "armature",
                  "20Hz_feedback", "commands", "terrain", "randomization", "policy", "CENet", "PPO"],
    "restoration": "model/Adam/adaptive LR/iteration; no simulator/curriculum/RNG/history/filter state",
}


def threshold_contract():
    directory = Path(__file__).resolve().parent
    names = ("foot_force_500.py", "foot_force_500_contract.py")
    return {**deepcopy(SETTINGS), "sources_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
