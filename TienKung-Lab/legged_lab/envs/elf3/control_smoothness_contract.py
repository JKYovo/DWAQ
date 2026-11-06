"""Local reward-only continuation; old height-force/checkpoint sources stay intact."""
from copy import deepcopy
import hashlib
from pathlib import Path


TASK = "elf3_dwaq_still_v2_height_force_v2"
VERSION = "elf3_height_force_v2_smooth_v1"
SETTINGS = {
    "version": VERSION,
    "parent_task": TASK,
    "action_smooth_weight": -.001,
    "parent_action_rate_weight": -.012,
    "action_smooth_to_action_rate_max_ratio": .2,
    "previous_dof_acc_weight": -2.5e-7,
    "dof_acc_multiplier": 1.2,
    "dof_acc_weight": -3.0e-7,
    "formula": "sum((a_t-a_t1)^2) + sum((a_t+a_t2-2*a_t1)^2) + 0.05*sum(abs(a_t))",
    "scope": "all 29 raw action-buffer channels, all commands; 50Hz, RewardManager dt once",
    "reference": "TienKung-Lab legged_lab/mdp/rewards_elf3.py:action_smoothness",
    "reset": "unchanged existing DelayBuffer/CircularBuffer reset and first-push padding",
    "new_action_or_observation_filter": False,
    "automatic_weight_changes": False,
    "automatic_training_stop_or_restart": False,
    "unchanged": ["action_rate_l2", "ankle_costs", "height_floor", "feet_force", "standing",
                  "asset", "PD", "armature", "20Hz_feedback", "terrain", "commands", "randomization",
                  "actor", "CENet", "PPO", "learning_rate", "sampling_interval"],
}


def smoothness_contract():
    directory = Path(__file__).resolve().parent
    names = ("control_smoothness.py", "control_smoothness_contract.py", "control_smoothness_rewards.py")
    return {**deepcopy(SETTINGS), "sources_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
