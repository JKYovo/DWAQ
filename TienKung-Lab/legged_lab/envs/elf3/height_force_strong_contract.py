"""Isolated stronger coefficients; keep the previous height-force experiment immutable."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path


VERSION = "elf3_dwaq_still_v2_height_force_v2"
SETTINGS = {
    "version": VERSION,
    "height_weight": -50.0,
    "feet_force_weight": -0.005,
    "feet_force_threshold_n": 400,
    "automatic_weight_escalation": False,
}


def height_force_strong_contract():
    directory = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("strong_height_parent_contract", directory / "height_force_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = module.height_force_contract()
    result = {**deepcopy(previous), **SETTINGS,
        "parent_task": "elf3_dwaq_still_v2_ankle_cost_v1",
        "requested_clean_parent_iteration": 38500,
        "previous_height_force_contract": previous,
        "changes_from_height_force_v1": ["height_weight:-2->-50", "feet_force_weight:-0.0035->-0.005",
                                         "feet_force_threshold_n:500->400"],
        "height_geometry_gates_and_zero_command_behavior_unchanged": True,
        "feet_force_formula_sampling_and_cap_unchanged": True,
        "no_new_reward_terms": True}
    for name in ("height_force_strong.py", "height_force_strong_contract.py"):
        result["sources_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return result
