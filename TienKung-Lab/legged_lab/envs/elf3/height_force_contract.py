"""Versioned walking-height experiment; old checkpoints/source remain unchanged."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path


VERSION = "elf3_dwaq_still_v2_height_force_v1"
SETTINGS = {
    "version":VERSION, "parent_task":"elf3_dwaq_still_v2_motion_quality_v1",
    "removed_rewards":["support_knee_excess_l2"],
    "action_rate_weight":-.012,
    "feet_force_previous_weight":-.003, "feet_force_weight":-.0035,
    "feet_force_threshold_n":500., "feet_force_cap_n":400.,
    "height_weight":-2., "height_deadband_m":.015,
    "height_target":"default upright torso_link-to-sole height from current ELF3 URDF/home pose",
    "height_formula":"terrain_gate * relu(target_height - .015 - torso_height_above_support)^2; metres squared",
    "command_epsilon":1e-6,
    "activation":"finite nonzero vx/vy/wz and actual positive-Fz support and valid jointly flat local terrain",
    "support_radius_m":.3, "near_radius_m":.15, "min_points_per_foot":3,
    "flat_full_span_m":.01, "flat_off_span_m":.03,
    "ground_estimate":"local finite scan mean per foot, weighted by actual positive Fz > 1 N",
    "invalid_scan_fallback":"disabled, never root height minus target",
    "height_scope":"moving only; all zero-command rewards/phase behavior unchanged",
    "mapping":"resolve names once in init; no per-step sensor.body_names calls",
    "actor_privileged_input_added":False, "new_PD_or_action_filter":False,
    "unchanged":["all_other_rewards","20Hz_velocity_feedback","asset","PD","armature","action_scale",
                 "commands","terrain","domain_randomization","PPO","learning_rate"],
}


def height_force_contract():
    directory=Path(__file__).resolve().parent
    # Keep checkpoint header checks independent of Isaac/package registration.
    spec=importlib.util.spec_from_file_location("height_force_contract_reference",directory/"height_force_reference.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=("height_force.py","height_force_contract.py","height_force_reference.py","height_force_rewards.py")
    return {**deepcopy(SETTINGS),"height_reference":module.default_height_reference(),
            "sources_sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}}
