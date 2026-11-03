"""Independent zero-command objective on the recommended ELF3 feedback task."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "elf3_dwaq_feedback_still_v2"
PARENT_TASK = "elf3_dwaq_feedback_v1"


def standing_settings():
    return {
        "version": VERSION,
        "parent_task": PARENT_TASK,
        "command_epsilon": 1e-6,
        "fixed_phase": [0., 0., 1., -1.],
        "phase_scope": "current_and_all_history_if_current_zero",
        "full_body_pose_weight": -1.0,
        "torso_upright_weight": 1.0,
        "joint_motion_weight": -0.5,
        "base_stillness_weight": 1.0,
        "base_linear_std": 0.05,
        "base_angular_std": 0.10,
        "pose_target": "asset.default_joint_pos",
        "replace_zero_group_pose_terms": True,
        "additional_hip_action_constraints": False,
        "lock_world_yaw": False,
    }


def standing_experiment_contract(settings):
    expected = standing_settings()
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported feedback standing v2 settings; version objective changes explicitly")
    root = Path(__file__).resolve().parent
    names = ("feedback_still_v2_contract.py", "feedback_still_v2_rewards.py",
             "feedback_still_v2_env.py", "feedback_still_v2_runner.py", "elf3_feedback_still_v2_config.py",
             "stillness.py", "standing_rewards.py")
    return {
        **deepcopy(expected),
        "activation": "strict current zero vx/vy/wz; no settled/contact/speed/terrain/timer gate",
        "pose_cost": "sum absolute actual q-default q over all 29 named joints; replaces four zero group costs",
        "upright": "Melody equivalent WXYZ roll/pitch and gravity-xy exponentials; yaw invariant",
        "joint_motion": "mean absolute actual raw dq over all 29 joints",
        "base_stillness": "exp(-sum(actual root lin vel squared)/0.05^2-sum(actual root ang vel squared)/0.10^2)",
        "moving_path": "recommended feedback-v1 rewards and periodic assistance unchanged",
        "initialization": "from scratch; no legacy model/optimizer migration",
        "additional_ppo_losses": False,
        "deployment": "fixed zero-command phase preprocessing embedded; raw 500 inputs / 29 actions unchanged",
        "sources_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }
