"""Private, zero-command actual leg-pose cost for the minimal standing task.

The original all-command leg-pose weight remains -0.02. Configure this new
term with weight -0.08 to obtain a total coefficient of -0.10 only at a true
zero command. This is a soft actual-position cost, not an action clamp or a
joint-position target, and it does not alter the existing stillness motion cost.
"""
from __future__ import annotations

import math

import torch


STAND_POSE_JOINT_NAMES = (
    "l_hip_y_joint",
    "l_knee_y_joint",
    "r_hip_y_joint",
    "r_knee_y_joint",
)


def stand_leg_pose_l1(env, asset_cfg, command_epsilon: float = 1e-6):
    """Sum the four actual leg-joint deviations only at a current zero command.

    RewardManager must resolve ``asset_cfg.joint_ids`` from the explicit names
    in ``STAND_POSE_JOINT_NAMES`` in simulator order. No support, settling,
    terrain, posture, velocity or history gate is applied, and no state is
    cached across calls or resets. Non-finite commands do not activate the cost.
    The home pose comes from the asset's actual default joint positions, not
    policy actions or position targets, so the term remains meaningful under
    position-tracking error and externally applied load.
    """
    if not math.isfinite(float(command_epsilon)) or command_epsilon < 0.0:
        raise ValueError("command epsilon must be finite and non-negative")
    command = env.command_generator.command[:, :3]
    if command.ndim != 2 or command.shape != (env.num_envs, 3):
        raise ValueError("stand_leg_pose_l1 requires three commands per environment")

    data = env.robot.data
    position = data.joint_pos[:, asset_cfg.joint_ids]
    home = data.default_joint_pos[:, asset_cfg.joint_ids]
    expected_shape = (env.num_envs, len(STAND_POSE_JOINT_NAMES))
    if position.ndim != 2 or tuple(position.shape) != expected_shape or home.shape != position.shape:
        raise ValueError("stand_leg_pose_l1 requires exactly four resolved hip-pitch/knee joints")

    active = torch.isfinite(command).all(dim=-1) & (command.abs().amax(dim=-1) <= command_epsilon)
    value = (position - home).abs().sum(dim=-1)
    return torch.where(active, value, torch.zeros_like(value))
