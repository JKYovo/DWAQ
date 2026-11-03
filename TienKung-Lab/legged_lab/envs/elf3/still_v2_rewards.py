"""Extra zero-command waist pose cost for the original still-task recipe.

The caller supplies -0.10 in addition to the inherited -0.20 waist/arm term.
This pure-Torch term reads actual q and native default q at identical indices.
It does not replace baseline rewards or introduce support/settling gates.
"""
from __future__ import annotations

import math
import operator

import torch


STAND_WAIST_JOINT_NAMES = ("waist_y_joint", "waist_x_joint", "waist_z_joint")


def stand_waist_pose_l1(env, asset_cfg, command_epsilon: float = 1e-6):
    """Sum three actual waist deviations at finite current zero vx/vy/wz.

    Resolve joint_ids from STAND_WAIST_JOINT_NAMES, preserve_order=True. The
    same native indices select actual and default positions. No target/action,
    velocity, phase, contact, posture, terrain or timer is read or modified.
    """
    if isinstance(command_epsilon, bool):
        raise ValueError("command epsilon must be finite and nonnegative")
    try:
        epsilon = float(command_epsilon)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("command epsilon must be finite and nonnegative") from error
    if not math.isfinite(epsilon) or epsilon < 0.:
        raise ValueError("command epsilon must be finite and nonnegative")
    command = env.command_generator.command
    if (not isinstance(command, torch.Tensor) or not command.is_floating_point()
            or tuple(command.shape) != (env.num_envs, 3)):
        raise ValueError("stand_waist_pose_l1 requires three floating point commands per environment")
    data = env.robot.data
    position, home = data.joint_pos, data.default_joint_pos
    if (not isinstance(position, torch.Tensor) or not position.is_floating_point()
            or position.ndim != 2 or position.shape[0] != env.num_envs
            or not isinstance(home, torch.Tensor) or not home.is_floating_point()
            or home.shape != position.shape):
        raise ValueError("Actual/default positions must be matching floating point [num_envs, joints] tensors")
    selection = asset_cfg.joint_ids
    if isinstance(selection, slice):
        indices = list(range(position.shape[1]))[selection]
    else:
        if isinstance(selection, torch.Tensor):
            if selection.ndim != 1 or selection.dtype == torch.bool or selection.is_floating_point():
                raise ValueError("Resolved waist joint indices must be one-dimensional integers")
            selection = selection.tolist()
        try:
            values = list(selection)
            if any(isinstance(value, bool) for value in values):
                raise ValueError("Boolean waist joint indices are not supported")
            indices = [operator.index(value) for value in values]
        except (TypeError, IndexError) as error:
            raise ValueError("Expected three resolved waist joint indices") from error
    if (len(indices) != 3 or len(set(indices)) != 3
            or any(index < 0 or index >= position.shape[1] for index in indices)):
        raise ValueError("Exactly three unique, in-range waist joint indices are required")
    if position.device != home.device or position.device != command.device:
        raise ValueError("Waist position/default and commands must share a device")
    active = torch.isfinite(command).all(-1) & (command.abs().amax(-1) <= epsilon)
    value = (position[:, indices] - home[:, indices]).abs().sum(-1)
    return torch.where(active, value, torch.zeros_like(value))
