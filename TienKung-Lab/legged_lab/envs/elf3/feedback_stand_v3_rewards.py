"""Actual-state default-pose and joint-motion costs at strict zero command.

The environment config applies the negative weights. Pose uses sum-L1; joint
motion uses mean absolute raw dq. There is no ratio score, upright target,
contact/settling gate, clock transform, action clamp or persistent state.
"""
from __future__ import annotations

import math
import operator

import torch


NUM_JOINTS = 29
COMMAND_EPSILON = 1e-6


def strict_zero_command(command, command_epsilon=COMMAND_EPSILON):
    if isinstance(command_epsilon, bool):
        raise ValueError("command_epsilon must be finite and nonnegative")
    try:
        epsilon = float(command_epsilon)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("command_epsilon must be finite and nonnegative") from error
    if not math.isfinite(epsilon) or epsilon < 0.:
        raise ValueError("command_epsilon must be finite and nonnegative")
    if (not isinstance(command, torch.Tensor) or not command.is_floating_point()
            or command.ndim != 2 or command.shape[1] != 3):
        raise ValueError("Expected floating point commands shaped [num_envs, 3]")
    return torch.isfinite(command).all(-1) & (command.abs().amax(-1) <= epsilon)


def _all_joint_indices(selection):
    if isinstance(selection, slice):
        indices = list(range(NUM_JOINTS))[selection]
    else:
        if isinstance(selection, torch.Tensor):
            if selection.ndim != 1 or selection.dtype == torch.bool or selection.is_floating_point():
                raise ValueError("joint_ids must be a one-dimensional integer selection")
            selection = selection.tolist()
        try:
            values = list(selection)
            if any(isinstance(value, bool) for value in values):
                raise ValueError("Boolean joint indices are not supported")
            indices = [operator.index(value) for value in values]
        except (TypeError, IndexError) as error:
            raise ValueError("Expected resolved joint indices") from error
    if len(indices) != NUM_JOINTS or set(indices) != set(range(NUM_JOINTS)):
        raise ValueError("Select all 29 unique, in-range native robot joints")
    return indices


def zero_command_default_pose_l1(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Return sum(abs(actual q - native default q)) only at current zero.

    Actual and default positions use the same resolved indices. The positive
    cost has no normalization/saturation; a configuration weight of -0.1 gives
    a -0.1*radian default-pose reward contribution. Invalid commands are off.
    Nonfinite actual/default q at zero is not silently hidden by a state gate;
    it remains visible to the training loop's numerical-safety checks.
    """
    command = env.command_generator.command
    active = strict_zero_command(command, command_epsilon)
    if command.shape[0] != env.num_envs:
        raise ValueError("Command batch must match env.num_envs")
    name = getattr(asset_cfg, "name", "robot")
    if hasattr(env, "scene"):
        data = env.scene[name].data
    elif name == "robot":
        data = env.robot.data
    else:
        raise ValueError("Only robot is available without env.scene")
    actual, default = data.joint_pos, data.default_joint_pos
    expected = (env.num_envs, NUM_JOINTS)
    for value in (actual, default):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or tuple(value.shape) != expected):
            raise ValueError("Actual/default joint positions must be [num_envs, 29] floating point tensors")
    if actual.device != default.device or actual.device != command.device:
        raise ValueError("Commands and actual/default joint positions must share a device")
    indices = _all_joint_indices(asset_cfg.joint_ids)
    cost = (actual[:, indices] - default[:, indices]).abs().sum(-1)
    return torch.where(active, cost, torch.zeros_like(cost))


def zero_command_joint_motion_l1(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Return mean(abs(actual raw dq)) over all 29 joints at current zero.

    This reads the articulation's physical joint_vel, not observed/filtered
    velocity, target velocity, actions or position targets. No settling/support
    or elapsed-time condition delays activation. The config weight is -0.5.
    """
    command = env.command_generator.command
    active = strict_zero_command(command, command_epsilon)
    if command.shape[0] != env.num_envs:
        raise ValueError("Command batch must match env.num_envs")
    name = getattr(asset_cfg, "name", "robot")
    if hasattr(env, "scene"):
        data = env.scene[name].data
    elif name == "robot":
        data = env.robot.data
    else:
        raise ValueError("Only robot is available without env.scene")
    actual = data.joint_vel
    if (not isinstance(actual, torch.Tensor) or not actual.is_floating_point()
            or tuple(actual.shape) != (env.num_envs, NUM_JOINTS)):
        raise ValueError("Actual joint velocity must be a [num_envs, 29] floating point tensor")
    if actual.device != command.device:
        raise ValueError("Commands and actual joint velocities must share a device")
    indices = _all_joint_indices(asset_cfg.joint_ids)
    cost = actual[:, indices].abs().mean(-1)
    return torch.where(active, cost, torch.zeros_like(cost))
