"""Pure-Torch, strict-command standing objectives for the feedback-v1 fork.

These terms read actual articulation state, never policy actions, position
targets, torques, contacts, a settling timer, or observation-filtered velocity.
All command gates are immediate and stateless. Resolved entity indices must be
provided by the reward manager; no policy/simulator joint-order assumptions are
made here. The environment/configuration owns reward weights.
"""
from __future__ import annotations

import math
import operator

import torch


COMMAND_EPSILON = 1e-6
NUM_JOINTS = 29


def _positive_scalar(value, name, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def strict_zero_command(command, command_epsilon=COMMAND_EPSILON):
    """Return an [N] mask for finite [vx, vy, wz] with max(abs(.)) <= eps.

    Non-finite command rows activate neither standing nor moving objectives.
    In particular, 0.1 m/s and pure yaw are moving, not a standing deadband.
    """
    epsilon = _positive_scalar(command_epsilon, "command_epsilon", allow_zero=True)
    if (not isinstance(command, torch.Tensor) or not command.is_floating_point()
            or command.ndim != 2 or command.shape[1] != 3):
        raise ValueError("Expected floating point commands shaped [num_envs, 3]")
    return torch.isfinite(command).all(-1) & (command.abs().amax(-1) <= epsilon)


def _commands(env, command_epsilon):
    command = env.command_generator.command
    if (not isinstance(command, torch.Tensor) or command.ndim != 2
            or command.shape != (env.num_envs, 3)):
        raise ValueError("Expected three velocity commands per environment")
    return command, strict_zero_command(command, command_epsilon)


def _asset(env, asset_cfg):
    name = getattr(asset_cfg, "name", "robot")
    if hasattr(env, "scene"):
        return env.scene[name]
    if name != "robot":
        raise ValueError("Only the robot fallback is available without an environment scene")
    return env.robot


def _indices(selection, size):
    """Validate static, resolved metadata, including duplicates and bool masks."""
    if isinstance(selection, slice):
        indices = list(range(size))[selection]
    else:
        if isinstance(selection, torch.Tensor):
            if selection.ndim != 1 or selection.dtype == torch.bool or selection.is_floating_point():
                raise ValueError("Resolved indices must be a one-dimensional integer selection")
        try:
            values = list(selection)
            if any(isinstance(value, bool) for value in values):
                raise ValueError("Boolean entity selections are not supported")
            indices = [operator.index(value) for value in values]
        except (TypeError, IndexError) as error:
            raise ValueError("Expected resolved entity indices") from error
        if any(value < -size or value >= size for value in indices):
            raise ValueError("Entity index out of range")
        indices = [value % size for value in indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Entity indices must be nonempty and unique")
    return indices


def _joint_values(env, asset_cfg, field, *, full_body=False):
    data = _asset(env, asset_cfg).data
    value = getattr(data, field)
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or value.shape[0] != env.num_envs):
        raise ValueError(f"Expected {field} shaped [num_envs, num_joints]")
    indices = _indices(asset_cfg.joint_ids, value.shape[1])
    if full_body and (value.shape[1] != NUM_JOINTS or len(indices) != NUM_JOINTS):
        raise ValueError("Full-body standing objectives require all 29 unique robot joints")
    return data, value[:, indices], indices


def _pose_cost(env, asset_cfg, *, full_body=False):
    data, position, indices = _joint_values(env, asset_cfg, "joint_pos", full_body=full_body)
    home = data.default_joint_pos
    if not isinstance(home, torch.Tensor) or home.shape != data.joint_pos.shape:
        raise ValueError("Default and actual joint-position arrays must have identical shapes")
    return (position - home[:, indices]).abs().sum(-1)


def full_body_default_pose_l1(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Zero-command sum L1 on all 29 actual q minus the asset default q."""
    _, active = _commands(env, command_epsilon)
    cost = _pose_cost(env, asset_cfg, full_body=True)
    return torch.where(active, cost, torch.zeros_like(cost))


def joint_deviation_l1_when_moving(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Original groupwise actual-q L1, active only for finite nonzero commands.

    Unlike the full-body standing terms this accepts any nonempty named group.
    A command transition needs no history, support or settling condition.
    """
    command, standing = _commands(env, command_epsilon)
    moving = torch.isfinite(command).all(-1) & ~standing
    cost = _pose_cost(env, asset_cfg)
    return torch.where(moving, cost, torch.zeros_like(cost))


def torso_upright(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Bounded zero-command upright score, equivalent to Melody's actual score.

    Quaternion order is WXYZ. Use physical ZYX roll/pitch, not world heading:
    (exp(-10*(abs(roll)+abs(pitch))) + exp(-20*norm(gravity_body.xy))) / 2.
    This is invariant to world-Z yaw rotations. Exactly one torso body must be
    selected. Invalid/nonfinite or zero-norm quaternions produce zero reward.
    Quaternion sign and small normalization error do not change the objective.
    """
    _, active = _commands(env, command_epsilon)
    data = _asset(env, asset_cfg).data
    quaternions = data.body_quat_w
    if (not isinstance(quaternions, torch.Tensor) or not quaternions.is_floating_point()
            or quaternions.ndim != 3 or quaternions.shape[0] != env.num_envs
            or quaternions.shape[-1] != 4):
        raise ValueError("Expected WXYZ body quaternions shaped [num_envs, num_bodies, 4]")
    indices = _indices(asset_cfg.body_ids, quaternions.shape[1])
    if len(indices) != 1:
        raise ValueError("The upright term requires exactly one resolved torso body")
    raw = quaternions[:, indices[0]]
    epsilon = torch.finfo(raw.dtype).eps
    norm = torch.linalg.vector_norm(raw, dim=-1)
    valid = torch.isfinite(raw).all(-1) & torch.isfinite(norm) & (norm > epsilon)
    identity = torch.zeros_like(raw)
    identity[:, 0] = 1.
    safe = torch.where(valid[:, None], raw, identity)
    unit = safe / torch.linalg.vector_norm(safe, dim=-1, keepdim=True)
    w, x, y, z = unit.unbind(-1)
    roll_y = 2. * (w * x + y * z)
    roll_x = 1. - 2. * (x.square() + y.square())
    # ZYX roll is undefined at the pitch pole; use its finite canonical limit
    # rather than atan2(0,0). This affects only a measure-zero singularity.
    regular = roll_y.abs() + roll_x.abs() > epsilon
    roll = torch.atan2(torch.where(regular, roll_y, torch.zeros_like(roll_y)),
                       torch.where(regular, roll_x, torch.ones_like(roll_x)))
    pitch_sine = 2. * (w * y - z * x)
    # Leave a one-ULP margin so asin gradients remain finite at +/-90 degrees.
    pitch = torch.asin(pitch_sine.clamp(-1. + epsilon, 1. - epsilon))
    gravity_xy = torch.stack((pitch_sine, -2. * (w * x + y * z)), dim=-1)
    score = .5 * (torch.exp(-10. * (roll.abs() + pitch.abs()))
                  + torch.exp(-20. * torch.linalg.vector_norm(gravity_xy, dim=-1)))
    return torch.where(active & valid, score, torch.zeros_like(score))


def stand_joint_motion(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Zero-command mean absolute actual/raw velocity of all 29 joints."""
    _, active = _commands(env, command_epsilon)
    _, velocity, _ = _joint_values(env, asset_cfg, "joint_vel", full_body=True)
    cost = velocity.abs().mean(-1)
    return torch.where(active, cost, torch.zeros_like(cost))


def _base_velocities(env, asset_cfg):
    data = _asset(env, asset_cfg).data
    linear, angular = data.root_lin_vel_b, data.root_ang_vel_b
    if (not isinstance(linear, torch.Tensor) or not isinstance(angular, torch.Tensor)
            or not linear.is_floating_point() or not angular.is_floating_point()
            or linear.shape != (env.num_envs, 3) or angular.shape != linear.shape):
        raise ValueError("Expected actual body-frame root linear/angular velocities shaped [num_envs, 3]")
    return linear, angular


def stand_base_motion(env, asset_cfg, command_epsilon=COMMAND_EPSILON):
    """Optional dense cost: sum(v_xyz^2) + .25*sum(omega_xyz^2), only at zero."""
    _, active = _commands(env, command_epsilon)
    linear, angular = _base_velocities(env, asset_cfg)
    cost = linear.square().sum(-1) + .25 * angular.square().sum(-1)
    return torch.where(active, cost, torch.zeros_like(cost))


def stand_base_stillness(env, asset_cfg, command_epsilon=COMMAND_EPSILON,
                         linear_std=.05, angular_std=.10):
    """Bounded [0,1] reward for actual base stillness, independent of joint q.

    exp(-sum(v_xyz^2)/linear_std^2 - sum(omega_xyz^2)/angular_std^2).
    Both full three-dimensional norms are invariant to a rotation of the
    reporting frame. No target action, support condition or timer can enable it.
    """
    linear_std = _positive_scalar(linear_std, "linear_std")
    angular_std = _positive_scalar(angular_std, "angular_std")
    _, active = _commands(env, command_epsilon)
    linear, angular = _base_velocities(env, asset_cfg)
    score = torch.exp(-(linear / linear_std).square().sum(-1)
                      - (angular / angular_std).square().sum(-1))
    return torch.where(active, score, torch.zeros_like(score))
