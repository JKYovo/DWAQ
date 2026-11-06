"""Named-joint, read-only support-knee threshold cost; no Isaac imports."""
import math

import torch


KNEE_JOINT_NAMES = ("l_knee_y_joint", "r_knee_y_joint")
FOOT_BODY_NAMES = ("l_ankle_x_link", "r_ankle_x_link")


def _ordered_ids(names, indices, expected, label):
    if isinstance(indices, slice):
        raise ValueError(f"{label} requires explicitly resolved indices")
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1:
            raise ValueError(f"{label} indices must be one-dimensional")
        indices = indices.tolist()
    indices = tuple(indices)
    if (len(indices) != 2 or len(set(indices)) != 2
            or any(type(i) is not int or i < 0 or i >= len(names) for i in indices)
            or tuple(names[i] for i in indices) != expected):
        raise ValueError(f"{label} must resolve the explicit left/right ELF3 names in order")
    return list(indices)


def support_knee_excess_l2(
    env, asset_cfg, sensor_cfg, threshold_deg=50.0,
    support_force_threshold=1.0, command_epsilon=1e-6,
):
    """Cost only excessive positive knee flexion on the corresponding support leg.

    The 50-degree threshold is an absolute actual joint angle, not an error
    relative to the 0.6-rad home pose. Both knees use positive flexion in ELF3.
    At or below the threshold, in swing, and at zero command the added cost is
    zero. Existing standing/pose objectives are not changed. This is a soft
    reward cost, not a hard joint limit or a fixed torso-height target.

    Read the existing control-end contact cache just like other rewards. Do
    not advance the sensor at each physics substep, maintain contact timers,
    modify actions/observations, or update any simulator state here.
    """
    if not math.isfinite(threshold_deg) or not 0 < threshold_deg < 180:
        raise ValueError("Knee threshold must be finite and between 0 and 180 degrees")
    if not math.isfinite(support_force_threshold) or support_force_threshold < 0:
        raise ValueError("Support threshold must be finite and nonnegative")
    if not math.isfinite(command_epsilon) or command_epsilon < 0:
        raise ValueError("Command epsilon must be finite and nonnegative")
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    joint_ids = _ordered_ids(asset.joint_names, asset_cfg.joint_ids, KNEE_JOINT_NAMES, "Knees")
    body_ids = _ordered_ids(sensor.body_names, sensor_cfg.body_ids, FOOT_BODY_NAMES, "Contact feet")
    q = asset.data.joint_pos[:, joint_ids]
    fz = sensor.data.net_forces_w[:, body_ids, 2]
    command = env.command_generator.command
    if q.ndim != 2 or q.shape[1] != 2 or fz.shape != q.shape or command.shape != (q.shape[0], 3):
        raise ValueError("Knees/Fz must be [env,2] and commands [env,3]")
    for value in (q, fz, command):
        torch._assert_async(torch.isfinite(value).all(), "Knee reward inputs must be finite")
    excess = torch.relu(q - math.radians(threshold_deg)).square()
    support = fz > support_force_threshold
    moving = command.abs().amax(dim=-1) > command_epsilon
    return (excess * support).sum(dim=-1) * moving
