"""Small, read-only landing and support-knee costs; no simulator imports.

Sole samples are verified vertices of the current foot STL, not recovered
PhysX contact points. Their world velocity includes foot rotation, rather
than confusing a link position with a COM linear velocity. No fixed world-Z
height, estimated body height, phase gate, or stateful timer is used here.
"""
from __future__ import annotations

import math

import torch


KNEE_JOINT_NAMES = ("l_knee_y_joint", "r_knee_y_joint")
SOLE_REFERENCE_POINTS = (
    (-0.075750000775, 0., -0.041000001132),
    (0.128939226270, -0.000694592716, -0.041000001132),
)


def _finite(value, name):
    torch._assert_async(torch.isfinite(value).all(), f"Nonfinite {name}")


def sole_kinematics(link_pos, link_quat, link_vel, ang_vel, local_points):
    """Transform local sole samples; inputs use link-origin velocity, wxyz q."""
    if (link_pos.ndim != 3 or link_pos.shape[1:] != (2, 3)
            or link_quat.shape != (*link_pos.shape[:2], 4)
            or link_vel.shape != link_pos.shape or ang_vel.shape != link_pos.shape
            or local_points.ndim != 2 or local_points.shape[1] != 3):
        raise ValueError("Expected two foot links and local [points, 3] samples")
    for value in (link_pos, link_quat, link_vel, ang_vel, local_points):
        _finite(value, "sole kinematics")
    local = local_points[None, None].expand(*link_pos.shape[:2], -1, -1)
    qxyz = link_quat[..., None, 1:].expand_as(local)
    twice_cross = 2 * torch.cross(qxyz, local, dim=-1)
    offset = local + link_quat[..., None, :1] * twice_cross + torch.cross(qxyz, twice_cross, dim=-1)
    position = link_pos.unsqueeze(-2) + offset
    velocity = link_vel.unsqueeze(-2) + torch.cross(ang_vel.unsqueeze(-2).expand_as(offset), offset, dim=-1)
    return position, velocity


def local_ground(points, ray_hits, radius=.10, height_tolerance=.03):
    """Nearest finite same-point terrain sample; ambiguous edges fail closed.

    Within ``radius``, a height spread above ``height_tolerance`` is ambiguous
    on the existing 10 cm grid. Disable that point instead of inventing ground
    from the robot/other foot. Invalid points return zero plus validity=False.
    """
    if radius <= 0 or height_tolerance < 0 or not math.isfinite(radius + height_tolerance):
        raise ValueError("Invalid local terrain query parameters")
    if (points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 3
            or ray_hits.ndim != 3 or ray_hits.shape[0] != points.shape[0]
            or ray_hits.shape[-1] != 3 or ray_hits.shape[1] < 1):
        raise ValueError("Expected [N,2,P,3] points and nonempty [N,R,3] terrain hits")
    finite = torch.isfinite(ray_hits).all(-1)[:, None, None] & torch.isfinite(points).all(-1)[..., None]
    distance2 = (points[..., None, :2] - ray_hits[:, None, None, :, :2]).square().sum(-1)
    nearby = finite & (distance2 <= radius**2)
    distances = torch.where(nearby, distance2, torch.inf)
    heights = ray_hits[:, None, None, :, 2].expand_as(distances)
    closest = distances.argmin(-1, keepdim=True)
    ground = heights.gather(-1, closest).squeeze(-1)
    low = torch.where(nearby, heights, torch.inf).amin(-1)
    high = torch.where(nearby, heights, -torch.inf).amax(-1)
    valid = nearby.any(-1) & ((high - low) <= height_tolerance)
    return torch.where(valid, ground, 0.), valid


def flat_gate(foot_pos, ray_hits, radius=.3, near_radius=.15, min_points=3, full_span=.01, off_span=.03):
    """Flatness of BOTH foot neighborhoods together; no stair-level loophole."""
    if not (0 < near_radius <= radius and 0 <= full_span < off_span and min_points >= 1):
        raise ValueError("Invalid flat terrain gate parameters")
    if (foot_pos.ndim != 3 or foot_pos.shape[1:] != (2, 3)
            or ray_hits.ndim != 3 or ray_hits.shape[0] != foot_pos.shape[0]
            or ray_hits.shape[-1] != 3 or ray_hits.shape[1] < 1):
        raise ValueError("Expected two feet and a nonempty terrain scan")
    finite = torch.isfinite(ray_hits).all(-1)[:, None] & torch.isfinite(foot_pos).all(-1)[..., None]
    distance2 = (foot_pos[..., None, :2] - ray_hits[:, None, :, :2]).square().sum(-1)
    nearby = finite & (distance2 <= radius**2)
    covered = (nearby.sum(-1) >= min_points) & (finite & (distance2 <= near_radius**2)).any(-1)
    heights = ray_hits[:, None, :, 2].expand_as(distance2)
    low = torch.where(nearby, heights, torch.inf).amin(dim=(-1, -2))
    high = torch.where(nearby, heights, -torch.inf).amax(dim=(-1, -2))
    gain = ((off_span - (high - low)) / (off_span - full_span)).clamp(0., 1.)
    return torch.where(covered.all(-1), gain, 0.)


def near_ground_descent_cost(env, clearance_threshold=.08, command_epsilon=1e-6):
    """Sum feet/mean sole samples of downward vz squared, near touchdown only.

    Taper linearly from one at ground to zero at 8 cm. Already contacting,
    upward, high, penetrated, invalid, and strict zero-command samples are off.
    Raw units are (m/s)^2, NOT Nm^2: coefficient ratios do not imply equal
    reward-contribution ratios. RewardManager applies control dt exactly once.
    """
    if clearance_threshold <= 0 or command_epsilon < 0 or not math.isfinite(clearance_threshold + command_epsilon):
        raise ValueError("Invalid landing threshold or command epsilon")
    height, velocity = env.soft_step_clearance, env.soft_step_sole_vz
    valid = env.soft_step_ground_valid & torch.isfinite(height) & torch.isfinite(velocity)
    active = (valid & (height >= 0) & (height < clearance_threshold)
              & ~env.soft_step_contact[..., None])
    # Mask BEFORE squaring: inf/nan from invalid scans must never poison zero.
    downward = torch.where(active, (-velocity).clamp(min=0.), 0.)
    taper = torch.where(active, (1 - height / clearance_threshold).clamp(0., 1.), 0.)
    value = (downward.square() * taper).mean(-1).sum(-1)
    command = env.command_generator.command
    moving = torch.isfinite(command).all(-1) & (command.abs().amax(-1) > command_epsilon)
    return torch.where(moving, value, 0.)


def support_knee_pose_l1(env, asset_cfg, command_epsilon=1e-6):
    """Only a small increment to existing actual knee/default L1 on support.

    Flat terrain, moving command, positive support Fz, upright (<20 degrees),
    no foot stumble. No swing/obstacle/recovery or zero-command reinforcement.
    Default knee remains the asset's 0.6 rad, never a straight-knee hard limit.
    """
    if command_epsilon < 0 or not math.isfinite(command_epsilon):
        raise ValueError("Invalid command epsilon")
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        raise ValueError("Explicit left/right knee joint IDs are required")
    ids = tuple(ids)
    names = tuple(env.robot.joint_names)
    if (len(ids) != 2 or any(not isinstance(i, int) or i < 0 or i >= len(names) for i in ids)
            or tuple(names[i] for i in ids) != KNEE_JOINT_NAMES):
        raise ValueError("Knee IDs must resolve left/right knee names in preserved order")
    data = env.robot.data
    deviation = (data.joint_pos[:, list(ids)] - data.default_joint_pos[:, list(ids)]).abs()
    _finite(deviation, "actual knee/default position")
    command = env.command_generator.command
    moving = torch.isfinite(command).all(-1) & (command.abs().amax(-1) > command_epsilon)
    upright = torch.isfinite(data.projected_gravity_b).all(-1) & (-data.projected_gravity_b[:, 2] > math.cos(math.radians(20)))
    active = moving & upright & ~env.soft_step_stumble
    value = (deviation * env.soft_step_contact).sum(-1) * env.soft_step_flat_gate
    return torch.where(active, value, 0.)
