"""Moving-only, one-sided terrain-relative height cost; no hard knee constraint."""
import math

import torch


def support_surface(foot_pos, hits, foot_fz, radius=.3, near_radius=.15,
                    min_points=3, flat_full=.01, flat_off=.03):
    """Use finite local scan points and positive actual support force.

    Both foot neighborhoods must be covered and jointly flat. A foot on each
    stair level must NOT appear flat by checking each foot separately. Invalid
    scans have no torso-derived fallback: return valid=False / gate=0 instead.
    No state, filter or full PhysX metadata query is performed.
    """
    if not (0 < near_radius <= radius and 0 <= flat_full < flat_off and min_points >= 1):
        raise ValueError("Invalid support-surface geometry parameters")
    if (foot_pos.ndim != 3 or foot_pos.shape[1:] != (2,3) or foot_fz.shape != foot_pos.shape[:2]
            or hits.ndim != 3 or hits.shape[0] != foot_pos.shape[0] or hits.shape[-1] != 3 or hits.shape[1] < 1):
        raise ValueError("Expected feet [N,2,3], forces [N,2], nonempty hits [N,R,3]")
    finite = hits.isfinite().all(-1)[:,None] & foot_pos.isfinite().all(-1)[...,None]
    distance2 = (foot_pos[...,None,:2] - hits[:,None,:,:2]).square().sum(-1)
    nearby = finite & (distance2 <= radius**2)
    count = nearby.sum(-1)
    covered = (count >= min_points) & (finite & (distance2 <= near_radius**2)).any(-1)
    z = hits[:,None,:,2].expand_as(distance2)
    safe_z = torch.where(nearby,z,0.)
    local_height = safe_z.sum(-1) / count.clamp(min=1)
    low = torch.where(nearby,z,torch.inf).amin(dim=(-1,-2))
    high = torch.where(nearby,z,-torch.inf).amax(dim=(-1,-2))
    flat = ((flat_off - (high-low)) / (flat_off-flat_full)).clamp(0.,1.)
    weights = torch.where(covered & foot_fz.isfinite() & (foot_fz>1.),foot_fz,0.)
    total = weights.sum(-1)
    valid = covered.all(-1) & (total>0.) & low.isfinite() & high.isfinite()
    ground = (local_height * weights).sum(-1) / total.clamp(min=1e-12)
    return torch.where(valid,ground,0.), torch.where(valid,flat,0.), valid


def moving_base_height_l2(env, target_height, deadband=.015, command_epsilon=1e-6):
    """Squared shortfall below (default torso/sole height - deadband), in m².

    This adapts the existing base_height L2 objective to ELF3: moving commands
    only, positive actual support, local ground, one-sided error. It does not
    reward jumping, pull an already-high torso down, or touch standing rewards.
    RewardManager applies the configured negative weight and control dt once.
    """
    if (not all(math.isfinite(x) for x in (target_height,deadband,command_epsilon))
            or target_height<=0 or not 0<=deadband<target_height or command_epsilon<0):
        raise ValueError("Invalid ELF3 moving-height parameters")
    command = env.command_generator.command
    height = env.height_force_relative_height
    gate = env.height_force_terrain_gate
    valid = env.height_force_ground_valid
    if command.shape != (height.shape[0],3) or gate.shape != height.shape or valid.shape != height.shape:
        raise ValueError("Height/gate/valid must be [N], command [N,3]")
    active = (command.isfinite().all(-1) & (command.abs().amax(-1)>command_epsilon)
              & valid & height.isfinite() & gate.isfinite())
    # Mask invalid slots BEFORE squaring, so disabled NaN scans cannot poison rewards.
    shortfall = torch.where(active,(target_height-deadband-height).clamp(min=0.),0.)
    return shortfall.square() * torch.where(active,gate,0.)
