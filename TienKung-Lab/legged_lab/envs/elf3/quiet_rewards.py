"""Task-private zero-command eligibility and actual arm-motion reward.

Eligibility uses only the current velocity command. There is deliberately no
support, torso-motion, tilt, history or already-settled precondition, and no
timer or per-episode eligibility cache. Policy observations and physics remain
unchanged.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def zero_command_mask(env, epsilon: float = 1e-6):
    """Return a fresh mask for finite zero vx/vy/yaw commands immediately."""
    if not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("command epsilon must be finite and non-negative")
    command = env.command_generator.command[:, :3]
    if command.ndim != 2 or command.shape != (env.num_envs, 3):
        raise ValueError("Zero-command eligibility requires three commands per environment")
    return torch.isfinite(command).all(-1) & (command.abs().amax(dim=-1) <= epsilon)


def upper_body_stillness(env, asset_cfg, velocity_std: float = 0.05):
    """Bounded positive reward for low *physical* selected-joint velocity.

    The RewardManager resolves ``asset_cfg.joint_ids`` in simulator order. Do
    not index raw simulator tensors using native policy action IDs, and do not
    substitute the filtered velocity seen by the Actor.
    """
    if not math.isfinite(float(velocity_std)) or velocity_std <= 0:
        raise ValueError("velocity_std must be finite and positive")
    epsilon = env.cfg.quiet_standing.get("command_epsilon", 1e-6)
    active = zero_command_mask(env, epsilon=epsilon)
    velocity = env.robot.data.joint_vel[:, asset_cfg.joint_ids]
    if velocity.ndim != 2 or velocity.shape[0] != env.num_envs or velocity.shape[1] == 0:
        raise ValueError("upper_body_stillness requires resolved nonempty joint indices")
    value = torch.exp(-torch.square(velocity / velocity_std).mean(dim=-1))
    return torch.where(active, value, torch.zeros_like(value))
