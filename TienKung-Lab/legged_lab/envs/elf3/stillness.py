"""Minimal, tensor-only zero-command treatment for the ELF3 DWAQ actor.

The phase substitution is an input transformation, not an action clamp or a
modification of the environment's clock. Keep raw history buffers unchanged so
that switching back to walking sees the same history as the baseline task.
"""
from __future__ import annotations

import math

import torch


COMMAND_EPSILON = 1e-6
FIXED_PHASE = (0.0, 0.0, 1.0, -1.0)


def prepare_still_inputs(
    observations: torch.Tensor,
    history: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fix current and all five historical phases at a current zero command.

    Only command entries 6:9 decide eligibility. Old moving commands in history
    deliberately do not delay activation. Returned tensors never alias the
    supplied observations/history; the original raw history remains available
    when a subsequent nonzero command resumes walking.
    """
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("command epsilon must be finite and non-negative")
    if observations.dim() != 2 or observations.shape[1] != 100:
        raise ValueError("ELF3 stillness requires observations shaped [batch, 100]")
    if history.dim() != 2 or history.shape[1] != 500 or history.shape[0] != observations.shape[0]:
        raise ValueError("ELF3 stillness requires history shaped [batch, 500]")

    command = observations[:, 6:9]
    active = torch.isfinite(command).all(dim=-1) & (command.abs().amax(dim=-1) <= epsilon)
    fixed = torch.tensor([0.0, 0.0, 1.0, -1.0], dtype=observations.dtype, device=observations.device)
    prepared_observations = observations.clone()
    prepared_history = history.clone().reshape(-1, 5, 100)
    prepared_observations[:, 96:100] = torch.where(
        active.unsqueeze(-1), fixed, prepared_observations[:, 96:100]
    )
    prepared_history[:, :, 96:100] = torch.where(
        active.reshape(-1, 1, 1), fixed, prepared_history[:, :, 96:100]
    )
    return prepared_observations, prepared_history.reshape(-1, 500)


def stand_joint_motion(env, asset_cfg, command_epsilon: float = COMMAND_EPSILON):
    """Dense zero-command L1 cost on actual, raw selected-joint velocities.

    ``asset_cfg.joint_ids`` must be resolved by the RewardManager in simulator
    order. Configure this term for all 29 ELF3 joints with a negative weight;
    no already-settled/contact/history criterion or exponential saturation is
    applied. The filtered velocity seen by the actor is not used here.
    """
    if not math.isfinite(float(command_epsilon)) or command_epsilon < 0.0:
        raise ValueError("command epsilon must be finite and non-negative")
    command = env.command_generator.command[:, :3]
    if command.ndim != 2 or command.shape != (env.num_envs, 3):
        raise ValueError("stand_joint_motion requires three commands per environment")
    velocity = env.robot.data.joint_vel[:, asset_cfg.joint_ids]
    if velocity.ndim != 2 or velocity.shape[0] != env.num_envs or velocity.shape[1] == 0:
        raise ValueError("stand_joint_motion requires nonempty resolved joint indices")
    active = torch.isfinite(command).all(dim=-1) & (command.abs().amax(dim=-1) <= command_epsilon)
    value = velocity.abs().mean(dim=-1)
    return torch.where(active, value, torch.zeros_like(value))
