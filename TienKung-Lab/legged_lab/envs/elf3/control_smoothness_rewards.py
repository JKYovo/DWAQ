"""TienKung-style action cost, without changing actions, PD, or history state."""
import torch


def action_smooth_parts(history):
    """Raw dimensionless sums over all 29 joints, oldest-to-newest history."""
    if history.ndim != 3 or history.shape[1] < 3 or history.shape[2] != 29:
        raise ValueError("action_smooth requires at least three frames of 29 actions")
    current, previous, older = history[:, -1], history[:, -2], history[:, -3]
    first = (current - previous).square().sum(-1)
    second = (current + older - 2 * previous).square().sum(-1)
    amplitude = .05 * current.abs().sum(-1)
    return first, second, amplitude


def action_smooth(env):
    # Read the existing ring once. It already handles wraparound and per-env
    # reset/first-frame padding. Never add another action or observation buffer.
    parts = action_smooth_parts(env.action_buffer._circular_buffer.buffer)
    return parts[0] + parts[1] + parts[2]
