"""Pure, mirror-invariant zero-command phase canonicalization for stand v3."""
from __future__ import annotations

import math

import torch


COMMAND_EPSILON = 1e-6
FIXED_PHASE = (0., 0., 1., 1.)


def prepare_stand_inputs(observations, history, epsilon=COMMAND_EPSILON):
    """Clone 100-D observations/500-D history and fix phases at current zero.

    The latest observation's three command entries determine eligibility for
    all five history frames immediately. Old moving commands do not delay the
    switch. Both legs use the same zero-command phase code, so sagittal mirror
    and this canonicalization commute. Nonzero-command rows remain bitwise
    unchanged. No input, state, action or deployed joint target is modified.
    """
    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be finite and nonnegative")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("epsilon must be finite and nonnegative") from error
    if not math.isfinite(epsilon) or epsilon < 0.:
        raise ValueError("epsilon must be finite and nonnegative")
    if (not isinstance(observations, torch.Tensor) or not observations.is_floating_point()
            or observations.ndim != 2 or observations.shape[1] != 100):
        raise ValueError("Expected floating point observations shaped [batch, 100]")
    if (not isinstance(history, torch.Tensor) or not history.is_floating_point()
            or history.shape != (observations.shape[0], 500)):
        raise ValueError("Expected floating point history shaped [batch, 500]")
    if history.device != observations.device:
        raise ValueError("Current observations and history must share a device")
    command = observations[:, 6:9]
    active = torch.isfinite(command).all(-1) & (command.abs().amax(-1) <= epsilon)
    result = observations.clone()
    result_history = history.clone().reshape(-1, 5, 100)
    result[:, 96:100] = torch.where(
        active[:, None], result.new_tensor(FIXED_PHASE), result[:, 96:100])
    result_history[:, :, 96:100] = torch.where(
        active[:, None, None], result_history.new_tensor(FIXED_PHASE), result_history[:, :, 96:100])
    return result, result_history.reshape(-1, 500)
