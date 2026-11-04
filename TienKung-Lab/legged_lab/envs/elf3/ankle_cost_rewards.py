"""Two isolated, read-only costs for the ELF3 ankle-effort experiment.

The torque channel is Isaac Lab's actuator ``applied_torque`` estimate. For an
implicit actuator it is not a measured hardware torque or an exact recovered
PhysX constraint force. The foot-speed channel must be populated once per
control step by the experiment environment from the mean of the four physical
substep speed norms, in the explicit left/right body order below.

No action filtering, feedback dynamics, standing objective, or simulator state
is changed here. These functions deliberately have no Isaac Sim imports so the
named-joint selection and reward mathematics can be tested on the CPU.
"""
from __future__ import annotations

import math

import torch


ANKLE_JOINT_NAMES = (
    "l_ankle_y_joint", "l_ankle_x_joint",
    "r_ankle_y_joint", "r_ankle_x_joint",
)
FOOT_BODY_NAMES = ("l_ankle_x_link", "r_ankle_x_link")


def _finite(value: torch.Tensor, name: str):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    # This assertion does not copy a GPU scalar to the CPU on every reward
    # evaluation. It is synchronous only in the CPU unit tests.
    torch._assert_async(torch.isfinite(value).all(), f"{name} must be finite")


def ankle_torque_l2(env, asset_cfg) -> torch.Tensor:
    """Return the sum of squared estimated torques for exactly four ankles.

    ``asset_cfg`` is a resolved SceneEntityCfg containing the four explicitly
    named ankle joints. Joint indices are never inferred from a G1 ordering.
    The cost applies at every command, including exact zero command.
    """
    asset = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    if isinstance(indices, slice):
        raise ValueError("ankle_torque_l2 requires explicit resolved ankle joint_ids, not a slice")
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1:
            raise ValueError("ankle joint_ids must be one-dimensional")
        indices = indices.tolist()
    indices = tuple(indices)
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("ankle_torque_l2 requires exactly four distinct ankle joint_ids")
    names = tuple(asset.joint_names)
    if any(not isinstance(index, int) or index < 0 or index >= len(names) for index in indices):
        raise ValueError("Invalid resolved ankle joint index")
    if set(names[index] for index in indices) != set(ANKLE_JOINT_NAMES):
        raise ValueError("Resolved ankle joint names do not match the four ELF3 ankle joints")
    torque = asset.data.applied_torque
    if torque.ndim != 2 or torque.shape[1] != len(names):
        raise ValueError("applied_torque must have shape [environment, simulator_joint]")
    selected = torque[:, list(indices)]
    _finite(selected, "ankle applied_torque")
    return selected.square().sum(dim=1)


def stance_foot_speed_cost(
    env,
    stance_threshold: float = 0.55,
    phase_sigma: float = 0.015,
    speed_scale: float = 200.0,
    command_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Bounded, continuous clock-stance speed cost; exact zero command is off.

    C = sum_feet I_stance(phase) * (1 - exp(-speed_scale * mean_speed**2)).

    ``I_stance`` is the periodic Gaussian smoothing of phase in [0, threshold).
    It is continuous at both the stance/swing boundary and phase wrap. This is
    equivalent in action-dependent gradients to the native TienKung positive
    stance-speed exponential, up to a phase-only constant. No actual-contact
    gate is added: the experiment matches the original clock-stance objective.
    """
    if not math.isfinite(stance_threshold) or not 0 < stance_threshold < 1:
        raise ValueError("stance_threshold must be strictly between zero and one")
    if not math.isfinite(phase_sigma) or not 0 < phase_sigma <= 0.1:
        raise ValueError("phase_sigma must be finite and in (0, 0.1]")
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError("speed_scale must be finite and positive")
    if not math.isfinite(command_epsilon) or command_epsilon < 0:
        raise ValueError("command_epsilon must be finite and nonnegative")
    if tuple(getattr(env, "ankle_cost_feet_body_names", ())) != FOOT_BODY_NAMES:
        raise ValueError("Foot-speed cache must explicitly use left/right ELF3 ankle-x body order")
    speed = env.ankle_cost_feet_speed_mean
    phase = env.leg_phase
    command = env.command_generator.command
    for value, name in ((speed, "mean foot speed"), (phase, "leg phase"), (command, "command")):
        _finite(value, name)
    if speed.ndim != 2 or speed.shape[1] != 2 or phase.shape != speed.shape:
        raise ValueError("Mean foot speed and leg phase must both have shape [environment, 2]")
    if command.shape != (speed.shape[0], 3):
        raise ValueError("command must have shape [environment, 3]")
    if phase.device != speed.device or command.device != speed.device:
        raise ValueError("Foot speed, phase, and command must be on the same device")
    torch._assert_async((speed >= 0).all(), "mean foot speed must be a nonnegative norm")

    wrapped = torch.remainder(phase, 1.0)
    denominator = math.sqrt(2.0) * phase_sigma
    stance = torch.zeros_like(wrapped)
    # The nearest three periods cover the Gaussian tails to numerical precision
    # for sigma <= 0.1, including both sides of the phase-0/phase-1 boundary.
    for shift in (-1.0, 0.0, 1.0):
        stance = stance + 0.5 * (
            torch.erf((stance_threshold + shift - wrapped) / denominator)
            - torch.erf((shift - wrapped) / denominator)
        )
    stance = stance.clamp(0.0, 1.0)
    cost = (stance * (-torch.expm1(-speed_scale * speed.square()))).sum(dim=1)
    moving = command.abs().amax(dim=1) > command_epsilon
    return torch.where(moving, cost, torch.zeros_like(cost))
