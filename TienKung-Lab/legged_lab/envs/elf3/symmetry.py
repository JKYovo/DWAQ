"""Sagittal mirror transforms for the ELF3 DWAQ policy contract."""

from __future__ import annotations

import torch

from legged_lab.assets.elf3 import JOINT_NAMES


OBS_DIM = 100
ARM_ACTION_IDS = tuple(range(JOINT_NAMES.index("l_shoulder_y_joint"), len(JOINT_NAMES)))


def _mirrored_name(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def _joint_mirror_sign(name: str) -> float:
    # Revolute joint coordinates are axial vectors. Under reflection across
    # the sagittal X-Z plane, rotations about X/Z change sign and Y does not.
    axis = name.rsplit("_", 2)[-2]
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Cannot infer ELF3 joint axis from {name!r}")
    return 1.0 if axis == "y" else -1.0


JOINT_MIRROR_IDS = tuple(JOINT_NAMES.index(_mirrored_name(name)) for name in JOINT_NAMES)
JOINT_MIRROR_SIGNS = tuple(_joint_mirror_sign(name) for name in JOINT_NAMES)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror normalized ELF3 actions without limiting their magnitude."""
    if actions.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} actions, got {actions.shape[-1]}")
    signs = actions.new_tensor(JOINT_MIRROR_SIGNS)
    return actions[..., JOINT_MIRROR_IDS] * signs


def mirror_observations(observations: torch.Tensor) -> torch.Tensor:
    """Mirror one or more 100-dimensional ELF3 actor observation frames."""
    if observations.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected {OBS_DIM} observations, got {observations.shape[-1]}")

    mirrored = observations.clone()
    # root_ang_vel_b is axial; projected gravity and linear command are polar.
    mirrored[..., 0:3] = observations[..., 0:3] * observations.new_tensor((-1.0, 1.0, -1.0))
    mirrored[..., 3:6] = observations[..., 3:6] * observations.new_tensor((1.0, -1.0, 1.0))
    # command = [vx, vy, wz], where wz is axial.
    mirrored[..., 6:9] = observations[..., 6:9] * observations.new_tensor((1.0, -1.0, -1.0))

    # joint position offsets, joint velocities, and previous actions.
    for start in (9, 38, 67):
        mirrored[..., start : start + 29] = mirror_actions(observations[..., start : start + 29])

    # [sin(left), sin(right), cos(left), cos(right)].
    mirrored[..., 96:100] = observations[..., (97, 96, 99, 98)]
    return mirrored


def mirror_history(history: torch.Tensor) -> torch.Tensor:
    """Mirror every frame in a flattened DWAQ observation history."""
    if history.shape[-1] % OBS_DIM:
        raise ValueError(f"History dimension {history.shape[-1]} is not divisible by {OBS_DIM}")
    original_shape = history.shape
    frames = history.reshape(*original_shape[:-1], -1, OBS_DIM)
    return mirror_observations(frames).reshape(original_shape)

