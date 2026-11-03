"""Private command gates; the inherited walking reward implementations stay intact.

The phase clock still runs and remains in the policy observation. At a true zero
command it must not pay the robot to alternate its contacts in place. These
functions have no timer/cache, so command changes and resets cannot leak state.
"""
import torch
from isaaclab.managers import SceneEntityCfg

from legged_lab.envs.elf3.standing_contract import COMMAND_EPSILON
from legged_lab.mdp.rewards import feet_swing_height, gait_phase_contact


def moving_command_mask(env, command_epsilon: float = COMMAND_EPSILON):
    """Include lateral motion and pure yaw; do not use a low-speed deadband."""
    return env.command_generator.command[:, :3].abs().amax(dim=-1) > command_epsilon


def gait_phase_contact_when_moving(
    env,
    sensor_cfg: SceneEntityCfg,
    stance_threshold: float = 0.55,
    command_epsilon: float = COMMAND_EPSILON,
):
    value = gait_phase_contact(env, sensor_cfg=sensor_cfg, stance_threshold=stance_threshold)
    return torch.where(moving_command_mask(env, command_epsilon), value, torch.zeros_like(value))


def feet_swing_height_when_moving(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08,
    command_epsilon: float = COMMAND_EPSILON,
):
    value = feet_swing_height(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg, target_height=target_height)
    return torch.where(moving_command_mask(env, command_epsilon), value, torch.zeros_like(value))
