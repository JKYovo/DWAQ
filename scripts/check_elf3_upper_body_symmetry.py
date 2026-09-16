"""Checks for the ELF3 upper-body DWAQ mirror loss."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import torch

from legged_lab.envs.elf3.symmetry import (
    ARM_ACTION_IDS,
    JOINT_MIRROR_IDS,
    mirror_actions,
    mirror_history,
    mirror_observations,
)
from rsl_rl.algorithms import DWAQPPO
from rsl_rl.modules import ActorCritic_DWAQ


torch.manual_seed(42)
obs = torch.randn(8, 100)
history = torch.randn(8, 500)
actions = torch.randn(8, 29)

torch.testing.assert_close(mirror_observations(mirror_observations(obs)), obs)
torch.testing.assert_close(mirror_history(mirror_history(history)), history)
torch.testing.assert_close(mirror_actions(mirror_actions(actions)), actions)

# Explicitly verify the shoulder pairs and axial-vector signs.
assert JOINT_MIRROR_IDS[15:18] == (22, 23, 24)
expected_right_shoulder = torch.stack((actions[:, 15], -actions[:, 16], -actions[:, 17]), dim=1)
torch.testing.assert_close(mirror_actions(actions)[:, 22:25], expected_right_shoulder)
assert ARM_ACTION_IDS == tuple(range(15, 29))

policy = ActorCritic_DWAQ(
    num_actor_obs=119,
    num_critic_obs=311,
    num_actions=29,
    cenet_in_dim=500,
    cenet_out_dim=19,
    obs_dim=100,
)
algorithm = DWAQPPO(policy=policy, obs_dim=100)
algorithm.configure_upper_body_symmetry(
    mirror_observations, mirror_history, mirror_actions, ARM_ACTION_IDS, coefficient=0.1
)
loss = algorithm._upper_body_symmetry_loss(obs, history)
assert loss.isfinite() and loss.item() >= 0
loss.backward()
output_weight_grad = policy.actor[-1].weight.grad
assert torch.isfinite(output_weight_grad).all()
assert torch.count_nonzero(output_weight_grad[:15]) == 0
assert torch.count_nonzero(output_weight_grad[15:]) > 0

print({"status": "passed", "raw_upper_body_symmetry_loss": loss.item(), "arm_action_ids": ARM_ACTION_IDS})
simulation_app.close()
