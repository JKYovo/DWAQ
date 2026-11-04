"""CPU tests of the actual new environment methods, without importing Isaac.

Only the environment class AST is compiled from the implementation. Its
inherited simulator parent is replaced with a controlled substep harness;
the implementation's methods themselves are not copied or rewritten.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


DIRECTORY = Path(__file__).resolve().parents[1] / "TienKung-Lab/legged_lab/envs/elf3"
REWARD_SPEC = importlib.util.spec_from_file_location("cpu_ankle_cost_rewards", DIRECTORY / "ankle_cost_rewards.py")
rewards = importlib.util.module_from_spec(REWARD_SPEC)
REWARD_SPEC.loader.exec_module(rewards)


class DummyStillFeedbackParent:
    """Minimal stand-in for the original environment/200 Hz feedback hook."""
    def __init__(self, num_envs=3):
        self.num_envs = num_envs
        self.device = "cpu"
        self.cfg = SimpleNamespace(sim=SimpleNamespace(decimation=4))
        self.robot = SimpleNamespace(
            data=SimpleNamespace(body_lin_vel_w=torch.zeros(num_envs, 6, 3)),
            find_bodies=self.find_bodies,
        )
        self.leg_phase = torch.tensor([[.2, .7]]).repeat(num_envs, 1)
        self.command_generator = SimpleNamespace(command=torch.tensor([[.3, 0., 0.]]).repeat(num_envs, 1))
        self.lookup_names = rewards.FOOT_BODY_NAMES
        self.lookup_calls = []
        self.feedback_updates = 0
        self.parent_reset_calls = []
        self.parent_init_calls = 0
        self.physical_samples = []
        self.reward_snapshots = []
        self.expected_result = (
            torch.randn(num_envs, 100), torch.randn(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {"observations": {"obs_hist": torch.randn(num_envs, 500)}},
        )

    def find_bodies(self, names, preserve_order):
        self.lookup_calls.append((tuple(names), preserve_order))
        return [4, 1], list(self.lookup_names)

    def init_buffers(self):
        self.parent_init_calls += 1

    def _post_physics_step(self):
        self.feedback_updates += 1

    def step(self, actions):
        self.seen_actions = actions
        self.seen_actions_copy = actions.clone()
        # Capture what the actual child cleared before invoking its parent.
        self.cache_at_step_start = (
            self._ankle_cost_speed_sum.clone(),
            self.ankle_cost_feet_speed_mean.clone(), self._ankle_cost_substeps,
        )
        for velocity in self.physical_samples:
            self.robot.data.body_lin_vel_w.copy_(velocity)
            self._post_physics_step()
        # The real parent computes rewards after all physical substeps, then
        # performs reset. Repeated reward reads must not update either cache.
        self.reward_snapshots = [self.ankle_cost_feet_speed_mean.clone()]
        rewards.stance_foot_speed_cost(self)
        rewards.stance_foot_speed_cost(self)
        self.reward_snapshots.append(self.ankle_cost_feet_speed_mean.clone())
        return self.expected_result

    def reset(self, env_ids):
        self.parent_reset_calls.append(torch.as_tensor(env_ids).clone())
        if len(env_ids):
            self.robot.data.body_lin_vel_w[env_ids] = 0.


def actual_environment_class():
    source = DIRECTORY / "ankle_cost.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    candidates = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                  and node.name == "Elf3DwaqAnkleCostEnv"]
    if len(candidates) != 1:
        raise AssertionError("Expected the actual uniquely defined ankle-cost environment class")
    node = candidates[0]
    if len(node.bases) != 1 or not isinstance(node.bases[0], ast.Name) or node.bases[0].id != "Elf3DwaqStillEnv":
        raise AssertionError("Original still-v2/feedback environment inheritance changed")
    namespace = {"torch": torch, "FOOT_BODY_NAMES": rewards.FOOT_BODY_NAMES,
                 "Elf3DwaqStillEnv": DummyStillFeedbackParent}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename=str(source), mode="exec"), namespace)
    return namespace["Elf3DwaqAnkleCostEnv"]


Environment = actual_environment_class()


def samples(env, multipliers=(1., 1., 1., 1.)):
    # Each foot's velocity reverses direction; mean(norm(v)) stays nonzero,
    # while norm(mean(v)) is zero when the four magnitudes are equal.
    left = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.))
    right = ((0., 2., 0.), (0., -2., 0.), (0., 0., 2.), (0., 0., -2.))
    scale = torch.arange(1, env.num_envs + 1).reshape(-1, 1)
    result = []
    for index, multiplier in enumerate(multipliers):
        velocity = torch.full((env.num_envs, 6, 3), 999.)
        velocity[:, 4] = scale * torch.tensor(left[index % 4]) * multiplier
        velocity[:, 1] = scale * torch.tensor(right[index % 4]) * multiplier
        result.append(velocity)
    return result


class TestAnkleCostActualEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.env.init_buffers()
        self.env.physical_samples = samples(self.env)
        self.action = torch.randn(self.env.num_envs, 29)

    def test_named_body_resolution_preserves_left_right(self):
        self.assertEqual(self.env.parent_init_calls, 1)
        self.assertEqual(self.env.lookup_calls, [(rewards.FOOT_BODY_NAMES, True)])
        self.assertEqual(tuple(self.env._ankle_cost_feet_ids), (4, 1))
        self.assertEqual(self.env.ankle_cost_feet_body_names, rewards.FOOT_BODY_NAMES)

    def test_mean_of_four_norms_not_norm_of_mean(self):
        self.env.step(self.action)
        expected = torch.tensor([[1., 2.], [2., 4.], [3., 6.]])
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean, expected)
        torch.testing.assert_close(self.env._ankle_cost_speed_sum, expected * 4)
        raw_mean = torch.stack(self.env.physical_samples).mean(dim=0)[:, [4, 1]].norm(dim=-1)
        torch.testing.assert_close(raw_mean, torch.zeros_like(raw_mean))
        self.assertFalse(torch.equal(raw_mean, self.env.ankle_cost_feet_speed_mean))

    def test_nonconstant_samples_correct_average_once(self):
        self.env.physical_samples = samples(self.env, (1., 2., 3., 4.))
        self.env.step(self.action)
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean,
                                   torch.tensor([[2.5, 5.], [5., 10.], [7.5, 15.]]))
        self.assertEqual(self.env._ankle_cost_substeps, 4)
        self.assertEqual(self.env.feedback_updates, 4)

    def test_reward_reads_do_not_update_mean_or_feedback(self):
        self.env.step(self.action)
        before = (self.env._ankle_cost_speed_sum.clone(), self.env.ankle_cost_feet_speed_mean.clone(),
                  self.env._ankle_cost_substeps, self.env.feedback_updates)
        for _ in range(3):
            rewards.stance_foot_speed_cost(self.env)
        torch.testing.assert_close(self.env.reward_snapshots[0], self.env.reward_snapshots[1])
        torch.testing.assert_close(self.env._ankle_cost_speed_sum, before[0])
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean, before[1])
        self.assertEqual((self.env._ankle_cost_substeps, self.env.feedback_updates), before[2:])

    def test_each_step_clears_previous_cache_before_parent(self):
        self.env._ankle_cost_speed_sum.fill_(123.)
        self.env.ankle_cost_feet_speed_mean.fill_(456.)
        self.env._ankle_cost_substeps = 99
        self.env.step(self.action)
        total, mean, count = self.env.cache_at_step_start
        torch.testing.assert_close(total, torch.zeros_like(total))
        torch.testing.assert_close(mean, torch.zeros_like(mean))
        self.assertEqual(count, 0)
        self.env.physical_samples = [torch.zeros_like(value) for value in self.env.physical_samples]
        self.env.step(self.action)
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean, torch.zeros(self.env.num_envs, 2))
        self.assertEqual(self.env.feedback_updates, 8)
        self.assertEqual(self.env._ankle_cost_substeps, 4)

    def test_reset_only_selected_rows_and_preserves_shared_step_counter(self):
        self.env.step(self.action)
        mean = self.env.ankle_cost_feet_speed_mean.clone()
        total = self.env._ankle_cost_speed_sum.clone()
        self.env.reset(torch.tensor([1]))
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean[[0, 2]], mean[[0, 2]])
        torch.testing.assert_close(self.env._ankle_cost_speed_sum[[0, 2]], total[[0, 2]])
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean[1], torch.zeros(2))
        torch.testing.assert_close(self.env._ankle_cost_speed_sum[1], torch.zeros(2))
        self.assertEqual(self.env._ankle_cost_substeps, 4)
        self.assertEqual(self.env.feedback_updates, 4)
        torch.testing.assert_close(self.env.parent_reset_calls[-1], torch.tensor([1]))

    def test_empty_reset_no_new_rows_touched(self):
        self.env.step(self.action)
        before = self.env.ankle_cost_feet_speed_mean.clone()
        self.env.reset(torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(self.env.ankle_cost_feet_speed_mean, before)

    def test_early_reset_before_new_buffers_is_safe(self):
        env = Environment()
        env.reset(torch.tensor([0]))
        self.assertFalse(hasattr(env, "ankle_cost_feet_speed_mean"))
        torch.testing.assert_close(env.parent_reset_calls[-1], torch.tensor([0]))

    def test_actions_and_parent_return_are_passed_through_unchanged(self):
        action_before = self.action.clone()
        result = self.env.step(self.action)
        self.assertIs(self.env.seen_actions, self.action)
        torch.testing.assert_close(self.action, action_before)
        torch.testing.assert_close(self.env.seen_actions_copy, action_before)
        self.assertIs(result, self.env.expected_result)
        for returned, expected in zip(result, self.env.expected_result):
            self.assertIs(returned, expected)

    def test_missing_or_extra_physical_substeps_rejected(self):
        for count in (0, 1, 3, 5):
            with self.subTest(count=count):
                env = Environment()
                env.init_buffers()
                env.physical_samples = samples(env, (1.,) * count)
                with self.assertRaisesRegex(RuntimeError, "every physical substep exactly once"):
                    env.step(self.action)
                self.assertEqual(env.feedback_updates, count)

    def test_wrong_foot_order_cannot_silently_enter_cache(self):
        env = Environment()
        env.lookup_names = tuple(reversed(rewards.FOOT_BODY_NAMES))
        with self.assertRaisesRegex(ValueError, "explicitly left/right"):
            env.init_buffers()


if __name__ == "__main__":
    unittest.main()
