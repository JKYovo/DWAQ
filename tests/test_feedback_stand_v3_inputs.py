"""CPU-only checks of mirror-invariant, strict-zero stand-v3 actor inputs."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "TienKung-Lab/legged_lab/envs/elf3"
SPEC = importlib.util.spec_from_file_location("cpu_feedback_stand_v3_inputs", DIRECTORY / "feedback_stand_v3_inputs.py")
inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inputs)


def actual_mirrors():
    contract = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
    source = DIRECTORY / "symmetry.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.ImportFrom) and node.module == "legged_lab.elf3_contract")]
    namespace = {"JOINT_NAMES": tuple(contract["joint_names"])}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["mirror_observations"], namespace["mirror_history"]


class FeedbackStandV3InputTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.obs = torch.randn(16, 100, generator=generator)
        self.history = torch.randn(16, 500, generator=generator)

    def test_zero_fixes_current_and_all_history_but_not_other_features(self):
        self.obs[:, 6:9] = 0.
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        expected = self.obs.new_tensor([0., 0., 1., 1.])
        torch.testing.assert_close(obs[:, 96:100], expected.expand(16, -1))
        torch.testing.assert_close(history.reshape(16, 5, 100)[:, :, 96:100], expected.expand(16, 5, -1))
        self.assertTrue(torch.equal(obs[:, :96], self.obs[:, :96]))
        self.assertTrue(torch.equal(history.reshape(16, 5, 100)[:, :, :96], self.history.reshape(16, 5, 100)[:, :, :96]))

    def test_moving_including_pure_yaw_bitwise_unchanged(self):
        self.obs[:8, 6:9] = self.obs.new_tensor([.1, 0., 0.])
        self.obs[8:, 6:9] = self.obs.new_tensor([0., 0., .1])
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertTrue(torch.equal(obs, self.obs))
        self.assertTrue(torch.equal(history, self.history))

    def test_all_signed_velocity_axes_and_forward_point_three_one_stay_unchanged(self):
        commands = self.obs.new_tensor([[.1, 0., 0.], [-.1, 0., 0.], [0., .1, 0.],
                                        [0., -.1, 0.], [0., 0., .1], [0., 0., -.1],
                                        [.3, 0., 0.], [1., 0., 0.]])
        self.obs[:, 6:9] = commands.repeat(2, 1)
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertTrue(torch.equal(obs, self.obs))
        self.assertTrue(torch.equal(history, self.history))

    def test_inputs_and_storage_never_mutated_or_aliased(self):
        self.obs[:, 6:9] = 0.
        original_obs, original_history = self.obs.clone(), self.history.clone()
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertTrue(torch.equal(self.obs, original_obs))
        self.assertTrue(torch.equal(self.history, original_history))
        self.assertNotEqual(obs.data_ptr(), self.obs.data_ptr())
        self.assertNotEqual(history.data_ptr(), self.history.data_ptr())
        obs.fill_(99.)
        history.fill_(-99.)
        self.assertTrue(torch.equal(self.obs, original_obs))
        self.assertTrue(torch.equal(self.history, original_history))

    def test_current_zero_applies_immediately_despite_old_moving_history(self):
        self.obs[:, 6:9] = 0.
        self.history.reshape(16, 5, 100)[:, :, 6:9] = .6
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertTrue(torch.equal(obs[:, 96:100], self.obs.new_tensor([0., 0., 1., 1.]).expand(16, -1)))
        self.assertTrue(bool((history.reshape(16, 5, 100)[:, :, 98:100] == 1.).all()))
        self.assertTrue(bool((history.reshape(16, 5, 100)[:, :, 6:9] == .6).all()))

    def test_exact_epsilon_boundary_and_nonfinite_commands(self):
        self.obs[:, 6:9] = 0.
        self.obs[0, 6] = 1e-6
        self.obs[1, 6] = 1.01e-6
        self.obs[2, 6] = float("nan")
        self.obs[3, 6] = float("inf")
        obs, history = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertEqual(obs[0, 96:100].tolist(), [0., 0., 1., 1.])
        self.assertTrue(torch.equal(obs[1:4, 96:100], self.obs[1:4, 96:100]))
        self.assertTrue(torch.equal(history[1:4], self.history[1:4]))

    def test_mirror_and_prepare_commute_for_zero_and_moving_full_history(self):
        self.obs[:8, 6:9] = 0.
        self.obs[8:12, 6:9] = self.obs.new_tensor([.1, 0., 0.])
        self.obs[12:, 6:9] = self.obs.new_tensor([0., 0., -.1])
        mirror, mirror_history = actual_mirrors()
        prepared_obs, prepared_history = inputs.prepare_stand_inputs(self.obs, self.history)
        mirrored_then_prepared = inputs.prepare_stand_inputs(mirror(self.obs), mirror_history(self.history))
        self.assertTrue(torch.equal(mirror(prepared_obs), mirrored_then_prepared[0]))
        self.assertTrue(torch.equal(mirror_history(prepared_history), mirrored_then_prepared[1]))

    def test_repeated_prepare_idempotent_and_moving_switch_restores_raw_history(self):
        self.obs[:, 6:9] = 0.
        first = inputs.prepare_stand_inputs(self.obs, self.history)
        second = inputs.prepare_stand_inputs(*first)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.obs[:, 6] = .1
        moving = inputs.prepare_stand_inputs(self.obs, self.history)
        self.assertTrue(torch.equal(moving[0], self.obs))
        self.assertTrue(torch.equal(moving[1], self.history))

    def test_train_eval_deterministic_without_cache(self):
        self.obs[:, 6:9] = 0.
        first = inputs.prepare_stand_inputs(self.obs, self.history)
        torch.manual_seed(987)
        second = inputs.prepare_stand_inputs(self.obs.clone(), self.history.clone())
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_zero_removes_phase_only_variation_from_current_and_five_frame_history(self):
        self.obs[:, 6:9] = 0.
        other_obs, other_history = self.obs.clone(), self.history.clone()
        other_obs[:, 96:100] += 1.23
        other_history.reshape(16, 5, 100)[:, :, 96:100] -= .47
        first = inputs.prepare_stand_inputs(self.obs, self.history)
        second = inputs.prepare_stand_inputs(other_obs, other_history)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.obs[:, 6] = other_obs[:, 6] = .3
        first = inputs.prepare_stand_inputs(self.obs, self.history)
        second = inputs.prepare_stand_inputs(other_obs, other_history)
        self.assertFalse(torch.equal(first[0][:, 96:100], second[0][:, 96:100]))
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_invalid_shapes_types_or_epsilon_fail(self):
        for obs, history in ((torch.zeros(16, 96), self.history), (self.obs, torch.zeros(16, 400)),
                             (self.obs, torch.zeros(15, 500)), (self.obs, torch.zeros(16, 5, 100)),
                             (torch.zeros(16, 100, dtype=torch.int64), self.history)):
            with self.subTest(obs_shape=obs.shape, history_shape=history.shape), self.assertRaises(ValueError):
                inputs.prepare_stand_inputs(obs, history)
        for epsilon in (-1., float("nan"), float("inf"), True, "invalid"):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                inputs.prepare_stand_inputs(self.obs, self.history, epsilon)


if __name__ == "__main__":
    unittest.main()
