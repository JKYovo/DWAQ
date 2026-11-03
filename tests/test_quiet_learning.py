"""CPU checks of stand-v2 phase auxiliaries and rollout-label alignment.

The actual policy, PPO, storage and private learning code run. Only package
import containers and the environment's command-mask provider are stubbed.
No Isaac, GPU, simulator, network, training process or robot is started.
"""
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Normal


ROOT = Path(__file__).resolve().parents[1]
RSL = ROOT / "TienKung-Lab/rsl_rl/rsl_rl"
ELF3 = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


policy_module = load_file("quiet_cpu_policy", RSL / "modules/actor_critic_DWAQ.py")
utils_stub = ModuleType("rsl_rl.utils")
utils_stub.split_and_pad_trajectories = lambda *args: (_ for _ in ()).throw(
    AssertionError("Feed-forward tests must not call recurrent trajectory padding"))
with patch.dict(sys.modules, {"rsl_rl.utils": utils_stub}):
    storage_module = load_file("quiet_cpu_storage", RSL / "storage/rollout_storage_dwaq.py")
modules_stub = ModuleType("rsl_rl.modules")
modules_stub.ActorCritic_DWAQ = policy_module.ActorCritic_DWAQ
storage_stub = ModuleType("rsl_rl.storage")
storage_stub.RolloutStorageDWAQ = storage_module.RolloutStorageDWAQ
with patch.dict(sys.modules, {"rsl_rl.modules": modules_stub, "rsl_rl.storage": storage_stub}):
    ppo_module = load_file("quiet_cpu_base_ppo", RSL / "algorithms/dwaq_ppo.py")
with patch.dict(sys.modules, {"rsl_rl.algorithms.dwaq_ppo": ppo_module,
                             "rsl_rl.storage": storage_stub}):
    quiet = load_file("quiet_cpu_private_learning", ELF3 / "quiet_learning.py")
contract = load_file("quiet_cpu_settings", ELF3 / "quiet_contract.py")


def inputs(batch=4, dtype=torch.float32):
    """Realistic clock layout; only proprioception is arbitrary test data."""
    history = torch.linspace(-.2, .2, batch * 500, dtype=dtype).reshape(batch, 5, 100).clone()
    history[:, :, 6:9] = 0
    left = torch.arange(batch, dtype=dtype)[:, None] * .17 + torch.arange(5, dtype=dtype)[None] * .025
    phase = torch.stack((left, left + .5), dim=-1) * (2 * math.pi)
    history[:, :, 96:98] = phase.sin()
    history[:, :, 98:100] = phase.cos()
    observations = history[:, -1].clone()
    return observations, history.reshape(batch, 500)


class CommandMaskProvider:
    def __init__(self, values):
        self.mask = torch.tensor(values, dtype=torch.bool)
        self.num_envs = len(values)

    def get_standing_mask(self):
        # Deliberately return the mutable provider buffer; PPO must snapshot it.
        return self.mask


class PhaseToyPolicy(nn.Module):
    """Independent output rows expose which action heads are directly supervised."""
    context_mode = "mean"

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 29)
        with torch.no_grad():
            self.head.weight[:, 0] = torch.linspace(.1, .9, 29)
            self.head.weight[:, 1] = torch.linspace(.3, .7, 29)
            self.head.bias.fill_(.2)
        self.calls = 0
        self.distribution = object()

    def deterministic_action_mean(self, observations, history):
        self.calls += 1
        return self.head(torch.stack((observations[:, 96], history[:, -4]), dim=-1))


def make_real_policy():
    return policy_module.ActorCritic_DWAQ(
        num_actor_obs=119, num_critic_obs=311, num_actions=29,
        cenet_in_dim=500, cenet_out_dim=19, obs_dim=100, context_mode="mean")


def make_private(policy=None, *, mask=(True, False, True, False), settings=None, steps=2):
    policy = PhaseToyPolicy() if policy is None else policy
    env = CommandMaskProvider(mask)
    algorithm = quiet.Elf3DwaqStandPPO(policy, obs_dim=100, device="cpu",
                                    num_learning_epochs=2, num_mini_batches=2,
                                    learning_rate=1e-4, schedule="fixed")
    algorithm.configure_standing(env, contract.quiet_settings() if settings is None else settings)
    algorithm.init_storage(len(mask), steps, [100], [311], [500], [29])
    return algorithm, env


class QuietLearningTests(unittest.TestCase):
    def setUp(self):
        original = Normal.set_default_validate_args
        self.addCleanup(setattr, Normal, "set_default_validate_args", original)
        torch.manual_seed(42)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_common_phase_rotation_changes_only_four_features_and_does_not_mutate_inputs(self):
        observations, history = inputs(dtype=torch.float64)
        originals = observations.clone(), history.clone()
        for offset in (math.pi / 2, math.pi, 3 * math.pi / 2):
            changed, changed_history = quiet.rotate_phase_inputs(observations, history, offset)
            self.assertEqual(changed.shape, observations.shape)
            self.assertEqual(changed_history.shape, history.shape)
            self.assertNotEqual(changed.data_ptr(), observations.data_ptr())
            self.assertNotEqual(changed_history.data_ptr(), history.data_ptr())
            self.assertTrue(torch.equal(changed[:, :96], observations[:, :96]))
            self.assertTrue(torch.equal(changed_history.reshape(-1, 100)[:, :96],
                                        history.reshape(-1, 100)[:, :96]))
            for raw, result in ((observations, changed),
                                (history.reshape(-1, 100), changed_history.reshape(-1, 100))):
                expected_sin = raw[:, 96:98] * math.cos(offset) + raw[:, 98:100] * math.sin(offset)
                expected_cos = raw[:, 98:100] * math.cos(offset) - raw[:, 96:98] * math.sin(offset)
                torch.testing.assert_close(result[:, 96:98], expected_sin, atol=1e-12, rtol=0)
                torch.testing.assert_close(result[:, 98:100], expected_cos, atol=1e-12, rtol=0)
                torch.testing.assert_close(result[:, 96:98].square()+result[:, 98:100].square(),
                                           torch.ones_like(expected_sin), atol=1e-12, rtol=0)
            torch.testing.assert_close(changed[:, 96:100], changed_history.reshape(-1, 5, 100)[:, -1, 96:100],
                                       atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(observations, originals[0]))
        self.assertTrue(torch.equal(history, originals[1]))

    def test_rotation_preserves_left_right_and_history_relative_phase(self):
        observations, history = inputs(dtype=torch.float64)
        for offset in (0., math.pi / 2, math.pi, 3 * math.pi / 2):
            changed, previous = quiet.rotate_phase_inputs(observations, history, offset)
            original_phase = torch.complex(history.reshape(-1, 5, 100)[:, :, 98:100],
                                           history.reshape(-1, 5, 100)[:, :, 96:98])
            shifted_phase = torch.complex(previous.reshape(-1, 5, 100)[:, :, 98:100],
                                          previous.reshape(-1, 5, 100)[:, :, 96:98])
            torch.testing.assert_close(shifted_phase[:, :, 1] / shifted_phase[:, :, 0],
                                       original_phase[:, :, 1] / original_phase[:, :, 0], atol=1e-12, rtol=0)
            torch.testing.assert_close(shifted_phase[:, 1:] / shifted_phase[:, :-1],
                                       original_phase[:, 1:] / original_phase[:, :-1], atol=1e-12, rtol=0)
            if offset == 0:
                self.assertTrue(torch.equal(changed, observations))
                self.assertTrue(torch.equal(previous, history))

    def test_reset_repeated_history_frames_stay_repeated_after_rotation(self):
        observations, history = inputs()
        history = observations[:, None].expand(-1, 5, -1).reshape(-1, 500).clone()
        changed, previous = quiet.rotate_phase_inputs(observations, history, math.pi / 2)
        expected = changed[:, None].expand(-1, 5, -1)
        self.assertTrue(torch.equal(previous.reshape(-1, 5, 100), expected))

    def test_wrong_layout_is_rejected(self):
        for obs_width, history_width in ((96, 500), (100, 2000)):
            with self.assertRaises(ValueError):
                quiet.rotate_phase_inputs(torch.zeros(2, obs_width), torch.zeros(2, history_width), .5)

    def test_current_zero_command_mask_distinguishes_tolerance_low_speed_and_yaw_only(self):
        observations, history = inputs(batch=8)
        observations[1, 6] = 1e-6
        observations[2, 7] = -1e-6
        observations[3, 6] = .1
        observations[4, 8] = .1
        observations[5, 7] = -.1
        history.reshape(8, 5, 100)[6, 0, 8] = -.3
        history.reshape(8, 5, 100)[7, 3, 6] = 2e-6
        mask = quiet.zero_command_observation_mask(observations)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [True, True, True, False, False, False, True, True])

    def test_first_zero_command_frame_is_active_despite_nonzero_command_history(self):
        algorithm, env = make_private(make_real_policy(), mask=[True] * 4)
        observations, history = inputs()
        # A newly requested stop must not wait for 5 zero-command history
        # frames, physical stillness or an elapsed settling-time gate.
        history.reshape(4, 5, 100)[:, :-1, 6] = .6
        history.reshape(4, 5, 100)[:, :-1, 8] = -.5
        critic = torch.zeros(4, 311)
        algorithm.act(observations, critic, critic, history)
        self.assertTrue(algorithm._pending_standing_mask.all().item())
        algorithm.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        self.assertTrue(algorithm.storage.standing_masks[0].all().item())
        algorithm.storage.batch_standing_mask = algorithm.storage.standing_masks[0].clone()
        loss = algorithm._additional_loss(observations, history, critic)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.)
        self.assertEqual(algorithm._standing_metric_sum[2].item(), 1.)

    def test_private_storage_preserves_all_base_batches_and_shuffles_gate_by_same_indices(self):
        storage = quiet.StandingRolloutStorage(5, 4, [100], [311], [500], [29], "cpu")
        baseline = storage_module.RolloutStorageDWAQ(5, 4, [100], [311], [500], [29], "cpu")
        identifiers = torch.arange(20).reshape(4, 5)
        fields = ("observations", "privileged_observations", "prev_critic_obs", "observation_history",
                  "actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma")
        for index, name in enumerate(fields):
            target = getattr(storage, name)
            target.copy_((identifiers + index * 100)[..., None].expand_as(target))
            getattr(baseline, name).copy_(target)
        storage.standing_masks.copy_(identifiers.remainder(3) == 0)
        torch.manual_seed(123)
        expected = list(baseline.mini_batch_generator(4, 2))
        torch.manual_seed(123)
        seen = []
        for batch, old_batch in zip(storage.mini_batch_generator(4, 2), expected):
            self.assertEqual(len(batch), 13)
            self.assert_nested_equal(batch, old_batch)
            ids = batch[0][:, 0].long()
            self.assertTrue(torch.equal(storage.batch_standing_mask, ids.remainder(3) == 0))
            seen.extend(ids.tolist())
        self.assertEqual(sorted(seen), sorted(list(range(20)) * 2))
        storage.step = 4
        storage.clear()
        self.assertEqual(storage.step, 0)
        self.assertFalse(storage.standing_masks.any().item())
        self.assertIsNone(storage.batch_standing_mask)

    def test_gate_captured_before_action_is_cloned_and_not_replaced_by_post_reset_state(self):
        algorithm, env = make_private(make_real_policy(), mask=(True, True, False, True))
        observations, history = inputs()
        observations[1, 6] = .1
        history.reshape(4, 5, 100)[3, 0, 6] = .1
        critic = torch.zeros(4, 311)
        algorithm.act(observations, critic, critic, history)
        expected_first = torch.tensor([True, False, False, True])
        env.mask[:] = torch.tensor([False, True, True, False])
        algorithm.process_env_step(torch.ones(4), torch.tensor([True, False, False, False]),
                                   {"standing_mask": env.mask, "reset_standing_mask": env.mask})
        self.assertTrue(torch.equal(algorithm.storage.standing_masks[0], expected_first))
        self.assertIsNone(algorithm._pending_standing_mask)
        # Fresh episode/current state cannot inherit the previous stored mask.
        observations, history = inputs()
        expected_second = env.mask.clone()
        algorithm.act(observations, critic, critic, history)
        env.mask.fill_(False)
        algorithm.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        self.assertTrue(torch.equal(algorithm.storage.standing_masks[1], expected_second))
        self.assertTrue(torch.equal(algorithm.storage.standing_masks[0], expected_first))

    def test_process_step_without_pre_action_gate_is_rejected(self):
        algorithm, env = make_private()
        with self.assertRaises(RuntimeError):
            algorithm.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {})

    def test_active_auxiliary_has_positive_losses_and_direct_gradients_only_on_arm_output_rows(self):
        algorithm, env = make_private()
        observations, history = inputs()
        algorithm.storage.batch_standing_mask = env.mask.clone()
        loss = algorithm._additional_loss(observations, history, torch.zeros(4, 311))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        self.assertGreater(algorithm._standing_metric_sum[0].item(), 0)
        self.assertGreater(algorithm._standing_metric_sum[1].item(), 0)
        self.assertAlmostEqual(algorithm._standing_metric_sum[2].item(), .5)
        loss.backward()
        gradient = algorithm.policy.head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue(torch.equal(gradient[:15], torch.zeros_like(gradient[:15])))
        self.assertGreater(gradient[15:].abs().sum().item(), 0)
        self.assertTrue(torch.equal(algorithm.policy.head.bias.grad[:15], torch.zeros(15)))

    def test_auxiliary_leaves_distribution_inputs_and_rng_untouched(self):
        algorithm, env = make_private(make_real_policy())
        observations, history = inputs()
        originals = observations.clone(), history.clone()
        algorithm.policy.act(observations, history)
        distribution = algorithm.policy.distribution
        mean, std = distribution.mean.clone(), distribution.stddev.clone()
        algorithm.storage.batch_standing_mask = env.mask.clone()
        before_rng = torch.random.get_rng_state().clone()
        algorithm._additional_loss(observations, history, torch.zeros(4, 311))
        self.assertIs(algorithm.policy.distribution, distribution)
        self.assertTrue(torch.equal(algorithm.policy.action_mean, mean))
        self.assertTrue(torch.equal(algorithm.policy.action_std, std))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertTrue(torch.equal(observations, originals[0]))
        self.assertTrue(torch.equal(history, originals[1]))

    def test_inactive_and_zero_coefficient_paths_do_not_run_extra_policy_forward(self):
        for inactive, zero_coefficients in ((True, False), (False, True)):
            settings = contract.quiet_settings()
            if zero_coefficients:
                settings["auxiliary"].update(phase_invariance_coeff=0., arm_pose_coeff=0.)
            mask = [not inactive] * 4
            algorithm, env = make_private(settings=settings, mask=mask)
            algorithm.storage.batch_standing_mask = env.mask.clone()
            observations, history = inputs()
            rng = torch.random.get_rng_state().clone()
            with patch.object(algorithm.policy, "deterministic_action_mean",
                              side_effect=AssertionError("Unexpected auxiliary policy forward")):
                self.assertIsNone(algorithm._additional_loss(observations, history, torch.zeros(4, 311)))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_current_motion_disables_auxiliary_even_if_provider_mask_is_true(self):
        algorithm, env = make_private(mask=[True] * 4)
        observations, history = inputs()
        observations[:2, 8] = .1  # Pure turning is motion.
        observations[2:, 6] = .1
        algorithm.storage.batch_standing_mask = env.mask.clone()
        with patch.object(algorithm.policy, "deterministic_action_mean",
                          side_effect=AssertionError("Motion rows must not be supervised")):
            self.assertIsNone(algorithm._additional_loss(observations, history, torch.zeros(4, 311)))

    def test_old_moving_command_history_does_not_disable_zero_command_auxiliary(self):
        algorithm, env = make_private(mask=[True] * 4)
        observations, history = inputs()
        history.reshape(4, 5, 100)[:, :-1, 6:9] = .3
        algorithm.storage.batch_standing_mask = env.mask.clone()
        loss = algorithm._additional_loss(observations, history, torch.zeros(4, 311))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.)
        self.assertEqual(algorithm._standing_metric_sum[2].item(), 1.)

    def test_base_noop_hook_does_not_touch_inputs_or_random_generator(self):
        algorithm = ppo_module.DWAQPPO(PhaseToyPolicy(), obs_dim=100, device="cpu")
        observations, history = inputs()
        rng = torch.random.get_rng_state().clone()
        with patch.object(algorithm.policy, "deterministic_action_mean",
                          side_effect=AssertionError("Legacy PPO must not add an auxiliary forward")):
            self.assertIsNone(algorithm._additional_loss(observations, history, torch.zeros(4, 311)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_active_auxiliaries_are_aggregated_inside_the_original_optimizer_updates(self):
        algorithm, env = make_private(make_real_policy(), mask=[True] * 4)
        for step in range(2):
            observations, history = inputs()
            critic = torch.linspace(-.1, .1, 4 * 311).reshape(4, 311)
            algorithm.act(observations, critic, critic, history)
            algorithm.process_env_step(torch.linspace(.1, 1., 4), torch.zeros(4, dtype=torch.bool), {})
        algorithm.compute_returns(torch.zeros(4, 311))
        before = {name: value.clone() for name, value in algorithm.policy.state_dict().items()}
        with patch.object(algorithm.optimizer, "step", wraps=algorithm.optimizer.step) as step:
            result = algorithm.update()
        self.assertEqual(step.call_count, algorithm.num_learning_epochs * algorithm.num_mini_batches)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertGreater(algorithm.last_standing_metrics["phase_invariance"], 0.)
        self.assertGreater(algorithm.last_standing_metrics["arm_pose"], 0.)
        self.assertEqual(algorithm.last_standing_metrics["active_fraction"], 1.)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in algorithm.policy.state_dict().items()))
        self.assertTrue(all(torch.isfinite(value).all().item()
                            for value in algorithm.policy.state_dict().values()))

    def test_zero_coefficients_preserve_full_base_ppo_update_parameters_optimizer_and_rng(self):
        policy = make_real_policy()
        baseline = ppo_module.DWAQPPO(copy.deepcopy(policy), obs_dim=100, device="cpu",
                                     num_learning_epochs=2, num_mini_batches=2,
                                     learning_rate=1e-4, schedule="fixed")
        baseline.init_storage(4, 2, [100], [311], [500], [29])
        settings = contract.quiet_settings()
        settings["auxiliary"].update(phase_invariance_coeff=0., arm_pose_coeff=0.)
        private, env = make_private(copy.deepcopy(policy), mask=[True] * 4, settings=settings)
        for step in range(2):
            observations, history = inputs()
            observations = observations + step * .01
            observations[:, 6:9] = 0
            critic = torch.linspace(-.1, .1, 4 * 311).reshape(4, 311)
            for algorithm in (baseline, private):
                torch.manual_seed(200 + step)
                algorithm.act(observations, critic, critic, history)
                algorithm.process_env_step(torch.linspace(.1, 1., 4), torch.zeros(4, dtype=torch.bool), {})
        final_critic = torch.zeros(4, 311)
        baseline.compute_returns(final_critic)
        private.compute_returns(final_critic)
        torch.manual_seed(987)
        expected_losses = baseline.update()
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(987)
        with patch.object(private.policy, "deterministic_action_mean",
                          side_effect=AssertionError("Zero coefficients must not add a policy forward")):
            actual_losses = private.update()
        self.assert_nested_equal(expected_losses, actual_losses)
        self.assert_nested_equal(baseline.policy.state_dict(), private.policy.state_dict())
        self.assert_nested_equal(baseline.optimizer.state_dict(), private.optimizer.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        self.assertEqual(private.last_standing_metrics,
                         {"phase_invariance": 0., "arm_pose": 0., "active_fraction": 1.})
        self.assertIsNone(private.storage.batch_standing_mask)
        self.assertFalse(private.storage.standing_masks.any().item())


if __name__ == "__main__":
    unittest.main()
