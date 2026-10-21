"""CPU-only DWAQ context checks; never import Isaac Sim or launch training."""

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.distributions import Normal


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy_module = load_module(
    "feedback_context_policy",
    "TienKung-Lab/rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py",
)
export_module = load_module(
    "feedback_context_exporter",
    "TienKung-Lab/legged_lab/scripts/export_dwaq_policy.py",
)


class FeedbackContextTests(unittest.TestCase):
    def setUp(self):
        # The legacy constructor overwrites this class attribute; do not leak that
        # pre-existing side effect into unrelated tests in the same interpreter.
        original = Normal.set_default_validate_args
        self.addCleanup(setattr, Normal, "set_default_validate_args", original)
        torch.manual_seed(42)

    def make_policy(self, **kwargs):
        return policy_module.ActorCritic_DWAQ(
            num_actor_obs=119,
            num_critic_obs=311,
            num_actions=29,
            cenet_in_dim=500,
            cenet_out_dim=19,
            obs_dim=100,
            **kwargs,
        )

    def inputs(self, batch=32):
        history = torch.randn(batch, 500)
        return history[:, -100:], history

    def assert_nonzero_finite_gradient(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_default_and_validation(self):
        self.assertEqual(self.make_policy().context_mode, "sampled")
        with self.assertRaisesRegex(ValueError, "context_mode"):
            self.make_policy(context_mode="invalid")

    def test_mean_frozen_rollout_recompute_ratio_is_exactly_one(self):
        policy = self.make_policy(context_mode="mean")
        observations, history = self.inputs(256)
        with torch.no_grad():
            torch.manual_seed(100)
            actions = policy.act(observations, history)
            old_log_prob = policy.get_actions_log_prob(actions).clone()
            old_mean, old_std = policy.action_mean.clone(), policy.action_std.clone()
            torch.manual_seed(200)
            new_actions = policy.act(observations, history)
            new_log_prob = policy.get_actions_log_prob(actions)
        self.assertTrue(torch.equal(old_mean, policy.action_mean))
        self.assertTrue(torch.equal(old_std, policy.action_std))
        self.assertTrue(torch.equal(old_log_prob, new_log_prob))
        self.assertTrue(torch.equal(torch.exp(new_log_prob - old_log_prob), torch.ones(256)))
        # PPO action exploration stays stochastic; only its context is deterministic.
        self.assertFalse(torch.equal(actions, new_actions))

    def test_mean_actor_does_not_call_stochastic_vae_or_decoder(self):
        policy = self.make_policy(context_mode="mean")
        observations, history = self.inputs()
        with patch.object(policy, "cenet_forward", side_effect=AssertionError("VAE called")), \
                patch.object(policy.decoder, "forward", side_effect=AssertionError("decoder called")), \
                patch.object(policy.encode_logvar_vel, "forward", side_effect=AssertionError("logvar called")), \
                patch.object(policy.encode_logvar_latent, "forward", side_effect=AssertionError("logvar called")):
            self.assertEqual(policy.act(observations, history).shape, (32, 29))

    def test_mean_ppo_gradient_reaches_encoder_and_both_mean_heads(self):
        policy = self.make_policy(context_mode="mean")
        observations, history = self.inputs()
        with torch.no_grad():
            actions = policy.act(observations, history)
        policy.act(observations, history)
        advantages = torch.linspace(-1.0, 1.0, observations.shape[0])
        loss = -(policy.get_actions_log_prob(actions) * advantages).mean()
        loss.backward()
        for parameter in (
            policy.encoder[0].weight, policy.encoder[2].weight,
            policy.encode_mean_vel.weight, policy.encode_mean_latent.weight,
            policy.actor[0].weight, policy.actor[-1].weight,
        ):
            self.assert_nonzero_finite_gradient(parameter)
        self.assertIsNone(policy.decoder[0].weight.grad)
        self.assertIsNone(policy.encode_logvar_vel.weight.grad)
        self.assertIsNone(policy.encode_logvar_latent.weight.grad)

    def test_auxiliary_vae_remains_stochastic_and_differentiable(self):
        policy = self.make_policy(context_mode="mean")
        observations, history = self.inputs()
        first = policy.cenet_forward(history)
        second = policy.cenet_forward(history)
        for index in (0, 1, 2):  # Context, sampled velocity, reconstruction.
            self.assertFalse(torch.equal(first[index], second[index]))
        for index in (3, 4, 5, 6):
            self.assertTrue(torch.equal(first[index], second[index]))
        _, sampled_vel, reconstruction, mean_vel, logvar_vel, mean_latent, logvar_latent = first
        kl = sum(
            (mean.square() + logvar.exp() - logvar - 1).mean()
            for mean, logvar in ((mean_vel, logvar_vel), (mean_latent, logvar_latent))
        )
        loss = (reconstruction - observations).square().mean() + sampled_vel.square().mean() + 0.1 * kl
        loss.backward()
        for parameter in (
            policy.encoder[0].weight, policy.decoder[0].weight,
            policy.encode_logvar_vel.weight, policy.encode_logvar_latent.weight,
        ):
            self.assert_nonzero_finite_gradient(parameter)

    def test_default_sampled_path_is_exact_legacy_random_sequence(self):
        policy = self.make_policy()
        observations, history = self.inputs()
        torch.manual_seed(777)
        actual_actions = policy.act(observations, history)
        actual_mean = policy.action_mean.clone()
        actual_log_prob = policy.get_actions_log_prob(actual_actions).clone()
        torch.manual_seed(777)
        code, _, _, _, _, _, _ = policy.cenet_forward(history)
        policy.update_distribution(torch.cat((code, observations), dim=-1))
        reference_actions = policy.distribution.sample()
        self.assertTrue(torch.equal(actual_actions, reference_actions))
        self.assertTrue(torch.equal(actual_mean, policy.action_mean))
        self.assertTrue(torch.equal(actual_log_prob, policy.get_actions_log_prob(reference_actions)))
        torch.manual_seed(778)
        policy.act(observations, history)
        self.assertFalse(torch.equal(actual_mean, policy.action_mean))

    def test_state_dict_layout_and_explicit_sampled_are_compatible(self):
        legacy = self.make_policy()
        sampled = self.make_policy(context_mode="sampled")
        mean = self.make_policy(context_mode="mean")
        legacy_state = legacy.state_dict()
        for policy in (sampled, mean):
            self.assertEqual(set(legacy_state), set(policy.state_dict()))
            self.assertEqual(
                {key: tuple(value.shape) for key, value in legacy_state.items()},
                {key: tuple(value.shape) for key, value in policy.state_dict().items()},
            )
            policy.load_state_dict(legacy_state, strict=True)
        observations, history = self.inputs()
        torch.manual_seed(121)
        actions = legacy.act(observations, history)
        torch.manual_seed(121)
        self.assertTrue(torch.equal(actions, sampled.act(observations, history)))
        self.assertEqual(mean.actor[0].in_features, 119)
        self.assertEqual(mean.actor[-1].out_features, 29)

    def test_train_eval_inference_and_actual_exporter_share_mean_path(self):
        policy = self.make_policy(context_mode="mean")
        exporter = export_module.DWAQPolicyExporter(100, 29, 5)
        state = policy.state_dict()
        exporter.load_state_dict({key: state[key] for key in exporter.state_dict()}, strict=True)
        for batch in (1, 3, 8):
            observations, history = self.inputs(batch)
            with torch.no_grad():
                policy.train()
                policy.act(observations, history)
                training_mean = policy.action_mean.clone()
                train_inference = policy.act_inference(observations, history)
                policy.eval()
                eval_inference = policy.act_inference(observations, history)
                exported = exporter(history)
                repeated = policy.act_inference(observations, history)
            for actual in (train_inference, eval_inference, exported, repeated):
                torch.testing.assert_close(actual, training_mean, rtol=0, atol=0)
            # Inference remains the same deterministic means even for old sampled tasks.
            policy.context_mode = "sampled"
            torch.testing.assert_close(policy.act_inference(observations, history), training_mean, rtol=0, atol=0)
            policy.context_mode = "mean"

    def test_runner_forwards_context_mode_without_default_override(self):
        source = ROOT / "TienKung-Lab/rsl_rl/rsl_rl/runners/dwaq_on_policy_runner.py"
        tree = ast.parse(source.read_text())
        allowlists = [
            ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "dwaq_supported_params" for target in node.targets)
        ]
        self.assertEqual(allowlists, [["activation", "init_noise_std", "context_mode"]])


if __name__ == "__main__":
    unittest.main()
