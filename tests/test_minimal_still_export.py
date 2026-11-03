"""CPU-only parity of minimal zero-command preprocessing and deployment exports.

Synthetic fixture checkpoints below are untrained test data in temporary
directories, not relabeled historical training checkpoints. No Isaac, GPU,
environment, deployment process or robot is started.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load_file("minimal_still_export_cpu", ROOT / "scripts/export_elf3_dwaq.py")
upstream = load_file("minimal_still_upstream_cpu", ROOT / "TienKung-Lab/legged_lab/scripts/export_dwaq_policy.py")
native = load_file("minimal_still_native_cpu", ROOT / "TienKung-Lab/rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py")


def settings():
    return {"version": "elf3_dwaq_still_v1", "command_epsilon": 1e-6,
            "fixed_phase": [0., 0., 1., -1.],
            "phase_scope": "current_and_all_history_if_current_zero",
            "joint_motion_weight": -.5}


def make_policy():
    torch.manual_seed(42)
    return upstream.DWAQPolicyExporter(100, 29, 5).eval()


def mixed_inputs():
    generator = torch.Generator().manual_seed(7)
    history = torch.randn(8, 5, 100, generator=generator) * .2
    # Old moving histories must not prevent activation on the current zero frame.
    history[:, :, 6:9] = torch.tensor([.6, .1, -.2])
    history[:3, -1, 6:9] = 0
    history[3, -1, 6:9] = 1e-6
    history[4, -1, 6:9] = -1e-6
    history[5, -1, 6:9] = 1.001e-6
    history[6, -1, 6:9] = torch.tensor([0., .1, 0.])
    history[7, -1, 6:9] = torch.tensor([0., 0., -.5])
    return history.reshape(8, 500)


class MinimalStillExportTests(unittest.TestCase):
    def test_wrapper_matches_native_helper_and_actor_mean_without_input_mutation(self):
        policy = make_policy()
        wrapped = export.MinimalStillPolicyExporter(policy).eval()
        history = mixed_inputs()
        original = history.clone()
        prepare = export.load_still_preprocessor()
        observations, prepared = prepare(history[:, -100:], history)
        model = native.ActorCritic_DWAQ(
            num_actor_obs=119, num_critic_obs=311, num_actions=29,
            cenet_in_dim=500, cenet_out_dim=19, obs_dim=100, context_mode="mean")
        model.load_state_dict(policy.state_dict(), strict=False)
        model.eval()
        with torch.inference_mode():
            actual = wrapped(history)
            expected = model.deterministic_action_mean(observations, prepared)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(history, original))
        self.assertTrue(torch.equal(actual, wrapped(history)))

    def test_moving_path_is_bitwise_original_for_all_actions(self):
        policy = make_policy()
        wrapped = export.MinimalStillPolicyExporter(policy).eval()
        history = mixed_inputs()[5:]
        with torch.inference_mode():
            self.assertTrue(torch.equal(wrapped(history), policy(history)))

    def test_zero_branch_clock_independent_but_body_features_remain_effective(self):
        policy = make_policy()
        wrapped = export.MinimalStillPolicyExporter(policy).eval()
        history = mixed_inputs()[:5]
        other = history.reshape(-1, 5, 100).clone()
        other[:, :, 96:100] = torch.randn_like(other[:, :, 96:100]) * 100
        other = other.reshape(-1, 500)
        with torch.inference_mode():
            self.assertTrue(torch.equal(wrapped(history), wrapped(other)))
            other[:, -100] += .2
            self.assertFalse(torch.equal(wrapped(history), wrapped(other)))
        prepare = export.load_still_preprocessor()
        obs, flat = prepare(history[:, -100:], history)
        self.assertTrue(torch.equal(flat.reshape(-1, 5, 100)[:, :, :96],
                                    history.reshape(-1, 5, 100)[:, :, :96]))
        self.assertTrue(torch.equal(obs[:, :96], history[:, -100:-4]))

    def test_torchscript_and_onnx_runtime_match_mixed_and_dynamic_batches(self):
        policy = export.MinimalStillPolicyExporter(make_policy()).eval()
        # Trace with entirely moving samples: zero handling must remain in graph.
        example = mixed_inputs()[5:]
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            pt = Path(directory) / "policy.pt"
            onnx = Path(directory) / "policy.onnx"
            torch.jit.trace(policy, example).save(str(pt))
            restored = torch.jit.load(str(pt))
            torch.onnx.export(policy, example, str(onnx), opset_version=17,
                              input_names=["history"], output_names=["actions"],
                              dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}})
            session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            for history in (mixed_inputs()[:1], mixed_inputs(), example, mixed_inputs()[:5]):
                expected = policy(history)
                torch.testing.assert_close(restored(history), expected, atol=1e-7, rtol=1e-6)
                actual = session.run(None, {"history": history.numpy()})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)

    def fixture_checkpoint(self, directory, standing=None):
        policy = make_policy()
        contract = json.loads(export.CONTRACT_PATH.read_text())
        metadata = {"asset_contract_sha256": hashlib.sha256(
                    json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                    "joint_names": contract["joint_names"], "num_obs": 100, "history_length": 5}
        if standing is not None:
            metadata["standing_experiment"] = copy.deepcopy(standing)
        path = Path(directory) / "untrained_fixture.pt"
        torch.save({"model_state_dict": policy.state_dict(), "infos": {"elf3_contract": metadata}}, path)
        return path, policy, metadata

    def test_loader_wraps_only_new_contract_after_strict_tensor_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path, baseline, expected_metadata = self.fixture_checkpoint(directory, settings())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, _, metadata = export.load_policy(path)
            self.assertIsInstance(loaded, export.MinimalStillPolicyExporter)
            self.assertEqual(metadata, expected_metadata)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            for key, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.policy.state_dict()[key]))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved["model_state_dict"].pop("actor.0.weight")
            torch.save(saved, path)
            with self.assertRaises(KeyError):
                export.load_policy(path)

    def test_legacy_checkpoints_use_original_exporter_and_exact_outputs(self):
        for old in (None, {"version": "elf3_stand_v1"},
                    {"version": "elf3_stand_v2"}, {"version": "elf3_stand_v2_command_only_v2"}):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as directory:
                path, baseline, _ = self.fixture_checkpoint(directory, old)
                loaded, _, _ = export.load_policy(path)
                self.assertNotIsInstance(loaded, export.MinimalStillPolicyExporter)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(loaded(mixed_inputs()), baseline(mixed_inputs())))

    def test_unknown_incomplete_or_changed_still_contract_rejected(self):
        invalid = [{"version": "elf3_dwaq_still_v2"}, {"version": "elf3_dwaq_still"}]
        for key in ("command_epsilon", "fixed_phase", "phase_scope", "joint_motion_weight"):
            missing = settings()
            missing.pop(key)
            invalid.append(missing)
        invalid.extend([settings() | {"command_epsilon": .01},
                        settings() | {"fixed_phase": [0., 0., 1., 1.]},
                        settings() | {"joint_motion_weight": -.1}])
        for item in invalid:
            with self.subTest(item=item), tempfile.TemporaryDirectory() as directory:
                path, _, _ = self.fixture_checkpoint(directory, item)
                with self.assertRaises(ValueError):
                    export.load_policy(path)

    def test_metadata_describes_builtin_phase_only_preprocessor_without_interface_change(self):
        original = {"standing_experiment": settings(), "retained": True}
        before = copy.deepcopy(original)
        fields = export.preprocessing_metadata(original)
        self.assertEqual(original, before)
        preprocess = fields["zero_command_preprocessing"]
        self.assertTrue(preprocess["embedded_in_model"])
        self.assertEqual(preprocess["scope"], "current_and_all_history_if_current_zero")
        self.assertEqual(preprocess["fixed_phase"], [0., 0., 1., -1.])
        self.assertFalse(preprocess["extra_velocity_filter"])
        self.assertFalse(preprocess["extra_action_filter"])
        self.assertFalse(preprocess["joint_lock"])
        self.assertEqual(export.preprocessing_metadata({}), {})


if __name__ == "__main__":
    unittest.main()
