"""CPU export parity for pose-only v3 and its immutable failed-v3 predecessor.

No historical model is relabeled or imported, and no Isaac environment, GPU,
training job or deployment process is started by these tests.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("stand_v3_existing_export_tests", ROOT / "tests/test_minimal_still_export.py")
contract = load("stand_v3_export_contract", ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_contract.py")
v2_contract = load("stand_v3_previous_contract", ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_still_v2_contract.py")


class LegacyFeedbackStandV3ExportTests(unittest.TestCase):
    def identity(self):
        source = ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py"
        return old.export.legacy_stand_v3_settings() | {
            "sources_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}

    def test_contract_version_phase_and_preprocessing_source_are_independent(self):
        identity = self.identity()
        self.assertEqual(identity["version"], "elf3_dwaq_feedback_stand_v3")
        self.assertEqual(identity["parent_task"], "elf3_dwaq_feedback_v1")
        self.assertEqual(identity["fixed_phase"], [0., 0., 1., 1.])
        source = ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py"
        self.assertEqual(identity["sources_sha256"][source.name], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(old.export.still_preprocessing_settings({"standing_experiment": identity}),
                         old.export.legacy_stand_v3_settings())
        self.assertEqual(len(old.export.legacy_stand_v3_settings()), 17)

    def test_wrapper_matches_native_mean_for_zero_moving_and_epsilon_boundaries(self):
        base = old.make_policy()
        policy = old.export.FeedbackStandV3PolicyExporter(base).eval()
        history = old.mixed_inputs()
        original = history.clone()
        prepare = old.export.load_stand_v3_preprocessor()
        obs, prepared = prepare(history[:, -100:], history)
        model = old.native.ActorCritic_DWAQ(
            num_actor_obs=119, num_critic_obs=311, num_actions=29,
            cenet_in_dim=500, cenet_out_dim=19, obs_dim=100, context_mode="mean")
        model.load_state_dict(base.state_dict(), strict=False)
        model.eval()
        with torch.inference_mode():
            actual = policy(history)
            expected = model.deterministic_action_mean(obs, prepared)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual, policy(history)))
        self.assertEqual(actual.shape, (8, 29))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.equal(history, original))

    def test_zero_phase_is_v3_pair_and_only_phase_features_change(self):
        history = old.mixed_inputs()
        obs, flat = old.export.load_stand_v3_preprocessor()(history[:, -100:], history)
        expected = history.reshape(8, 5, 100).clone()
        # Current zero command activates the entire history even if older
        # frames contain commands; exact positive/negative epsilon are zero.
        expected[:5, :, 96:100] = torch.tensor([0., 0., 1., 1.])
        self.assertTrue(torch.equal(flat, expected.flatten(1)))
        self.assertTrue(torch.equal(obs, expected[:, -1]))
        self.assertTrue(torch.equal(flat.reshape(8, 5, 100)[:, :, :96],
                                    history.reshape(8, 5, 100)[:, :, :96]))

    def test_moving_is_bitwise_original_and_zero_is_clock_independent(self):
        base = old.make_policy()
        policy = old.export.FeedbackStandV3PolicyExporter(base).eval()
        history = old.mixed_inputs()
        changed = history[:5].reshape(5, 5, 100).clone()
        changed[:, :, 96:100] += 100
        with torch.inference_mode():
            self.assertTrue(torch.equal(policy(history[5:]), base(history[5:])))
            self.assertTrue(torch.equal(policy(history[:5]), policy(changed.flatten(1))))
            changed[:, -1, 9] += .2
            self.assertFalse(torch.equal(policy(history[:5]), policy(changed.flatten(1))))

    def test_loader_selects_only_v3_wrapper_and_preserves_every_tensor_and_file(self):
        fixture = old.MinimalStillExportTests()
        with tempfile.TemporaryDirectory() as directory:
            path, baseline, metadata = fixture.fixture_checkpoint(directory, self.identity())
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            policy, _, result = old.export.load_policy(path)
            self.assertIsInstance(policy, old.export.FeedbackStandV3PolicyExporter)
            self.assertNotIsInstance(policy, old.export.MinimalStillPolicyExporter)
            self.assertEqual(result, metadata)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            for key, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, policy.policy.state_dict()[key]), key)

    def test_existing_v2_keeps_old_wrapper_and_minus_one_right_cosine(self):
        fixture = old.MinimalStillExportTests()
        identity = v2_contract.standing_experiment_contract(v2_contract.standing_settings())
        with tempfile.TemporaryDirectory() as directory:
            path, base, _ = fixture.fixture_checkpoint(directory, identity)
            policy, _, result = old.export.load_policy(path)
            self.assertIsInstance(policy, old.export.MinimalStillPolicyExporter)
            self.assertNotIsInstance(policy, old.export.FeedbackStandV3PolicyExporter)
            self.assertEqual(result["standing_experiment"]["fixed_phase"], [0., 0., 1., -1.])
            history = old.mixed_inputs()
            with torch.inference_mode():
                obs, prepared = old.export.load_still_preprocessor()(history[:, -100:], history)
                self.assertTrue(torch.equal(policy(history), base(prepared)))
                self.assertTrue(torch.equal(obs[:5, 96:100], torch.tensor([0., 0., 1., -1.]).expand(5, -1)))

    def test_missing_or_changed_strict_settings_are_rejected(self):
        identity = self.identity()
        for key in old.export.legacy_stand_v3_settings():
            if key == "version":
                continue  # Missing version is indistinguishable from a legacy contract.
            with self.subTest(missing=key):
                invalid = copy.deepcopy(identity)
                invalid.pop(key)
                with self.assertRaises(ValueError):
                    old.export.still_preprocessing_settings({"standing_experiment": invalid})
        for change in ({"fixed_phase": [0., 0., 1., -1.]}, {"command_epsilon": .1},
                       {"parent_task": "elf3_dwaq_feedback_still_v2"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": identity | change})

    def test_unknown_stand_versions_are_rejected_not_treated_as_legacy(self):
        for version in ("elf3_dwaq_feedback_stand", "elf3_dwaq_feedback_stand_v4",
                        "elf3_dwaq_feedback_stand_v3_typo"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": {"version": version}})

    def test_v2_identity_cannot_be_reinterpreted_as_v3(self):
        invalid = v2_contract.standing_experiment_contract(v2_contract.standing_settings())
        invalid["version"] = "elf3_dwaq_feedback_stand_v3"
        with self.assertRaises(ValueError):
            old.export.still_preprocessing_settings({"standing_experiment": invalid})

    def test_missing_or_changed_v3_preprocessor_digest_rejected(self):
        for digest in (None, "0" * 64):
            identity = self.identity()
            if digest is None:
                identity["sources_sha256"].pop("feedback_stand_v3_inputs.py")
            else:
                identity["sources_sha256"]["feedback_stand_v3_inputs.py"] = digest
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "preprocessing source"):
                old.export.still_preprocessing_settings({"standing_experiment": identity})

    def test_metadata_describes_v3_builtin_phase_without_extra_deployment_filter(self):
        original = {"standing_experiment": self.identity()}
        before = copy.deepcopy(original)
        result = old.export.preprocessing_metadata(original)["zero_command_preprocessing"]
        self.assertEqual(original, before)
        self.assertEqual(result["fixed_phase"], [0., 0., 1., 1.])
        self.assertEqual(result["scope"], old.export.legacy_stand_v3_settings()["phase_scope"])
        self.assertTrue(result["embedded_in_model"])
        self.assertFalse(result["extra_velocity_filter"])
        self.assertFalse(result["extra_action_filter"])
        self.assertFalse(result["joint_lock"])

    def test_torchscript_onnx_dynamic_mixed_batches_and_epsilon_match(self):
        policy = old.export.FeedbackStandV3PolicyExporter(old.make_policy()).eval()
        moving = old.mixed_inputs()[5:]
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            pt, onnx = Path(directory) / "policy.pt", Path(directory) / "policy.onnx"
            # Trace with no zero samples: the zero branch must stay dynamic.
            torch.jit.trace(policy, moving).save(str(pt))
            restored = torch.jit.load(str(pt))
            torch.onnx.export(policy, moving, str(onnx), opset_version=17,
                              input_names=["history"], output_names=["actions"],
                              dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}})
            session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            for history in (old.mixed_inputs()[:1], old.mixed_inputs(), moving, old.mixed_inputs()[:5]):
                expected = policy(history)
                torch.testing.assert_close(restored(history), expected, atol=1e-7, rtol=1e-6)
                actual = session.run(None, {"history": history.numpy()})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


class FeedbackStandV3ZeroPhaseExportTests(unittest.TestCase):
    VERSION = "elf3_dwaq_feedback_stand_v3_zero_phase_v1"
    SOURCES = ("feedback_stand_v3_contract.py", "feedback_stand_v3_env.py",
               "elf3_feedback_stand_v3_config.py", "feedback_stand_v3_rewards.py",
               "feedback_stand_v3_runner.py", "feedback_stand_v3_inputs.py", "standing_rewards.py")

    def identity(self):
        return contract.stand_experiment_contract(contract.stand_settings())

    def fixture(self, directory):
        return old.MinimalStillExportTests().fixture_checkpoint(directory, self.identity())

    def test_new_contract_explicitly_requires_zero_phase_preprocessing(self):
        identity = self.identity()
        self.assertEqual(identity["version"], self.VERSION)
        self.assertEqual(identity["parent_task"], "elf3_dwaq_feedback_v1")
        self.assertEqual(identity["default_pose_weight"], -.1)
        self.assertEqual(identity["joint_motion_weight"], -.5)
        self.assertEqual(identity["pose_target"], "asset.default_joint_pos")
        self.assertTrue(identity["baseline_pose_preserved"])
        self.assertEqual(identity["actor_inputs"], "zero_command_fixed_phase_only")
        self.assertFalse(identity["additional_ppo_losses"])
        self.assertFalse(identity["zero_command_periodic_rewards"])
        self.assertIs(identity["no_phase_preprocessing"], False)
        self.assertEqual(identity["fixed_phase"], [0., 0., 1., 1.])
        self.assertEqual(identity["phase_scope"], "current_and_all_history_frames")
        for name in self.SOURCES:
            self.assertEqual(identity["sources_sha256"][name], hashlib.sha256(
                (ROOT / "TienKung-Lab/legged_lab/envs/elf3" / name).read_bytes()).hexdigest())
        self.assertEqual(old.export.still_preprocessing_settings({"standing_experiment": identity}),
                         contract.stand_settings())
        prep = old.export.preprocessing_metadata({"standing_experiment": identity})["zero_command_preprocessing"]
        self.assertTrue(prep["embedded_in_model"])
        self.assertEqual(prep["fixed_phase"], [0., 0., 1., 1.])
        self.assertEqual(prep["scope"], "current_and_all_history_frames")
        self.assertFalse(prep["extra_velocity_filter"])
        self.assertFalse(prep["extra_action_filter"])
        self.assertFalse(prep["joint_lock"])

    def test_loader_uses_v3_preprocessor_not_the_old_minus_one_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path, baseline, metadata = self.fixture(directory)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(old.export, "load_still_preprocessor", side_effect=AssertionError("old still")):
                policy, _, result = old.export.load_policy(path)
            self.assertIsInstance(policy, old.export.FeedbackStandV3PolicyExporter)
            self.assertNotIsInstance(policy, old.export.MinimalStillPolicyExporter)
            self.assertTrue(hasattr(policy, "prepare_inputs"))
            self.assertEqual(result, metadata)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(set(policy.policy.state_dict()), set(baseline.state_dict()))
            for key, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, policy.policy.state_dict()[key]), key)

    def test_zero_history_phase_fixed_and_moving_native_mean_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path, baseline, _ = self.fixture(directory)
            policy, _, _ = old.export.load_policy(path)
            native = old.native.ActorCritic_DWAQ(
                num_actor_obs=119, num_critic_obs=311, num_actions=29,
                cenet_in_dim=500, cenet_out_dim=19, obs_dim=100, context_mode="mean")
            native.load_state_dict(baseline.state_dict(), strict=False)
            native.eval()
            history = old.mixed_inputs()
            original = history.clone()
            changed = history.reshape(8, 5, 100).clone()
            changed[:, :, 96:100] += 1.
            changed = changed.flatten(1)
            with torch.inference_mode():
                for value in (history, changed, history[:1], history[:5], history[5:]):
                    obs, prepared = old.export.load_stand_v3_preprocessor()(value[:, -100:], value)
                    actual = policy(value)
                    self.assertEqual(actual.shape, (len(value), 29))
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(torch.equal(actual, baseline(prepared)))
                    self.assertTrue(torch.equal(actual, native.deterministic_action_mean(obs, prepared)))
                    self.assertTrue(torch.equal(actual, policy(value)))
                self.assertTrue(torch.equal(policy(history[:5]), policy(changed[:5])))
                self.assertTrue(torch.equal(policy(history[5:]), baseline(history[5:])))
            obs, prepared = old.export.load_stand_v3_preprocessor()(history[:, -100:], history)
            expected = history.reshape(8, 5, 100).clone()
            expected[:5, :, 96:100] = torch.tensor([0., 0., 1., 1.])
            self.assertTrue(torch.equal(prepared, expected.flatten(1)))
            self.assertTrue(torch.equal(obs, expected[:, -1]))
            self.assertTrue(torch.equal(history, original))

    def test_incomplete_changed_or_legacy_contaminated_pose_settings_rejected(self):
        identity = self.identity()
        for key in contract.stand_settings():
            if key == "version":
                continue
            invalid = copy.deepcopy(identity)
            invalid.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": invalid})
        for changes in ({"default_pose_weight": -.2}, {"joint_motion_weight": -.1},
                        {"baseline_pose_preserved": False},
                        {"baseline_pose_preserved": 1}, {"actor_inputs": "feedback_v1_unmodified"},
                        {"additional_ppo_losses": True}, {"fixed_phase": [0., 0., 1., -1.]},
                        {"phase_scope": "current_and_all_history_if_current_zero"},
                        {"zero_command_periodic_rewards": True}, {"no_phase_preprocessing": True},
                        {"score_weight": 1.}, {"component_weights": {"pose": 1.}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": identity | changes})

    def test_pose_source_schema_fingerprints_are_required_and_exact(self):
        for name in self.SOURCES:
            for missing in (True, False):
                invalid = self.identity()
                if missing:
                    invalid["sources_sha256"].pop(name)
                else:
                    invalid["sources_sha256"][name] = "0" * 64
                with self.subTest(source=name, missing=missing), self.assertRaisesRegex(ValueError, "source"):
                    old.export.still_preprocessing_settings({"standing_experiment": invalid})
        invalid = self.identity()
        invalid.pop("sources_sha256")
        with self.assertRaisesRegex(ValueError, "source"):
            old.export.still_preprocessing_settings({"standing_experiment": invalid})

    def test_pose_legacy_v3_and_v2_versions_cannot_be_interchanged(self):
        invalid = [LegacyFeedbackStandV3ExportTests().identity() | {"version": self.VERSION},
                   self.identity() | {"version": "elf3_dwaq_feedback_stand_v3"},
                   v2_contract.standing_experiment_contract(v2_contract.standing_settings()) | {"version": self.VERSION},
                   self.identity() | {"version": "elf3_dwaq_feedback_stand_v3_default_pose_v1"}]
        for identity in invalid:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": identity})
        with self.assertRaises(ValueError):
            old.export.still_preprocessing_settings({"standing_experiment": {"version": self.VERSION + "_typo"}})

    def test_pose_tensor_loading_remains_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved["model_state_dict"].pop("actor.0.weight")
            torch.save(saved, path)
            with self.assertRaises(KeyError):
                old.export.load_policy(path)

    def test_zero_phase_preprocessing_commutes_with_sagittal_mirror(self):
        mirror = old.load_file("stand_v3_zero_phase_export_mirror",
                               ROOT / "TienKung-Lab/legged_lab/envs/elf3/symmetry.py")
        prepare = old.export.load_stand_v3_preprocessor()
        history = old.mixed_inputs()
        obs, prepared = prepare(history[:, -100:], history)
        mirrored_obs, mirrored_hist = prepare(mirror.mirror_observations(history[:, -100:]),
                                               mirror.mirror_history(history))
        self.assertTrue(torch.equal(mirrored_obs, mirror.mirror_observations(obs)))
        self.assertTrue(torch.equal(mirrored_hist, mirror.mirror_history(prepared)))

    def test_intermediate_default_pose_version_is_explicitly_rejected(self):
        identity = {"version": "elf3_dwaq_feedback_stand_v3_default_pose_v1",
                    "actor_inputs": "feedback_v1_unmodified", "no_phase_preprocessing": True}
        with self.assertRaisesRegex(ValueError, "Intermediate default-pose"):
            old.export.still_preprocessing_settings({"standing_experiment": identity})
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = old.MinimalStillExportTests().fixture_checkpoint(directory, identity)
            with self.assertRaisesRegex(ValueError, "Intermediate default-pose"):
                old.export.load_policy(path)

    def test_zero_phase_torchscript_onnx_dynamic_batch_has_dynamic_command_branch(self):
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path, _, _ = self.fixture(directory)
            policy, _, _ = old.export.load_policy(path)
            pt, model_path = Path(directory) / "policy.pt", Path(directory) / "policy.onnx"
            example = old.mixed_inputs()[5:]
            torch.jit.trace(policy, example).save(str(pt))
            restored = torch.jit.load(str(pt))
            torch.onnx.export(policy, example, str(model_path), opset_version=17,
                              input_names=["history"], output_names=["actions"],
                              dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}})
            graph = onnx.load(str(model_path)).graph
            self.assertEqual([value.name for value in graph.input], ["history"])
            self.assertEqual(graph.input[0].type.tensor_type.shape.dim[1].dim_value, 500)
            self.assertEqual(graph.output[0].type.tensor_type.shape.dim[1].dim_value, 29)
            self.assertIn("Where", [node.op_type for node in graph.node])
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            changed = old.mixed_inputs().reshape(8, 5, 100).clone()
            changed[:, :, 96:100] += 1.
            for history in (old.mixed_inputs()[:1], old.mixed_inputs(), example,
                            old.mixed_inputs()[:5], changed.flatten(1)):
                expected = policy(history)
                torch.testing.assert_close(restored(history), expected, atol=1e-7, rtol=1e-6)
                actual = session.run(None, {"history": history.numpy()})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
