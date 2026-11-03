"""CPU-only still-v2 objective validation with unchanged still-v1 deployment.

Fixture checkpoints are newly created untrained tensors in temporary folders,
not historical weights relabeled as still-v2. No Isaac/GPU/training is started.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("still_v2_export_old_tests", ROOT / "tests/test_minimal_still_export.py")
contract = load("still_v2_export_contract", DIRECTORY / "still_v2_contract.py")
still_contract = load("still_v2_export_parent_contract", DIRECTORY / "still_contract.py")


class StillV2ExportTests(unittest.TestCase):
    SOURCES = ("still_v2_contract.py", "still_v2_rewards.py", "elf3_still_v2_config.py",
               "still_v2_runner.py", "still_pose_rewards.py")

    def identity(self):
        return contract.pose_experiment_contract(contract.pose_settings())

    def metadata(self):
        return {"standing_experiment": still_contract.still_experiment_contract(still_contract.still_settings()),
                "still_v2_experiment": self.identity()}

    def fixture(self, directory, metadata=None):
        metadata = self.metadata() if metadata is None else metadata
        path, base, result = old.MinimalStillExportTests().fixture_checkpoint(
            directory, metadata.get("standing_experiment"))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in metadata.items():
            saved["infos"]["elf3_contract"][key] = copy.deepcopy(value)
            result[key] = copy.deepcopy(value)
        torch.save(saved, path)
        return path, base, result

    def test_objective_version_weights_and_original_phase_contract_are_distinct(self):
        metadata = self.metadata()
        identity = metadata["still_v2_experiment"]
        self.assertEqual(identity["version"], "elf3_dwaq_still_v2")
        self.assertEqual(identity["parent_task"], "elf3_dwaq_still")
        self.assertEqual(identity["reference_task"], "elf3_dwaq_still_pose_v1")
        self.assertEqual(identity["command_epsilon"], 1e-6)
        self.assertEqual(identity["leg_incremental_weight"], -.13)
        self.assertEqual(identity["leg_baseline_weight"], -.02)
        self.assertEqual(identity["leg_total_zero_weight"], -.15)
        self.assertEqual(identity["waist_incremental_weight"], -.10)
        self.assertEqual(identity["waist_baseline_weight"], -.20)
        self.assertEqual(identity["waist_total_zero_weight"], -.30)
        self.assertEqual(identity["target"], "asset.default_joint_pos")
        self.assertEqual(len(identity["leg_joint_names"]), 4)
        self.assertEqual(len(identity["waist_joint_names"]), 3)
        settings = old.export.still_preprocessing_settings(metadata)
        self.assertEqual(settings["fixed_phase"], [0., 0., 1., -1.])
        self.assertEqual(settings["joint_motion_weight"], -.5)
        self.assertEqual(settings["command_epsilon"], identity["command_epsilon"])

    def test_loader_uses_only_original_still_wrapper_and_preserves_tensors_metadata_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path, base, metadata = self.fixture(directory)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(old.export, "load_stand_v3_preprocessor", side_effect=AssertionError("v3 must not be used")):
                policy, _, actual = old.export.load_policy(path)
            self.assertIsInstance(policy, old.export.MinimalStillPolicyExporter)
            self.assertNotIsInstance(policy, old.export.FeedbackStandV3PolicyExporter)
            self.assertEqual(actual, metadata)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            for key, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, policy.policy.state_dict()[key]), key)

    def test_zero_moving_epsilon_and_history_match_original_native_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path, base, _ = self.fixture(directory)
            policy, _, _ = old.export.load_policy(path)
            native = old.native.ActorCritic_DWAQ(
                num_actor_obs=119, num_critic_obs=311, num_actions=29,
                cenet_in_dim=500, cenet_out_dim=19, obs_dim=100, context_mode="mean")
            native.load_state_dict(base.state_dict(), strict=False)
            native.eval()
            history = old.mixed_inputs()
            before = history.clone()
            prepare = old.export.load_still_preprocessor()
            obs, prepared = prepare(history[:, -100:], history)
            self.assertTrue(torch.equal(prepared.reshape(8, 5, 100)[:5, :, 96:100],
                                        torch.tensor([0., 0., 1., -1.]).expand(5, 5, 4)))
            with torch.inference_mode():
                self.assertTrue(torch.equal(policy(history), native.deterministic_action_mean(obs, prepared)))
                self.assertTrue(torch.equal(policy(history[5:]), base(history[5:])))
                changed = history[:5].reshape(5, 5, 100).clone()
                changed[:, :, 96:100] += 5.
                self.assertTrue(torch.equal(policy(history[:5]), policy(changed.flatten(1))))
            self.assertTrue(torch.equal(history, before))

    def test_missing_changed_and_nonfinite_objective_settings_rejected(self):
        for key in contract.pose_settings():
            metadata = self.metadata()
            metadata["still_v2_experiment"].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings(metadata)
        for changes in ({"leg_incremental_weight": -.08}, {"leg_total_zero_weight": -.10},
                        {"waist_incremental_weight": -.20}, {"waist_total_zero_weight": -.20},
                        {"command_epsilon": float("nan")}, {"command_epsilon": .1},
                        {"parent_task": "elf3_dwaq_feedback_still_v2"},
                        {"target": "world_height"}, {"leg_joint_names": ["l_knee_y_joint"]}):
            metadata = self.metadata()
            metadata["still_v2_experiment"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings(metadata)

    def test_unknown_null_and_feedback_v2_objective_versions_rejected(self):
        for experiment in (None, {}, {"version": "elf3_dwaq_still_v3"},
                           {"version": "elf3_dwaq_feedback_still_v2"},
                           {"version": "elf3_dwaq_still_v2_typo"}):
            metadata = self.metadata()
            metadata["still_v2_experiment"] = experiment
            with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings(metadata)

    def test_new_objective_cannot_be_attached_to_other_phase_tasks(self):
        for standing in (None, {}, {"version": "elf3_dwaq_feedback_still_v2"},
                         {"version": "elf3_dwaq_feedback_stand_v3"},
                         {"version": "elf3_dwaq_feedback_stand_v3_zero_phase_v1"}):
            metadata = self.metadata()
            metadata["standing_experiment"] = standing
            with self.subTest(standing=standing), self.assertRaisesRegex(ValueError, "original still-v1"):
                old.export.still_preprocessing_settings(metadata)
        metadata = self.metadata()
        metadata["standing_experiment"]["fixed_phase"] = [0., 0., 1., 1.]
        with self.assertRaises(ValueError):
            old.export.still_preprocessing_settings(metadata)

    def test_objective_sources_and_original_input_helper_digest_are_enforced(self):
        for name in self.SOURCES:
            for missing in (True, False):
                metadata = self.metadata()
                sources = metadata["still_v2_experiment"]["sources_sha256"]
                if missing:
                    sources.pop(name)
                else:
                    sources[name] = "0" * 64
                with self.subTest(source=name, missing=missing), self.assertRaisesRegex(ValueError, "source"):
                    old.export.still_preprocessing_settings(metadata)
        for missing in (True, False):
            metadata = self.metadata()
            sources = metadata["standing_experiment"]["sources_sha256"]
            if missing:
                sources.pop("stillness.py")
            else:
                sources["stillness.py"] = "0" * 64
            with self.subTest(original_helper_missing=missing), self.assertRaisesRegex(ValueError, "preprocessing source"):
                old.export.still_preprocessing_settings(metadata)

    def test_absent_new_metadata_preserves_historical_still_and_pose_interfaces(self):
        for extra in ({}, {"zero_command_pose_experiment": {"version": "elf3_dwaq_still_pose_v1"}}):
            metadata = {"standing_experiment": still_contract.still_experiment_contract(still_contract.still_settings()), **extra}
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                path, base, _ = self.fixture(directory, metadata)
                policy, _, _ = old.export.load_policy(path)
                self.assertIsInstance(policy, old.export.MinimalStillPolicyExporter)
                history = old.mixed_inputs()
                with torch.inference_mode():
                    self.assertTrue(torch.equal(policy(history), old.export.MinimalStillPolicyExporter(base)(history)))

    def test_metadata_retains_old_phase_and_raw_500_to_29_interface(self):
        metadata = self.metadata()
        before = copy.deepcopy(metadata)
        description = old.export.preprocessing_metadata(metadata)["zero_command_preprocessing"]
        self.assertEqual(metadata, before)
        self.assertEqual(description["fixed_phase"], [0., 0., 1., -1.])
        self.assertFalse(description["extra_velocity_filter"])
        self.assertFalse(description["extra_action_filter"])
        self.assertFalse(description["joint_lock"])

    def test_torchscript_and_onnx_match_zero_moving_and_dynamic_batches(self):
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
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            for history in (old.mixed_inputs()[:1], old.mixed_inputs(), example, old.mixed_inputs()[:5]):
                expected = policy(history)
                self.assertEqual(expected.shape, (len(history), 29))
                torch.testing.assert_close(restored(history), expected, atol=1e-7, rtol=1e-6)
                actual = session.run(None, {"history": history.numpy()})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
