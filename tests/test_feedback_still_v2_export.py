"""The v2 zero preprocessing is required, with old export behavior preserved."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("v2_export_existing_tests", ROOT / "tests/test_minimal_still_export.py")
contract = load("v2_export_contract", ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_still_v2_contract.py")


class FeedbackStillV2ExportTests(unittest.TestCase):
    def test_complete_contract_has_all_source_fingerprints(self):
        settings = contract.standing_settings()
        identity = contract.standing_experiment_contract(settings)
        self.assertEqual(identity["parent_task"], "elf3_dwaq_feedback_v1")
        self.assertFalse(identity["additional_hip_action_constraints"])
        self.assertFalse(identity["lock_world_yaw"])
        self.assertGreaterEqual(len(identity["sources_sha256"]), 7)
        self.assertEqual(old.export.still_preprocessing_settings({"standing_experiment": identity}), settings)

    def test_loader_embeds_preprocessor_and_keeps_moving_policy(self):
        fixture = old.MinimalStillExportTests()
        with tempfile.TemporaryDirectory() as directory:
            path, baseline, _ = fixture.fixture_checkpoint(
                directory, contract.standing_experiment_contract(contract.standing_settings()))
            policy, _, _ = old.export.load_policy(path)
            self.assertIsInstance(policy, old.export.MinimalStillPolicyExporter)
            moving = old.mixed_inputs()[5:]
            zero = old.mixed_inputs()[:3]
            altered = zero.reshape(-1, 5, 100).clone()
            altered[:, :, 96:100] += 20
            with torch.inference_mode():
                self.assertTrue(torch.equal(policy(moving), baseline(moving)))
                self.assertTrue(torch.equal(policy(zero), policy(altered.flatten(1))))

    def test_incomplete_changed_unknown_contract_rejected(self):
        invalid = [{"version": "elf3_dwaq_feedback_still_v3"}]
        for key in contract.standing_settings():
            candidate = contract.standing_settings()
            candidate.pop(key)
            if key == "version":
                continue
            invalid.append(candidate)
        invalid.extend([contract.standing_settings() | {"full_body_pose_weight": -.1},
                        contract.standing_settings() | {"lock_world_yaw": True}])
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                old.export.still_preprocessing_settings({"standing_experiment": settings})

    def test_changed_preprocessing_source_rejected(self):
        identity = contract.standing_experiment_contract(contract.standing_settings())
        identity["sources_sha256"]["stillness.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "preprocessing source"):
            old.export.still_preprocessing_settings({"standing_experiment": identity})


if __name__ == "__main__":
    unittest.main()
