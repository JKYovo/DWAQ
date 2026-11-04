"""CPU-only public-entry guards; no simulator, GPU, network or training starts."""
from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ankle_cost_public_launcher_cpu", ROOT / "scripts/train_elf3_ankle_cost.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def recipe(runner="Elf3DwaqAnkleCostRunner"):
    return {"runner_class_name": runner, "seed": 42, "save_interval": 500,
            "policy": {"context_mode": "mean", "init_noise_std": 1.},
            "algorithm": {"class_name": "DWAQPPO", "learning_rate": 1e-4, "entropy_coef": .01},
            "num_steps_per_env": 24, "empirical_normalization": False,
            "upper_body_mirror_loss_coeff": .1, "upper_body_pose_loss_coeff": .02,
            "device": "cuda:0", "run_name": "example", "max_iterations": 100001,
            "resume": False, "logger": "tensorboard", "experiment_name": "elf3_dwaq_still_v2"}


def completed_checkpoint(mode="resume", iteration=100):
    contract = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
    metadata = {"asset_contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                "joint_names": contract["joint_names"], "num_obs": 100, "history_length": 5,
                "standing_experiment": {"version": "elf3_dwaq_still_v1"},
                "still_v2_experiment": {"version": "elf3_dwaq_still_v2"},
                "feedback_experiment": {"version": "elf3_feedback_v1",
                                        "velocity_feedback": {"mode": "lpf20", "cutoff_hz": 20.}}}
    if mode == "resume":
        metadata["reward_stage_experiment"] = launcher.stage_contract_helpers().reward_stage_contract()
    return {"iter": iteration, "model_state_dict": {"actor.weight": torch.ones(2, 2)},
            "optimizer_state_dict": {"state": {0: {"step": torch.tensor(3.), "exp_avg": torch.zeros(2, 2)}},
                                     "param_groups": [{"lr": 8e-5, "params": [0]}]},
            "infos": {"elf3_contract": metadata,
                      "training_recipe": recipe("Elf3DwaqStillV2Runner" if mode == "parent" else launcher.RUNNER_CLASS)}}


class PublicParserTests(unittest.TestCase):
    def args(self, *items):
        return launcher.build_parser().parse_args(list(items))

    def test_safe_fresh_defaults_need_no_private_logs_or_account(self):
        args = self.args()
        self.assertEqual((args.num_envs, args.iterations, args.seed), (64, 20, 42))
        self.assertIsNone(args.swanlab_project)
        prepared = launcher.prepare_checkpoint(args)
        self.assertEqual((prepared["mode"], prepared["start_iteration"], prepared["learning_updates"],
                          prepared["target_iteration"]), ("fresh", 0, 20, 19))
        self.assertIsNone(prepared["path"])

    def test_mutually_exclusive_modes_and_fixed_seed(self):
        for items in (("--resume", "one.pt", "--parent-checkpoint", "two.pt"), ("--seed", "123")):
            with self.subTest(items=items), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*items)

    def test_parent_digest_required_and_cannot_be_used_with_other_modes(self):
        for items in (("--parent-checkpoint", "model.pt"), ("--parent-sha256", "a" * 64),
                      ("--resume", "model.pt", "--parent-sha256", "a" * 64),
                      ("--parent-checkpoint", "model.pt", "--parent-sha256", "bad")):
            with self.subTest(items=items), self.assertRaises(ValueError):
                launcher.validate_args(self.args(*items))
        args = launcher.validate_args(self.args("--parent-checkpoint", "model.pt", "--parent-sha256", "A" * 64))
        self.assertEqual(args.parent_sha256, "a" * 64)

    def test_reject_invalid_counts_names_devices_and_project(self):
        for items in (("--num-envs", "0"), ("--iterations", "0"), ("--run-name", ".."),
                      ("--run-name", "../run"), ("--run-name", "/absolute"), ("--run-name", r"dir\name"),
                      ("--run-name", ""), ("--device", "cuda:-1"), ("--swanlab-project", " ")):
            with self.subTest(items=items), self.assertRaises(ValueError):
                launcher.validate_args(self.args(*items))

    def test_no_learning_rate_or_reward_override_cli(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--learning-rate", "0.001")

    def test_total_iteration_budget_and_no_off_by_one(self):
        self.assertEqual(launcher.update_budget(20), {"start_iteration": 0, "learning_updates": 20, "target_iteration": 19})
        self.assertEqual(launcher.update_budget(100001, 18500),
                         {"start_iteration": 18501, "learning_updates": 81500, "target_iteration": 100000})
        for total, done in ((20, 19), (20, 20), (0, None), (20, True), (20, -1)):
            with self.subTest(total=total, done=done), self.assertRaises(ValueError):
                launcher.update_budget(total, done)

    def test_unique_new_run_directory_never_reuses_source(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, "ROOT", Path(temp)):
            args = self.args("--run-name", "portable")
            directory = launcher.choose_run_directory(args, stamp="2026-11-04_12-00-00", pid=123)
            self.assertEqual(directory.parent, Path(temp) / "logs" / launcher.TASK)
            self.assertEqual(directory.name, "2026-11-04_12-00-00_portable_123")
            self.assertFalse(directory.exists())
            directory.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                launcher.choose_run_directory(args, stamp="2026-11-04_12-00-00", pid=123)


class CheckpointGuardTests(unittest.TestCase):
    def test_completed_stage_and_parent_have_correct_next_iteration(self):
        for mode in ("resume", "parent"):
            with self.subTest(mode=mode):
                report = launcher.validate_checkpoint_header(completed_checkpoint(mode), mode, 200)
                self.assertEqual((report["completed_iteration"], report["start_iteration"], report["learning_updates"]), (100, 101, 99))

    def test_wrong_stage_source_version_layout_and_feedback_rejected(self):
        fixtures = []
        saved = completed_checkpoint(); saved["infos"]["elf3_contract"]["reward_stage_experiment"]["version"] = "wrong"; fixtures.append(saved)
        saved = completed_checkpoint(); saved["infos"]["elf3_contract"]["reward_stage_experiment"]["sources_sha256"]["ankle_cost.py"] = "bad"; fixtures.append(saved)
        saved = completed_checkpoint(); saved["infos"]["elf3_contract"]["history_length"] = 20; fixtures.append(saved)
        saved = completed_checkpoint(); saved["infos"]["elf3_contract"]["feedback_experiment"]["velocity_feedback"]["mode"] = "ideal"; fixtures.append(saved)
        saved = completed_checkpoint(); saved["infos"]["elf3_contract"]["still_v2_experiment"]["version"] = "wrong"; fixtures.append(saved)
        for index, saved in enumerate(fixtures):
            with self.subTest(fixture=index), self.assertRaises(ValueError):
                launcher.validate_checkpoint_header(saved, "resume", 200)
        with self.assertRaises(ValueError):
            launcher.validate_checkpoint_header(completed_checkpoint(), "parent", 200)
        with self.assertRaises(ValueError):
            launcher.validate_checkpoint_header(completed_checkpoint("parent"), "resume", 200)

    def test_initial_random_empty_adam_and_nonfinite_rejected(self):
        saved = completed_checkpoint(); saved["infos"]["initialization_only"] = True
        with self.assertRaises(ValueError):
            launcher.validate_checkpoint_header(saved, "resume", 200)
        saved = completed_checkpoint(); saved["optimizer_state_dict"]["state"] = {}
        with self.assertRaises(ValueError):
            launcher.validate_checkpoint_header(saved, "resume", 200)
        saved = completed_checkpoint(); saved["model_state_dict"]["actor.weight"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            launcher.validate_checkpoint_header(saved, "resume", 200)

    def test_invalid_completed_iteration_and_learning_rate_rejected(self):
        for iteration in (-1, True, 200):
            saved = completed_checkpoint(iteration=iteration)
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                launcher.validate_checkpoint_header(saved, "resume", 200)
        for value in (None, 0., -1., float("nan")):
            saved = completed_checkpoint(); saved["optimizer_state_dict"]["param_groups"][0]["lr"] = value
            with self.subTest(lr=value), self.assertRaises(ValueError):
                launcher.validate_checkpoint_header(saved, "resume", 200)

    def test_recipe_allows_only_bookkeeping_not_ppo_or_auxiliaries(self):
        saved = completed_checkpoint()
        current = deepcopy(saved["infos"]["training_recipe"])
        current.update(device="cuda:1", run_name="new_run", max_iterations=200001)
        self.assertTrue(launcher.validate_saved_recipe(saved, current, "resume")["full_training_recipe_matched"])
        for field, value in (("algorithm", {"learning_rate": .001}), ("policy", {"context_mode": "sample"}),
                             ("num_steps_per_env", 12), ("upper_body_pose_loss_coeff", .1)):
            altered = deepcopy(current); altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                launcher.validate_saved_recipe(saved, altered, "resume")
        parent = completed_checkpoint("parent")
        launcher.validate_saved_recipe(parent, recipe(), "parent")

    def test_parent_sha_checked_and_source_checkpoint_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model_100.pt"
            torch.save(completed_checkpoint("parent"), path)
            digest = launcher.sha256(path)
            args = launcher.build_parser().parse_args(["--parent-checkpoint", str(path), "--parent-sha256", digest,
                                                       "--iterations", "200"])
            result = launcher.prepare_checkpoint(args)
            self.assertEqual(result["mode"], "parent")
            self.assertEqual(result["start_iteration"], 101)
            self.assertEqual(launcher.sha256(path), digest)
            args.parent_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                launcher.prepare_checkpoint(args)

    def test_exact_learning_rate_and_next_iteration_restoration_guard(self):
        saved = completed_checkpoint()
        check = Mock()
        algorithm = SimpleNamespace(policy=SimpleNamespace(state_dict=lambda: saved["model_state_dict"]),
                                    optimizer=SimpleNamespace(state_dict=lambda: saved["optimizer_state_dict"],
                                                              param_groups=deepcopy(saved["optimizer_state_dict"]["param_groups"])),
                                    learning_rate=8e-5)
        runner = SimpleNamespace(alg=algorithm, current_learning_iteration=101, _assert_tree_equal=check)
        self.assertTrue(launcher.verify_restored_state(runner, saved)["model_and_Adam_exactly_restored"])
        self.assertEqual(check.call_count, 2)
        algorithm.learning_rate = 1e-4
        with self.assertRaisesRegex(ValueError, "learning rate"):
            launcher.verify_restored_state(runner, saved)
        algorithm.learning_rate = 8e-5; runner.current_learning_iteration = 100
        with self.assertRaisesRegex(ValueError, "iteration"):
            launcher.verify_restored_state(runner, saved)

    def test_public_entry_does_not_import_private_resume_or_bind_account(self):
        text = (ROOT / "scripts/train_elf3_ankle_cost.py").read_text()
        self.assertNotIn("resume_elf3_still_v2", text)
        self.assertNotIn("resume_elf3_ankle_cost", text)
        self.assertNotIn("vckg3ugg", text)
        self.assertNotIn("/home/cheng", text)
        self.assertIn("initialization_only", text)


if __name__ == "__main__":
    unittest.main()
