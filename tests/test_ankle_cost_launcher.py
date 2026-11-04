"""CPU-only continuation guards; no Isaac app, GPU work or training starts."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ankle_cost_launcher_cpu", ROOT / "scripts/resume_elf3_ankle_cost.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "logs/original_run"
        (self.run / "params").mkdir(parents=True)
        self.parent = self.run / "model_2000.pt"
        self.parent.write_bytes(b"original read-only checkpoint")
        for name in ("initialization.json", "params/env.yaml", "params/agent.yaml"):
            (self.run / name).write_text("original protected metadata")
        protected = {str(p): digest(p) for p in (self.parent, self.run / "initialization.json",
                     self.run / "params/env.yaml", self.run / "params/agent.yaml")}
        self.parent_report = {
            "checkpoint": str(self.parent), "checkpoint_sha256": digest(self.parent),
            "original_log_dir": str(self.run), "completed_iteration": 2000,
            "swanlab_run_path": "Rainbowow/DWAQ/vckg3ugg", "cloud_ack_iteration_before_resume": 2150,
            "elapsed_time_log_offset": 6001,
            "cloud_recomputed_prefix_policy": "keep existing ACK",
            "environment_resume_note": "physics/history/RNG reset, weights and Adam restored",
            "source_audit": {"all_training_dependencies_unchanged": True},
        }
        self.saved = {"iter": 2000, "model_state_dict": {"model": "CPU fixture"},
                      "optimizer_state_dict": {"state": {0: {"step": 20}}, "param_groups": [{"lr": 1e-4}]}}
        baseline = (self.run, {"task": "original"}, {"device": "cuda:0"}, self.saved, protected, self.parent_report)
        self.original = SimpleNamespace(preflight=unittest.mock.Mock(return_value=baseline),
            sha256=digest, require_finite_tree=unittest.mock.Mock())
        self.identity = {"version": "test_ankle_cost_v1", "weight": -.0005}

        def validate(value):
            if value != self.identity:
                raise ValueError("reward stage differs")
            return self.identity
        self.helper = SimpleNamespace(reward_stage_contract=lambda: copy.deepcopy(self.identity),
                                      validate_reward_stage_contract=validate)
        self.patchers = [patch.object(launcher, "ROOT", self.root),
                         patch.object(launcher, "original_resume", return_value=self.original),
                         patch.object(launcher, "stage_contract_module", return_value=self.helper),
                         patch.object(launcher, "canonical_sha256", side_effect=fingerprint)]
        for item in self.patchers:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temporary.cleanup)

    def args(self, *extra):
        return launcher.build_parser().parse_args(["--checkpoint", str(self.parent),
                                                   "--source-run", str(self.run), *extra])

    def stage_fixture(self):
        recipe = {"seed": 42, "policy": {"context_mode": "mean"}, "save_interval": 500}
        contract = {"feedback_experiment": {"recipe": "unchanged"},
                    "reward_stage_experiment": copy.deepcopy(self.identity)}
        saved = {"iter": 2500, "model_state_dict": {"model": "fixture"},
                 "optimizer_state_dict": {"state": {0: {"step": 25}}, "param_groups": [{"lr": 1e-4}]},
                 "infos": {"elf3_contract": contract, "training_recipe": recipe}}
        initialization = {"smoke": False, "num_envs": 16384, "seed": 42,
            "target_iteration": 100000, "save_interval": 500, "device": "cuda:0",
            "source_checkpoint_sha256": self.parent_report["checkpoint_sha256"],
            "source_run": str(self.run), "experiment_contract": copy.deepcopy(contract),
            "agent_configuration_sha256": fingerprint(recipe)}
        return saved, initialization

    def test_checkpoint_is_required_and_formal_controls_cannot_be_overridden(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.build_parser().parse_args([])
        for extra in (("--num-envs", "16384"), ("--num-envs", "64"), ("--iterations", "20")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launcher.resolve_args(self.args(*extra))

    def test_formal_preflight_preserves_original_recipe_and_output_separation(self):
        result = launcher.preflight(self.args())
        report = result["report"]
        self.assertEqual((report["num_envs"], report["seed"], report["save_interval"]), (16384, 42, 500))
        self.assertEqual((report["next_iteration"], report["updates_requested"],
                          report["expected_last_iteration"]), (2001, 98000, 100000))
        self.assertEqual(report["swanlab_run_path"], "Rainbowow/DWAQ/vckg3ugg")
        self.assertTrue(report["swanlab_enabled"])
        self.assertFalse(report["new_swanlab_run"])
        self.assertFalse(report["automatic_training_restart"])
        self.assertTrue(result["stage_dir"].is_relative_to(self.run / "stages"))
        self.assertFalse(result["stage_dir"].exists())
        passed = self.original.preflight.call_args.args[0]
        self.assertEqual((passed.log_dir, passed.checkpoint, passed.target_iteration, passed.device),
                         (self.run, self.parent, 100000, "cuda:0"))
        for path, expected in result["protected"].items():
            self.assertEqual(digest(path), expected)

    def test_smoke_is_64_env_20_updates_no_swan_and_never_original_log_directory(self):
        result = launcher.preflight(self.args("--smoke"))
        report = result["report"]
        self.assertEqual((report["num_envs"], report["updates_requested"], report["expected_last_iteration"]),
                         (64, 20, 2020))
        self.assertFalse(report["swanlab_enabled"])
        self.assertFalse(report["smoke_weights_allowed_for_formal_training"])
        self.assertTrue(result["stage_dir"].is_relative_to(self.root / "artifacts"))
        self.assertTrue(result["stage_dir"].name.startswith("ankle_cost_smoke_"))
        self.assertFalse(result["stage_dir"].exists())

    def test_smoke_overrides_are_explicit_and_positive(self):
        result = launcher.preflight(self.args("--smoke", "--num-envs", "8", "--iterations", "2"))
        self.assertEqual((result["report"]["num_envs"], result["report"]["updates_requested"]), (8, 2))
        for extra in (("--num-envs", "0"), ("--iterations", "-1")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launcher.preflight(self.args("--smoke", *extra))

    def test_stage_path_guards_prevent_original_or_existing_directory_reuse(self):
        for directory in (self.run, self.root, self.run / "stages"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                launcher.preflight(self.args("--stage-dir", str(directory)))
        existing = self.run / "stages/already_exists"
        existing.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            launcher.preflight(self.args("--stage-dir", str(existing)))
        with self.assertRaises(ValueError):
            launcher.preflight(self.args("--smoke", "--stage-dir", str(self.root / "artifacts/not_smoke")))

    def test_formal_stage_checkpoint_resume_and_Adam_preflight(self):
        saved, initialization = self.stage_fixture()
        directory = self.run / "stages/formal_stage"
        (directory / "params").mkdir(parents=True)
        checkpoint = directory / "model_2500.pt"
        checkpoint.write_bytes(b"new reward stage checkpoint")
        (directory / "initialization.json").write_text(json.dumps(initialization))
        for name in ("env.yaml", "agent.yaml"):
            (directory / "params" / name).write_text("stage immutable params")
        with patch("torch.load", return_value=saved):
            result = launcher.preflight(self.args("--resume-stage", str(checkpoint)))
        self.assertEqual((result["report"]["next_iteration"], result["report"]["updates_requested"]), (2501, 97500))
        self.assertEqual(result["stage_dir"], directory)
        self.assertTrue(result["report"]["resuming_stage"])
        self.assertIn(str(directory / "initialization.json"), result["protected"])
        self.assertEqual(self.original.require_finite_tree.call_count, 2)

    def test_smoke_or_mismatched_stage_cannot_resume_formal(self):
        saved, initial = self.stage_fixture()
        cases = [("smoke", True), ("num_envs", 64), ("seed", 0), ("target_iteration", 200000),
                 ("save_interval", 100), ("device", "cpu"), ("source_checkpoint_sha256", "wrong"),
                 ("source_run", "another run"), ("experiment_contract", {})]
        for name, value in cases:
            candidate = copy.deepcopy(initial)
            candidate[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                launcher.validate_stage_checkpoint(saved, candidate, self.parent_report, self.args(), self.helper)
        for iteration in (True, 1999, 100000):
            candidate = copy.deepcopy(saved)
            candidate["iter"] = iteration
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                launcher.validate_stage_checkpoint(candidate, initial, self.parent_report, self.args(), self.helper)
        for field in ("initialization_only", "recipe", "stage", "Adam"):
            candidate = copy.deepcopy(saved)
            if field == "initialization_only":
                candidate["infos"][field] = True
            elif field == "recipe":
                candidate["infos"]["training_recipe"]["seed"] = 0
            elif field == "stage":
                candidate["infos"]["elf3_contract"]["reward_stage_experiment"]["weight"] = 0
            else:
                candidate["optimizer_state_dict"]["state"] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                launcher.validate_stage_checkpoint(candidate, initial, self.parent_report, self.args(), self.helper)

    def test_smoke_cannot_resume_a_formal_or_smoke_stage(self):
        with self.assertRaises(ValueError):
            launcher.resolve_args(self.args("--smoke", "--resume-stage", str(self.parent)))

    def test_verify_only_never_calls_runtime_or_creates_stage(self):
        output = io.StringIO()
        with patch.object(launcher, "run", side_effect=AssertionError("No Isaac/training in CPU verify")) as runtime:
            with redirect_stdout(output):
                launcher.main(["--checkpoint", str(self.parent), "--source-run", str(self.run), "--verify-config-only"])
        runtime.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["next_iteration"], 2001)
        self.assertFalse(Path(report["stage_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
