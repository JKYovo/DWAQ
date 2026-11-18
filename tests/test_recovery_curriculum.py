"""Course handoff, interruption and relocated checkpoint identity, without Isaac."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import recovery_curriculum_plan as plan
import train_elf3_recovery_curriculum as launch

spec = importlib.util.spec_from_file_location("recovery_course_contract_cpu",
    ROOT/"TienKung-Lab/legged_lab/envs/elf3/recovery_course_contract.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def checkpoint(iteration, runner="Elf3DwaqRewardCurriculumRunner"):
    result = {"iter": iteration, "optimizer_state_dict": {"state": {0: "trained"}}, "infos": {
        "training_recipe": {"runner_class_name": runner,
                            "reward_curriculum": {"ramp_start": 20000, "ramp_end": 30000}},
        "elf3_contract": {}}}
    if runner == "Elf3DwaqRecoveryInitRunner":
        result["infos"]["elf3_contract"]["recovery_initialization_experiment"] = {}
        result["infos"]["recovery_training_state"] = {}
    return result


class RecoveryCourseTests(unittest.TestCase):
    def test_fresh_course_exact_boundary_and_update_count(self):
        stages = plan.stage_plan()
        self.assertEqual([(s["first_iteration"], s["last_iteration"], s["updates"])
                          for s in stages], [(0, 50000, 50001), (50001, 80000, 30000)])
        self.assertEqual(sum(s["updates"] for s in stages), 80001)
        self.assertEqual(stages[1]["mode"], "parent")

    def test_resume_before_at_and_after_boundary(self):
        cases = [(49999, "curriculum", [(50000, 50000), (50001, 80000)]),
                 (50000, "curriculum_boundary", [(50001, 80000)]),
                 (60000, "recovery", [(60001, 80000)]),
                 (79999, "recovery", [(80000, 80000)])]
        for completed, mode, expected in cases:
            with self.subTest(completed=completed):
                stages = plan.stage_plan(80001, completed, mode)
                self.assertEqual([(s["first_iteration"], s["last_iteration"]) for s in stages], expected)
                self.assertEqual(sum(s["updates"] for s in stages), 80000-completed)
        self.assertEqual(plan.stage_plan(60003, 60000, "historical_flat60000")[0]["mode"], "parent")
        self.assertEqual(plan.stage_plan(20)[0]["updates"], 20)

    def test_invalid_budgets_and_mixed_source_stage_rejected(self):
        for count in (0, -1, 80002, True, 2.5):
            with self.assertRaises(ValueError):
                plan.stage_plan(count)
        for completed, kind in ((80000, "recovery"), (50000, "recovery"),
                                (60000, "curriculum_boundary"), (False, "curriculum")):
            with self.assertRaises(ValueError):
                plan.stage_plan(80001, completed, kind)

    def test_checkpoint_headers_distinguish_parent_from_amp_resume(self):
        self.assertEqual(plan.checkpoint_stage(checkpoint(50000)), "curriculum_boundary")
        self.assertEqual(plan.checkpoint_stage(checkpoint(20000)), "curriculum")
        self.assertEqual(plan.checkpoint_stage(checkpoint(64000, "Elf3DwaqRecoveryInitRunner")), "recovery")
        bad = [checkpoint(50000, "DWAQOnPolicyRunner"), checkpoint(50001), checkpoint(True)]
        for field, value in (("initialization_only", True), ("recovery_training_state", None)):
            row = checkpoint(60000, "Elf3DwaqRecoveryInitRunner")
            if field == "recovery_training_state":
                del row["infos"][field]
            else:
                row["infos"][field] = value
            bad.append(row)
        row = checkpoint(50000); row["infos"]["training_recipe"]["reward_curriculum"]["ramp_start"] = 5000
        bad.append(row)
        for saved in bad:
            with self.assertRaises(ValueError):
                plan.checkpoint_stage(saved)

    def test_relocated_bank_allowed_but_physics_hash_or_sampling_drift_rejected(self):
        expected = {"physics_dt": .005, "recovery_initialization_experiment": {
            "bank": {"path": "/new/bank.npz", "sha256": "bank", "metadata_sha256": "metadata", "poses": 4096},
            "sources_sha256": {"core": "source"}, "velocity_probabilities": [.75, .2, .05]}}
        original = deepcopy(expected)
        original["recovery_initialization_experiment"]["bank"]["path"] = "/old/bank.npz"
        saved = {"infos": {"elf3_contract": original}}
        identity.validate_portable_contract(saved, expected)
        self.assertEqual(saved["infos"]["elf3_contract"], original)
        for mutate in (lambda c: c.update(physics_dt=.002),
                       lambda c: c["recovery_initialization_experiment"]["bank"].update(sha256="other"),
                       lambda c: c["recovery_initialization_experiment"]["bank"].update(metadata_sha256="other"),
                       lambda c: c["recovery_initialization_experiment"].update(velocity_probabilities=[1., 0., 0.]),
                       lambda c: c["recovery_initialization_experiment"]["sources_sha256"].update(core="other")):
            changed = deepcopy(original); mutate(changed)
            with self.assertRaises(ValueError):
                identity.validate_portable_contract({"infos": {"elf3_contract": changed}}, expected)

    def test_child_commands_pin_parent_and_preserve_total_budget(self):
        args = launch.parser().parse_args(["--headless", "--swanlab-project", "DWAQ"])
        stages = plan.stage_plan()
        first = launch.command(args, stages[0], "test_walking")
        self.assertEqual(first[first.index("--iterations")+1], "50001")
        self.assertNotIn("--parent-checkpoint", first)
        second = launch.command(args, stages[1], "test_recovery", "/tmp/model_50000.pt", "a"*64)
        self.assertEqual(second[second.index("--iterations")+1], "80001")
        self.assertEqual(second[second.index("--parent-sha256")+1], "a"*64)
        self.assertEqual(second[second.index("--parent-num-envs")+1], "16384")
        args.num_envs = 4096
        other = launch.command(args, stages[1], "test_recovery", "/tmp/model_50000.pt", "a"*64)
        self.assertEqual(other[other.index("--parent-num-envs")+1], "4096")
        self.assertIn("--headless", second)
        self.assertEqual(second[second.index("--swanlab-project")+1], "DWAQ")

    def test_failed_child_never_starts_recovery_or_retries(self):
        args = launch.parser().parse_args([])
        prepared = {"source": None, "sha256": None, "plan": plan.stage_plan(), "source_stage": None}
        with tempfile.TemporaryDirectory() as directory, patch.object(launch, "ROOT", Path(directory)), \
                patch.object(launch.subprocess, "Popen") as popen:
            child = popen.return_value
            child.pid, child.wait.return_value, child.poll.return_value = 1234, 9, 9
            with self.assertRaisesRegex(RuntimeError, "next stage was not started"):
                launch.execute(args, prepared)
            self.assertEqual(popen.call_count, 1)

    def test_successful_handoff_uses_exact_child_checkpoint_and_sha256(self):
        args = launch.parser().parse_args(["--num-envs", "4096"])
        prepared = {"source": None, "sha256": None, "plan": plan.stage_plan(), "source_stage": None}
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch.object(launch, "ROOT", Path(directory)):
            def start(argv, cwd):
                calls.append(argv)
                pid = 4100+len(calls)
                name = argv[argv.index("--run-name")+1]
                namespace = "elf3_dwaq_still_v2_curriculum_v1" if len(calls) == 1 else plan.RECOVERY_TASK
                target = int(argv[argv.index("--iterations")+1])-1
                destination = Path(directory)/"logs"/namespace/f"test_{name}_{pid}"/f"model_{target}.pt"
                def wait():
                    destination.parent.mkdir(parents=True)
                    destination.write_bytes(f"completed child{pid}".encode())
                    return 0
                return SimpleNamespace(pid=pid, wait=wait, poll=lambda: 0)
            with patch.object(launch.subprocess, "Popen", side_effect=start):
                launch.execute(args, prepared)
            self.assertEqual(len(calls), 2)
            second = calls[1]
            source = Path(second[second.index("--parent-checkpoint")+1])
            self.assertEqual(source.name, "model_50000.pt")
            self.assertEqual(second[second.index("--parent-sha256")+1], launch.sha256(source))
            self.assertEqual(second[second.index("--parent-num-envs")+1], "4096")
            import json
            manifests = list((Path(directory)/"logs"/plan.TASK).glob("*/course.json"))
            self.assertEqual(len(manifests), 1)
            result = json.loads(manifests[0].read_text())
            self.assertTrue(result["success"])
            self.assertTrue(result["final_checkpoint"].endswith("model_80000.pt"))


if __name__ == "__main__":
    unittest.main()
