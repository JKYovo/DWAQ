"""Portable release checks; no private logs, checkpoints, GPU or Isaac startup."""
import ast
from copy import copy, deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
        spec.loader.exec_module(module)
    return module


launcher = load(ROOT / "scripts/train_elf3_force500.py", "force500_public_launcher_test")
contract = load(DIR / "foot_force_500_contract.py", "force500_public_contract_test")
rewards = load(DIR / "foot_force_peak_rewards.py", "force500_public_rewards_test")


class ParentEnv:
    def init_buffers(self):
        self.device, self.num_envs = "cpu", 2
        self.cfg = SimpleNamespace(sim=SimpleNamespace(decimation=4))
        self.contact_sensor = SimpleNamespace(
            find_bodies=lambda names, preserve_order: ([2, 0], tuple(names)),
            data=SimpleNamespace(net_forces_w=torch.zeros(2, 3, 3)))
        self.feedback_calls = 0

    def _post_physics_step(self):
        self.feedback_calls += 1

    def reset(self, ids):
        pass

    def step(self, values):
        for left, right in values:
            self.contact_sensor.data.net_forces_w[:, 2, 2] = left
            self.contact_sensor.data.net_forces_w[:, 0, 2] = right
            self._post_physics_step()
        return rewards.body_force_control_peak(self, None, threshold=500., max_reward=400.)


scope = {"copy": copy, "deepcopy": deepcopy, "torch": torch,
         "Elf3DwaqControlSmoothnessEnv": ParentEnv, "Elf3DwaqControlSmoothnessRunner": object,
         "FOOT_BODY_NAMES": ("l_ankle_x_link", "r_ankle_x_link"),
         "accumulate_peak": rewards.accumulate_peak}
tree = ast.parse((DIR / "foot_force_peak.py").read_text())
env_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Elf3DwaqFootForcePeakEnv")
exec(compile(ast.Module(body=[env_class], type_ignores=[]), str(DIR / "foot_force_peak.py"), "exec"), scope)


class Force500PublicTests(unittest.TestCase):
    def test_scratch_is_default_with_no_parent_or_account_dependency(self):
        args = launcher.build_parser().parse_args([])
        self.assertIsNone(args.resume)
        self.assertIsNone(args.swanlab_project)
        prepared = launcher.prepare_checkpoint(args)
        self.assertEqual((prepared["mode"], prepared["start_iteration"], prepared["target_iteration"]),
                         ("fresh", 0, 19))
        self.assertIsNone(prepared["path"])
        self.assertNotIn("--parent-checkpoint", launcher.build_parser()._option_string_actions)

    def test_total_budget_and_strict_arguments(self):
        args = launcher.build_parser().parse_args(["--num-envs", "16384", "--iterations", "100001"])
        self.assertEqual(launcher.prepare_checkpoint(args)["target_iteration"], 100000)
        for flags in (["--num-envs", "0"], ["--iterations", "0"], ["--run-name", "../bad"]):
            with self.assertRaises(ValueError):
                launcher.prepare_checkpoint(launcher.build_parser().parse_args(flags))

    def test_foreign_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_0.pt"
            torch.save({"infos": {"elf3_contract": {}}}, path)
            args = launcher.build_parser().parse_args(["--resume", str(path)])
            with self.assertRaisesRegex(ValueError, "exact500N"):
                launcher.prepare_checkpoint(args)

    def test_declared_sources_complete_and_portable(self):
        for name in launcher.RUNTIME_FILES:
            source = DIR / name
            self.assertTrue(source.is_file(), name)
            self.assertNotIn("/home/cheng/", source.read_text())
        identity = contract.threshold_contract()
        self.assertEqual(identity["threshold_n"], 500)
        self.assertEqual(identity["feet_force_weight"], -.005)
        for name, digest in identity["sources_sha256"].items():
            self.assertEqual(hashlib.sha256((DIR / name).read_bytes()).hexdigest(), digest)
        source = (ROOT / "scripts/train_elf3_force500.py").read_text()
        self.assertNotIn("/home/cheng/", source)
        self.assertNotIn("Rainbowow", source)

    def test_boundaries_and_cap_match_actual500_formula(self):
        values = torch.tensor([0., 399., 400., 424., 499., 500., 501., 600., 900., 1000.])
        raw = rewards.force_excess(values, threshold=500., max_reward=400.)
        torch.testing.assert_close(raw, torch.tensor([0., 0., 0., 0., 0., 0., 1., 100., 400., 400.]))
        self.assertAlmostEqual(float(raw[7] * -.005 * .02), -.01, places=7)

    def test_every_substep_and_noncontiguous_named_feet(self):
        for pulse in range(4):
            env = scope["Elf3DwaqFootForcePeakEnv"]()
            env.init_buffers()
            values = [(300., 0.)] * 4
            values[pulse] = (700., 0.)
            torch.testing.assert_close(env.step(values), torch.full((2,), 200.))
            self.assertEqual(env.feedback_calls, 4)
            env.reset([0])
            self.assertEqual(float(env.force_peak_metric[0]), 0.)
            self.assertEqual(float(env.force_peak_metric[1]), 700.)

    def test_no_fictitious_simultaneous_peak_and_no_cross_step_leak(self):
        env = scope["Elf3DwaqFootForcePeakEnv"]()
        env.init_buffers()
        torch.testing.assert_close(env.step([(500., 0.), (0., 500.), (0., 0.), (0., 0.)]), torch.zeros(2))
        env.step([(900., 0.)] * 4)
        torch.testing.assert_close(env.step([(200., 0.)] * 4), torch.zeros(2))

    def test_latest_factory_changes_only_threshold_and_runner_identity(self):
        tree = ast.parse((DIR / "foot_force_500.py").read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_configs")
        writes = [ast.unparse(n.targets[0]) for n in factory.body if isinstance(n, ast.Assign)]
        self.assertEqual(writes, ["(cfg, agent)", "cfg.reward.feet_force.params['threshold']", "agent.runner_class_name"])


if __name__ == "__main__":
    unittest.main()
