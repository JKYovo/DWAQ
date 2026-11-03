"""CPU-only lifecycle tests of the real stand-v1 launcher.

All Popen calls are intercepted. The only files created are temporary fixtures;
no Isaac import, training, network, SwanLab SDK, ROS node or robot is started.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/train_elf3_stand.py"
NATIVE_PID = 27341
PARENT_PID = 111
STAMP = "20261030_135900"
RUN_NAME = "launcher_unit"
SUFFIX = f"{RUN_NAME}_{STAMP}_{PARENT_PID}"


def load_launcher():
    """Import the actual launcher, replacing only its preflight dependencies."""
    release = ModuleType("release_0918_checks")
    release.check_source_manifest = Mock(return_value={"verified": True})
    feedback = ModuleType("train_elf3_feedback")
    feedback.source_fingerprints = Mock(return_value={"mock_source": "a" * 64})
    spec = importlib.util.spec_from_file_location("standing_launcher_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"release_0918_checks": release,
                                 "train_elf3_feedback": feedback}):
        spec.loader.exec_module(module)
    return module


class LauncherFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "logs/elf3_dwaq_stand_v1" / f"2026-10-30_13-59-00_{SUFFIX}"
        self.module = load_launcher()
        # The launcher hashes its own source relative to ROOT. Use a temporary
        # fixture copy as that path without altering any repository file.
        fake_source = self.root / "scripts/train_elf3_stand.py"
        fake_source.parent.mkdir(parents=True)
        fake_source.write_bytes(SOURCE.read_bytes())
        self.module.__file__ = str(fake_source)
        self.module.ROOT = self.root
        self.module.os = SimpleNamespace(getpid=lambda: PARENT_PID)
        self.module.datetime = SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: STAMP))
        self.child = Mock()
        self.child.pid = NATIVE_PID
        self.child.poll.return_value = None
        self.child.wait.return_value = 0
        self.sidecar = SimpleNamespace(pid=99999)
        self.launches = []
        self.sidecar_request = None
        self.output = io.StringIO()

    def write_params(self, *, model_zero=True, target=True, agent_changes=None):
        params = self.directory / "params"
        params.mkdir(parents=True, exist_ok=True)
        agent = {"runner_class_name": "Elf3DwaqStandRunner", "experiment_name": "elf3_dwaq_stand_v1",
                 "seed": 42, "max_iterations": 2, "save_interval": 500, "resume": False,
                 "logger": "tensorboard", "upper_body_mirror_loss_coeff": .1,
                 "upper_body_pose_loss_coeff": .02, "policy": {"context_mode": "mean"},
                 "algorithm": {"learning_rate": 1e-4}}
        agent.update(agent_changes or {})
        env = {"scene": {"num_envs": 64}, "velocity_feedback": {"mode": "lpf20", "cutoff_hz": 20.},
               "standing_experiment": {"version": "elf3_stand_v1", "mode": "walking_zero_command_fix",
                                       "command_epsilon": 1e-6}}
        (params / "env.yaml").write_text(yaml.safe_dump(env))
        (params / "agent.yaml").write_text(yaml.safe_dump(agent))
        if model_zero:
            (self.directory / "model_0.pt").write_bytes(b"inert checkpoint fixture")
        if target:
            (self.directory / "model_1.pt").write_bytes(b"inert final checkpoint fixture")

    def popen(self, command, **kwargs):
        command = list(command)
        self.launches.append((command, kwargs))
        if command[1].endswith("/legged_lab/scripts/train.py"):
            return self.child
        if command[1].endswith("/scripts/pose_scratch_swanlab.py"):
            # Provenance/request must be complete before sidecar discovery.
            self.sidecar_request = json.loads((self.directory / "swanlab_request.json").read_text())
            assert (self.directory / "initialization.json").is_file()
            return self.sidecar
        raise AssertionError(f"Unexpected subprocess: {command}")

    def run(self, *, swanlab=False, sleep=None):
        argv = [str(SOURCE), "--num-envs", "64", "--iterations", "2", "--seed", "42",
                "--run-name", RUN_NAME, "--swanlab-project", "DWAQ" if swanlab else ""]
        def no_sleep(seconds):
            raise AssertionError("Unexpected wait in an already-discoverable case")
        with patch.object(self.module.subprocess, "Popen", side_effect=self.popen), \
             patch.object(self.module.time, "sleep", side_effect=sleep or no_sleep), \
             patch.object(sys, "argv", argv), \
             contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output):
            self.module.main()

    def read(self, name):
        return json.loads((self.directory / name).read_text())

    def assert_no_stop_or_restart(self, test):
        self.child.kill.assert_not_called()
        self.child.terminate.assert_not_called()
        self.child.send_signal.assert_not_called()
        native = [command for command, kwargs in self.launches if command[1].endswith("/legged_lab/scripts/train.py")]
        test.assertEqual(len(native), 1)
        self.child.wait.assert_called_once_with()


class StandingLauncherTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return LauncherFixture(temporary.name)

    def test_native_arguments_are_from_scratch_without_native_swanlab_or_resume(self):
        case = self.fixture()
        case.write_params()
        case.run()
        command, kwargs = case.launches[0]
        self.assertEqual(command[command.index("--task") + 1], "elf3_dwaq_stand_v1")
        self.assertEqual(command[command.index("--num_envs") + 1], "64")
        self.assertEqual(command[command.index("--max_iterations") + 1], "2")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(command[command.index("--logger") + 1], "tensorboard")
        self.assertIn("--headless", command)
        for forbidden in ("--resume", "--checkpoint", "--load_run", "--reset_optimizer_on_resume",
                          "--swanlab_project", "--swanlab-project"):
            self.assertNotIn(forbidden, command)
        self.assertEqual(kwargs, {"cwd": case.root})
        init = case.read("initialization.json")
        self.assertTrue(init["from_scratch"])
        self.assertIsNone(init["checkpoint_loaded"])
        self.assertEqual(init["initial_iteration"], 0)
        self.assertEqual(init["target_iteration"], 1)
        self.assertFalse(init["automatic_training_stop"])
        self.assertFalse(init["automatic_training_restart"])
        self.assertEqual(len(case.launches), 1)
        case.assert_no_stop_or_restart(self)

    def test_completed_short_run_is_discovered_even_when_child_already_exited(self):
        case = self.fixture()
        case.write_params()
        case.child.poll.return_value = 0
        case.run()
        completed = case.read("training_exit.json")
        self.assertTrue(completed["success"])
        self.assertEqual(completed["child_returncode"], 0)
        self.assertTrue(completed["target_checkpoint_exists"])
        case.assert_no_stop_or_restart(self)

    def test_files_published_at_first_exit_observation_are_not_missed(self):
        case = self.fixture()
        def child_exits():
            # Reproduces the former race: glob found no directory, then the
            # native child completed and poll returned exit=0. Taking poll's
            # status snapshot BEFORE discovery makes these files discoverable.
            case.write_params()
            return 0
        case.child.poll.side_effect = child_exits
        case.run()
        self.assertTrue(case.read("training_exit.json")["success"])
        case.assert_no_stop_or_restart(self)

    def test_opened_partial_yaml_is_not_read_before_first_checkpoint_barrier(self):
        case = self.fixture()
        params = case.directory / "params"
        params.mkdir(parents=True)
        (params / "agent.yaml").write_text("policy: [unterminated")
        (params / "env.yaml").write_text("")
        waits = []
        def finish_native_startup(seconds):
            waits.append(seconds)
            self.assertEqual(len(waits), 1)
            case.write_params()
        case.run(sleep=finish_native_startup)
        self.assertEqual(waits, [1])
        self.assertTrue(case.read("training_exit.json")["success"])
        case.assert_no_stop_or_restart(self)

    def test_sidecar_receives_native_training_pid_and_does_not_change_child_command(self):
        case = self.fixture()
        case.write_params()
        case.run(swanlab=True)
        self.assertEqual(len(case.launches), 2)
        command, kwargs = case.launches[1]
        self.assertEqual(command[command.index("--training-pid") + 1], str(NATIVE_PID))
        self.assertEqual(command[command.index("--log-dir") + 1], str(case.directory))
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(case.sidecar_request["project"], "DWAQ")
        self.assertEqual(case.sidecar_request["config"]["training_pid"], NATIVE_PID)
        self.assertEqual(case.sidecar_request["config"]["launcher_pid"], PARENT_PID)
        self.assertEqual(case.read("initialization.json")["training_pid"], NATIVE_PID)
        self.assertTrue(case.read("training_exit.json")["success"])
        case.assert_no_stop_or_restart(self)

    def test_nonzero_child_exit_records_failure_without_restart_or_kill(self):
        case = self.fixture()
        case.write_params()
        case.child.wait.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "exit 7"):
            case.run()
        completed = case.read("training_exit.json")
        self.assertFalse(completed["success"])
        self.assertEqual(completed["child_returncode"], 7)
        self.assertFalse(completed["auto_restart"])
        self.assertTrue((case.directory / "launcher_failure.txt").is_file())
        case.assert_no_stop_or_restart(self)

    def test_exit_zero_without_final_checkpoint_is_not_reported_as_success(self):
        case = self.fixture()
        case.write_params(target=False)
        with self.assertRaisesRegex(RuntimeError, "before target checkpoint"):
            case.run()
        completed = case.read("training_exit.json")
        self.assertFalse(completed["success"])
        self.assertEqual(completed["child_returncode"], 0)
        self.assertFalse(completed["target_checkpoint_exists"])
        case.assert_no_stop_or_restart(self)

    def test_child_exit_without_any_saved_configuration_is_reported(self):
        case = self.fixture()
        case.child.poll.return_value = 9
        case.child.wait.return_value = 9
        with self.assertRaisesRegex(RuntimeError, "before configuration was saved.*exit 9"):
            case.run()
        self.assertFalse(case.directory.exists())
        case.assert_no_stop_or_restart(self)


if __name__ == "__main__":
    unittest.main()
