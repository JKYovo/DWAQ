"""CPU checks of real native-play command helpers/loop; no AppLauncher import."""
import argparse
import ast
from contextlib import redirect_stderr
import copy
import hashlib
import io
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as S
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "TienKung-Lab/legged_lab/scripts/play.py"
ISAAC = ROOT / "IsaacLab/source/isaaclab/isaaclab"


def tree():
    return ast.parse(PLAY.read_text())


def execute_nodes(nodes, namespace, filename=PLAY):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


def configure():
    function = next(node for node in tree().body
                    if isinstance(node, ast.FunctionDef) and node.name == "configure_fixed_play_command")
    namespace = {"math": math}
    execute_nodes([function], namespace)
    return namespace[function.name]


def command_config():
    return S(asset_name="robot", heading_command=True, rel_heading_envs=1., rel_standing_envs=.2,
             heading_control_stiffness=.5, resampling_time_range=(10., 10.),
             debug_vis=False, ranges=S(lin_vel_x=(1., 1.), lin_vel_y=(0., 0.),
                                        ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)))


def command_simulator(cfg, count=128):
    """Execute the real CPU resampling/postprocessing methods, not substitutes."""
    namespace = {"torch": torch, "math_utils": S(wrap_to_pi=lambda x: (x+math.pi)%(2*math.pi)-math.pi)}
    velocity = ISAAC / "envs/mdp/commands/velocity_command.py"
    module = ast.parse(velocity.read_text())
    owner = next(node for node in module.body
                 if isinstance(node, ast.ClassDef) and node.name == "UniformVelocityCommand")
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_resample_command", "_update_command")]
    execute_nodes(methods, namespace, velocity)
    manager = ISAAC / "managers/command_manager.py"
    owner = next(node for node in ast.parse(manager.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "CommandTerm")
    execute_nodes([next(node for node in owner.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_resample")], namespace, manager)
    simulator = S(cfg=cfg, num_envs=count, device="cpu", vel_command_b=torch.zeros(count, 3),
                  heading_target=torch.rand(count)*6-3,
                  is_heading_env=torch.ones(count, dtype=torch.bool),
                  is_standing_env=torch.zeros(count, dtype=torch.bool),
                  time_left=torch.zeros(count), command_counter=torch.zeros(count, dtype=torch.long),
                  robot=S(data=S(heading_w=torch.rand(count)*6-3)))
    for name in ("_resample_command", "_update_command", "_resample"):
        setattr(simulator, name, MethodType(namespace[name], simulator))
    return simulator


def parse_new_options(arguments):
    """Execute actual option registrations and validations, without starting Kit."""
    module = tree()
    parser = argparse.ArgumentParser()
    registrations = []
    validations = []
    for node in module.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add_argument" and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value in ("--command", "--max_steps")):
            registrations.append(node)
        if (isinstance(node, ast.If) and any(isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute) and item.func.attr == "error"
                for item in ast.walk(node))):
            validations.append(node)
    execute_nodes(registrations, {"parser": parser})
    args, _ = parser.parse_known_args(arguments)
    execute_nodes(validations, {"parser": parser, "args_cli": args, "math": math})
    return args


class PlayFixedCommandTests(unittest.TestCase):
    def test_none_preserves_complete_legacy_configuration(self):
        cfg = command_config()
        before = copy.deepcopy(vars(cfg))
        self.assertIsNone(configure()(cfg, None))
        self.assertEqual(vars(cfg), before)
        self.assertEqual(cfg.ranges.lin_vel_x, (1., 1.))
        self.assertTrue(cfg.heading_command)

    def test_zero_positive_forward_and_pure_turn_configuration(self):
        for values in ([0., 0., 0.], [.3, 0., 0.], [0., 0., .1], [-.3, .2, -.5]):
            with self.subTest(values=values):
                cfg = command_config()
                original = values.copy()
                result = configure()(cfg, values)
                self.assertEqual(result, tuple(values))
                self.assertEqual(values, original)
                self.assertFalse(cfg.heading_command)
                self.assertEqual(cfg.rel_heading_envs, 0.)
                self.assertEqual(cfg.rel_standing_envs, 1. if values == [0., 0., 0.] else 0.)
                for name, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), values):
                    self.assertEqual(getattr(cfg.ranges, name), (value, value))

    def test_only_exact_zero_gets_standing_sampler_flag(self):
        cfg = command_config()
        configure()(cfg, [-0., 0., -0.])
        self.assertEqual(cfg.rel_standing_envs, 1.)
        configure()(cfg, [1e-9, 0., 0.])
        self.assertEqual(cfg.rel_standing_envs, 0.)

    def test_nonfinite_and_bad_lengths_rejected_before_configuration_mutation(self):
        for values in ([0., 0.], [0., 0., 0., 0.], [], [float("nan"), 0., 0.],
                       [0., float("inf"), 0.], [0., 0., -float("inf")]):
            cfg = command_config()
            before = copy.deepcopy(vars(cfg))
            with self.subTest(values=values), self.assertRaises(ValueError):
                configure()(cfg, values)
            self.assertEqual(vars(cfg), before)

    def test_actual_sampler_keeps_initial_frame_and_repeated_resets_fixed(self):
        for values in ([0., 0., 0.], [.3, 0., 0.], [0., 0., .1]):
            with self.subTest(values=values):
                torch.manual_seed(42)
                cfg = command_config()
                fixed = configure()(cfg, values)
                simulator = command_simulator(cfg)
                all_ids = torch.arange(simulator.num_envs)
                expected = torch.tensor(fixed).expand(simulator.num_envs, -1)
                # Reset calls resample without update; even that first frame
                # must already be exact because all three ranges are fixed.
                simulator._resample(all_ids)
                self.assertTrue(torch.equal(simulator.vel_command_b, expected))
                for iteration in range(12):
                    simulator.robot.data.heading_w.uniform_(-math.pi, math.pi)
                    simulator.heading_target.uniform_(-math.pi, math.pi)
                    simulator._update_command()
                    self.assertTrue(torch.equal(simulator.vel_command_b, expected))
                    ids = all_ids if iteration % 2 else all_ids[::3]
                    simulator._resample(ids)
                    self.assertTrue(torch.equal(simulator.vel_command_b, expected))
                    simulator._update_command()
                    self.assertTrue(torch.equal(simulator.vel_command_b, expected))

    def test_cli_options_defaults_and_finite_values(self):
        args = parse_new_options([])
        self.assertIsNone(args.command)
        self.assertIsNone(args.max_steps)
        args = parse_new_options(["--command", "0", "0", "0", "--max_steps", "3"])
        self.assertEqual(args.command, [0., 0., 0.])
        self.assertEqual(args.max_steps, 3)
        for options in (["--command", "nan", "0", "0"], ["--command", "0", "inf", "0"],
                        ["--command", "0", "0", "-inf"], ["--command", "0", "0"],
                        ["--max_steps", "0"], ["--max_steps", "-1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_new_options(options)

    def test_cli_validation_precedes_actual_app_launch(self):
        module = tree()
        launch = next(node for node in ast.walk(module) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "AppLauncher")
        validations = [node for node in module.body if isinstance(node, ast.If)
                       and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                               and item.func.attr == "error" for item in ast.walk(node))]
        self.assertGreaterEqual(len(validations), 2)
        self.assertTrue(all(node.lineno < launch.lineno for node in validations))

    def test_fixed_configuration_runs_after_legacy_ranges_before_env_creation(self):
        function = next(node for node in tree().body if isinstance(node, ast.FunctionDef) and node.name == "play")
        fixed = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "configure_fixed_play_command")
        construction = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "env" for target in node.targets))
        legacy = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Attribute) and target.attr in ("lin_vel_x", "lin_vel_y")
                          for target in node.targets)]
        self.assertTrue(legacy)
        self.assertTrue(all(node.lineno < fixed.lineno for node in legacy))
        self.assertLess(fixed.lineno, construction.lineno)

    def test_real_loop_respects_max_steps_and_entry_finally_closes_app(self):
        module = tree()
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "play")
        loop = next(node for node in function.body if isinstance(node, ast.While))
        for dwaq in (False, True):
            calls = []

            def step(action):
                calls.append(action)
                if len(calls) > 5:
                    raise AssertionError("Bounded play failed to stop")
                return torch.zeros(2, 100), torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

            namespace = {"simulation_app": S(is_running=lambda: True), "torch": torch,
                         "args_cli": S(max_steps=3), "completed_steps": 0,
                         "env": S(step=step, command_generator=S(command=torch.zeros(2, 3))),
                         "obs": torch.zeros(2, 100), "extras": {},
                         "use_dwaq_policy": dwaq, "use_depth_policy": False,
                         "policy": lambda *args: torch.zeros(2, 29),
                         "fixed_command_tensor": torch.zeros(3), "fixed_command": (0., 0., 0.),
                         "print": lambda *args, **kwargs: None}
            execute_nodes([loop], namespace)
            self.assertEqual(len(calls), 3)
            self.assertEqual(namespace["completed_steps"], 3)
        entry = next(node for node in module.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        guarded = next(node for node in entry.body if isinstance(node, ast.Try))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "simulation_app"
                            and node.func.attr == "close" for statement in guarded.finalbody for node in ast.walk(statement)))

    def test_training_does_not_import_play_helper_and_helper_never_writes_training(self):
        paths = [ROOT / "scripts/train_elf3_feedback_stand_v3.py",
                 ROOT / "TienKung-Lab/legged_lab/scripts/train.py"]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(any(alias.name == "configure_fixed_play_command" for alias in node.names))
                    self.assertFalse((node.module or "").endswith(".scripts.play"))
        configure()(command_config(), [0., 0., 0.])
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
