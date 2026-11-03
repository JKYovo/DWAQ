"""CPU-only guards for the scratch-only v3 launcher; no Isaac/training imports."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/train_elf3_feedback_stand_v3.py"
SPEC = importlib.util.spec_from_file_location("feedback_stand_v3_launcher_cpu", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class Config(SimpleNamespace):
    def to_dict(self):
        def convert(value):
            if isinstance(value, Config):
                return {key: convert(item) for key, item in vars(value).items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return copy.deepcopy(value)
        return convert(self)

    def copy(self):
        return copy.deepcopy(self)


def fixture():
    rewards = Config(gait_phase_contact=Config(func="original_gait", weight=.2, params={}),
                     feet_swing_height=Config(func="original_swing", weight=-.2, params={}),
                     tracking_lin_vel=Config(func="tracking", weight=2., params={}))
    for suffix in ("hip", "ankle", "arms", "legs"):
        setattr(rewards, f"joint_deviation_{suffix}", Config(func="original_pose", weight=-.2, params={}))
    parent_cfg = Config(scene=Config(num_envs=4096, seed=42, max_init_terrain_level=5,
                                    robot=Config(armature=.01, velocity_limit=20.)),
                        device="cuda:0", reward=rewards, noise=Config(enabled=True),
                        commands=Config(resampling_time_range=(10., 10.), rel_standing_envs=.2),
                        velocity_feedback={"mode": "lpf20", "cutoff_hz": 20.})
    parent_agent = Config(experiment_name=launcher.PARENT_TASK, wandb_project=launcher.PARENT_TASK,
                          runner_class_name="Elf3DwaqFeedbackRunner", seed=42, device="cuda:0",
                          max_iterations=100001, save_interval=500, run_name="", logger="tensorboard",
                          resume=False, upper_body_mirror_loss_coeff=.1, upper_body_pose_loss_coeff=.02,
                          policy=Config(context_mode="mean"),
                          algorithm=Config(learning_rate=1e-4, min_learning_rate=1e-5,
                                           max_learning_rate=3e-4, schedule="adaptive"))
    cfg, agent = parent_cfg.copy(), parent_agent.copy()
    cfg.strict_standing_v3 = {"command_epsilon": 1e-6}
    for name in ("gait_phase_contact", "feet_swing_height"):
        getattr(cfg.reward, name).func = "zero_gated"
        getattr(cfg.reward, name).params["command_epsilon"] = 1e-6
    for name, weight in (("zero_default_pose_l1", -.1), ("zero_joint_motion", -.5)):
        setattr(cfg.reward, name, Config(func=name, weight=weight, params={}))
    agent.experiment_name = agent.wandb_project = launcher.TASK
    agent.runner_class_name = launcher.RUNNER_CLASS
    return parent_cfg, parent_agent, cfg, agent


class FeedbackStandV3LauncherTests(unittest.TestCase):
    def test_safe_defaults_and_count_semantics(self):
        args = launcher.build_parser().parse_args([])
        self.assertEqual((args.num_envs, args.iterations, args.seed), (64, 20, 42))
        self.assertIsNone(args.swanlab_project)
        self.assertFalse(args.verify_config_only)
        self.assertFalse(args.use_cached_visual)
        args = launcher.build_parser().parse_args(["--iterations", "100001"])
        self.assertEqual(args.iterations - 1, 100000)

    def test_no_checkpoint_or_resume_or_other_task_cli(self):
        for argument in ("--resume", "--checkpoint", "--task", "--load-run"):
            with self.subTest(argument=argument), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.build_parser().parse_args([argument, "old_training"])

    def test_explicit_parent_recipe_and_runtime_changes_pass(self):
        parent_cfg, parent_agent, registered_cfg, registered_agent = fixture()
        cfg, agent = registered_cfg.copy(), registered_agent.copy()
        cfg.scene.num_envs = 64
        agent.max_iterations, agent.run_name = 20, "cpu_guard_test"
        runtime = launcher.validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent)
        parent = launcher.validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)
        self.assertTrue(runtime["from_scratch"])
        self.assertFalse(runtime["checkpoint_load_supported"])
        self.assertEqual(parent["parent_task"], launcher.PARENT_TASK)
        paths = {item["path"] for item in parent["environment_changes"]}
        self.assertTrue(launcher.PARENT_REWARD_PATHS.issubset(paths))
        self.assertNotIn("scene.max_init_terrain_level", paths)

    def test_parent_guard_rejects_unapproved_changes(self):
        cases = {
            "tracking_weight": lambda e, a: setattr(e.reward.tracking_lin_vel, "weight", 3.),
            "gait_weight": lambda e, a: setattr(e.reward.gait_phase_contact, "weight", .4),
            "pose_old_weight": lambda e, a: setattr(e.reward.joint_deviation_legs, "weight", -.1),
            "pose_zero_gate": lambda e, a: setattr(e.reward.joint_deviation_arms, "func", "moving_only"),
            "noise": lambda e, a: setattr(e.noise, "enabled", False),
            "armature": lambda e, a: setattr(e.scene.robot, "armature", 0.),
            "velocity_limit": lambda e, a: setattr(e.scene.robot, "velocity_limit", 100.),
            "terrain_initial_level": lambda e, a: setattr(e.scene, "max_init_terrain_level", 1),
            "command_interval": lambda e, a: setattr(e.commands, "resampling_time_range", (1., 1.)),
            "standing_sampling": lambda e, a: setattr(e.commands, "rel_standing_envs", .3),
            "algorithm_lr": lambda e, a: setattr(a.algorithm, "learning_rate", .001),
            "policy_context": lambda e, a: setattr(a.policy, "context_mode", "sample"),
            "feedback": lambda e, a: e.velocity_feedback.update(mode="ideal"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                parent_cfg, parent_agent, cfg, agent = fixture()
                mutate(cfg, agent)
                with self.assertRaises(ValueError):
                    launcher.validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)

    def test_runtime_guard_rejects_even_registered_reward_recipe_changes(self):
        for kind in ("pose", "motion", "gait_function", "swing_epsilon"):
            _, _, registered_cfg, registered_agent = fixture()
            cfg, agent = registered_cfg.copy(), registered_agent.copy()
            if kind == "pose": cfg.reward.zero_default_pose_l1.weight = -2.
            elif kind == "motion": cfg.reward.zero_joint_motion.weight = -1.
            elif kind == "gait_function": cfg.reward.gait_phase_contact.func = "original_gait"
            elif kind == "swing_epsilon": cfg.reward.feet_swing_height.params["command_epsilon"] = .1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                launcher.validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent)

    def test_parent_guard_rejects_legacy_standing_fields_and_stacked_rewards(self):
        for field in ("strict_standing_v2", "zero_full_body_pose", "zero_torso_upright",
                      "zero_base_stillness", "zero_stand_score"):
            with self.subTest(field=field):
                parent_cfg, parent_agent, cfg, agent = fixture()
                if field.startswith("strict_"):
                    setattr(cfg, field, {"version": "legacy"})
                else:
                    setattr(cfg.reward, field, Config(func="legacy", weight=1., params={}))
                with self.assertRaises(ValueError):
                    launcher.validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)

    def test_source_fingerprints_explicitly_cover_the_complete_new_experiment(self):
        expected = {
            "scripts/train_elf3_feedback_stand_v3.py",
            "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_contract.py",
            "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_rewards.py",
            "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py",
            "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_env.py",
            "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_runner.py",
            "TienKung-Lab/legged_lab/envs/elf3/elf3_feedback_stand_v3_config.py",
            "TienKung-Lab/legged_lab/envs/elf3/standing_rewards.py",
        }
        self.assertTrue(expected.issubset(launcher.EXPERIMENT_SOURCE_PATHS))
        self.assertEqual(len(launcher.EXPERIMENT_SOURCE_PATHS), len(set(launcher.EXPERIMENT_SOURCE_PATHS)))
        self.assertEqual(launcher.TASK, "elf3_dwaq_feedback_stand_v3")
        self.assertEqual(launcher.PARENT_TASK, "elf3_dwaq_feedback_v1")
        self.assertEqual(launcher.RUNNER_CLASS, "Elf3DwaqFeedbackStandV3Runner")
        self.assertEqual(launcher.PARENT_REWARD_PATHS, {
            "reward.zero_default_pose_l1", "reward.zero_joint_motion",
            "reward.gait_phase_contact.func", "reward.gait_phase_contact.params.command_epsilon",
            "reward.feet_swing_height.func", "reward.feet_swing_height.params.command_epsilon",
        })

    def test_runtime_guard_rejects_resume_and_wrong_runner_and_interval(self):
        for attribute, value in (("resume", True), ("save_interval", 100),
                                 ("runner_class_name", "Elf3DwaqStillPoseRunner"),
                                 ("upper_body_pose_loss_coeff", .1), ("logger", "swanlab")):
            with self.subTest(attribute=attribute):
                _, _, cfg, registered_agent = fixture()
                agent = registered_agent.copy()
                setattr(agent, attribute, value)
                with self.assertRaises(ValueError):
                    launcher.validate_runtime_overrides(cfg, agent, cfg.copy(), registered_agent)

    def test_fresh_runner_requires_zero_iteration_and_empty_adam(self):
        fresh = SimpleNamespace(current_learning_iteration=0,
                                alg=SimpleNamespace(optimizer=SimpleNamespace(state={})))
        launcher.assert_fresh_runner(fresh)
        fresh.current_learning_iteration = 100000
        with self.assertRaises(ValueError):
            launcher.assert_fresh_runner(fresh)
        fresh.current_learning_iteration = 0
        fresh.alg.optimizer.state[0] = {"step": 1}
        with self.assertRaises(ValueError):
            launcher.assert_fresh_runner(fresh)

    def test_report_encodes_real_parent_delta_slices_functions_and_nested_config(self):
        parent_cfg, parent_agent, cfg, agent = fixture()
        # Runtime entity resolution adds slices, while reward callbacks remain
        # functions in the real config. Both appear in parent audit deltas.
        cfg.reward.zero_default_pose_l1.params["asset_cfg"] = Config(
            name="robot", joint_ids=slice(None), joint_names=["joint"])
        cfg.reward.zero_default_pose_l1.func = launcher.validate_parent_recipe
        audit = launcher.validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)
        report = {"initialization": {"configuration_audit": audit}, "nested_config":
                  Config(selection=slice(1, 29, 2), callback=launcher.report_default)}
        decoded = json.loads(launcher.report_json(report, indent=2))
        delta = next(item for item in decoded["initialization"]["configuration_audit"]["environment_changes"]
                     if item["path"] == "reward.zero_default_pose_l1")
        self.assertEqual(delta["after"]["params"]["asset_cfg"]["joint_ids"], {"slice": [None, None, None]})
        self.assertEqual(delta["after"]["func"], f"{launcher.__name__}.validate_parent_recipe")
        self.assertEqual(decoded["nested_config"]["selection"], {"slice": [1, 29, 2]})
        self.assertEqual(decoded["nested_config"]["callback"], f"{launcher.__name__}.report_default")
        self.assertIsInstance(cfg.reward.zero_default_pose_l1.params["asset_cfg"].joint_ids, slice)
        self.assertIs(cfg.reward.zero_default_pose_l1.func, launcher.validate_parent_recipe)
        with self.assertRaises(TypeError):
            launcher.report_json({"unknown": object()})

    def test_launcher_contains_no_training_load_or_restart_path(self):
        import ast
        tree = ast.parse(SOURCE.read_text())
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {"load", "initialize_from_still", "kill", "terminate", "send_signal"}
        self.assertFalse([node.attr for node in calls if isinstance(node, ast.Attribute) and node.attr in forbidden])
        learns = [node for node in calls if isinstance(node, ast.Attribute) and node.attr == "learn"]
        self.assertEqual(len(learns), 1)

    def test_visual_alias_is_only_entered_after_app_launch_and_closed_before_app(self):
        import ast
        tree = ast.parse(SOURCE.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        main_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Name) and node.func.id == "cached_visual_alias"
                             for node in main_calls))
        run_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        app_launch = next(node for node in run_calls
                          if isinstance(node.func, ast.Name) and node.func.id == "AppLauncher")
        alias = next(node for node in run_calls
                     if isinstance(node.func, ast.Name) and node.func.id == "cached_visual_alias")
        closes = {node.func.value.id: node.lineno for node in run_calls
                  if isinstance(node.func, ast.Attribute) and node.func.attr == "close"
                  and isinstance(node.func.value, ast.Name)}
        self.assertLess(app_launch.lineno, alias.lineno)
        self.assertLess(closes["visual_context"], closes["app"])


if __name__ == "__main__":
    unittest.main()
