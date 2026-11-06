"""CPU integration guards for the isolated feedback experiment.

Only the low-level RSL runner and simulator configuration containers are mocked.
The actual ELF3 runner save/load/contract methods and launcher validation execute.
No Isaac module, GPU operation, environment, training, or robot is started.
"""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "TienKung-Lab/legged_lab"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_file("integration_feedback_contract", LAB / "envs/elf3/feedback_contract.py")
robot_contract = load_file("integration_robot_contract", LAB / "elf3_contract.py")
launcher = load_file("integration_feedback_launcher", ROOT / "scripts/train_elf3_feedback.py")


class ConfigNode(SimpleNamespace):
    def to_dict(self):
        def convert(value):
            if isinstance(value, ConfigNode):
                return {key: convert(item) for key, item in vars(value).items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(convert(item) for item in value)
            return value
        return convert(self)


def feedback_settings(mode="lpf20"):
    return {"mode": mode, "cutoff_hz": 20.0,
            "model_version": "joint_velocity_tustin_v1", "latest_critic_contacts": True}


class ScalarWriter:
    def __init__(self):
        self.values = []

    def add_scalar(self, key, value, step):
        self.values.append((key, float(value), step))


class TensorOnlyRslRunner:
    """Provide storage plumbing under the real old/new ELF3 runner classes."""
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg = env, copy.deepcopy(train_cfg)
        self.alg = SimpleNamespace(policy=SimpleNamespace(context_mode=train_cfg["policy"]["context_mode"]),
                                   configure_upper_body_symmetry=self.configure_symmetry)
        self.current_learning_iteration = 0
        self.lowlevel_load_count = 0
        self.logged = []
        self.writer = ScalarWriter()

    def configure_symmetry(self, **kwargs):
        self.symmetry_arguments = kwargs

    def save(self, path, infos=None):
        torch.save({"iter": self.current_learning_iteration, "infos": infos}, path)

    def _resolve_checkpoint_path(self, path):
        return str(path)

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.current_learning_iteration = saved["iter"]
        self.lowlevel_load_count += 1
        self.last_load_optimizer = load_optimizer
        return saved["infos"]

    def log(self, locs, width=80, pad=35):
        self.logged.append((locs, width, pad))


def load_real_runner():
    rsl_module = ModuleType("rsl_rl.runners")
    rsl_module.DWAQOnPolicyRunner = TensorOnlyRslRunner
    modules = {"rsl_rl.runners": rsl_module,
               "legged_lab.elf3_contract": robot_contract,
               "legged_lab.envs.elf3.feedback_contract": contract}
    with patch.dict(sys.modules, modules):
        symmetry = load_file("integration_actual_symmetry", LAB / "envs/elf3/symmetry.py")
        with patch.dict(sys.modules, {"legged_lab.envs.elf3.symmetry": symmetry}):
            old = load_file("integration_actual_old_runner", LAB / "envs/elf3/runner.py")
            with patch.dict(sys.modules, {"legged_lab.envs.elf3.runner": old}):
                new = load_file("integration_actual_new_runner", LAB / "envs/elf3/feedback_runner.py")
    return new.Elf3DwaqFeedbackRunner


Runner = load_real_runner()


def runner_environment(mode="lpf20"):
    return ConfigNode(
        cfg=ConfigNode(velocity_feedback=feedback_settings(mode), reward=ConfigNode(track=ConfigNode(weight=2.0)),
                       robot=ConfigNode(gait_phase=ConfigNode(period=.8, offset=.5)), device="cpu",
                       scene=ConfigNode(num_envs=2, robot=ConfigNode(spawn=ConfigNode(usd_path="/mock/elf3.usd")))),
        physics_dt=.005, step_dt=.02, num_obs=100, num_privileged_obs=311, num_obs_hist=5,
        robot=ConfigNode(data=ConfigNode(joint_vel=torch.tensor([[1., -1.]]))),
        velocity_feedback_model=ConfigNode(value=torch.tensor([[.5, -.5]])))


def runner_recipe():
    return {"policy": {"context_mode": "mean"}, "empirical_normalization": False,
            "seed": 42, "num_steps_per_env": 24,
            "upper_body_mirror_loss_coeff": .1, "upper_body_pose_loss_coeff": .02,
            "algorithm": {"learning_rate": 1e-4, "min_learning_rate": 1e-5, "max_learning_rate": 3e-4}}


def actual_differences():
    source = ast.parse((ROOT / "scripts/melody_launch_audit.py").read_text())
    method = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "differences")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual differences helper>", "exec"), namespace)
    return namespace["differences"]


def configuration_pair(mode="lpf20"):
    baseline_env = ConfigNode(
        scene=ConfigNode(num_envs=4096, seed=42,
                         robot=ConfigNode(actuators={"legs": {"armature": .01, "effort_limit_sim": 100}}),
                         terrain_generator=ConfigNode(curriculum=True)),
        device="cuda:0", sim=ConfigNode(dt=.005, decimation=4),
        commands=ConfigNode(resampling_time_range=(10., 10.)),
        domain_rand=ConfigNode(action_delay=ConfigNode(enable=True)),
        reward=ConfigNode(track=ConfigNode(weight=2.0, params={"std": .5})))
    baseline_agent = ConfigNode(
        policy=ConfigNode(context_mode="sampled"), algorithm=ConfigNode(
            learning_rate=1e-4, min_learning_rate=1e-5, max_learning_rate=3e-4, schedule="adaptive"),
        upper_body_mirror_loss_coeff=.1, upper_body_pose_loss_coeff=.02,
        runner_class_name="Elf3DwaqUpperBodySymmetryRunner", experiment_name="elf3_dwaq_upper_symmetry_pose",
        wandb_project="elf3_dwaq_upper_symmetry_pose", max_iterations=20000, save_interval=50,
        run_name="", resume=False, seed=42, device="cuda:0", logger="tensorboard")
    env, agent = copy.deepcopy(baseline_env), copy.deepcopy(baseline_agent)
    env.velocity_feedback = feedback_settings(mode)
    env.scene.num_envs = 64
    agent.policy.context_mode = "mean"
    agent.runner_class_name = "Elf3DwaqFeedbackRunner"
    agent.experiment_name = agent.wandb_project = "elf3_dwaq_feedback_v1"
    agent.max_iterations, agent.save_interval = 20, 500
    agent.run_name = "cpu_guard"
    return env, agent, baseline_env, baseline_agent


class ContractIntegrationTests(unittest.TestCase):
    def test_contract_layout_rates_feedback_and_deployment_boundary(self):
        value = contract.experiment_contract(feedback_settings(), .005, .02)
        self.assertEqual(value["version"], "elf3_feedback_v1")
        self.assertEqual(value["observation"], {"frame_size": 100, "history_length": 5, "flat_size": 500,
                                              "joint_velocity_slice": [38, 67],
                                              "history_order": "oldest_to_newest"})
        self.assertEqual(value["feedback_update_dt"], .005)
        self.assertEqual(value["actor_sample_dt"], .02)
        self.assertEqual(value["critic_force_history_index"], 0)
        self.assertEqual(value["deployment"], {"num_actions": 29, "extra_velocity_filter": False,
                                             "extra_action_filter": False})
        ideal = contract.experiment_contract(feedback_settings("ideal"), .005, .02)
        self.assertNotEqual(value, ideal)
        altered = copy.deepcopy(value)
        altered["velocity_feedback"]["mode"] = "ideal"
        self.assertEqual(ideal, altered)

    def test_contract_rejects_unknown_version_modes_rates_and_context(self):
        cases = [({"mode": "delay20"}, .005, .02, "mean"),
                 ({"cutoff_hz": float("nan")}, .005, .02, "mean"),
                 ({"cutoff_hz": 100}, .005, .02, "mean"),
                 ({"model_version": "unknown"}, .005, .02, "mean"),
                 ({"latest_critic_contacts": False}, .005, .02, "mean"),
                 ({}, .01, .02, "mean"), ({}, .005, .04, "mean"), ({}, .005, .02, "sampled")]
        for update, physics_dt, control_dt, context in cases:
            with self.subTest(update=update, physics_dt=physics_dt, control_dt=control_dt, context=context):
                settings = feedback_settings() | update
                with self.assertRaises(ValueError):
                    contract.experiment_contract(settings, physics_dt, control_dt, context)
        for settings in ({"mode": "ideal"}, feedback_settings() | {"typo": 0}):
            with self.assertRaises(ValueError):
                contract.experiment_contract(settings, .005, .02)

    def test_fingerprint_is_order_stable_and_detects_recipe_change(self):
        first = {"a": [1, 2], "b": {"x": 3}, "s": slice(1, 4)}
        second = {"s": slice(1, 4), "b": {"x": 3}, "a": (1, 2)}
        self.assertEqual(contract.canonical_sha256(first), contract.canonical_sha256(second))
        second["b"]["x"] = 4
        self.assertNotEqual(contract.canonical_sha256(first), contract.canonical_sha256(second))
        with self.assertRaises(ValueError):
            contract.canonical_sha256({"bad": float("nan")})


class RunnerIntegrationTests(unittest.TestCase):
    def test_actual_runner_save_preserves_robot_and_experiment_contract_and_recipe(self):
        runner = Runner(runner_environment(), runner_recipe())
        runner.current_learning_iteration = 499
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_499.pt"
            extra = {"note": "retained"}
            runner.save(path, extra)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(extra, {"note": "retained"})
        self.assertEqual(saved["iter"], 499)
        self.assertEqual(saved["infos"]["training_recipe"], runner.cfg)
        self.assertEqual(saved["infos"]["note"], "retained")
        self.assertEqual(saved["infos"]["elf3_contract"], runner.contract())
        self.assertEqual(runner.contract()["joint_names"], robot_contract.CONTRACT["joint_names"])
        self.assertEqual(runner.contract()["upper_body_symmetry"]["mirror_loss_coeff"], .1)
        self.assertEqual(runner.contract()["upper_body_symmetry"]["pose_loss_coeff"], .02)
        self.assertEqual(runner.contract()["feedback_experiment"]["unchanged_reward_sha256"],
                         contract.canonical_sha256(runner.env.cfg.reward.to_dict()))
        self.assertEqual(runner.symmetry_arguments["coefficient"], .1)
        self.assertEqual(runner.symmetry_arguments["pose_coefficient"], .02)

    def test_actual_runner_resume_starts_after_last_completed_index(self):
        writer = Runner(runner_environment(), runner_recipe())
        writer.current_learning_iteration = 999
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_999.pt"
            writer.save(path)
            restored = Runner(runner_environment(), runner_recipe())
            restored.load(path, load_optimizer=False)
        self.assertEqual(restored.current_learning_iteration, 1000)
        self.assertEqual(restored.lowlevel_load_count, 1)
        self.assertFalse(restored.last_load_optimizer)

    def test_old_amp_or_other_mode_checkpoints_fail_before_lowlevel_load(self):
        runner = Runner(runner_environment(), runner_recipe())
        wrong_mode = Runner(runner_environment("ideal"), runner_recipe())
        old_contract = copy.deepcopy(runner.contract())
        old_contract.pop("feedback_experiment")
        cases = {
            "missing_infos": {"iter": 0},
            "old_dwaq": {"iter": 0, "infos": {"elf3_contract": old_contract}},
            "amp": {"iter": 0, "infos": {"elf3_contract": {"robot": "elf3", "algorithm": "AMP"}}},
            "other_mode": {"iter": 0, "infos": {"elf3_contract": wrong_mode.contract()}},
            "initial_snapshot": {"iter": 0, "infos": {"elf3_contract": runner.contract(),
                                                       "initialization_only": True}},
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, saved in cases.items():
                with self.subTest(name=name):
                    path = Path(folder) / f"{name}.pt"
                    torch.save(saved, path)
                    with self.assertRaises(ValueError):
                        runner.load(path)
                    self.assertEqual(runner.lowlevel_load_count, 0)
                    self.assertEqual(runner.current_learning_iteration, 0)

    def test_reward_contract_change_rejects_resume(self):
        writer = Runner(runner_environment(), runner_recipe())
        reader = Runner(runner_environment(), runner_recipe())
        reader.env.cfg.reward.track.weight = 3.0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "different_reward.pt"
            writer.save(path)
            with self.assertRaises(ValueError):
                reader.load(path)
        self.assertEqual(reader.lowlevel_load_count, 0)

    def test_recipe_hash_rejects_algorithm_seed_and_rollout_changes(self):
        writer = Runner(runner_environment(), runner_recipe())
        for key, value in (("algorithm", runner_recipe()["algorithm"] | {"learning_rate": .001}),
                           ("seed", 43), ("num_steps_per_env", 16)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                reader = Runner(runner_environment(), runner_recipe() | {key: value})
                path = Path(folder) / "incompatible_recipe.pt"
                writer.save(path)
                with self.assertRaises(ValueError):
                    reader.load(path)
                self.assertEqual(reader.lowlevel_load_count, 0)

    def test_recipe_hash_normalizes_device_and_asset_location_without_mutating_config(self):
        first = Runner(runner_environment(), runner_recipe())
        second = Runner(runner_environment(), runner_recipe())
        second.env.cfg.device = "cuda:1"
        second.env.cfg.scene.robot.spawn.usd_path = "/relocated/same/elf3.usd"
        before = copy.deepcopy(second.env.cfg.to_dict())
        self.assertEqual(first.contract(), second.contract())
        self.assertEqual(before, second.env.cfg.to_dict())

    def test_runner_rejects_sampled_context_and_empirical_normalization(self):
        recipe = runner_recipe()
        recipe["policy"]["context_mode"] = "sampled"
        with self.assertRaises(ValueError):
            Runner(runner_environment(), recipe)
        recipe = runner_recipe() | {"empirical_normalization": True}
        with self.assertRaises(ValueError):
            Runner(runner_environment(), recipe)

    def test_logging_keeps_episode_aggregation_cpu_and_rejects_nonfinite(self):
        runner = Runner(runner_environment(), runner_recipe())
        locs = {"ep_infos": [{"reward": torch.tensor([1., 3.])}, {"reward": 5.}],
                "it": 12, "mean_value_loss": .1}
        runner.log(locs)
        self.assertEqual(runner.logged[0][0]["ep_infos"], [])
        self.assertEqual(len(locs["ep_infos"]), 2)  # input left intact
        values = {key: value for key, value, step in runner.writer.values}
        self.assertEqual(values["Episode/reward"], 3.)
        self.assertEqual(values["Feedback/raw_velocity_rms"], 1.)
        self.assertEqual(values["Feedback/reporting_error_rms"], .5)
        for bad in ({"ep_infos": [{"bad": torch.tensor([math.inf])}], "it": 1},
                    {"ep_infos": [], "it": 1, "mean_autoenc_loss": math.nan}):
            with self.assertRaises(FloatingPointError):
                runner.log(bad)


class LauncherAndConfigIntegrationTests(unittest.TestCase):
    def setUp(self):
        audit = ModuleType("melody_launch_audit")
        audit.differences = actual_differences()
        self.modules = patch.dict(sys.modules, {
            "melody_launch_audit": audit, "legged_lab.envs.elf3.feedback_contract": contract})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_whitelisted_effective_configs_are_accepted_for_both_modes(self):
        for mode in ("ideal", "lpf20"):
            with self.subTest(mode=mode):
                result = launcher.validate_effective_config(*configuration_pair(mode))
                self.assertTrue(result["reward_asset_PD_commands_terrain_and_algorithm_unchanged"])
                self.assertEqual(result["experiment"]["velocity_feedback"]["mode"], mode)

    def test_unapproved_asset_reward_command_randomization_and_algorithm_changes_fail(self):
        changes = {
            "armature": lambda env, agent: env.scene.robot.actuators["legs"].update(armature=.02),
            "effort": lambda env, agent: env.scene.robot.actuators["legs"].update(effort_limit_sim=150),
            "command_period": lambda env, agent: setattr(env.commands, "resampling_time_range", (1., 1.)),
            "reward": lambda env, agent: setattr(env.reward.track, "weight", 3.),
            "terrain": lambda env, agent: setattr(env.scene.terrain_generator, "curriculum", False),
            "delay": lambda env, agent: setattr(env.domain_rand.action_delay, "enable", False),
            "physics_dt": lambda env, agent: setattr(env.sim, "dt", .01),
            "learning_rate": lambda env, agent: setattr(agent.algorithm, "learning_rate", .001),
            "min_lr": lambda env, agent: setattr(agent.algorithm, "min_learning_rate", 1e-6),
            "schedule": lambda env, agent: setattr(agent.algorithm, "schedule", "fixed"),
            "context": lambda env, agent: setattr(agent.policy, "context_mode", "sampled"),
            "runner": lambda env, agent: setattr(agent, "runner_class_name", "Elf3DwaqUpperBodySymmetryRunner"),
            "mirror": lambda env, agent: setattr(agent, "upper_body_mirror_loss_coeff", .2),
            "pose": lambda env, agent: setattr(agent, "upper_body_pose_loss_coeff", .04),
            "auto_resume": lambda env, agent: setattr(agent, "resume", True),
            "save_interval": lambda env, agent: setattr(agent, "save_interval", 100),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                env, agent, baseline_env, baseline_agent = configuration_pair()
                change(env, agent)
                with self.assertRaises(ValueError):
                    launcher.validate_effective_config(env, agent, baseline_env, baseline_agent)

    def test_parser_defaults_are_short_and_tasks_opt_in(self):
        parser = launcher.build_parser()
        args = parser.parse_args([])
        self.assertEqual((args.num_envs, args.iterations, args.seed), (64, 20, 42))
        self.assertIsNone(args.resume)
        # Historical identities remain readable, but only feedback is public.
        self.assertEqual(tuple(launcher.TASKS), ("elf3_dwaq_feedback_v1",))
        self.assertTrue(set(launcher.TASKS).issubset(contract.TASKS))
        self.assertTrue(parser.parse_args(["--verify-config-only"]).verify_config_only)

    def test_config_ast_only_extends_existing_upper_body_task_and_changes_context(self):
        tree = ast.parse((LAB / "envs/elf3/elf3_feedback_config.py").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertEqual(ast.unparse(classes["Elf3DwaqFeedbackEnvCfg"].bases[0]), "Elf3DwaqEnvCfg")
        self.assertEqual(ast.unparse(classes["Elf3DwaqFeedbackAgentCfg"].bases[0]),
                         "Elf3DwaqUpperBodySymmetryPoseAgentCfg")
        def fields(cls):
            return {node.target.id: ast.literal_eval(node.value) for node in cls.body
                    if isinstance(node, ast.AnnAssign)}
        filtered = fields(classes["Elf3DwaqFeedbackEnvCfg"])
        ideal = fields(classes["Elf3DwaqConsistentEnvCfg"])
        self.assertEqual(filtered, {"velocity_feedback": feedback_settings("lpf20")})
        self.assertEqual(ideal, {"velocity_feedback": feedback_settings("ideal")})
        agent = classes["Elf3DwaqFeedbackAgentCfg"]
        values = fields(agent)
        self.assertEqual(values["runner_class_name"], "Elf3DwaqFeedbackRunner")
        self.assertEqual((values["max_iterations"], values["save_interval"]), (100001, 500))
        methods = [node for node in agent.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual([node.name for node in methods], ["__post_init__"])
        self.assertEqual([ast.unparse(node) for node in methods[0].body],
                         ["super().__post_init__()", "self.policy.context_mode = 'mean'"])

    def test_new_tasks_register_their_exact_environment_agent_and_runner(self):
        tree = ast.parse((LAB / "envs/__init__.py").read_text())
        registered = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
                if node.args and isinstance(node.args[0], ast.Constant):
                    registered[node.args[0].value] = [ast.unparse(value) for value in node.args[1:]]
        self.assertEqual(registered["elf3_dwaq_feedback_v1"],
                         ["Elf3DwaqFeedbackEnv", "Elf3DwaqFeedbackEnvCfg()", "Elf3DwaqFeedbackAgentCfg()"])
        self.assertEqual(set(registered), {"elf3_dwaq_feedback_v1",
                                          "elf3_dwaq_still_v2", "elf3_dwaq_still_v2_ankle_cost_v1",
                                          "elf3_dwaq_still_v2_force500_v1",
                                          "elf3_dwaq_still_v2_curriculum_v1"})


if __name__ == "__main__":
    unittest.main()
