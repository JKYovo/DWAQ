"""CPU integration boundaries for the isolated quiet-standing v2 task.

The actual config, contract, runner and exporter execute. Only simulator
configuration containers and the low-level checkpoint plumbing are stubbed.
No Isaac application, environment, training, GPU operation or robot is started.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "TienKung-Lab/legged_lab"
ELF3 = LAB / "envs/elf3"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standing = load_file("quiet_integration_standing_helpers", ROOT / "tests/test_standing_task.py")
helpers = standing.helpers
play = standing.play


def load_quiet_contract():
    return load_file("quiet_integration_contract", ELF3 / "quiet_contract.py")


def load_quiet_rewards(contract):
    with patch.dict(sys.modules, standing.import_stubs() | {
        "legged_lab.envs.elf3.quiet_contract": contract,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
    }):
        return load_file("quiet_integration_rewards", ELF3 / "quiet_rewards.py")


class QuietContractIntegrationTests(unittest.TestCase):
    def test_settings_are_independent_and_identity_preserves_every_setting(self):
        module = load_quiet_contract()
        first, second = module.quiet_settings(), module.quiet_settings()
        self.assertIsNot(first, second)
        original = copy.deepcopy(first)
        identity = module.quiet_experiment_contract(first)
        self.assertEqual(module.VERSION, "elf3_stand_v2_command_only_v2")
        self.assertNotIn("gate", first)
        self.assertEqual(first["activation"],
                         {"mode": "zero_command_only", "history_requirement": "none"})
        self.assertEqual(first, original)
        self.assertNotEqual(identity, standing.standing_contract.standing_experiment_contract(standing.settings()))
        self.assertIn("v2", str(identity))
        sources = identity["sources_sha256"]
        self.assertIn("quiet_learning.py", sources)
        self.assertIn("quiet_env.py", sources)
        self.assertEqual(sources["quiet_runner.py"],
                         hashlib.sha256((ELF3 / "quiet_runner.py").read_bytes()).hexdigest())
        # The source hash supplements settings; it cannot replace them.
        def contains(tree, key, value):
            if isinstance(tree, dict):
                if key in tree and tree[key] == value:
                    return True
                return any(contains(item, key, value) for item in tree.values())
            return False
        for key, value in first.items():
            self.assertTrue(contains(identity, key, value), key)
        first["unexpected"] = True
        self.assertEqual(second, original)
        with self.assertRaises((ValueError, TypeError)):
            module.quiet_experiment_contract(first)

    def test_unknown_old_incomplete_and_nonfinite_settings_fail_closed(self):
        module = load_quiet_contract()
        settings = module.quiet_settings()
        invalid = [None, {}, standing.settings(), settings | {"unknown": True}]
        for key in settings:
            altered = copy.deepcopy(settings)
            altered.pop(key)
            invalid.append(altered)
        for key, value in settings.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                invalid.append(settings | {key: float("nan")})
                invalid.append(settings | {key: float("inf")})
        # Any old "already settled" recipe must fail closed, rather than
        # silently resume under the new command-only semantics.
        old_gated = copy.deepcopy(settings)
        old_gated["version"] = "elf3_stand_v2"
        old_gated.pop("activation")
        old_gated["gate"] = {"stable_time_s": .5, "max_tilt_deg": 20.,
                             "max_lin_speed": .1, "max_ang_speed": .5,
                             "min_foot_force_z": 5.}
        invalid.append(old_gated)
        for group in ("activation", "auxiliary", "physical"):
            for key, value in settings[group].items():
                altered = copy.deepcopy(settings)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    altered[group][key] = value + .001
                elif isinstance(value, list):
                    altered[group][key] = value + [.123]
                elif isinstance(value, str):
                    altered[group][key] = value + "_unexpected"
                else:
                    continue
                invalid.append(altered)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises((ValueError, TypeError)):
                module.quiet_experiment_contract(item)


def quiet_configuration_classes():
    contract = load_quiet_contract()
    rewards = load_quiet_rewards(contract)
    _, FeedbackEnv, FeedbackAgent = standing.configuration_classes()
    utils = ModuleType("isaaclab.utils")
    utils.configclass = standing.stub_configclass
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = standing.EntityCfg
    managers.RewardTermCfg = helpers.ConfigNode
    feedback = ModuleType("legged_lab.envs.elf3.elf3_feedback_config")
    feedback.Elf3DwaqFeedbackEnvCfg, feedback.Elf3DwaqFeedbackAgentCfg = FeedbackEnv, FeedbackAgent
    with patch.dict(sys.modules, standing.import_stubs() | {
        "isaaclab.utils": utils,
        "isaaclab.managers": managers,
        "legged_lab.envs.elf3.elf3_feedback_config": feedback,
        "legged_lab.elf3_contract": helpers.robot_contract,
        "legged_lab.envs.elf3.quiet_contract": contract,
        "legged_lab.envs.elf3.quiet_rewards": rewards,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
    }):
        config = load_file("quiet_integration_config", ELF3 / "elf3_stand_v2_config.py")
    return config, FeedbackEnv, FeedbackAgent, contract, rewards


class QuietConfigurationIntegrationTests(unittest.TestCase):
    def test_reward_changes_are_private_and_old_configuration_is_not_mutated(self):
        module, Env, _, contract, rewards = quiet_configuration_classes()
        baseline = Env()
        before = copy.deepcopy(baseline.to_dict())
        config = module.Elf3DwaqStandV2EnvCfg()
        changed = config.to_dict()
        self.assertEqual(changed.pop("quiet_standing"), contract.quiet_settings())
        extra = changed["reward"].pop("upper_body_stillness")
        self.assertIs(extra["func"], rewards.upper_body_stillness)
        self.assertEqual(extra["weight"], .3)
        for name in ("gait_phase_contact", "feet_swing_height"):
            term = changed["reward"][name]
            self.assertIs(term["func"], getattr(standing.standing_rewards, name + "_when_moving"))
            self.assertEqual(term["params"].pop("command_epsilon"), 1e-6)
            term["func"] = before["reward"][name]["func"]
        self.assertEqual(changed, before)
        self.assertEqual(baseline.to_dict(), before)
        self.assertEqual(Env().to_dict(), before)

    def test_agent_only_changes_private_identity_and_algorithm_class(self):
        module, _, Agent, _, _ = quiet_configuration_classes()
        baseline = Agent().to_dict()
        changed = module.Elf3DwaqStandV2AgentCfg().to_dict()
        for key, value in (("experiment_name", "elf3_dwaq_stand_v2"),
                           ("wandb_project", "elf3_dwaq_stand_v2"),
                           ("runner_class_name", "Elf3DwaqStandV2Runner")):
            self.assertEqual(changed[key], value)
            changed[key] = baseline[key]
        self.assertEqual(changed["algorithm"].pop("class_name"), "Elf3DwaqStandPPO")
        baseline["algorithm"].pop("class_name", None)
        self.assertEqual(changed, baseline)

    def test_v2_and_v1_have_distinct_task_environment_and_runner_registration(self):
        tree = ast.parse((LAB / "envs/__init__.py").read_text())
        registrations = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "register" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                registrations[node.args[0].value] = [ast.unparse(arg) for arg in node.args[1:]]
        self.assertEqual(registrations["elf3_dwaq_stand_v2"],
                         ["Elf3DwaqStandV2Env", "Elf3DwaqStandV2EnvCfg()", "Elf3DwaqStandV2AgentCfg()"])
        self.assertEqual(registrations["elf3_dwaq_stand_v1"],
                         ["Elf3DwaqFeedbackEnv", "Elf3DwaqStandEnvCfg()", "Elf3DwaqStandAgentCfg()"])
        self.assertEqual(registrations["elf3_dwaq_feedback_v1"],
                         ["Elf3DwaqFeedbackEnv", "Elf3DwaqFeedbackEnvCfg()", "Elf3DwaqFeedbackAgentCfg()"])


def load_runner_fixture():
    contract = load_quiet_contract()
    rewards = load_quiet_rewards(contract)

    class FeedbackFixture(helpers.Runner):
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            super().__init__(env, train_cfg, log_dir=log_dir, device=device)
            self.alg.configure_standing = self.configure_standing
            self.alg.last_standing_metrics = {}

        def configure_standing(self, env, settings):
            self.standing_arguments = (env, copy.deepcopy(settings))

    class Elf3DwaqStandPPO:
        pass

    feedback = ModuleType("legged_lab.envs.elf3.feedback_runner")
    feedback.Elf3DwaqFeedbackRunner = FeedbackFixture
    ppo = ModuleType("legged_lab.envs.elf3.quiet_learning")
    ppo.Elf3DwaqStandPPO = Elf3DwaqStandPPO
    with patch.dict(sys.modules, standing.import_stubs() | {
        "legged_lab.envs.elf3.feedback_runner": feedback,
        "legged_lab.envs.elf3.quiet_contract": contract,
        "legged_lab.envs.elf3.quiet_rewards": rewards,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.envs.elf3.quiet_learning": ppo,
    }):
        actual = load_file("quiet_integration_runner", ELF3 / "quiet_runner.py")
    return actual.Elf3DwaqStandV2Runner, contract, rewards


def quiet_runner_environment(contract, rewards):
    env = standing.runner_environment()
    env.cfg.quiet_standing = contract.quiet_settings()
    env.cfg.reward.upper_body_stillness = helpers.ConfigNode(
        func=rewards.upper_body_stillness, weight=.3,
        params={"velocity_std": env.cfg.quiet_standing["physical"]["velocity_std"]})
    return env


class QuietRunnerIntegrationTests(unittest.TestCase):
    def test_runner_rejects_wrong_functions_coefficients_and_physical_reward_identity(self):
        Runner, contract, rewards = load_runner_fixture()
        alterations = {
            "contact_function": lambda env: setattr(env.cfg.reward.gait_phase_contact, "func", standing.base_rewards.gait_phase_contact),
            "swing_epsilon": lambda env: env.cfg.reward.feet_swing_height.params.update(command_epsilon=.01),
            "arm_function": lambda env: setattr(env.cfg.reward.upper_body_stillness, "func", None),
            "arm_weight": lambda env: setattr(env.cfg.reward.upper_body_stillness, "weight", .4),
            "arm_velocity_std": lambda env: env.cfg.reward.upper_body_stillness.params.update(velocity_std=.1),
            "phase_coefficient": lambda env: env.cfg.quiet_standing["auxiliary"].update(phase_invariance_coeff=2.),
        }
        for name, alter in alterations.items():
            env = quiet_runner_environment(contract, rewards)
            alter(env)
            with self.subTest(change=name), self.assertRaises(ValueError):
                Runner(env, helpers.runner_recipe())

    def test_save_resume_configure_standing_and_deployment_contract(self):
        Runner, contract, rewards = load_runner_fixture()
        env = quiet_runner_environment(contract, rewards)
        recipe = helpers.runner_recipe()
        recipe["algorithm"]["class_name"] = "Elf3DwaqStandPPO"
        runner = Runner(env, recipe)
        self.assertIs(runner.standing_arguments[0], env)
        self.assertEqual(runner.standing_arguments[1], contract.quiet_settings())
        identity = runner.contract()
        self.assertEqual(identity["standing_experiment"],
                         contract.quiet_experiment_contract(contract.quiet_settings()))
        self.assertEqual((identity["num_obs"], identity["history_length"], len(identity["joint_names"])),
                         (100, 5, 29))
        self.assertEqual(identity["feedback_experiment"]["deployment"],
                         {"num_actions": 29, "extra_velocity_filter": False, "extra_action_filter": False})
        runner.current_learning_iteration = 499
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_499.pt"
            infos = {"retained": True}
            runner.save(path, infos)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            reader = Runner(quiet_runner_environment(contract, rewards), recipe)
            reader.load(path, load_optimizer=False)
        self.assertEqual(infos, {"retained": True})
        self.assertEqual(saved["infos"]["elf3_contract"], identity)
        self.assertEqual(saved["infos"]["training_recipe"], recipe)
        self.assertEqual(reader.current_learning_iteration, 500)
        self.assertEqual(reader.lowlevel_load_count, 1)
        self.assertFalse(reader.last_load_optimizer)

    def test_v1_feedback_and_v2_checkpoints_reject_each_other_before_model_load(self):
        Runner, contract, rewards = load_runner_fixture()
        env = quiet_runner_environment(contract, rewards)
        recipe = helpers.runner_recipe()
        recipe["algorithm"]["class_name"] = "Elf3DwaqStandPPO"
        quiet = Runner(env, recipe)
        # Deliberately share config to isolate top-level experiment identity.
        legacy = [standing.Runner(env, recipe), helpers.Runner(env, recipe)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quiet.pt"
            quiet.save(path)
            for index, old in enumerate(legacy):
                other = Path(folder) / f"old_{index}.pt"
                old.save(other)
                with self.subTest(version=index):
                    with self.assertRaises(ValueError):
                        quiet.load(other)
                    with self.assertRaises(ValueError):
                        old.load(path)
                    self.assertEqual(old.lowlevel_load_count, 0)
        self.assertEqual(quiet.lowlevel_load_count, 0)

    def test_play_relaxes_only_recipe_hash_not_quiet_identity_or_rewards(self):
        Runner, contract, rewards = load_runner_fixture()
        identity = Runner(quiet_runner_environment(contract, rewards), helpers.runner_recipe()).contract()
        saved = {"iter": 499, "infos": {"elf3_contract": copy.deepcopy(identity)}}
        expected = copy.deepcopy(identity)
        expected["feedback_experiment"]["training_recipe_sha256"] = "f" * 64
        before = copy.deepcopy((saved, expected))
        play.validate_play_checkpoint(saved, expected)
        self.assertEqual((saved, expected), before)
        changes = [
            lambda value: value.pop("standing_experiment"),
            lambda value: value.update(standing_experiment=standing.standing_contract.standing_experiment_contract(standing.settings())),
            lambda value: value["standing_experiment"].update(unexpected=True),
            lambda value: value["feedback_experiment"].update(unchanged_reward_sha256="0" * 64),
            lambda value: value.update(history_length=20),
        ]
        for index, change in enumerate(changes):
            actual = copy.deepcopy(identity)
            change(actual)
            with self.subTest(change=index), self.assertRaises(ValueError):
                play.validate_play_checkpoint({"iter": 499, "infos": {"elf3_contract": actual}}, expected)

    def test_logging_reports_raw_arm_motion_only_for_zero_command_environments(self):
        Runner, contract, rewards = load_runner_fixture()
        env = quiet_runner_environment(contract, rewards)
        env.robot.data.joint_vel = torch.zeros(2, 29)
        env.robot.data.joint_vel[0, 15:] = .02
        env.robot.data.joint_vel[1, 15:] = 10.
        env.robot.data.default_joint_pos = torch.zeros(2, 29)
        env.robot.data.joint_pos = torch.zeros(2, 29)
        env.robot.data.joint_pos[0, 15:] = .1
        env.robot.data.joint_pos[1, 15:] = 10.
        env.velocity_feedback_model.value = torch.zeros(2, 29)
        env.quiet_arm_joint_ids = list(range(15, 29))
        env.get_standing_mask = lambda: torch.tensor([True, False])
        runner = Runner(env, helpers.runner_recipe())
        runner.alg.last_standing_metrics = {"phase_invariance": .123, "arm_pose": .456, "active_fraction": .5}
        runner.log({"ep_infos": [], "it": 12})
        values = {key: value for key, value, step in runner.writer.values}
        self.assertEqual(values["Standing/phase_invariance"], .123)
        self.assertEqual(values["Standing/zero_command_env_fraction"], .5)
        self.assertNotIn("Standing/settled_env_fraction", values)
        self.assertAlmostEqual(values["Standing/raw_arm_velocity_rms"], .02, places=6)
        self.assertAlmostEqual(values["Standing/actual_arm_pose_error_rms"], .1, places=6)


class QuietAlgorithmWiringTests(unittest.TestCase):
    def test_actual_lowlevel_constructor_uses_private_algorithm_without_global_injection(self):
        source = ROOT / "TienKung-Lab/rsl_rl/rsl_rl/runners/dwaq_on_policy_runner.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DWAQOnPolicyRunner")

        class ActorCritic_DWAQ(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.arguments = kwargs

        class DWAQPPO:
            def __init__(self, **kwargs):
                self.arguments = kwargs

            def init_storage(self, **kwargs):
                self.storage_arguments = kwargs

        class Elf3DwaqStandPPO(DWAQPPO):
            pass

        namespace = {"torch": torch, "DWAQPPO": DWAQPPO,
                     "ActorCritic_DWAQ": ActorCritic_DWAQ,
                     "rsl_rl": SimpleNamespace(__file__="/cpu/stub/rsl_rl.py")}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                     str(source), "exec"), namespace)
        Base = namespace["DWAQOnPolicyRunner"]

        def configure(instance):
            instance.is_distributed = False
            instance.gpu_global_rank = 0

        class PrivateRunner(Base):
            algorithm_class = Elf3DwaqStandPPO

        env = helpers.runner_environment()
        env.num_actions, env.num_envs = 29, 2
        recipe = helpers.runner_recipe() | {"save_interval": 500}
        with patch.object(Base, "_configure_multi_gpu", configure):
            default = Base(env, copy.deepcopy(recipe))
            private_recipe = copy.deepcopy(recipe)
            private_recipe["algorithm"]["class_name"] = "Elf3DwaqStandPPO"
            private = PrivateRunner(env, private_recipe)
        self.assertIs(type(default.alg), DWAQPPO)
        self.assertIs(type(private.alg), Elf3DwaqStandPPO)
        self.assertEqual(private.alg.storage_arguments["obs_hist_shape"], [500])
        self.assertEqual(private.alg.storage_arguments["action_shape"], [29])
        wrong = copy.deepcopy(private_recipe)
        wrong["algorithm"]["class_name"] = "DWAQPPO"
        with patch.object(Base, "_configure_multi_gpu", configure), self.assertRaises(ValueError):
            PrivateRunner(env, wrong)


class QuietExportIntegrationTests(unittest.TestCase):
    def test_actual_exporter_preserves_quiet_metadata_and_fixed_500_to_29_interface(self):
        Runner, contract, rewards = load_runner_fixture()
        identity = Runner(quiet_runner_environment(contract, rewards), helpers.runner_recipe()).contract()
        upstream = load_file("quiet_export_upstream", LAB / "scripts/export_dwaq_policy.py")
        exporter = load_file("quiet_export_checked", ROOT / "scripts/export_elf3_dwaq.py")
        torch.manual_seed(42)
        policy = upstream.DWAQPolicyExporter(num_obs=100, num_actions=29, dwaq_obs_history_length=5)
        snapshot = {"iter": 499, "model_state_dict": policy.state_dict(),
                    "infos": {"elf3_contract": identity}}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model_499.pt"
            torch.save(snapshot, checkpoint)
            loaded, asset_contract, metadata = exporter.load_policy(checkpoint)
            sample = torch.randn(3, 500)
            with torch.inference_mode():
                actual, expected = loaded(sample), policy(sample)
            self.assertEqual(tuple(actual.shape), (3, 29))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(metadata["standing_experiment"], identity["standing_experiment"])
            self.assertEqual(asset_contract["joint_names"], identity["joint_names"])


if __name__ == "__main__":
    unittest.main()
