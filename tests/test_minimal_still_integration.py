"""CPU integration guards for the minimal, original-network zero-command task.

Execute the real private config, runner, contract and export loader. Only Isaac
configuration containers and low-level checkpoint I/O are stubbed. No simulator,
GPU operation, training run or robot is started by these tests.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
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


standing = load_file("minimal_still_standing_helpers", ROOT / "tests/test_standing_task.py")
helpers = standing.helpers
play = standing.play


def runner_recipe():
    recipe = helpers.runner_recipe()
    recipe["algorithm"]["class_name"] = "DWAQPPO"
    return recipe


def modules():
    contract = load_file("minimal_still_contract", ELF3 / "still_contract.py")
    with patch.dict(sys.modules, standing.import_stubs() | {
        "legged_lab.envs.elf3.still_contract": contract,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
    }):
        stillness = load_file("minimal_still_operations", ELF3 / "stillness.py")
    return contract, stillness


def configuration_classes(contract, stillness):
    _, FeedbackEnv, FeedbackAgent = standing.configuration_classes()
    utils = ModuleType("isaaclab.utils")
    utils.configclass = standing.stub_configclass
    managers = ModuleType("isaaclab.managers")
    # Isaac config serialization recursively expands SceneEntityCfg. A plain
    # object would stringify with a memory address and invent a recipe mismatch.
    class AssetEntityCfg(helpers.ConfigNode):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
    managers.SceneEntityCfg = AssetEntityCfg
    managers.RewardTermCfg = helpers.ConfigNode
    feedback = ModuleType("legged_lab.envs.elf3.elf3_feedback_config")
    feedback.Elf3DwaqFeedbackEnvCfg = FeedbackEnv
    feedback.Elf3DwaqFeedbackAgentCfg = FeedbackAgent
    with patch.dict(sys.modules, standing.import_stubs() | {
        "isaaclab.utils": utils,
        "isaaclab.managers": managers,
        "legged_lab.envs.elf3.elf3_feedback_config": feedback,
        "legged_lab.elf3_contract": helpers.robot_contract,
        "legged_lab.envs.elf3.still_contract": contract,
        "legged_lab.envs.elf3.stillness": stillness,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
    }):
        actual = load_file("minimal_still_configuration", ELF3 / "elf3_still_config.py")
    return actual, FeedbackEnv, FeedbackAgent


def runner_fixture():
    contract, stillness = modules()
    config, Env, Agent = configuration_classes(contract, stillness)
    feedback = ModuleType("legged_lab.envs.elf3.feedback_runner")
    feedback.Elf3DwaqFeedbackRunner = helpers.Runner
    with patch.dict(sys.modules, standing.import_stubs() | {
        "legged_lab.envs.elf3.feedback_runner": feedback,
        "legged_lab.envs.elf3.still_contract": contract,
        "legged_lab.envs.elf3.stillness": stillness,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.elf3_contract": helpers.robot_contract,
    }):
        runner = load_file("minimal_still_actual_runner", ELF3 / "still_runner.py")
    env = helpers.runner_environment()
    private = config.Elf3DwaqStillEnvCfg()
    env.cfg.stillness = copy.deepcopy(private.stillness)
    env.cfg.reward = copy.deepcopy(private.reward)
    return runner.Elf3DwaqStillRunner, env, contract, stillness


class MinimalStillContractTests(unittest.TestCase):
    def test_explicit_settings_are_fresh_complete_and_source_bound(self):
        contract, _ = modules()
        first, second = contract.still_settings(), contract.still_settings()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(first["version"], "elf3_dwaq_still_v1")
        self.assertEqual(first["command_epsilon"], 1e-6)
        self.assertEqual(first["fixed_phase"], [0., 0., 1., -1.])
        self.assertEqual(first["phase_scope"], "current_and_all_history_if_current_zero")
        self.assertEqual(first["joint_motion_weight"], -.5)
        identity = contract.still_experiment_contract(first)
        self.assertEqual(first, second)
        self.assertEqual(identity["version"], "elf3_dwaq_still_v1")
        for name, digest in identity["sources_sha256"].items():
            if (ELF3 / name).is_file():
                self.assertEqual(digest, hashlib.sha256((ELF3 / name).read_bytes()).hexdigest())
        first["fixed_phase"][0] = 1.
        self.assertEqual(second["fixed_phase"], [0., 0., 1., -1.])

    def test_missing_unknown_old_and_altered_contracts_fail_closed(self):
        contract, _ = modules()
        expected = contract.still_settings()
        invalid = [None, {}, standing.settings(), expected | {"unknown": True}]
        for key in expected:
            incomplete = copy.deepcopy(expected)
            incomplete.pop(key)
            invalid.append(incomplete)
        invalid += [expected | {"command_epsilon": value}
                    for value in (0, .001, True, float("nan"), float("inf"))]
        invalid += [expected | {"joint_motion_weight": -.4},
                    expected | {"fixed_phase": [1., 0., 0., -1.]},
                    expected | {"phase_scope": "current_only"}]
        for value in invalid:
            with self.subTest(settings=value), self.assertRaises((ValueError, TypeError)):
                contract.still_experiment_contract(value)


class MinimalStillConfigurationTests(unittest.TestCase):
    def test_only_two_gait_gates_one_actual_motion_term_and_identity_change(self):
        contract, stillness = modules()
        config, FeedbackEnv, _ = configuration_classes(contract, stillness)
        old = FeedbackEnv()
        baseline = copy.deepcopy(old.to_dict())
        private = config.Elf3DwaqStillEnvCfg()
        changed = private.to_dict()
        self.assertEqual(changed.pop("stillness"), contract.still_settings())
        motion = changed["reward"].pop("stand_joint_motion")
        self.assertIs(motion["func"], stillness.stand_joint_motion)
        self.assertEqual(motion["weight"], -.5)
        self.assertEqual(motion["params"]["command_epsilon"], 1e-6)
        asset = private.reward.stand_joint_motion.params["asset_cfg"]
        self.assertEqual(asset.name, "robot")
        self.assertEqual(tuple(asset.joint_names), helpers.robot_contract.JOINT_NAMES)
        self.assertIs(asset.preserve_order, True)
        self.assertEqual(set(motion["params"]), {"asset_cfg", "command_epsilon"})
        for name in ("gait_phase_contact", "feet_swing_height"):
            term = changed["reward"][name]
            self.assertIs(term["func"], getattr(standing.standing_rewards, name + "_when_moving"))
            self.assertEqual(term["params"].pop("command_epsilon"), 1e-6)
            term["func"] = baseline["reward"][name]["func"]
        self.assertEqual(changed, baseline)
        self.assertEqual(old.to_dict(), baseline)
        self.assertEqual(FeedbackEnv().to_dict(), baseline)
        self.assertEqual(private.commands.rel_standing_envs, .2)
        self.assertEqual(private.commands.resampling_time_range, (10., 10.))

    def test_agent_keeps_original_ppo_network_losses_and_hyperparameters(self):
        contract, stillness = modules()
        config, _, FeedbackAgent = configuration_classes(contract, stillness)
        original = FeedbackAgent().to_dict()
        changed = config.Elf3DwaqStillAgentCfg().to_dict()
        for key, value in (("experiment_name", "elf3_dwaq_still"),
                           ("wandb_project", "elf3_dwaq_still"),
                           ("runner_class_name", "Elf3DwaqStillRunner")):
            self.assertEqual(changed[key], value)
            changed[key] = original[key]
        self.assertEqual(changed, original)
        self.assertEqual(changed["upper_body_mirror_loss_coeff"], .1)
        self.assertEqual(changed["upper_body_pose_loss_coeff"], .02)
        self.assertEqual(changed["algorithm"]["min_learning_rate"], 1e-5)
        self.assertEqual(changed["algorithm"]["max_learning_rate"], 3e-4)
        self.assertEqual(changed["policy"]["context_mode"], "mean")

    def test_new_registration_does_not_replace_feedback_or_existing_stand_tasks(self):
        tree = ast.parse((LAB / "envs/__init__.py").read_text())
        found = {}
        counts = {}
        for call in ast.walk(tree):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "register" and call.args
                    and isinstance(call.args[0], ast.Constant)):
                name = call.args[0].value
                counts[name] = counts.get(name, 0) + 1
                found[name] = [ast.unparse(arg) for arg in call.args[1:]]
        expected = {
            "elf3_dwaq_still": ["Elf3DwaqStillEnv", "Elf3DwaqStillEnvCfg()", "Elf3DwaqStillAgentCfg()"],
            "elf3_dwaq_feedback_v1": ["Elf3DwaqFeedbackEnv", "Elf3DwaqFeedbackEnvCfg()", "Elf3DwaqFeedbackAgentCfg()"],
            "elf3_dwaq_stand_v1": ["Elf3DwaqFeedbackEnv", "Elf3DwaqStandEnvCfg()", "Elf3DwaqStandAgentCfg()"],
            "elf3_dwaq_stand_v2": ["Elf3DwaqStandV2Env", "Elf3DwaqStandV2EnvCfg()", "Elf3DwaqStandV2AgentCfg()"],
        }
        for name, arguments in expected.items():
            self.assertEqual(counts[name], 1)
            self.assertEqual(found[name], arguments)


class MinimalStillRunnerTests(unittest.TestCase):
    def test_private_reward_recipe_and_original_algorithm_are_validated_before_load(self):
        alterations = {
            "gait_function": lambda env: setattr(env.cfg.reward.gait_phase_contact, "func", standing.base_rewards.gait_phase_contact),
            "gait_epsilon": lambda env: env.cfg.reward.feet_swing_height.params.update(command_epsilon=.01),
            "motion_function": lambda env: setattr(env.cfg.reward.stand_joint_motion, "func", None),
            "motion_weight": lambda env: setattr(env.cfg.reward.stand_joint_motion, "weight", -.4),
            "motion_epsilon": lambda env: env.cfg.reward.stand_joint_motion.params.update(command_epsilon=.1),
            "motion_joints": lambda env: setattr(env.cfg.reward.stand_joint_motion.params["asset_cfg"], "joint_names", list(reversed(helpers.robot_contract.JOINT_NAMES))),
        }
        for name, alter in alterations.items():
            Runner, env, _, _ = runner_fixture()
            alter(env)
            with self.subTest(change=name), self.assertRaises(ValueError):
                Runner(env, runner_recipe())
        for name in ("Elf3DwaqStandPPO", "AMP"):
            Runner, env, _, _ = runner_fixture()
            recipe = runner_recipe()
            recipe["algorithm"]["class_name"] = name
            with self.subTest(algorithm=name), self.assertRaises(ValueError):
                Runner(env, recipe)

    def test_runner_uses_baseline_algorithm_and_checkpoints_complete_identity(self):
        Runner, env, contract, _ = runner_fixture()
        # No v2 private PPO, storage, optimizer or auxiliary setup is allowed.
        self.assertNotIn("algorithm_class", vars(Runner))
        source = ast.parse((ELF3 / "still_runner.py").read_text())
        imports = [node.module or "" for node in ast.walk(source) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("quiet_learning" in name for name in imports))
        recipe = runner_recipe()
        runner = Runner(env, recipe)
        self.assertFalse(hasattr(runner.alg, "configure_standing"))
        identity = runner.contract()
        self.assertEqual(identity["standing_experiment"], contract.still_experiment_contract(env.cfg.stillness))
        self.assertEqual((identity["num_obs"], identity["history_length"], len(identity["joint_names"])), (100, 5, 29))
        self.assertEqual(runner.symmetry_arguments["coefficient"], .1)
        self.assertEqual(runner.symmetry_arguments["pose_coefficient"], .02)
        runner.current_learning_iteration = 499
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_499.pt"
            infos = {"retained": True}
            runner.save(path, infos)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            Reader, other, _, _ = runner_fixture()
            reader = Reader(other, recipe)
            reader.load(path, load_optimizer=False)
        self.assertEqual(infos, {"retained": True})
        self.assertEqual(saved["infos"]["elf3_contract"], identity)
        self.assertEqual(saved["infos"]["training_recipe"], recipe)
        self.assertEqual(reader.current_learning_iteration, 500)
        self.assertEqual(reader.lowlevel_load_count, 1)
        self.assertFalse(reader.last_load_optimizer)

    def test_original_and_previous_standing_contracts_reject_before_loading_weights(self):
        Runner, env, _, _ = runner_fixture()
        runner = Runner(env, runner_recipe())
        identity = runner.contract()
        old_contracts = []
        original = copy.deepcopy(identity)
        original.pop("standing_experiment")
        old_contracts.append(original)
        v1 = copy.deepcopy(identity)
        v1["standing_experiment"] = standing.standing_contract.standing_experiment_contract(standing.settings())
        old_contracts.append(v1)
        v2 = copy.deepcopy(identity)
        v2["standing_experiment"]["version"] = "elf3_stand_v2_command_only_v2"
        old_contracts.append(v2)
        with tempfile.TemporaryDirectory() as folder:
            for index, wrong in enumerate(old_contracts):
                path = Path(folder) / f"old_{index}.pt"
                torch.save({"iter": 499, "infos": {"elf3_contract": wrong}}, path)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    runner.load(path)
                self.assertEqual(runner.lowlevel_load_count, 0)

    def test_play_relaxes_recipe_hash_only_not_phase_semantics(self):
        Runner, env, _, _ = runner_fixture()
        identity = Runner(env, runner_recipe()).contract()
        saved = {"iter": 499, "infos": {"elf3_contract": copy.deepcopy(identity)}}
        expected = copy.deepcopy(identity)
        expected["feedback_experiment"]["training_recipe_sha256"] = "f" * 64
        play.validate_play_checkpoint(saved, expected)
        for key, value in (("version", "elf3_stand_v1"),
                           ("fixed_phase", [1., 0., 0., -1.]),
                           ("phase_scope", "current_only"),
                           ("joint_motion_weight", -.4)):
            altered = copy.deepcopy(identity)
            altered["standing_experiment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                play.validate_play_checkpoint({"iter": 499, "infos": {"elf3_contract": altered}}, expected)


class MinimalStillExportTests(unittest.TestCase):
    def test_real_export_loader_uses_fixed_phase_for_zero_only_and_keeps_500_to_29(self):
        Runner, env, _, _ = runner_fixture()
        identity = Runner(env, runner_recipe()).contract()
        upstream = load_file("minimal_still_export_upstream", LAB / "scripts/export_dwaq_policy.py")
        exporter = load_file("minimal_still_export_checked", ROOT / "scripts/export_elf3_dwaq.py")
        torch.manual_seed(42)
        raw = upstream.DWAQPolicyExporter(num_obs=100, num_actions=29, dwaq_obs_history_length=5).eval()
        snapshot = {"iter": 499, "model_state_dict": raw.state_dict(), "infos": {"elf3_contract": identity}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_499.pt"
            torch.save(snapshot, path)
            policy, asset_contract, metadata = exporter.load_policy(path)
        sample = torch.randn(3, 500)
        frames = sample.view(3, 5, 100)
        frames[0, -1, 6:9] = 0.
        frames[1, -1, 6:9] = torch.tensor([0., 0., .1])
        frames[2, -1, 6:9] = torch.tensor([.1, 0., 0.])
        before = sample.clone()
        prepared = sample.clone()
        prepared.view(3, 5, 100)[0, :, 96:100] = torch.tensor([0., 0., 1., -1.])
        with torch.inference_mode():
            actual = policy(sample)
            expected = raw(prepared)
        self.assertEqual(tuple(actual.shape), (3, 29))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        self.assertTrue(torch.equal(sample, before))
        self.assertEqual(metadata["standing_experiment"], identity["standing_experiment"])
        self.assertEqual(asset_contract["joint_names"], identity["joint_names"])
        # Both branches must survive graph tracing, not freeze to the example's
        # command state and silently make deployment differ from training.
        traced = torch.jit.trace(policy, sample)
        switched = sample.clone().view(3, 5, 100)
        switched[0, -1, 6] = .1
        switched[1, -1, 6:9] = 0.
        switched = switched.reshape(3, 500)
        with torch.inference_mode():
            torch.testing.assert_close(traced(switched), policy(switched), atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
