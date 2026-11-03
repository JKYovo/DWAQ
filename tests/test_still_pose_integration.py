"""CPU integration of the independent zero-only leg-pose continuation task.

Real config, contract, runner and export code execute over the existing minimal
task's Isaac-container/checkpoint-I/O stubs. No simulator, GPU or robot starts.
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


def load_file(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


minimal = load_file("still_pose_old_integration_helpers", ROOT / "tests/test_minimal_still_integration.py")
standing = minimal.standing
helpers = minimal.helpers
pose_rewards = load_file("still_pose_actual_rewards", ELF3 / "still_pose_rewards.py")

# Freeze the already-trained task's source identity, not merely its settings.
# This new experiment must not rewrite the parent model's inference contract.
OLD_STILL_SOURCE_SHA256 = {
    "still_contract.py": "68f10a802f3223ddda221c525a685b8dbfec768c208d076e7276369ba38c835c",
    "stillness.py": "6da6ddf7c7685d1bce2e31c045b76cc2be390bc4fd8897d660324ef685f256f6",
    "still_env.py": "a7798c1dc83c2ca1e1538c0c6951b2a30a65c2e4334aef9aca267933cc35b384",
    "still_runner.py": "b414526f88d4a8bc52a8a4f91a0b4d096bae6f1cd727bc95878a981cb7307adb",
    "elf3_still_config.py": "4f15901816fe21bfb25d2cc9c7a919507cd9788f4c23390713af88e4fd38cbad",
}


def actual_leg_deviation_function():
    tree = ast.parse((LAB / "mdp/rewards.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "joint_deviation_l1_always")
    module = ModuleType("legged_lab.mdp.rewards")
    module.torch, module.SceneEntityCfg = torch, standing.EntityCfg
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])),
                 "<actual baseline leg deviation>", "exec"), module.__dict__)
    return module.joint_deviation_l1_always


def configuration_fixture():
    old_contract, stillness = minimal.modules()
    old_config, FeedbackEnv, FeedbackAgent = minimal.configuration_classes(old_contract, stillness)
    # The earlier minimal test's compact baseline omitted these unchanged
    # fields. Supply realistic values for the stricter full-recipe audit.
    FeedbackEnv.device = "cpu"
    FeedbackEnv.scene.num_envs = 4096
    FeedbackEnv.scene.robot = helpers.ConfigNode(spawn=helpers.ConfigNode(usd_path="/mock/elf3.usd"))
    FeedbackEnv.reward.joint_deviation_legs = helpers.ConfigNode(
        func=actual_leg_deviation_function(), weight=-.02,
        params={"asset_cfg": helpers.ConfigNode(name="robot",
                    joint_names=["[lr]_hip_y_joint", "[lr]_knee_y_joint"], preserve_order=False)})
    FeedbackAgent.algorithm.class_name = "DWAQPPO"
    FeedbackAgent.seed = 42
    FeedbackAgent.num_steps_per_env = 24
    FeedbackAgent.empirical_normalization = False
    managers = ModuleType("isaaclab.managers")
    class EntityCfg(helpers.ConfigNode):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
    managers.SceneEntityCfg = EntityCfg
    managers.RewardTermCfg = helpers.ConfigNode
    utils = ModuleType("isaaclab.utils")
    utils.configclass = standing.stub_configclass
    stubs = standing.import_stubs() | {
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "legged_lab.envs.elf3.still_pose_rewards": pose_rewards,
        "legged_lab.envs.elf3.still_contract": old_contract,
        "legged_lab.envs.elf3.stillness": stillness,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.envs.elf3.elf3_still_config": old_config,
        "legged_lab.elf3_contract": helpers.robot_contract,
    }
    with patch.dict(sys.modules, stubs):
        contract = load_file("still_pose_actual_contract", ELF3 / "still_pose_contract.py")
        stubs["legged_lab.envs.elf3.still_pose_contract"] = contract
        with patch.dict(sys.modules, stubs):
            config = load_file("still_pose_actual_config", ELF3 / "elf3_still_pose_config.py")
    return config, contract, old_config, old_contract, stillness, stubs


def runner_fixture():
    config, contract, old_config, old_contract, stillness, stubs = configuration_fixture()
    feedback = ModuleType("legged_lab.envs.elf3.feedback_runner")
    feedback.Elf3DwaqFeedbackRunner = helpers.Runner
    stubs["legged_lab.envs.elf3.feedback_runner"] = feedback
    rsl = ModuleType("rsl_rl.runners")
    rsl.DWAQOnPolicyRunner = helpers.TensorOnlyRslRunner
    stubs["rsl_rl.runners"] = rsl
    stubs["legged_lab.envs.elf3.feedback_contract"] = helpers.contract
    with patch.dict(sys.modules, stubs):
        old_runner = load_file("still_pose_actual_parent_runner", ELF3 / "still_runner.py")
        stubs["legged_lab.envs.elf3.still_runner"] = old_runner
        with patch.dict(sys.modules, stubs):
            runner = load_file("still_pose_actual_runner", ELF3 / "still_pose_runner.py")
    env = helpers.runner_environment()
    env.cfg = config.Elf3DwaqStillPoseEnvCfg()
    env.cfg.scene.num_envs = 2  # Allowed separately recorded smoke override.
    env.num_envs = 2
    ids = [27, 4, 20, 8]
    env.robot.find_joints = lambda names, preserve_order: (ids, list(names))
    recipe = config.Elf3DwaqStillPoseAgentCfg().to_dict()
    return runner, env, recipe, contract, old_contract


class StillPoseContractTests(unittest.TestCase):
    def test_old_still_sources_have_not_been_modified(self):
        for name, digest in OLD_STILL_SOURCE_SHA256.items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((ELF3 / name).read_bytes()).hexdigest(), digest)

    def test_explicit_joint_target_and_incremental_total_weights(self):
        _, contract, _, _, _, _ = configuration_fixture()
        first, other = contract.pose_settings(), contract.pose_settings()
        self.assertEqual(first, {
            "version": "elf3_dwaq_still_pose_v1", "command_epsilon": 1e-6,
            "joint_names": list(pose_rewards.STAND_POSE_JOINT_NAMES),
            "incremental_weight": -.08, "baseline_weight": -.02,
            "total_zero_weight": -.10, "target": "asset.default_joint_pos"})
        self.assertIsNot(first, other)
        identity = contract.pose_experiment_contract(first)
        for key in first:
            self.assertEqual(identity[key], first[key])
        self.assertEqual(identity["moving_pose_weight"], -.02)
        self.assertFalse(identity["height_braking_PD_asset_and_actor_changes"])
        for name, digest in identity["sources_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((ELF3 / name).read_bytes()).hexdigest())
        first["joint_names"].reverse()
        self.assertEqual(other["joint_names"], list(pose_rewards.STAND_POSE_JOINT_NAMES))

    def test_unknown_incomplete_or_changed_pose_contract_fails_closed(self):
        _, contract, _, _, _, _ = configuration_fixture()
        baseline = contract.pose_settings()
        invalid = [None, {}, baseline | {"extra": True}, baseline | {"incremental_weight": -.1},
                   baseline | {"command_epsilon": .1}, baseline | {"target": "zero_action"},
                   baseline | {"joint_names": list(reversed(baseline["joint_names"]))},
                   baseline | {"total_zero_weight": -.12}, baseline | {"baseline_weight": -.04}]
        for key in baseline:
            altered = copy.deepcopy(baseline)
            altered.pop(key)
            invalid.append(altered)
        for candidate in invalid:
            with self.subTest(settings=candidate), self.assertRaises((TypeError, ValueError)):
                contract.pose_experiment_contract(candidate)


class StillPoseConfigurationTests(unittest.TestCase):
    def test_new_environment_adds_only_one_zero_pose_reward_and_its_settings(self):
        config, contract, old_config, _, _, _ = configuration_fixture()
        baseline = old_config.Elf3DwaqStillEnvCfg()
        before = copy.deepcopy(baseline.to_dict())
        candidate = config.Elf3DwaqStillPoseEnvCfg()
        changed = candidate.to_dict()
        self.assertEqual(changed.pop("zero_command_pose"), contract.pose_settings())
        term = changed["reward"].pop("stand_leg_pose")
        self.assertIs(term["func"], pose_rewards.stand_leg_pose_l1)
        self.assertEqual(term["weight"], -.08)
        self.assertEqual(term["params"]["command_epsilon"], 1e-6)
        self.assertEqual(set(term["params"]), {"asset_cfg", "command_epsilon"})
        asset = candidate.reward.stand_leg_pose.params["asset_cfg"]
        self.assertEqual(asset.name, "robot")
        self.assertEqual(asset.joint_names, list(pose_rewards.STAND_POSE_JOINT_NAMES))
        self.assertIs(asset.preserve_order, True)
        self.assertEqual(changed, before)
        self.assertEqual(baseline.to_dict(), before)
        self.assertEqual(old_config.Elf3DwaqStillEnvCfg().to_dict(), before)
        self.assertEqual(candidate.reward.joint_deviation_legs.weight, -.02)
        self.assertEqual(candidate.reward.stand_joint_motion.weight, -.5)
        self.assertEqual(candidate.commands.rel_standing_envs, .2)
        self.assertEqual(candidate.commands.resampling_time_range, (10., 10.))

    def test_agent_changes_identity_only_preserving_full_learning_recipe(self):
        config, _, old_config, _, _, _ = configuration_fixture()
        baseline = old_config.Elf3DwaqStillAgentCfg().to_dict()
        candidate = config.Elf3DwaqStillPoseAgentCfg().to_dict()
        for key, expected in (("experiment_name", "elf3_dwaq_still_pose_v1"),
                              ("wandb_project", "elf3_dwaq_still_pose_v1"),
                              ("runner_class_name", "Elf3DwaqStillPoseRunner")):
            self.assertEqual(candidate[key], expected)
            candidate[key] = baseline[key]
        self.assertEqual(candidate, baseline)
        self.assertEqual(candidate["algorithm"]["class_name"], "DWAQPPO")
        self.assertEqual(candidate["upper_body_mirror_loss_coeff"], .1)
        self.assertEqual(candidate["upper_body_pose_loss_coeff"], .02)
        self.assertEqual(candidate["policy"]["context_mode"], "mean")

    def test_strict_config_audit_rejects_unrelated_rewards_or_hyperparameters(self):
        for key in ("baseline_weight", "motion_weight", "commands", "assets", "learning_rate", "sampling", "seed"):
            module, env, recipe, _, _ = runner_fixture()
            if key == "baseline_weight":
                env.cfg.reward.joint_deviation_legs.weight = -.1
            elif key == "motion_weight":
                env.cfg.reward.stand_joint_motion.weight = -.4
            elif key == "commands":
                env.cfg.commands.resampling_time_range = (5., 5.)
            elif key == "assets":
                env.cfg.scene.robot.spawn.usd_path = "/different/robot.usd"
            elif key == "learning_rate":
                recipe["algorithm"]["learning_rate"] = .001
            elif key == "sampling":
                env.cfg.commands.rel_standing_envs = .3
            elif key == "seed":
                recipe["seed"] = 7
            with self.subTest(change=key), self.assertRaises(ValueError):
                module.validate_pose_configuration(env.cfg, recipe)


class StillPoseRunnerAndExportTests(unittest.TestCase):
    def test_private_reward_weight_function_epsilon_and_joint_order_are_validated(self):
        changes = {
            "weight": lambda term: setattr(term, "weight", -.1),
            "function": lambda term: setattr(term, "func", None),
            "epsilon": lambda term: term.params.update(command_epsilon=.1),
            "joint_order": lambda term: setattr(term.params["asset_cfg"], "joint_names",
                                                 list(reversed(pose_rewards.STAND_POSE_JOINT_NAMES))),
        }
        for key, alter in changes.items():
            module, env, recipe, _, _ = runner_fixture()
            alter(env.cfg.reward.stand_leg_pose)
            with self.subTest(change=key), self.assertRaises(ValueError):
                module.Elf3DwaqStillPoseRunner(env, recipe)

    def test_identity_keeps_parent_still_contract_and_adds_independent_pose_contract(self):
        module, env, recipe, contract, old_contract = runner_fixture()
        runner = module.Elf3DwaqStillPoseRunner(env, recipe)
        identity = runner.contract()
        self.assertEqual(identity["standing_experiment"], old_contract.still_experiment_contract(env.cfg.stillness))
        self.assertEqual(identity["standing_experiment"]["sources_sha256"], OLD_STILL_SOURCE_SHA256)
        self.assertEqual(identity["zero_command_pose_experiment"],
                         contract.pose_experiment_contract(env.cfg.zero_command_pose))
        self.assertEqual((identity["num_obs"], identity["history_length"], len(identity["joint_names"])), (100, 5, 29))
        self.assertEqual(runner._pose_joint_ids, [27, 4, 20, 8])
        self.assertEqual(runner.symmetry_arguments["coefficient"], .1)
        self.assertEqual(runner.symmetry_arguments["pose_coefficient"], .02)
        self.assertNotIn("algorithm_class", vars(module.Elf3DwaqStillPoseRunner))
        audit = module.validate_pose_configuration(env.cfg, recipe)
        self.assertTrue(audit["original_algorithm_policy_PD_asset_commands_and_terrain"])
        self.assertTrue(audit["smoke_num_envs_override"])

    def test_new_checkpoint_roundtrip_preserves_both_identities_and_strict_resume(self):
        module, env, recipe, _, _ = runner_fixture()
        writer = module.Elf3DwaqStillPoseRunner(env, recipe)
        writer.current_learning_iteration = 100499
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_100499.pt"
            writer.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            reader_module, reader_env, reader_recipe, _, _ = runner_fixture()
            reader = reader_module.Elf3DwaqStillPoseRunner(reader_env, reader_recipe)
            reader.load(path, load_optimizer=False)
            self.assertEqual(reader.current_learning_iteration, 100500)
            self.assertEqual(reader.lowlevel_load_count, 1)
            self.assertEqual(saved["infos"]["elf3_contract"], writer.contract())
            self.assertEqual(saved["infos"]["training_recipe"], recipe)
            # The ordinary resume path must not silently accept the old task.
            saved["infos"]["elf3_contract"].pop("zero_command_pose_experiment")
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                reader.load(path)
            self.assertEqual(reader.lowlevel_load_count, 1)

    def test_export_still_wraps_zero_phase_and_preserves_new_training_metadata(self):
        module, env, recipe, _, _ = runner_fixture()
        identity = module.Elf3DwaqStillPoseRunner(env, recipe).contract()
        upstream = load_file("still_pose_export_upstream", LAB / "scripts/export_dwaq_policy.py")
        export = load_file("still_pose_export_loader", ROOT / "scripts/export_elf3_dwaq.py")
        torch.manual_seed(42)
        raw = upstream.DWAQPolicyExporter(num_obs=100, num_actions=29, dwaq_obs_history_length=5).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_test_only.pt"
            torch.save({"iter": 100499, "model_state_dict": raw.state_dict(),
                        "infos": {"elf3_contract": identity}}, path)
            wrapped, _, metadata = export.load_policy(path)
        self.assertIsInstance(wrapped, export.MinimalStillPolicyExporter)
        self.assertEqual(metadata["standing_experiment"], identity["standing_experiment"])
        self.assertEqual(metadata["zero_command_pose_experiment"], identity["zero_command_pose_experiment"])
        self.assertTrue(export.preprocessing_metadata(metadata)["zero_command_preprocessing"]["embedded_in_model"])
        history = torch.randn(3, 5, 100)
        history[:, :, 6:9] = torch.tensor([.6, .1, .2])
        history[0, -1, 6:9] = 0.
        history[1, -1, 6:9] = torch.tensor([0., 0., .1])
        history[2, -1, 6:9] = torch.tensor([.1, 0., 0.])
        prepared = history.clone()
        prepared[0, :, 96:100] = torch.tensor([0., 0., 1., -1.])
        flat, original = history.reshape(3, 500), history.clone()
        with torch.inference_mode():
            actual = wrapped(flat)
            self.assertEqual(tuple(actual.shape), (3, 29))
            torch.testing.assert_close(actual, raw(prepared.reshape(3, 500)), atol=0., rtol=0.)
            alternate = history.clone()
            alternate[0, :, 96:100] += 100.
            torch.testing.assert_close(wrapped(alternate.reshape(3, 500))[0], actual[0], atol=0., rtol=0.)
            torch.testing.assert_close(actual[1:], raw(flat)[1:], atol=0., rtol=0.)
        self.assertTrue(torch.equal(history, original))


if __name__ == "__main__":
    unittest.main()
