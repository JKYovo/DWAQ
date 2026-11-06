"""CPU integration of actual still-v2 config/runner over unchanged still-v1.

Only Isaac containers and bottom-level RSL storage are mocked. No simulator,
GPU, training run or robot is started. Real serializer callbacks are tested.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
ELF3 = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


minimal = load("still_v2_minimal_helpers", ROOT / "tests/test_minimal_still_integration.py")
standing, helpers = minimal.standing, minimal.helpers
old_contract, stillness = minimal.modules()
contract = load("still_v2_actual_contract", ELF3 / "still_v2_contract.py")
leg_rewards = load("still_v2_actual_legacy_leg", ELF3 / "still_pose_rewards.py")
waist_rewards = load("still_v2_actual_waist", ELF3 / "still_v2_rewards.py")


def fixture():
    parent_config, FeedbackEnv, FeedbackAgent = minimal.configuration_classes(old_contract, stillness)
    FeedbackEnv.device = FeedbackAgent.device = "cpu"
    FeedbackEnv.scene.num_envs = 4096
    FeedbackEnv.scene.seed = FeedbackAgent.seed = 42
    FeedbackEnv.scene.max_init_terrain_level = 5
    FeedbackEnv.scene.robot = helpers.ConfigNode(armature=.01, velocity_limit=20., implicit_PD=True)
    FeedbackAgent.resume, FeedbackAgent.logger, FeedbackAgent.run_name = False, "tensorboard", ""
    FeedbackAgent.algorithm.class_name = "DWAQPPO"
    for name, weight in (("hip", -.3), ("ankle", -.2), ("arms", -.2), ("legs", -.02)):
        setattr(FeedbackEnv.reward, "joint_deviation_" + name, helpers.ConfigNode(
            func="original_joint_deviation_l1_always", weight=weight, params={
                "asset_cfg": helpers.ConfigNode(name="robot", joint_names=[name], preserve_order=False)}))
    ParentEnv, ParentAgent = parent_config.Elf3DwaqStillEnvCfg, parent_config.Elf3DwaqStillAgentCfg
    utils = ModuleType("isaaclab.utils")
    utils.configclass = standing.stub_configclass
    managers = ModuleType("isaaclab.managers")

    class Entity(helpers.ConfigNode):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)

    managers.SceneEntityCfg, managers.RewardTermCfg = Entity, helpers.ConfigNode
    feedback = ModuleType("legged_lab.envs.elf3.feedback_runner")
    feedback.Elf3DwaqFeedbackRunner = helpers.Runner
    stubs = standing.import_stubs() | {
        "isaaclab.utils": utils, "isaaclab.managers": managers,
        "legged_lab.elf3_contract": helpers.robot_contract,
        "legged_lab.envs.elf3.elf3_still_config": parent_config,
        "legged_lab.envs.elf3.still_contract": old_contract,
        "legged_lab.envs.elf3.stillness": stillness,
        "legged_lab.envs.elf3.still_pose_rewards": leg_rewards,
        "legged_lab.envs.elf3.still_v2_contract": contract,
        "legged_lab.envs.elf3.still_v2_rewards": waist_rewards,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.envs.elf3.feedback_runner": feedback,
        "legged_lab.envs.elf3.feedback_contract": helpers.contract,
    }
    with patch.dict(sys.modules, stubs):
        parent_runner = load("still_v2_actual_parent_runner", ELF3 / "still_runner.py")
        stubs["legged_lab.envs.elf3.still_runner"] = parent_runner
        config = load("still_v2_actual_config", ELF3 / "elf3_still_v2_config.py")
        stubs["legged_lab.envs.elf3.elf3_still_v2_config"] = config
        with patch.dict(sys.modules, stubs):
            runner = load("still_v2_actual_runner", ELF3 / "still_v2_runner.py")
    runner._test_stubs, runner._parent_runner = stubs, parent_runner
    return config, ParentEnv, ParentAgent, runner


def runner_fixture():
    config, _, _, module = fixture()
    private = config.Elf3DwaqStillV2EnvCfg()
    env = helpers.runner_environment()
    env.num_envs = 2
    env.cfg.stillness = copy.deepcopy(private.stillness)
    env.cfg.still_v2_pose = copy.deepcopy(private.still_v2_pose)
    env.cfg.reward = copy.deepcopy(private.reward)
    names = list(helpers.robot_contract.JOINT_NAMES[::2] + helpers.robot_contract.JOINT_NAMES[1::2])
    env.robot.find_joints = lambda requested, preserve_order=True: ([names.index(name) for name in requested], requested)
    env.robot.data = helpers.ConfigNode(joint_pos=torch.zeros(2, 29), default_joint_pos=torch.zeros(2, 29),
        joint_vel=torch.zeros(2, 29), projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2),
        root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3))
    env.velocity_feedback_model.value = torch.zeros(2, 29)
    env.command_generator = SimpleNamespace(command=torch.tensor([[0., 0., 0.], [.1, 0., 0.]]))
    recipe = helpers.runner_recipe()
    recipe["algorithm"]["class_name"] = "DWAQPPO"
    with patch.dict(sys.modules, module._test_stubs):
        runner = module.Elf3DwaqStillV2Runner(env, recipe)
    return runner, module


class StillV2IntegrationTests(unittest.TestCase):
    def test_direct_still_parent_only_two_added_rewards_and_environment_count(self):
        config, ParentEnv, _, _ = fixture()
        self.assertEqual(config.Elf3DwaqStillV2EnvCfg.__bases__, (ParentEnv,))
        original = ParentEnv().to_dict()
        cfg, candidate = config.Elf3DwaqStillV2EnvCfg(), config.Elf3DwaqStillV2EnvCfg().to_dict()
        self.assertEqual(candidate.pop("still_v2_pose"), contract.pose_settings())
        self.assertEqual(candidate["scene"]["num_envs"], 16384)
        candidate["scene"]["num_envs"] = original["scene"]["num_envs"]
        for name, function, weight, names in (
            ("stand_leg_pose", leg_rewards.stand_leg_pose_l1, -.13, leg_rewards.STAND_POSE_JOINT_NAMES),
            ("stand_waist_pose", waist_rewards.stand_waist_pose_l1, -.1, waist_rewards.STAND_WAIST_JOINT_NAMES)):
            term = candidate["reward"].pop(name)
            self.assertIs(term["func"], function)
            self.assertEqual(term["weight"], weight)
            self.assertEqual(set(term["params"]), {"asset_cfg", "command_epsilon"})
            self.assertEqual(term["params"]["command_epsilon"], 1e-6)
            entity = getattr(cfg.reward, name).params["asset_cfg"]
            self.assertEqual(tuple(entity.joint_names), tuple(names))
            self.assertTrue(entity.preserve_order)
        self.assertEqual(candidate, original)
        self.assertEqual(ParentEnv().to_dict(), original)
        self.assertEqual(cfg.stillness, old_contract.still_settings())
        self.assertEqual(cfg.stillness["fixed_phase"], [0., 0., 1., -1.])
        self.assertEqual(cfg.reward.stand_joint_motion.weight, -.5)
        self.assertEqual(cfg.commands.rel_standing_envs, .2)
        self.assertEqual(cfg.commands.resampling_time_range, (10., 10.))

    def test_agent_is_parent_still_policy_ppo_and_auxiliaries_with_identity_only(self):
        config, _, ParentAgent, _ = fixture()
        self.assertEqual(config.Elf3DwaqStillV2AgentCfg.__bases__, (ParentAgent,))
        original = ParentAgent().to_dict()
        candidate = config.Elf3DwaqStillV2AgentCfg().to_dict()
        for key, value in (("experiment_name", "elf3_dwaq_still_v2"),
                           ("wandb_project", "elf3_dwaq_still_v2"),
                           ("runner_class_name", "Elf3DwaqStillV2Runner")):
            self.assertEqual(candidate[key], value)
            candidate[key] = original[key]
        self.assertEqual(candidate, original)
        self.assertEqual(candidate["algorithm"]["class_name"], "DWAQPPO")
        self.assertEqual(candidate["algorithm"]["min_learning_rate"], 1e-5)
        self.assertEqual(candidate["algorithm"]["max_learning_rate"], 3e-4)
        self.assertEqual(candidate["upper_body_mirror_loss_coeff"], .1)
        self.assertEqual(candidate["upper_body_pose_loss_coeff"], .02)

    def test_contract_source_bound_and_no_migration_or_new_standing_phase(self):
        settings = contract.pose_settings()
        identity = contract.pose_experiment_contract(settings)
        self.assertEqual(identity["version"], "elf3_dwaq_still_v2")
        self.assertEqual(identity["parent_task"], "elf3_dwaq_still")
        self.assertEqual(identity["reference_task"], "elf3_dwaq_still_pose_v1")
        self.assertEqual(identity["initialization"], "from_scratch")
        self.assertAlmostEqual(settings["leg_incremental_weight"] + settings["leg_baseline_weight"], -.15)
        self.assertAlmostEqual(settings["waist_incremental_weight"] + settings["waist_baseline_weight"], -.30)
        self.assertEqual(identity["moving_leg_weight"], -.02)
        self.assertEqual(identity["moving_waist_weight"], -.20)
        for name, digest in identity["sources_sha256"].items():
            self.assertEqual(hashlib.sha256((ELF3 / name).read_bytes()).hexdigest(), digest)
        for change in ({"leg_incremental_weight": -.08}, {"waist_incremental_weight": 0.},
                       {"version": "elf3_dwaq_still_pose_v1"}, {"fixed_phase": [0., 0., 1., 1.]},
                       {"extra": True}, {"command_epsilon": .1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract.pose_experiment_contract(settings | change)

    def test_full_configuration_audit_allows_runtime_environment_count_but_not_recipe_changes(self):
        config, _, _, module = fixture()
        cfg, agent = config.Elf3DwaqStillV2EnvCfg(), config.Elf3DwaqStillV2AgentCfg().to_dict()
        cfg.scene.num_envs, agent["max_iterations"], agent["run_name"] = 64, 20, "technical_check"
        with patch.dict(sys.modules, module._test_stubs):
            self.assertEqual(module.validate_still_v2_configuration(cfg, agent)["initialization"], "from_scratch")
            for kind in ("leg_weight", "waist_weight", "old_pose", "gait", "motion", "phase", "sampling",
                         "command_interval", "armature", "terrain", "algorithm", "seed", "auxiliary"):
                e, a = copy.deepcopy(cfg), copy.deepcopy(agent)
                if kind == "leg_weight": e.reward.stand_leg_pose.weight = -.08
                elif kind == "waist_weight": e.reward.stand_waist_pose.weight = -.2
                elif kind == "old_pose": e.reward.joint_deviation_legs.weight = -.1
                elif kind == "gait": e.reward.gait_phase_contact.weight = .4
                elif kind == "motion": e.reward.stand_joint_motion.weight = -1.
                elif kind == "phase": e.stillness["fixed_phase"] = [0., 0., 1., 1.]
                elif kind == "sampling": e.commands.rel_standing_envs = .3
                elif kind == "command_interval": e.commands.resampling_time_range = (5., 5.)
                elif kind == "armature": e.scene.robot.armature = 0.
                elif kind == "terrain": e.scene.max_init_terrain_level = 1
                elif kind == "algorithm": a["algorithm"]["learning_rate"] = .001
                elif kind == "seed": a["seed"] = 123
                elif kind == "auxiliary": a["upper_body_pose_loss_coeff"] = .1
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    module.validate_still_v2_configuration(e, a)

    def test_reward_guard_rejects_callback_weight_epsilon_entity_and_extra_term_errors(self):
        config, _, _, module = fixture()
        for name in ("stand_leg_pose", "stand_waist_pose"):
            for kind in ("function", "weight", "epsilon", "extra_param", "joint_names", "preserve_order", "extra_reward"):
                cfg = config.Elf3DwaqStillV2EnvCfg()
                term = getattr(cfg.reward, name)
                if kind == "function": term.func = lambda *args: 0.
                elif kind == "weight": term.weight = -10.
                elif kind == "epsilon": term.params["command_epsilon"] = .1
                elif kind == "extra_param": term.params["settled"] = True
                elif kind == "joint_names": term.params["asset_cfg"].joint_names.reverse()
                elif kind == "preserve_order": term.params["asset_cfg"].preserve_order = False
                elif kind == "extra_reward": cfg.reward.zero_stand_score = copy.deepcopy(term)
                with patch.dict(sys.modules, module._test_stubs):
                    with self.subTest(term=name, kind=kind), self.assertRaises(ValueError):
                        module.validate_reward_identity(cfg)

    def test_actual_isaac_serializer_accepts_valid_callbacks_and_rejects_same_name_impostors(self):
        namespace = {"torch": torch, "inspect": inspect, "re": re}
        utils = ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
        for file, name in (("string.py", "callable_to_string"), ("dict.py", "class_to_dict")):
            tree = ast.parse((utils / file).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            selected = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
            exec(compile(selected, str(utils / file), "exec"), namespace)
        config, _, _, module = fixture()
        with patch.object(helpers.ConfigNode, "to_dict", namespace["class_to_dict"]), patch.dict(sys.modules, module._test_stubs):
            cfg = config.Elf3DwaqStillV2EnvCfg()
            serialized = cfg.reward.to_dict()
            module.validate_reward_identity(cfg)
            module.validate_still_v2_configuration(cfg, config.Elf3DwaqStillV2AgentCfg().to_dict())
            for name in ("stand_leg_pose", "stand_waist_pose"):
                original = getattr(cfg.reward, name).func
                self.assertEqual(serialized[name]["func"], f"{original.__module__}:{original.__name__}")

                def impostor(*args, **kwargs):
                    raise AssertionError("An identity guard must not execute a substituted reward")

                impostor.__module__, impostor.__name__, impostor.__qualname__ = original.__module__, original.__name__, original.__qualname__
                altered = copy.deepcopy(cfg)
                getattr(altered.reward, name).func = impostor
                self.assertEqual(altered.reward.to_dict(), serialized)
                with self.subTest(term=name), self.assertRaises(ValueError):
                    module.validate_reward_identity(altered)

    def test_runner_keeps_original_still_contract_and_rejects_legacy_checkpoints(self):
        writer, module = runner_fixture()
        self.assertEqual(type(writer).__bases__, (module._parent_runner.Elf3DwaqStillRunner,))
        self.assertFalse(hasattr(writer, "initialize_from_still"))
        identity = writer.contract()
        self.assertEqual(identity["standing_experiment"], old_contract.still_experiment_contract(writer.env.cfg.stillness))
        self.assertEqual(identity["still_v2_experiment"], contract.pose_experiment_contract(writer.env.cfg.still_v2_pose))
        self.assertEqual(writer.symmetry_arguments["coefficient"], .1)
        self.assertEqual(writer.symmetry_arguments["pose_coefficient"], .02)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.pt"
            writer.current_learning_iteration = 499
            writer.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for kind in ("still100000", "still_pose102500", "stand_v3"):
                broken = copy.deepcopy(saved)
                metadata = broken["infos"]["elf3_contract"]
                metadata.pop("still_v2_experiment")
                if kind == "still_pose102500": metadata["zero_command_pose_experiment"] = {"version": "elf3_dwaq_still_pose_v1"}
                elif kind == "stand_v3": metadata["standing_experiment"]["version"] = "elf3_dwaq_feedback_stand_v3_zero_phase_v1"
                legacy = Path(folder) / (kind + ".pt")
                torch.save(broken, legacy)
                reader, _ = runner_fixture()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    reader.load(legacy, load_optimizer=False)
                self.assertEqual(reader.lowlevel_load_count, 0)
            reader, _ = runner_fixture()
            reader.load(path, load_optimizer=False)
            self.assertEqual(reader.current_learning_iteration, 500)

    def test_constructor_allows_inference_terrain_override_but_training_guard_does_not(self):
        runner, module = runner_fixture()
        runner.env.cfg.scene.terrain = "play_plane"
        with patch.dict(sys.modules, module._test_stubs):
            module.validate_reward_identity(runner.env.cfg)
        # Full recipe audit belongs to launcher, not __init__: play legitimately
        # changes terrain/noise before construction and uses inference validation.
        tree = ast.parse((ELF3 / "still_v2_runner.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == "validate_still_v2_configuration" for node in ast.walk(constructor)))

    def test_added_logs_use_zero_rows_and_actual_named_native_joint_values_only(self):
        runner, _ = runner_fixture()
        data = runner.env.robot.data
        leg_ids, waist_ids = runner._still_v2_leg_ids, runner._still_v2_waist_ids
        data.joint_pos[0, leg_ids] = torch.tensor([-.4, .8, -.35, .75])
        data.default_joint_pos[0, leg_ids] = torch.tensor([-.3, .6, -.3, .6])
        data.joint_pos[0, waist_ids] = torch.tensor([.1, -.2, .3])
        data.joint_vel[0] = .7
        # A moving environment with much larger values must not enter zero logs.
        data.joint_pos[1], data.joint_vel[1] = 99., 99.
        angle = torch.deg2rad(torch.tensor(20.))
        data.projected_gravity_b[0] = torch.tensor([torch.sin(angle), 0., -torch.cos(angle)])
        before_q, before_home, before_dq = data.joint_pos.clone(), data.default_joint_pos.clone(), data.joint_vel.clone()
        runner.log({"it": 12, "ep_infos": []})
        metrics = {key: value for key, value, iteration in runner.writer.values if iteration == 12}
        self.assertAlmostEqual(metrics["StillV2/snapshot_leg_default_pose_l1_sum"], .5, places=6)
        self.assertAlmostEqual(metrics["StillV2/snapshot_waist_default_pose_l1_sum"], .6, places=6)
        self.assertAlmostEqual(metrics["StillV2/snapshot_raw_joint_velocity_abs_mean"], .7, places=6)
        self.assertAlmostEqual(metrics["StillV2/snapshot_torso_tilt_deg"], 20., places=4)
        settings = runner.env.cfg.still_v2_pose
        for names, ids in ((settings["leg_joint_names"], leg_ids), (settings["waist_joint_names"], waist_ids)):
            for name, index in zip(names, ids):
                self.assertEqual(metrics[f"StillV2/snapshot/{name}/actual_rad"], float(data.joint_pos[0, index]))
                self.assertEqual(metrics[f"StillV2/snapshot/{name}/default_error_rad"],
                                 float(data.joint_pos[0, index]-data.default_joint_pos[0, index]))
        self.assertTrue(torch.equal(data.joint_pos, before_q))
        self.assertTrue(torch.equal(data.default_joint_pos, before_home))
        self.assertTrue(torch.equal(data.joint_vel, before_dq))

    def test_original_still_v2_is_restored_and_ankle_env_keeps_original_still_path(self):
        tree = ast.parse((ROOT / "TienKung-Lab/legged_lab/envs/__init__.py").read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
                   and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "elf3_dwaq_still_v2"]
        self.assertEqual(len(matches), 1)
        self.assertEqual([ast.unparse(node) for node in matches[0].args[1:]],
                         ["Elf3DwaqStillEnv", "Elf3DwaqStillV2EnvCfg()", "Elf3DwaqStillV2AgentCfg()"])
        ankle = ast.parse((ROOT / "TienKung-Lab/legged_lab/envs/elf3/ankle_cost.py").read_text())
        env = next(node for node in ankle.body if isinstance(node, ast.ClassDef)
                   and node.name == "Elf3DwaqAnkleCostEnv")
        self.assertEqual([ast.unparse(base) for base in env.bases], ["Elf3DwaqStillEnv"])


if __name__ == "__main__":
    unittest.main()
