"""CPU guards for zero-command costs and private periodic-drive removal.

The real config, reward, contract, runner and environment execute under local
Isaac container/RSL stubs. No simulation, GPU, training or robot is started.
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
VERSION = "elf3_dwaq_feedback_stand_v3_zero_phase_v1"
LEGACY_TERMS = ("zero_stand_score", "zero_full_body_pose", "zero_torso_upright",
                "zero_base_stillness")


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


minimal = load("minimal_pose_integration_helpers", ROOT / "tests/test_minimal_still_integration.py")
env_helpers = load("minimal_pose_environment_helpers", ROOT / "tests/test_minimal_still_environment.py")
standing, helpers = minimal.standing, minimal.helpers
rewards = load("minimal_pose_actual_rewards", ELF3 / "feedback_stand_v3_rewards.py")
contract = load("minimal_pose_actual_contract", ELF3 / "feedback_stand_v3_contract.py")
inputs = load("zero_phase_actual_inputs", ELF3 / "feedback_stand_v3_inputs.py")


def fixture():
    _, ParentEnv, ParentAgent = standing.configuration_classes()
    ParentEnv.device = ParentAgent.device = "cpu"
    ParentEnv.scene.num_envs = 4096
    ParentEnv.scene.seed = ParentAgent.seed = 42
    ParentEnv.scene.max_init_terrain_level = 5
    ParentEnv.scene.robot = helpers.ConfigNode(armature=.01, velocity_limit=20., implicit_PD=True)
    ParentAgent.resume, ParentAgent.logger = False, "tensorboard"
    ParentAgent.run_name = ""
    ParentAgent.algorithm.class_name = "DWAQPPO"
    for name, weight in (("hip", -.3), ("ankle", -.2), ("arms", -.2), ("legs", -.02)):
        setattr(ParentEnv.reward, "joint_deviation_" + name,
                helpers.ConfigNode(func="original_joint_deviation_l1_always", weight=weight,
                    params={"asset_cfg": helpers.ConfigNode(name="robot", joint_names=[name], preserve_order=False)}))
    utils = ModuleType("isaaclab.utils")
    utils.configclass = standing.stub_configclass
    managers = ModuleType("isaaclab.managers")

    class Entity(helpers.ConfigNode):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)

    managers.SceneEntityCfg, managers.RewardTermCfg = Entity, helpers.ConfigNode
    feedback = ModuleType("legged_lab.envs.elf3.elf3_feedback_config")
    feedback.Elf3DwaqFeedbackEnvCfg, feedback.Elf3DwaqFeedbackAgentCfg = ParentEnv, ParentAgent
    base_runner = ModuleType("legged_lab.envs.elf3.feedback_runner")
    base_runner.Elf3DwaqFeedbackRunner = helpers.Runner
    stubs = standing.import_stubs() | {
        "isaaclab.utils": utils, "isaaclab.managers": managers,
        "legged_lab.elf3_contract": helpers.robot_contract,
        "legged_lab.envs.elf3.elf3_feedback_config": feedback,
        "legged_lab.envs.elf3.feedback_stand_v3_rewards": rewards,
        "legged_lab.envs.elf3.feedback_stand_v3_contract": contract,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.envs.elf3.feedback_runner": base_runner,
        "legged_lab.envs.elf3.feedback_contract": helpers.contract,
    }
    with patch.dict(sys.modules, stubs):
        config = load("minimal_pose_actual_config", ELF3 / "elf3_feedback_stand_v3_config.py")
        stubs["legged_lab.envs.elf3.elf3_feedback_stand_v3_config"] = config
        with patch.dict(sys.modules, stubs):
            runner = load("minimal_pose_actual_runner", ELF3 / "feedback_stand_v3_runner.py")
    runner._test_stubs = stubs
    return config, ParentEnv, ParentAgent, runner


def real_environment():
    parent = ModuleType("legged_lab.envs.elf3.elf3_feedback_env")
    parent.Elf3DwaqFeedbackEnv = env_helpers.helpers.FeedbackEnv
    with patch.dict(sys.modules, {"legged_lab.envs.elf3.elf3_feedback_env": parent,
                                 "legged_lab.envs.elf3.feedback_stand_v3_inputs": inputs}):
        cls = load("minimal_pose_actual_env", ELF3 / "feedback_stand_v3_env.py").Elf3DwaqFeedbackStandV3Env
    base = env_helpers.environment()
    env = cls.__new__(cls)
    env.__dict__.update(base.__dict__)
    env.cfg.strict_standing_v3 = contract.stand_settings()
    return env


def runner_fixture():
    config, _, _, runner_module = fixture()
    private = config.Elf3DwaqFeedbackStandV3EnvCfg()
    env = helpers.runner_environment()
    env.num_envs = 2
    env.cfg.strict_standing_v3 = copy.deepcopy(private.strict_standing_v3)
    env.cfg.reward = copy.deepcopy(private.reward)
    for name in ("zero_default_pose_l1", "zero_joint_motion"):
        getattr(env.cfg.reward, name).params["asset_cfg"].joint_ids = list(range(29))
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: getattr(env.cfg.reward, name))
    env.robot.find_joints = lambda names, preserve_order=True: (
        [helpers.robot_contract.JOINT_NAMES.index(name) for name in names], names)
    env.robot.data.joint_vel = torch.zeros(2, 29)
    env.velocity_feedback_model.value = torch.zeros(2, 29)
    recipe = helpers.runner_recipe()
    recipe["algorithm"]["class_name"] = "DWAQPPO"
    with patch.dict(sys.modules, runner_module._test_stubs):
        runner = runner_module.Elf3DwaqFeedbackStandV3Runner(env, recipe)
    return runner, runner_module


class FeedbackStandV3IntegrationTests(unittest.TestCase):
    def test_direct_parent_recipe_preserved_except_two_costs_and_two_zero_gait_gates(self):
        config, ParentEnv, _, _ = fixture()
        self.assertEqual(config.Elf3DwaqFeedbackStandV3EnvCfg.__bases__, (ParentEnv,))
        before = ParentEnv().to_dict()
        cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
        candidate = cfg.to_dict()
        self.assertEqual(candidate.pop("strict_standing_v3"), contract.stand_settings())
        added = candidate["reward"].pop("zero_default_pose_l1")
        self.assertIs(added["func"], rewards.zero_command_default_pose_l1)
        self.assertEqual(added["weight"], -.1)
        motion = candidate["reward"].pop("zero_joint_motion")
        self.assertIs(motion["func"], rewards.zero_command_joint_motion_l1)
        self.assertEqual(motion["weight"], -.5)
        for name in ("gait_phase_contact", "feet_swing_height"):
            term = candidate["reward"][name]
            self.assertIs(term["func"], getattr(standing.standing_rewards, name + "_when_moving"))
            self.assertEqual(term["params"].pop("command_epsilon"), 1e-6)
            term["func"] = before["reward"][name]["func"]
        self.assertEqual(candidate["scene"]["num_envs"], 16384)
        candidate["scene"]["num_envs"] = before["scene"]["num_envs"]
        self.assertEqual(candidate, before)
        self.assertEqual(ParentEnv().to_dict(), before)
        self.assertEqual(cfg.commands.rel_standing_envs, .2)
        self.assertEqual(cfg.commands.resampling_time_range, (10., 10.))
        self.assertEqual(cfg.scene.max_init_terrain_level, 5)
        self.assertEqual(cfg.robot.gait_phase.period, .8)
        self.assertEqual(cfg.scene.robot.armature, .01)
        self.assertEqual(cfg.velocity_feedback["mode"], "lpf20")
        for name in LEGACY_TERMS:
            self.assertFalse(hasattr(cfg.reward, name))
        for name in ("joint_deviation_hip", "joint_deviation_ankle",
                     "joint_deviation_arms", "joint_deviation_legs"):
            self.assertEqual(getattr(cfg.reward, name).to_dict(), before["reward"][name])
            self.assertNotIn("command_epsilon", getattr(cfg.reward, name).params)

    def test_only_two_new_terms_use_named_all_joints_without_torso_or_settled_params(self):
        config, ParentEnv, _, _ = fixture()
        cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
        self.assertEqual(set(vars(cfg.reward)) - set(vars(ParentEnv().reward)),
                         {"zero_default_pose_l1", "zero_joint_motion"})
        for name, setting, weight in (("zero_default_pose_l1", "default_pose_weight", -.1),
                                      ("zero_joint_motion", "joint_motion_weight", -.5)):
            term = getattr(cfg.reward, name)
            self.assertEqual(set(term.params), {"asset_cfg", "command_epsilon"})
            self.assertEqual(term.params["command_epsilon"], 1e-6)
            entity = term.params["asset_cfg"]
            self.assertEqual(entity.name, "robot")
            self.assertEqual(tuple(entity.joint_names), helpers.robot_contract.JOINT_NAMES)
            self.assertTrue(entity.preserve_order)
            self.assertFalse(getattr(entity, "body_names", None))
            self.assertEqual(term.weight, contract.stand_settings()[setting])
            self.assertEqual(term.weight, weight)

    def test_two_gait_rewards_are_zero_at_strict_zero_and_exact_parent_for_moving_commands(self):
        config, _, _, _ = fixture()
        cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
        env, sensor, asset = standing.reward_environment(
            [[0., 0., 0.], [1e-6, -1e-6, 1e-6], [.1, 0., 0.], [0., .1, 0.], [0., 0., -.5]])
        moving = torch.tensor([False, False, True, True, True])
        for name in ("gait_phase_contact", "feet_swing_height"):
            term = getattr(cfg.reward, name)
            kwargs = dict(term.params)
            kwargs["sensor_cfg"] = sensor
            if name == "feet_swing_height": kwargs["asset_cfg"] = asset
            actual = term.func(env, **kwargs)
            kwargs.pop("command_epsilon")
            original = getattr(standing.base_rewards, name)(env, **kwargs)
            self.assertTrue(torch.equal(actual[~moving], torch.zeros_like(actual[~moving])))
            self.assertTrue(torch.equal(actual[moving], original[moving]))

    def test_zero_phase_preprocessing_commutes_with_real_upper_body_mirror_inputs(self):
        with patch.dict(sys.modules, {"legged_lab.elf3_contract": helpers.robot_contract}):
            symmetry = load("zero_phase_actual_symmetry", ELF3 / "symmetry.py")
        generator = torch.Generator().manual_seed(42)
        obs = torch.randn(5, 100, generator=generator, dtype=torch.float64)
        hist = torch.randn(5, 500, generator=generator, dtype=torch.float64)
        obs[:, 6:9] = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [0., 0., -.5],
                                     [1e-6, -1e-6, 1e-6], [0., .1, 0.]])
        prepared_obs, prepared_hist = inputs.prepare_stand_inputs(obs, hist)
        before_mirror = (symmetry.mirror_observations(prepared_obs), symmetry.mirror_history(prepared_hist))
        after_mirror = inputs.prepare_stand_inputs(symmetry.mirror_observations(obs), symmetry.mirror_history(hist))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_mirror, after_mirror)))

    def test_new_reward_zero_gate_is_additive_and_does_not_write_actual_state_or_inputs(self):
        config, _, _, _ = fixture()
        term = config.Elf3DwaqFeedbackStandV3EnvCfg().reward.zero_default_pose_l1
        term.params["asset_cfg"].joint_ids = list(reversed(range(29)))
        q = torch.arange(7 * 29, dtype=torch.float64).reshape(7, 29) / 100
        home = torch.full_like(q, .123)
        command = torch.tensor([[0., 0., 0.], [1e-6, -1e-6, 1e-6],
                                [.1, 0., 0.], [0., 0., -.1],
                                [1.0001e-6, 0., 0.], [float("nan"), 0., 0.],
                                [float("inf"), 0., 0.]], dtype=torch.float64)
        state = SimpleNamespace(joint_pos=q, default_joint_pos=home)
        env = SimpleNamespace(num_envs=7, scene={"robot": SimpleNamespace(data=state)},
                              command_generator=SimpleNamespace(command=command))
        old_q, old_home, old_command = q.clone(), home.clone(), command.clone()
        actual = term.func(env, **term.params)
        expected = torch.zeros(7, dtype=q.dtype)
        expected[:2] = (q[:2] - home[:2]).abs().sum(-1)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(q, old_q))
        self.assertTrue(torch.equal(home, old_home))
        torch.testing.assert_close(command, old_command, rtol=0, atol=0, equal_nan=True)
        # With 20 ms control the pose term contributes exactly -0.1*cost*0.02.
        torch.testing.assert_close(actual * term.weight * .02, expected * -.002)

    def test_joint_motion_uses_actual_raw_mean_absolute_velocity_with_immediate_zero_gate(self):
        config, _, _, _ = fixture()
        term = config.Elf3DwaqFeedbackStandV3EnvCfg().reward.zero_joint_motion
        term.params["asset_cfg"].joint_ids = list(reversed(range(29)))
        dq = torch.arange(8 * 29, dtype=torch.float64).reshape(8, 29) / 10 - 7.
        command = torch.tensor([[0., 0., 0.], [1e-6, -1e-6, 1e-6], [.1, 0., 0.],
                                [0., 0., -.1], [1.0001e-6, 0., 0.],
                                [float("nan"), 0., 0.], [float("inf"), 0., 0.],
                                [0., 0., 0.]], dtype=torch.float64)
        # No q, default pose, contact, torso, body-speed or settling state exists.
        env = SimpleNamespace(num_envs=8, scene={"robot": SimpleNamespace(
            data=SimpleNamespace(joint_vel=dq))},
            velocity_feedback_model=SimpleNamespace(value=torch.zeros_like(dq)),
            command_generator=SimpleNamespace(command=command))
        before = dq.clone()
        actual = term.func(env, **term.params)
        expected = torch.zeros(8, dtype=dq.dtype)
        expected[[0, 1, 7]] = dq[[0, 1, 7]].abs().mean(-1)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float(actual[0]), 1.)  # No "already settled" enabling threshold.
        self.assertTrue(torch.equal(dq, before))
        torch.testing.assert_close(actual * term.weight * .02, expected * -.01)
        dq.neg_()
        torch.testing.assert_close(term.func(env, **term.params), expected)

    def test_agent_keeps_all_parent_network_ppo_learning_and_upper_body_auxiliary_settings(self):
        config, _, ParentAgent, _ = fixture()
        self.assertEqual(config.Elf3DwaqFeedbackStandV3AgentCfg.__bases__, (ParentAgent,))
        original = ParentAgent().to_dict()
        candidate = config.Elf3DwaqFeedbackStandV3AgentCfg().to_dict()
        for key, value in (("experiment_name", "elf3_dwaq_feedback_stand_v3"),
                           ("wandb_project", "elf3_dwaq_feedback_stand_v3"),
                           ("runner_class_name", "Elf3DwaqFeedbackStandV3Runner")):
            self.assertEqual(candidate[key], value)
            candidate[key] = original[key]
        self.assertEqual(candidate, original)
        self.assertEqual(candidate["upper_body_mirror_loss_coeff"], .1)
        self.assertEqual(candidate["upper_body_pose_loss_coeff"], .02)
        self.assertEqual(candidate["algorithm"]["class_name"], "DWAQPPO")
        self.assertEqual(candidate["algorithm"]["min_learning_rate"], 1e-5)
        self.assertEqual(candidate["algorithm"]["max_learning_rate"], 3e-4)

    def test_version_contract_records_zero_only_fixed_phase_without_complex_objective(self):
        settings = contract.stand_settings()
        self.assertEqual(settings["version"], VERSION)
        self.assertEqual(settings["parent_task"], "elf3_dwaq_feedback_v1")
        self.assertEqual(settings["default_pose_weight"], -.1)
        self.assertEqual(settings["joint_motion_weight"], -.5)
        self.assertTrue(settings["baseline_pose_preserved"])
        self.assertEqual(settings["actor_inputs"], "zero_command_fixed_phase_only")
        self.assertEqual(settings["fixed_phase"], [0., 0., 1., 1.])
        self.assertEqual(settings["phase_scope"], "current_and_all_history_frames")
        self.assertFalse(settings["zero_command_periodic_rewards"])
        self.assertFalse(settings["additional_ppo_losses"])
        for name in ("component_weights", "score_weight", "pose_std",
                     "reset_standing_command_before_first_observation"):
            self.assertNotIn(name, settings)
        identity = contract.stand_experiment_contract(settings)
        self.assertIn("feedback_stand_v3_inputs.py", identity["sources_sha256"])
        for name, digest in identity["sources_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((ELF3 / name).read_bytes()).hexdigest())
        for change in ({"extra": True}, {"default_pose_weight": -1.}, {"joint_motion_weight": -1.},
                       {"command_epsilon": .1},
                       {"actor_inputs": "feedback_v1_unmodified"}, {"baseline_pose_preserved": False},
                       {"fixed_phase": [0., 0., 1., -1.]}, {"zero_command_periodic_rewards": True},
                       {"version": "elf3_dwaq_feedback_stand_v3"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract.stand_experiment_contract(settings | change)
        old = load("minimal_pose_old_still_contract", ELF3 / "feedback_still_v2_contract.py")
        with self.assertRaises(ValueError):
            contract.stand_experiment_contract(old.standing_settings())

    def test_configuration_guard_rejects_any_extra_reward_gate_and_parent_recipe_mutation(self):
        config, _, _, runner = fixture()
        cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
        agent = config.Elf3DwaqFeedbackStandV3AgentCfg().to_dict()
        cfg.scene.num_envs = 64
        agent["max_iterations"], agent["run_name"] = 20, "technical_check"
        with patch.dict(sys.modules, runner._test_stubs):
            audit = runner.validate_stand_configuration(cfg, agent)
            self.assertEqual(audit["initialization"], "from_scratch")
            for kind in ("pose_weight", "motion_weight", "gait", "gait_gate", "group_gate", "group_weight",
                         "command", "sampling", "armature", "terrain", "algorithm", "old_score", "old_settings"):
                e, a = copy.deepcopy(cfg), copy.deepcopy(agent)
                if kind == "pose_weight": e.reward.zero_default_pose_l1.weight = -1.
                elif kind == "motion_weight": e.reward.zero_joint_motion.weight = -1.
                elif kind == "gait": e.reward.gait_phase_contact.weight = .4
                elif kind == "gait_gate": e.reward.gait_phase_contact.params["command_epsilon"] = .1
                elif kind == "group_gate": e.reward.joint_deviation_arms.func = lambda *args: 0.
                elif kind == "group_weight": e.reward.joint_deviation_legs.weight = -.1
                elif kind == "command": e.commands.resampling_time_range = (5., 5.)
                elif kind == "sampling": e.commands.rel_standing_envs = .3
                elif kind == "armature": e.scene.robot.armature = 0.
                elif kind == "terrain": e.scene.max_init_terrain_level = 1
                elif kind == "algorithm": a["algorithm"]["learning_rate"] = .001
                elif kind == "old_score": e.reward.zero_stand_score = copy.deepcopy(e.reward.zero_default_pose_l1)
                elif kind == "old_settings": e.strict_standing_v2 = {"version": "old"}
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    runner.validate_stand_configuration(e, a)

    def test_private_reward_guard_rejects_legacy_stacking_wrong_weight_function_and_named_joints(self):
        config, _, _, runner = fixture()
        for kind in ("legacy", "function", "weight", "motion_weight", "motion_function", "motion_epsilon",
                     "gait_function", "swing_epsilon",
                     "epsilon", "missing_param",
                     "extra_param", "joint_order", "preserve_order"):
            cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
            term = cfg.reward.zero_default_pose_l1
            if kind == "legacy": cfg.reward.zero_stand_score = copy.deepcopy(term)
            elif kind == "function": term.func = lambda *args, **kwargs: 0.
            elif kind == "weight": term.weight = -1.
            elif kind == "motion_weight": cfg.reward.zero_joint_motion.weight = -1.
            elif kind == "motion_function": cfg.reward.zero_joint_motion.func = lambda *args, **kwargs: 0.
            elif kind == "motion_epsilon": cfg.reward.zero_joint_motion.params["command_epsilon"] = .1
            elif kind == "gait_function": cfg.reward.gait_phase_contact.func = standing.base_rewards.gait_phase_contact
            elif kind == "swing_epsilon": cfg.reward.feet_swing_height.params["command_epsilon"] = .1
            elif kind == "epsilon": term.params["command_epsilon"] = .1
            elif kind == "missing_param": term.params.pop("command_epsilon")
            elif kind == "extra_param": term.params["settled"] = True
            elif kind == "joint_order": term.params["asset_cfg"].joint_names.reverse()
            elif kind == "preserve_order": term.params["asset_cfg"].preserve_order = False
            with patch.dict(sys.modules, runner._test_stubs):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    runner.validate_reward_identity(cfg)

    def test_real_isaac_callable_serialization_accepts_valid_gates_but_rejects_same_name_impostors(self):
        # Execute the actual CPU-only serializers rather than the lightweight
        # ConfigNode serializer, which intentionally leaves functions intact.
        utils = ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
        namespace = {"torch": torch, "inspect": inspect, "re": re}
        for filename, function_name in (("string.py", "callable_to_string"),
                                         ("dict.py", "class_to_dict")):
            tree = ast.parse((utils / filename).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == function_name)
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            selected = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
            exec(compile(selected, str(utils / filename), "exec"), namespace)
        config, _, _, runner = fixture()
        with patch.object(helpers.ConfigNode, "to_dict", namespace["class_to_dict"]), \
             patch.dict(sys.modules, runner._test_stubs):
            cfg = config.Elf3DwaqFeedbackStandV3EnvCfg()
            serialized = cfg.reward.to_dict()
            runner.validate_reward_identity(cfg)
            runner.validate_stand_configuration(cfg, config.Elf3DwaqFeedbackStandV3AgentCfg().to_dict())
            for name in ("gait_phase_contact", "feet_swing_height"):
                real_function = getattr(cfg.reward, name).func
                expected_name = f"{real_function.__module__}:{real_function.__name__}"
                self.assertEqual(serialized[name]["func"], expected_name)

                def impostor(*args, **kwargs):
                    raise AssertionError("The identity guard must never execute a substituted reward")

                # A same-name replacement serializes identically. Hash/name
                # matching alone must not bypass live function identity checks.
                impostor.__module__ = real_function.__module__
                impostor.__name__ = real_function.__name__
                impostor.__qualname__ = real_function.__qualname__
                altered = copy.deepcopy(cfg)
                getattr(altered.reward, name).func = impostor
                self.assertIsNot(getattr(altered.reward, name).func, real_function)
                self.assertEqual(altered.reward.to_dict(), serialized)
                with self.subTest(impostor=name), self.assertRaises(ValueError):
                    runner.validate_reward_identity(altered)

    def test_actual_environment_changes_only_actor_returns_not_physics_feedback_or_reset(self):
        env = real_environment()
        parent = env_helpers.helpers.FeedbackEnv
        self.assertEqual(type(env).__bases__, (parent,))
        for method in ("init_buffers", "reset", "compute_current_observations", "_post_physics_step"):
            self.assertIs(getattr(type(env), method), getattr(parent, method))
        tree = ast.parse((ELF3 / "feedback_stand_v3_env.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(methods, {"_prepare", "get_observations", "step"})
        self.assertFalse(hasattr(env, "pop_stand_rollout_counts"))

    def test_zero_fixes_current_and_all_history_phase_only_moving_and_raw_buffers_unchanged(self):
        env = real_environment()
        for command in ([0., 0., 0.], [.1, 0., 0.], [0., 0., -.1]):
            with self.subTest(command=command):
                env.command_generator.command[0] = torch.tensor(command)
                env.raw_observations[:, 6:9] = env.command_generator.command
                original_obs, original_hist = env.raw_observations.clone(), env.raw_history.clone()
                with patch.object(env_helpers.helpers.TensorOnlyBase, "get_observations",
                                  env_helpers.actual_g1_method("get_observations"), create=True):
                    obs, hist = env.get_observations()
                expected_obs, expected_hist = inputs.prepare_stand_inputs(original_obs, original_hist)
                self.assertTrue(torch.equal(obs, expected_obs))
                self.assertTrue(torch.equal(hist, expected_hist))
                if not any(command):
                    self.assertEqual(obs[0, 96:100].tolist(), [0., 0., 1., 1.])
                    self.assertTrue(torch.equal(obs[0, :96], original_obs[0, :96]))
                    self.assertTrue(torch.equal(hist.reshape(2, 5, 100)[0, :, :96],
                                                original_hist.reshape(2, 5, 100)[0, :, :96]))
                else:
                    self.assertTrue(torch.equal(obs, original_obs))
                    self.assertTrue(torch.equal(hist, original_hist))
                self.assertTrue(torch.equal(env.raw_observations, original_obs))
                self.assertTrue(torch.equal(env.raw_history, original_hist))
                self.assertIs(env.extras["observations"]["critic"], env.raw_critic)

    def test_actual_step_preserves_parent_policy_order_and_all_returned_fields(self):
        env = real_environment()
        actions = torch.randn(2, 29, dtype=torch.float64)
        reward, done = torch.tensor([1., 2.]), torch.tensor([False, True])
        seen = []

        def parent_step(parent, simulator_actions):
            seen.append(simulator_actions.clone())
            parent.raw_observations[:, 96:100] = .3
            parent.raw_history.reshape(2, 5, 100)[:, :, 96:100] = .4
            parent.extras = {"observations": {"critic": parent.raw_critic, "obs_hist": parent.raw_history},
                             "log": {"parent": 2.}}
            return parent.raw_observations, reward, done, parent.extras

        with patch.object(env_helpers.helpers.TensorOnlyBase, "step", parent_step, create=True):
            obs, returned_reward, returned_done, extras = env.step(actions)
        self.assertTrue(torch.equal(seen[0], actions[:, env.sim_to_policy]))
        self.assertEqual(obs[0, 96:100].tolist(), [0., 0., 1., 1.])
        self.assertTrue(torch.equal(obs[1], env.raw_observations[1]))
        self.assertTrue(torch.equal(extras["observations"]["obs_hist"][1], env.raw_history[1]))
        self.assertTrue((env.raw_history.reshape(2, 5, 100)[:, :, 96:100] == .4).all())
        self.assertIs(extras["observations"]["critic"], env.raw_critic)
        self.assertIs(returned_reward, reward)
        self.assertIs(returned_done, done)
        self.assertEqual(extras["log"], {"parent": 2.})
        self.assertTrue((env.raw_observations[:, 96:100] == .3).all())

    def test_reset_is_exact_parent_behavior_without_extra_first_frame_command_rewrite(self):
        env = real_environment()
        env.command_generator.command.fill_(.4)
        env.command_generator.is_standing_env = torch.tensor([True, True])
        before_command = env.command_generator.command.clone()
        before_other_feedback = env.velocity_feedback_model.value[1].clone()
        env.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(env.command_generator.command, before_command))
        self.assertTrue(torch.equal(env.velocity_feedback_model.value[0], env.robot.data.joint_vel[0]))
        self.assertTrue(torch.equal(env.velocity_feedback_model.value[1], before_other_feedback))

    def test_actual_runner_contract_rejects_both_old_standing_versions_before_lowlevel_load(self):
        writer, _ = runner_fixture()
        identity = writer.contract()
        self.assertEqual(identity["standing_experiment"]["version"], VERSION)
        self.assertEqual(identity["standing_experiment"]["default_pose_weight"], -.1)
        self.assertEqual(writer.symmetry_arguments["coefficient"], .1)
        self.assertEqual(writer.symmetry_arguments["pose_coefficient"], .02)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.pt"
            writer.current_learning_iteration = 499
            writer.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for version in ("elf3_dwaq_feedback_still_v2", "elf3_dwaq_feedback_stand_v3",
                            "elf3_dwaq_feedback_stand_v3_default_pose_v1"):
                broken = copy.deepcopy(saved)
                broken["infos"]["elf3_contract"]["standing_experiment"]["version"] = version
                invalid = Path(folder) / (version + ".pt")
                torch.save(broken, invalid)
                reader, _ = runner_fixture()
                with self.subTest(version=version), self.assertRaises(ValueError):
                    reader.load(invalid, load_optimizer=False)
                self.assertEqual(reader.lowlevel_load_count, 0)
            reader, _ = runner_fixture()
            reader.load(path, load_optimizer=False)
            self.assertEqual(reader.lowlevel_load_count, 1)
            self.assertEqual(reader.current_learning_iteration, 500)

    def test_historical_v3_is_no_longer_publicly_registered(self):
        tree = ast.parse((ROOT / "TienKung-Lab/legged_lab/envs/__init__.py").read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
                   and node.args and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == "elf3_dwaq_feedback_stand_v3"]
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
