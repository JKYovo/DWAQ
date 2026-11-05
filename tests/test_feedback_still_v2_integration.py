"""Execute the v2 config/contract/env with CPU-only Isaac container stubs."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
ELF3 = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


minimal = load("feedback_still_v2_minimal_helpers", ROOT / "tests/test_minimal_still_integration.py")
env_helpers = load("feedback_still_v2_environment_helpers", ROOT / "tests/test_minimal_still_environment.py")
standing, helpers = minimal.standing, minimal.helpers
rewards = load("feedback_still_v2_actual_rewards", ELF3 / "feedback_still_v2_rewards.py")
contract = load("feedback_still_v2_actual_contract", ELF3 / "feedback_still_v2_contract.py")


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
        "legged_lab.envs.elf3.feedback_still_v2_rewards": rewards,
        "legged_lab.envs.elf3.feedback_still_v2_contract": contract,
        "legged_lab.envs.elf3.standing_rewards": standing.standing_rewards,
        "legged_lab.envs.elf3.feedback_runner": base_runner,
        "legged_lab.envs.elf3.feedback_contract": helpers.contract,
    }
    with patch.dict(sys.modules, stubs):
        config = load("feedback_still_v2_actual_config", ELF3 / "elf3_feedback_still_v2_config.py")
        stubs["legged_lab.envs.elf3.elf3_feedback_still_v2_config"] = config
        with patch.dict(sys.modules, stubs):
            runner = load("feedback_still_v2_actual_runner", ELF3 / "feedback_still_v2_runner.py")
    runner._test_stubs = stubs
    return config, ParentEnv, ParentAgent, runner


def real_environment():
    parent = ModuleType("legged_lab.envs.elf3.elf3_feedback_env")
    parent.Elf3DwaqFeedbackEnv = env_helpers.helpers.FeedbackEnv
    with patch.dict(sys.modules, {
        "legged_lab.envs.elf3.elf3_feedback_env": parent,
        "legged_lab.envs.elf3.stillness": env_helpers.stillness,
    }):
        cls = load("feedback_still_v2_actual_env", ELF3 / "feedback_still_v2_env.py").Elf3DwaqFeedbackStillV2Env
    base = env_helpers.environment()
    env = cls.__new__(cls)
    env.__dict__.update(base.__dict__)
    env.cfg.strict_standing_v2 = contract.standing_settings()
    return env


class FeedbackStillV2IntegrationTests(unittest.TestCase):
    def test_direct_parent_and_exact_moving_recipe_unchanged(self):
        config, ParentEnv, _, _ = fixture()
        self.assertEqual(config.Elf3DwaqFeedbackStillV2EnvCfg.__bases__, (ParentEnv,))
        original = ParentEnv().to_dict()
        cfg = config.Elf3DwaqFeedbackStillV2EnvCfg()
        candidate = cfg.to_dict()
        self.assertEqual(candidate.pop("strict_standing_v2"), contract.standing_settings())
        expected_new = {"zero_full_body_pose": (rewards.full_body_default_pose_l1, -1.),
                        "zero_torso_upright": (rewards.torso_upright, 1.),
                        "zero_joint_motion": (rewards.stand_joint_motion, -.5),
                        "zero_base_stillness": (rewards.stand_base_stillness, 1.)}
        for name, (function, weight) in expected_new.items():
            actual = candidate["reward"].pop(name)
            self.assertIs(actual["func"], function)
            self.assertEqual(actual["weight"], weight)
        for name in ("gait_phase_contact", "feet_swing_height", "joint_deviation_hip",
                     "joint_deviation_ankle", "joint_deviation_arms", "joint_deviation_legs"):
            current = candidate["reward"][name]
            self.assertEqual(current["params"].pop("command_epsilon"), 1e-6)
            if name.startswith("joint_deviation"):
                self.assertIs(current["func"], rewards.joint_deviation_l1_when_moving)
            current["func"] = original["reward"][name]["func"]
        self.assertEqual(candidate, original)
        self.assertEqual(ParentEnv().to_dict(), original)
        for name in ("zero_full_body_pose", "zero_joint_motion"):
            asset = getattr(cfg.reward, name).params["asset_cfg"]
            self.assertEqual(tuple(asset.joint_names), helpers.robot_contract.JOINT_NAMES)
            self.assertTrue(asset.preserve_order)
        self.assertEqual(cfg.scene.max_init_terrain_level, 5)
        self.assertEqual(cfg.robot.gait_phase.period, .8)
        self.assertFalse(any("height" in name or "brak" in name or "stand_leg" in name
                             for name in vars(cfg.reward) if name != "feet_swing_height"))

    def test_agent_changes_identity_only(self):
        config, _, ParentAgent, _ = fixture()
        original = ParentAgent().to_dict()
        candidate = config.Elf3DwaqFeedbackStillV2AgentCfg().to_dict()
        self.assertEqual(config.Elf3DwaqFeedbackStillV2AgentCfg.__bases__, (ParentAgent,))
        for key, value in (("experiment_name", "elf3_dwaq_feedback_still_v2"),
                           ("wandb_project", "elf3_dwaq_feedback_still_v2"),
                           ("runner_class_name", "Elf3DwaqFeedbackStillV2Runner")):
            self.assertEqual(candidate[key], value)
            candidate[key] = original[key]
        self.assertEqual(candidate, original)
        self.assertEqual(candidate["upper_body_mirror_loss_coeff"], .1)
        self.assertEqual(candidate["upper_body_pose_loss_coeff"], .02)
        self.assertEqual(candidate["policy"]["context_mode"], "mean")
        self.assertEqual(candidate["algorithm"]["class_name"], "DWAQPPO")

    def test_contract_is_source_bound_and_rejects_unknown_or_changed_settings(self):
        settings = contract.standing_settings()
        identity = contract.standing_experiment_contract(settings)
        self.assertEqual(identity["parent_task"], "elf3_dwaq_feedback_v1")
        self.assertTrue(identity["replace_zero_group_pose_terms"])
        self.assertFalse(identity["additional_ppo_losses"])
        self.assertFalse(identity["lock_world_yaw"])
        for name, digest in identity["sources_sha256"].items():
            self.assertEqual(hashlib.sha256((ELF3 / name).read_bytes()).hexdigest(), digest)
        for change in ({"extra": True}, {"full_body_pose_weight": -.1}, {"command_epsilon": .1},
                       {"lock_world_yaw": True}, {"replace_zero_group_pose_terms": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract.standing_experiment_contract(settings | change)

    def test_exact_recipe_audit_allows_only_runtime_controls(self):
        config, _, _, runner = fixture()
        cfg, agent = config.Elf3DwaqFeedbackStillV2EnvCfg(), config.Elf3DwaqFeedbackStillV2AgentCfg().to_dict()
        cfg.scene.num_envs = 64
        agent["max_iterations"], agent["run_name"] = 20, "smoke"
        with patch.dict(sys.modules, runner._test_stubs):
            self.assertEqual(runner.validate_standing_configuration(cfg, agent)["initialization"], "from_scratch")
            for kind in ("pose", "gait", "command", "armature", "terrain", "algorithm"):
                e, a = copy.deepcopy(cfg), copy.deepcopy(agent)
                if kind == "pose": e.reward.zero_full_body_pose.weight = -2.
                elif kind == "gait": e.reward.gait_phase_contact.weight = .4
                elif kind == "command": e.commands.resampling_time_range = (5., 5.)
                elif kind == "armature": e.scene.robot.armature = 0.
                elif kind == "terrain": e.scene.max_init_terrain_level = 1
                elif kind == "algorithm": a["algorithm"]["learning_rate"] = .001
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    runner.validate_standing_configuration(e, a)

    def test_actual_env_zero_phase_moving_and_reset_preserve_raw_history(self):
        env = real_environment()
        raw_obs, raw_history = env.raw_observations.clone(), env.raw_history.clone()
        with patch.object(env_helpers.helpers.TensorOnlyBase, "get_observations",
                          env_helpers.actual_g1_method("get_observations"), create=True):
            obs, hist = env.get_observations()
            self.assertEqual(obs[0, 96:100].tolist(), list(env_helpers.stillness.FIXED_PHASE))
            self.assertTrue(torch.equal(hist.reshape(2, 5, 100)[0, :, 96:100],
                torch.tensor(env_helpers.stillness.FIXED_PHASE, dtype=hist.dtype).expand(5, 4)))
            self.assertTrue(torch.equal(obs[1], raw_obs[1]))
            self.assertTrue(torch.equal(hist[1], raw_history[1]))
            self.assertTrue(torch.equal(env.raw_observations, raw_obs))
            self.assertTrue(torch.equal(env.raw_history, raw_history))
            env.command_generator.command[0] = torch.tensor([0., 0., .1])
            env.raw_observations[:, 6:9] = env.command_generator.command
            env.reset(torch.tensor([0]))
            obs, hist = env.get_observations()
            self.assertTrue(torch.equal(obs, env.raw_observations))
            self.assertTrue(torch.equal(hist, env.raw_history))

    def test_actual_env_step_only_transforms_actor_return_after_parent(self):
        env = real_environment()
        actions = torch.randn(2, 29, dtype=torch.float64)
        rewards_tensor = torch.tensor([1., 2.])
        dones = torch.tensor([False, True])
        sent = []
        def parent_step(parent, simulator_actions):
            sent.append(simulator_actions.clone())
            parent.raw_observations[:, 96:100] = .3
            parent.raw_history.reshape(2, 5, 100)[:, :, 96:100] = .4
            parent.extras = {"observations": {"critic": parent.raw_critic,
                                              "obs_hist": parent.raw_history}, "log": {"parent": 2.}}
            return parent.raw_observations, rewards_tensor, dones, parent.extras
        with patch.object(env_helpers.helpers.TensorOnlyBase, "step", parent_step, create=True):
            obs, reward, done, extras = env.step(actions)
        self.assertTrue(torch.equal(sent[0], actions[:, env.sim_to_policy]))
        self.assertIs(reward, rewards_tensor)
        self.assertIs(done, dones)
        self.assertIs(extras["observations"]["critic"], env.raw_critic)
        self.assertEqual(extras["log"], {"parent": 2.})
        self.assertEqual(obs[0, 96:100].tolist(), list(env_helpers.stillness.FIXED_PHASE))
        self.assertTrue(torch.equal(obs[1], env.raw_observations[1]))
        self.assertTrue(torch.equal(extras["observations"]["obs_hist"][1], env.raw_history[1]))
        self.assertTrue((env.raw_observations[:, 96:100] == .3).all())
        self.assertTrue((env.raw_history.reshape(2, 5, 100)[:, :, 96:100] == .4).all())

    def test_historical_feedback_still_v2_is_no_longer_publicly_registered(self):
        tree = ast.parse((ROOT / "TienKung-Lab/legged_lab/envs/__init__.py").read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
                   and node.args and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == "elf3_dwaq_feedback_still_v2"]
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
