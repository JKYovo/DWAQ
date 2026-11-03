"""CPU guards for the opt-in walking task's zero-command gait reward fix.

The real inherited reward functions, private wrappers, contracts and ELF3
runner methods execute. Simulator configuration containers and bottom-level
checkpoint storage are mocked; no Isaac import, GPU, training or robot runs.
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


class EntityCfg:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


def package(name):
    result = ModuleType(name)
    result.__path__ = []
    return result


def load_real_rewards():
    source = LAB / "mdp/rewards.py"
    selected = {"gait_phase_contact", "feet_swing_height"}
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in functions} == selected
    module = ModuleType("legged_lab.mdp.rewards")
    module.__file__ = str(source)
    module.torch = torch
    module.SceneEntityCfg = EntityCfg
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])),
                 str(source), "exec"), module.__dict__)
    return module


base_rewards = load_real_rewards()
standing_contract = load_file("standing_cpu_contract", ELF3 / "standing_contract.py")
helpers = load_file("standing_cpu_feedback_helpers", ROOT / "tests/test_feedback_integration.py")
play = load_file("standing_cpu_play_validator", LAB / "utils/elf3_play_contract.py")


def import_stubs():
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = EntityCfg
    mdp = package("legged_lab.mdp")
    mdp.rewards = base_rewards
    return {
        "isaaclab": package("isaaclab"), "isaaclab.managers": managers,
        "legged_lab": package("legged_lab"), "legged_lab.envs": package("legged_lab.envs"),
        "legged_lab.envs.elf3": package("legged_lab.envs.elf3"),
        "legged_lab.mdp": mdp, "legged_lab.mdp.rewards": base_rewards,
        "legged_lab.envs.elf3.standing_contract": standing_contract,
    }


with patch.dict(sys.modules, import_stubs()):
    standing_rewards = load_file("standing_cpu_actual_rewards", ELF3 / "standing_rewards.py")


def load_real_standing_runner():
    feedback_runner = ModuleType("legged_lab.envs.elf3.feedback_runner")
    feedback_runner.Elf3DwaqFeedbackRunner = helpers.Runner
    with patch.dict(sys.modules, import_stubs() | {
        "legged_lab.envs.elf3.feedback_runner": feedback_runner,
        "legged_lab.envs.elf3.standing_rewards": standing_rewards,
    }):
        return load_file("standing_cpu_actual_runner", ELF3 / "standing_runner.py").Elf3DwaqStandRunner


Runner = load_real_standing_runner()


def settings():
    return {"version": "elf3_stand_v1", "mode": "walking_zero_command_fix", "command_epsilon": 1e-6}


class Scene(dict):
    pass


def reward_environment(commands):
    commands = torch.tensor(commands, dtype=torch.float64)
    count = len(commands)
    phases = torch.arange(count, dtype=torch.float64) / max(1, count)
    forces = torch.zeros(count, 4, 3, dtype=torch.float64)
    positions = torch.zeros(count, 8, 3, dtype=torch.float64)
    for index in range(count):
        # Sensor and articulation IDs deliberately differ. Include all four
        # contact patterns and terrain-dependent world heights.
        if index % 4 in (0, 2): forces[index, 1, 2] = 100.
        if index % 4 in (0, 1): forces[index, 3, 2] = 100.
        positions[index, 2, 2] = -.05 + .07 * index
        positions[index, 6, 2] = .20 - .03 * index
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
    robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=positions))
    scene = Scene(robot=robot)
    scene.sensors = {"contact_sensor": sensor}
    env = SimpleNamespace(scene=scene, command_generator=SimpleNamespace(command=commands),
                          leg_phase=torch.stack((phases, (phases + .5) % 1), dim=1))
    return env, EntityCfg("contact_sensor", body_ids=[1, 3]), EntityCfg("robot", body_ids=[2, 6])


def runner_environment():
    env = helpers.runner_environment()
    env.cfg.standing_experiment = settings()
    env.cfg.reward.gait_phase_contact = helpers.ConfigNode(
        func=standing_rewards.gait_phase_contact_when_moving, weight=.2,
        params={"sensor_cfg": helpers.ConfigNode(name="contact_sensor", body_ids=[0, 1]),
                "stance_threshold": .55, "command_epsilon": 1e-6})
    env.cfg.reward.feet_swing_height = helpers.ConfigNode(
        func=standing_rewards.feet_swing_height_when_moving, weight=-.2,
        params={"sensor_cfg": helpers.ConfigNode(name="contact_sensor", body_ids=[0, 1]),
                "asset_cfg": helpers.ConfigNode(name="robot", body_ids=[0, 1]),
                "target_height": .08, "command_epsilon": 1e-6})
    return env


class CommandGateTests(unittest.TestCase):
    def test_exact_zero_tolerance_low_speed_reverse_and_pure_yaw(self):
        commands = [[0, 0, 0], [1e-6, -1e-6, 1e-6], [-1e-6, 0, 0],
                    [1.00001e-6, 0, 0], [0, -1.00001e-6, 0], [0, 0, -1.00001e-6],
                    [.1, 0, 0], [-.1, 0, 0], [0, .1, 0], [0, -.1, 0],
                    [0, 0, .1], [0, 0, -.1]]
        env, _, _ = reward_environment(commands)
        actual = standing_rewards.moving_command_mask(env)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(tuple(actual.shape), (len(commands),))
        self.assertEqual(actual.tolist(), [False, False, False] + [True] * 9)

    def test_only_first_three_command_components_are_used(self):
        env, _, _ = reward_environment([[0, 0, 0, 1], [0, 0, .1, 0]])
        self.assertEqual(standing_rewards.moving_command_mask(env).tolist(), [False, True])

    def test_all_zero_commands_disable_both_terms_at_every_phase(self):
        env, sensor, asset = reward_environment([[0, 0, 0]] * 40)
        for function, kwargs in (
            (standing_rewards.gait_phase_contact_when_moving, {"sensor_cfg": sensor}),
            (standing_rewards.feet_swing_height_when_moving, {"sensor_cfg": sensor, "asset_cfg": asset}),
        ):
            with self.subTest(function=function.__name__):
                self.assertTrue(torch.equal(function(env, **kwargs), torch.zeros(40, dtype=torch.float64)))

    def test_mixed_batch_retains_exact_parent_reward_and_parameter_forwarding(self):
        env, sensor, asset = reward_environment([[0, 0, 0], [.1, 0, 0], [0, 0, -.1],
                                                [0, -.3, 0], [1e-6, 0, 0], [-.6, .2, .5]])
        mask = standing_rewards.moving_command_mask(env)
        cases = (
            (standing_rewards.gait_phase_contact_when_moving, base_rewards.gait_phase_contact,
             {"sensor_cfg": sensor, "stance_threshold": .73}),
            (standing_rewards.feet_swing_height_when_moving, base_rewards.feet_swing_height,
             {"sensor_cfg": sensor, "asset_cfg": asset, "target_height": .12}),
        )
        for gated, original, kwargs in cases:
            with self.subTest(function=gated.__name__):
                expected = original(env, **kwargs)
                actual = gated(env, **kwargs)
                self.assertTrue(torch.equal(actual[mask], expected[mask]))
                self.assertTrue(torch.equal(actual[~mask], torch.zeros_like(expected[~mask])))

    def test_custom_epsilon_is_forwarded_without_hidden_low_speed_deadband(self):
        env, sensor, asset = reward_environment([[.01, 0, 0], [.1, 0, 0], [.100001, 0, 0], [0, 0, -.2]])
        self.assertEqual(standing_rewards.moving_command_mask(env, command_epsilon=.1).tolist(),
                         [False, False, True, True])
        for gated, original, kwargs in (
            (standing_rewards.gait_phase_contact_when_moving, base_rewards.gait_phase_contact,
             {"sensor_cfg": sensor}),
            (standing_rewards.feet_swing_height_when_moving, base_rewards.feet_swing_height,
             {"sensor_cfg": sensor, "asset_cfg": asset}),
        ):
            actual = gated(env, command_epsilon=.1, **kwargs)
            self.assertTrue(torch.equal(actual[:2], torch.zeros_like(actual[:2])))
            self.assertTrue(torch.equal(actual[2:], original(env, **kwargs)[2:]))

    def test_calls_do_not_mutate_inputs_and_command_resets_leave_no_state(self):
        env, sensor, asset = reward_environment([[0, 0, 0], [.1, 0, 0], [0, 0, -.1], [0, -.2, 0]])
        command = env.command_generator.command.clone()
        phase = env.leg_phase.clone()
        force = env.scene.sensors["contact_sensor"].data.net_forces_w.clone()
        positions = env.scene["robot"].data.body_pos_w.clone()
        before_keys = set(vars(env))
        first_contact = standing_rewards.gait_phase_contact_when_moving(env, sensor)
        first_swing = standing_rewards.feet_swing_height_when_moving(env, sensor, asset)
        env.command_generator.command.zero_()
        self.assertFalse(standing_rewards.gait_phase_contact_when_moving(env, sensor).any())
        self.assertFalse(standing_rewards.feet_swing_height_when_moving(env, sensor, asset).any())
        env.command_generator.command.copy_(command)
        self.assertTrue(torch.equal(standing_rewards.gait_phase_contact_when_moving(env, sensor), first_contact))
        self.assertTrue(torch.equal(standing_rewards.feet_swing_height_when_moving(env, sensor, asset), first_swing))
        self.assertEqual(set(vars(env)), before_keys)
        self.assertTrue(torch.equal(env.leg_phase, phase))
        self.assertTrue(torch.equal(env.scene.sensors["contact_sensor"].data.net_forces_w, force))
        self.assertTrue(torch.equal(env.scene["robot"].data.body_pos_w, positions))

    def test_existing_phase_incentive_is_removed_not_the_phase_clock(self):
        env, sensor, _ = reward_environment([[0, 0, 0]] * 40)
        force = env.scene.sensors["contact_sensor"].data.net_forces_w
        force[:, [1, 3], 2] = 100.
        old_double_support = base_rewards.gait_phase_contact(env, sensor)
        self.assertAlmostEqual(float(old_double_support.mean()), 1.1)
        self.assertTrue(torch.equal(standing_rewards.gait_phase_contact_when_moving(env, sensor),
                                    torch.zeros(40, dtype=torch.float64)))
        force[:, [1, 3], 2] = 100. * (env.leg_phase < .55).to(force.dtype)
        self.assertTrue(torch.equal(base_rewards.gait_phase_contact(env, sensor), torch.full((40,), 2.)))
        self.assertFalse(standing_rewards.gait_phase_contact_when_moving(env, sensor).any())
        self.assertGreater(float(env.leg_phase.std()), 0.)


class StandingContractTests(unittest.TestCase):
    def test_exact_settings_source_hashes_and_walking_semantics(self):
        source = settings()
        original = copy.deepcopy(source)
        result = standing_contract.standing_experiment_contract(source)
        self.assertEqual(source, original)
        self.assertEqual(result["version"], standing_contract.VERSION)
        self.assertEqual(result["mode"], standing_contract.MODE)
        self.assertEqual(result["command_epsilon"], standing_contract.COMMAND_EPSILON)
        self.assertIs(result["zero_command_gait_rewards_disabled"], True)
        self.assertIs(result["supports_walking"], True)
        self.assertNotIn("zero_command_only", result)
        self.assertEqual(result["reward_source_sha256"], hashlib.sha256((ELF3 / "standing_rewards.py").read_bytes()).hexdigest())
        self.assertEqual(result["base_reward_source_sha256"], hashlib.sha256((LAB / "mdp/rewards.py").read_bytes()).hexdigest())

    def test_old_unknown_incomplete_and_nonfinite_settings_are_rejected(self):
        invalid = [None, {}, {"version": "elf3_stand_v1"}, settings() | {"unexpected": True},
                   settings() | {"version": "old"}, settings() | {"mode": "standing_only"}]
        invalid.extend(settings() | {"command_epsilon": value}
                       for value in (0, -1e-6, 1e-5, True, None, float("nan"), float("inf")))
        for value in invalid:
            with self.subTest(settings=value), self.assertRaises((ValueError, TypeError)):
                standing_contract.standing_experiment_contract(value)


class StandingRunnerTests(unittest.TestCase):
    def test_actual_save_resume_and_unchanged_feedback_observation_deployment(self):
        runner = Runner(runner_environment(), helpers.runner_recipe())
        identity = runner.contract()
        self.assertEqual(identity["standing_experiment"], standing_contract.standing_experiment_contract(settings()))
        feedback = identity["feedback_experiment"]
        self.assertEqual(feedback["velocity_feedback"]["mode"], "lpf20")
        self.assertEqual(feedback["actor_context"], "encoder_mean_velocity_and_mean_latent")
        self.assertEqual(feedback["observation"]["flat_size"], 500)
        self.assertEqual((identity["num_obs"], identity["history_length"], len(identity["joint_names"])), (100, 5, 29))
        self.assertFalse(feedback["deployment"]["extra_velocity_filter"])
        self.assertFalse(feedback["deployment"]["extra_action_filter"])
        runner.current_learning_iteration = 499
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model_499.pt"
            extra = {"retained": "yes"}
            runner.save(checkpoint, extra)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            reader = Runner(runner_environment(), helpers.runner_recipe())
            reader.load(checkpoint, load_optimizer=False)
        self.assertEqual(extra, {"retained": "yes"})
        self.assertEqual(saved["infos"]["elf3_contract"], identity)
        self.assertEqual(saved["infos"]["training_recipe"], runner.cfg)
        self.assertEqual(reader.current_learning_iteration, 500)
        self.assertEqual(reader.lowlevel_load_count, 1)
        self.assertFalse(reader.last_load_optimizer)

    def test_runner_rejects_missing_private_functions_or_epsilon_before_load(self):
        for name in ("gait_phase_contact", "feet_swing_height"):
            for alteration in ("function", "epsilon"):
                with self.subTest(term=name, alteration=alteration):
                    env = runner_environment()
                    term = getattr(env.cfg.reward, name)
                    if alteration == "function": term.func = getattr(base_rewards, name)
                    else: term.params["command_epsilon"] = .1
                    with self.assertRaises(ValueError):
                        Runner(env, helpers.runner_recipe())

    def test_new_and_original_feedback_runners_reject_each_others_checkpoints(self):
        # Give the old runner identical reward/config objects to isolate the
        # new top-level identity rather than relying only on reward hash drift.
        new = Runner(runner_environment(), helpers.runner_recipe())
        old = helpers.Runner(runner_environment(), helpers.runner_recipe())
        self.assertEqual(new.contract()["feedback_experiment"], old.contract()["feedback_experiment"])
        with tempfile.TemporaryDirectory() as folder:
            new_path, old_path = Path(folder) / "new.pt", Path(folder) / "old.pt"
            new.save(new_path); old.save(old_path)
            with self.assertRaises(ValueError): new.load(old_path)
            with self.assertRaises(ValueError): old.load(new_path)
        self.assertEqual(new.lowlevel_load_count, 0)
        self.assertEqual(old.lowlevel_load_count, 0)

    def test_play_allows_recipe_only_change_but_not_old_or_altered_identity(self):
        identity = Runner(runner_environment(), helpers.runner_recipe()).contract()
        saved = {"iter": 499, "infos": {"elf3_contract": copy.deepcopy(identity)}}
        expected = copy.deepcopy(identity)
        expected["feedback_experiment"]["training_recipe_sha256"] = "f" * 64
        before = copy.deepcopy((saved, expected))
        play.validate_play_checkpoint(saved, expected)
        self.assertEqual((saved, expected), before)
        old = copy.deepcopy(identity); old.pop("standing_experiment")
        for actual, wanted in ((old, identity), (identity, old)):
            with self.subTest(old_side=actual is old), self.assertRaises(ValueError):
                play.validate_play_checkpoint({"iter": 1, "infos": {"elf3_contract": actual}}, wanted)
        for key, value in (("mode", "standing_only"), ("command_epsilon", .1),
                           ("reward_source_sha256", "0" * 64), ("base_reward_source_sha256", "1" * 64)):
            with self.subTest(field=key):
                altered = copy.deepcopy(identity); altered["standing_experiment"][key] = value
                with self.assertRaises(ValueError):
                    play.validate_play_checkpoint({"iter": 1, "infos": {"elf3_contract": altered}}, identity)


def stub_configclass(cls):
    """Copy configuration defaults per instance; never import simulator config."""
    def initialize(self, **overrides):
        for parent in reversed(type(self).__mro__):
            for key, value in vars(parent).items():
                if not key.startswith("_") and not callable(value):
                    setattr(self, key, copy.deepcopy(value))
        for key, value in overrides.items(): setattr(self, key, value)
        self.__post_init__()
    cls.__init__ = initialize
    return cls


def configuration_classes():
    @stub_configclass
    class FeedbackEnv(helpers.ConfigNode):
        reward = helpers.ConfigNode(
            gait_phase_contact=helpers.ConfigNode(func=base_rewards.gait_phase_contact, weight=.2,
                params={"sensor_cfg": "feet", "stance_threshold": .55}),
            feet_swing_height=helpers.ConfigNode(func=base_rewards.feet_swing_height, weight=-.2,
                params={"sensor_cfg": "feet", "asset_cfg": "feet", "target_height": .08}),
            feet_slide=helpers.ConfigNode(weight=-.25), track=helpers.ConfigNode(weight=2., params={"std": .5}))
        velocity_feedback = helpers.feedback_settings()
        robot = helpers.ConfigNode(gait_phase=helpers.ConfigNode(enable=True, period=.8, offset=.5),
            dwaq_obs_history_length=5, actor_obs_history_length=1)
        commands = helpers.ConfigNode(resampling_time_range=(10., 10.), rel_standing_envs=.2)
        scene = helpers.ConfigNode(terrain="unchanged", robot="unchanged")
        domain_rand = helpers.ConfigNode(action_delay="unchanged")
        def __post_init__(self): pass

    @stub_configclass
    class FeedbackAgent(helpers.ConfigNode):
        policy = helpers.ConfigNode(context_mode="mean")
        algorithm = helpers.ConfigNode(learning_rate=1e-4, min_learning_rate=1e-5, max_learning_rate=3e-4)
        experiment_name = wandb_project = "elf3_dwaq_feedback_v1"
        runner_class_name = "Elf3DwaqFeedbackRunner"
        upper_body_mirror_loss_coeff = .1
        upper_body_pose_loss_coeff = .02
        max_iterations = 100001
        save_interval = 500
        def __post_init__(self): pass

    utils = ModuleType("isaaclab.utils"); utils.configclass = stub_configclass
    feedback_config = ModuleType("legged_lab.envs.elf3.elf3_feedback_config")
    feedback_config.Elf3DwaqFeedbackEnvCfg = FeedbackEnv
    feedback_config.Elf3DwaqFeedbackAgentCfg = FeedbackAgent
    with patch.dict(sys.modules, import_stubs() | {
        "isaaclab.utils": utils, "legged_lab.envs.elf3.elf3_feedback_config": feedback_config,
        "legged_lab.envs.elf3.standing_rewards": standing_rewards,
    }):
        actual = load_file("standing_cpu_actual_config", ELF3 / "elf3_stand_config.py")
    return actual, FeedbackEnv, FeedbackAgent


class StandingConfigurationTests(unittest.TestCase):
    def test_only_two_terms_and_independent_identity_change(self):
        module, Env, Agent = configuration_classes()
        old = Env(); old_before = copy.deepcopy(old.to_dict())
        cfg = module.Elf3DwaqStandEnvCfg()
        altered = cfg.to_dict(); self.assertEqual(altered.pop("standing_experiment"), settings())
        for name in ("gait_phase_contact", "feet_swing_height"):
            term = altered["reward"][name]
            self.assertIs(term["func"], getattr(standing_rewards, name + "_when_moving"))
            term["func"] = old_before["reward"][name]["func"]
            self.assertEqual(term["params"].pop("command_epsilon"), 1e-6)
        self.assertEqual(altered, old_before)
        self.assertEqual(old.to_dict(), old_before)
        self.assertEqual(Env().to_dict(), old_before)
        self.assertEqual(cfg.robot.gait_phase.period, .8)
        self.assertTrue(cfg.robot.gait_phase.enable)
        agent = module.Elf3DwaqStandAgentCfg().to_dict(); original = Agent().to_dict()
        for key, value in (("experiment_name", "elf3_dwaq_stand_v1"),
                           ("wandb_project", "elf3_dwaq_stand_v1"),
                           ("runner_class_name", "Elf3DwaqStandRunner")):
            self.assertEqual(agent[key], value); agent[key] = original[key]
        self.assertEqual(agent, original)

    def test_task_registration_uses_feedback_environment_and_own_agent(self):
        tree = ast.parse((LAB / "envs/__init__.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
                 and node.args and isinstance(node.args[0], ast.Constant)
                 and node.args[0].value == "elf3_dwaq_stand_v1"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[1]), "Elf3DwaqFeedbackEnv")
        self.assertEqual(ast.unparse(calls[0].args[2]), "Elf3DwaqStandEnvCfg()")
        self.assertEqual(ast.unparse(calls[0].args[3]), "Elf3DwaqStandAgentCfg()")


if __name__ == "__main__":
    unittest.main()
