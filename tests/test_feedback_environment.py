"""CPU-only feedback/lifecycle tests; never import or start Isaac Sim.

The environment tests run the actual ELF3 class and the actual base observation
method against a tiny tensor-only parent/robot. Physics-hook placement is checked
from the G1 AST. These are not substitutes for an Isaac Sim smoke test.
"""
import ast
import cmath
import importlib.util
import math
from pathlib import Path
import sys
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


feedback = load_file("isolated_tensor_feedback", LAB / "envs/elf3/feedback.py")
contract = load_file("isolated_elf3_feedback_contract", LAB / "elf3_contract.py")


def actual_base_observation_method():
    source = ast.parse((LAB / "envs/g1/g1_dwaq_env.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "G1DwaqEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "compute_current_observations")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual G1 observation method>", "exec"), namespace)
    return namespace[method.name]


class TensorOnlyBase:
    """Only lifecycle plumbing is mocked; observation layout comes from G1 source."""

    compute_current_observations = actual_base_observation_method()

    def init_buffers(self):
        self.num_actions = 29
        self.dwaq_obs_history_length = self.cfg.robot.dwaq_obs_history_length
        self.feet_cfg = SimpleNamespace(body_ids=[0, 2])
        self.feet_pos_in_body = torch.arange(self.num_envs * 6, dtype=torch.float64).reshape(self.num_envs, 2, 3)
        self.feet_vel_in_body = self.feet_pos_in_body + 10
        self.obs_scales = SimpleNamespace(ang_vel=2.0, projected_gravity=1.0, commands=1.0,
                                         joint_pos=1.0, joint_vel=.05, actions=1.0,
                                         lin_vel=2.0, feet_pos=1.0, feet_vel=.1, contact_force=.01)
        self.action_buffer = SimpleNamespace(_circular_buffer=SimpleNamespace(
            buffer=torch.arange(self.num_envs * 29, dtype=torch.float64).reshape(self.num_envs, 1, 29)))
        self.leg_phase = torch.tensor([[.1, .6]] * self.num_envs, dtype=torch.float64)
        self.init_obs_buffer()

    def init_obs_buffer(self):
        # This is the construction-time dynamic dispatch that requires the new
        # model to be initialized before calling the actual Elf3.init_buffers.
        actor, critic = self.compute_current_observations()
        self._num_obs = actor.shape[-1]
        self._num_privileged_obs = critic.shape[-1]

    def _compute_feet_state(self):
        pass

    def _post_physics_step(self):
        pass

    def reset(self, env_ids):
        # Stand in for reset events + write_data_to_sim + sim.forward. A correct
        # subclass must initialize from these new values, not the old episode.
        self.robot.data.joint_vel[env_ids] = -4.0


def load_environment_classes():
    base_module = ModuleType("legged_lab.envs.g1.g1_dwaq_env")
    base_module.G1DwaqEnv = TensorOnlyBase
    modules = {"legged_lab.envs.g1.g1_dwaq_env": base_module,
               "legged_lab.elf3_contract": contract,
               "legged_lab.envs.elf3.feedback": feedback}
    with patch.dict(sys.modules, modules):
        parent = load_file("isolated_real_elf3_env", LAB / "envs/elf3/elf3_dwaq_env.py")
        parent_module = ModuleType("legged_lab.envs.elf3.elf3_dwaq_env")
        parent_module.Elf3DwaqEnv = parent.Elf3DwaqEnv
        with patch.dict(sys.modules, {"legged_lab.envs.elf3.elf3_dwaq_env": parent_module}):
            child = load_file("isolated_real_feedback_env", LAB / "envs/elf3/elf3_feedback_env.py")
    return parent.Elf3DwaqEnv, child.Elf3DwaqFeedbackEnv


ParentEnv, FeedbackEnv = load_environment_classes()


def make_environment(mode="lpf20", feet_info=True, feet_force=True, latest=True, history=5):
    env = FeedbackEnv.__new__(FeedbackEnv)
    env.num_envs, env.device, env.physics_dt = 2, "cpu", .005
    env.cfg = SimpleNamespace(
        velocity_feedback={"mode": mode, "cutoff_hz": 20.0,
                           "model_version": feedback.FEEDBACK_MODEL_VERSION,
                           "latest_critic_contacts": latest},
        robot=SimpleNamespace(dwaq_obs_history_length=history, gait_phase=SimpleNamespace(enable=True)),
        scene=SimpleNamespace(height_scanner=SimpleNamespace(enable_height_scan=True, critic_only=True),
                              privileged_info=SimpleNamespace(enable_feet_info=feet_info,
                                                              enable_feet_contact_force=feet_force,
                                                              enable_root_height=True)))
    # Deliberately non-identity PhysX and different sensor/articulation body order.
    sim_names = list(contract.JOINT_NAMES[::2] + contract.JOINT_NAMES[1::2])
    joint_vel = torch.arange(58, dtype=torch.float64).reshape(2, 29) / 10
    robot = SimpleNamespace(
        joint_names=sim_names, body_names=[contract.CONTRACT["physical_root"], *reversed(contract.FOOT_BODIES)],
        data=SimpleNamespace(joint_vel=joint_vel, default_joint_vel=torch.ones_like(joint_vel) * .13,
                             joint_pos=joint_vel + .2, default_joint_pos=torch.ones_like(joint_vel) * .1,
                             root_ang_vel_b=torch.ones(2, 3, dtype=torch.float64),
                             projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2, dtype=torch.float64),
                             root_lin_vel_b=torch.ones(2, 3, dtype=torch.float64) * .3,
                             root_pos_w=torch.tensor([[0., 0., 1.05]] * 2, dtype=torch.float64)))
    robot.find_joints = lambda names, preserve_order: ([sim_names.index(name) for name in names], names)
    robot.find_bodies = lambda names, preserve_order: ([robot.body_names.index(name) for name in names], names)
    env.robot = robot
    sensor_names = [contract.FOOT_BODIES[0], "some_other_collision_link", contract.FOOT_BODIES[1]]
    forces = torch.arange(54, dtype=torch.float64).reshape(2, 3, 3, 3) + 1
    env.contact_sensor = SimpleNamespace(body_names=sensor_names,
                                        data=SimpleNamespace(net_forces_w_history=forces))
    env.contact_sensor.find_bodies = lambda names, preserve_order: (
        [sensor_names.index(name) for name in names], names)
    env.command_generator = SimpleNamespace(command=torch.ones(2, 3, dtype=torch.float64) * .1)
    env.init_buffers()
    return env


class TensorFeedbackTests(unittest.TestCase):
    def test_prewarped_transfer_function(self):
        b0, b1, a1 = feedback.tustin_lowpass_coefficients(20, .005)
        self.assertAlmostEqual(b0, .24523727525278555, places=14)
        self.assertEqual(b0, b1)
        self.assertAlmostEqual(a1, -.5095254494944288, places=14)
        z_inverse = cmath.exp(-2j * math.pi * 20 * .005)
        transfer = (b0 + b1 * z_inverse) / (1 + a1 * z_inverse)
        self.assertAlmostEqual(abs(transfer), 1 / math.sqrt(2), places=14)
        self.assertAlmostEqual(cmath.phase(transfer), -math.pi / 4, places=14)
        self.assertAlmostEqual((b0 + b1) / (1 + a1), 1.0, places=14)
        # A fixed delay has unit gain, unlike this measured-feedback hypothesis.
        self.assertNotAlmostEqual(abs(transfer), 1.0, places=3)

    def test_sinusoid_matches_transfer_function(self):
        model = feedback.JointVelocityFeedback(torch.zeros(1, 1, dtype=torch.float64), .005, "lpf20")
        time = torch.arange(2000, dtype=torch.float64) * .005
        signal = torch.sin(2 * math.pi * 20 * time)
        outputs = [model.update(value.reshape(1, 1)).clone().reshape(()) for value in signal]
        actual = torch.stack(outputs)[500:]
        expected = .5 * torch.sin(2 * math.pi * 20 * time[500:]) - .5 * torch.cos(2 * math.pi * 20 * time[500:])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_ideal_is_exact_and_detached(self):
        model = feedback.JointVelocityFeedback(torch.zeros(2, 29), .005, "ideal")
        value = torch.randn(2, 29, requires_grad=True)
        result = model.update(value)
        self.assertTrue(torch.equal(result, value))
        self.assertFalse(result.requires_grad)
        self.assertEqual(model.update_count, 1)

    def test_reset_is_per_environment_and_uses_current_velocity(self):
        model = feedback.JointVelocityFeedback(torch.zeros(2, 2), .005, "lpf20")
        model.update(torch.ones(2, 2))
        untouched = model.value[1].clone()
        current = torch.tensor([[3., 3.], [1., 1.]])
        model.reset(current, torch.tensor([0]))
        self.assertTrue(torch.equal(model.value[0], current[0]))
        self.assertTrue(torch.equal(model.value[1], untouched))
        self.assertEqual(model.steps_since_reset.tolist(), [0, 1])
        model.update(current)
        torch.testing.assert_close(model.value[0], current[0], atol=1e-6, rtol=0)
        model.reset(current, [])
        self.assertEqual(model.reset_count, 1)
        model.reset(current)
        self.assertTrue(torch.equal(model.value, current))
        self.assertEqual(model.steps_since_reset.tolist(), [0, 0])

    def test_invalid_configuration_is_rejected(self):
        for cutoff, dt in ((0, .005), (100, .005), (20, 0), (20, float("nan"))):
            with self.subTest(cutoff=cutoff, dt=dt), self.assertRaises(ValueError):
                feedback.tustin_lowpass_coefficients(cutoff, dt)
        for settings in ({"mode": "delay20"}, {"model_version": "unknown"}, {"cutoff_hz": -1}, {"typo": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                feedback.resolve_feedback_settings(settings)
        with self.assertRaises(TypeError):
            feedback.resolve_feedback_settings({"latest_critic_contacts": 1})


class EnvironmentFeedbackTests(unittest.TestCase):
    def test_construction_dynamic_dispatch_and_contract(self):
        env = make_environment()
        self.assertEqual(env._num_obs, 100)
        self.assertEqual(env.dwaq_obs_history_length, 5)
        self.assertEqual(env.feedback_update_count, 0)
        self.assertTrue(torch.equal(env.velocity_feedback_model.value, env.robot.data.joint_vel))
        with self.assertRaisesRegex(ValueError, "5 x 100"):
            make_environment(history=20)

    def test_ideal_parent_actor_and_critic_are_bitwise_identical_when_latest_disabled(self):
        env = make_environment(mode="ideal", latest=False)
        for offset in (0, 7):
            # Even if data was externally changed without a physics hook, ideal
            # mode strictly follows the original method, not a stale cache.
            env.robot.data.joint_vel.add_(offset)
            actual = env.compute_current_observations()
            expected = ParentEnv.compute_current_observations(env)
            for left, right in zip(actual, expected):
                self.assertTrue(torch.equal(left, right))

    def test_physics_substeps_not_observation_reads_advance_feedback(self):
        env = make_environment(latest=False)
        model = env.velocity_feedback_model
        expected, previous = model.value.clone(), model.value.clone()
        physical_positions = env.robot.data.joint_pos.clone()
        actions = env.action_buffer._circular_buffer.buffer.clone()
        for step in range(4):
            env.robot.data.joint_vel.fill_(step + 1)
            current = env.robot.data.joint_vel
            expected = model.b0 * current + model.b1 * previous - model.a1 * expected
            previous = current.clone()
            env._post_physics_step()
        for _ in range(5):
            actor, critic = env.compute_current_observations()
            expected_velocity = ((expected - env.robot.data.default_joint_vel)[:, env.policy_joint_ids]
                                 * env.obs_scales.joint_vel)
            torch.testing.assert_close(actor[:, 38:67], expected_velocity, atol=1e-14, rtol=0)
            self.assertTrue(torch.equal(critic[:, :100], actor))
        self.assertEqual(env.feedback_update_count, 4)
        self.assertEqual(model.steps_since_reset.tolist(), [4, 4])
        self.assertTrue(torch.equal(env.robot.data.joint_pos, physical_positions))
        self.assertTrue(torch.equal(env.action_buffer._circular_buffer.buffer, actions))
        parent_actor, parent_critic = ParentEnv.compute_current_observations(env)
        self.assertTrue(torch.equal(actor[:, :38], parent_actor[:, :38]))
        self.assertTrue(torch.equal(actor[:, 67:], parent_actor[:, 67:]))
        self.assertTrue(torch.equal(critic[:, 100:], parent_critic[:, 100:]))

    def test_latest_force_all_privileged_flag_combinations(self):
        for feet_info in (False, True):
            for feet_force in (False, True):
                with self.subTest(feet_info=feet_info, feet_force=feet_force):
                    env = make_environment(mode="ideal", feet_info=feet_info, feet_force=feet_force)
                    actor, critic = env.compute_current_observations()
                    parent_actor, expected = ParentEnv.compute_current_observations(env)
                    if feet_force:
                        start = 105 + (12 if feet_info else 0)
                        expected[:, start:start + 6] = (env.contact_sensor.data.net_forces_w_history[
                            :, 0, env.feet_cfg.body_ids, :].reshape(2, 6) * env.obs_scales.contact_force)
                    self.assertTrue(torch.equal(actor, parent_actor))
                    self.assertTrue(torch.equal(critic, expected))
                    self.assertTrue(torch.equal(critic[:, 103:105], torch.ones(2, 2, dtype=critic.dtype)))
                    self.assertTrue(torch.equal(critic[:, -1], env.robot.data.root_pos_w[:, 2]))
                    self.assertEqual(env.feedback_update_count, 0)

    def test_reset_after_base_reset_prevents_cross_episode_state(self):
        env = make_environment()
        env.robot.data.joint_vel.fill_(10.)
        env._post_physics_step()
        old_other_env = env.velocity_feedback_model.value[1].clone()
        env.reset(torch.tensor([0]))
        model = env.velocity_feedback_model
        self.assertTrue(torch.equal(model.value[0], torch.full((29,), -4., dtype=torch.float64)))
        self.assertTrue(torch.equal(model.value[1], old_other_env))
        self.assertEqual(model.steps_since_reset.tolist(), [0, 1])
        env.compute_current_observations()
        self.assertEqual(env.feedback_update_count, 1)
        env._post_physics_step()
        torch.testing.assert_close(model.value[0], env.robot.data.joint_vel[0], atol=1e-14, rtol=0)
        resets = model.reset_count
        env.reset(torch.tensor([], dtype=torch.long))
        self.assertEqual(model.reset_count, resets)

    def test_base_hook_is_noop_and_immediately_after_scene_update(self):
        source = ast.parse((LAB / "envs/g1/g1_dwaq_env.py").read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "G1DwaqEnv")
        hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_post_physics_step")
        self.assertEqual(len(hook.body), 2)  # docstring + pass
        self.assertIsInstance(hook.body[-1], ast.Pass)
        step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        calls = []
        for loop in (node for node in ast.walk(step) if isinstance(node, ast.For)):
            calls.extend(ast.unparse(statement.value.func) for statement in loop.body
                         if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call))
        index = calls.index("self.scene.update")
        self.assertEqual(calls[index + 1], "self._post_physics_step")
        self.assertEqual(calls.count("self._post_physics_step"), 1)


if __name__ == "__main__":
    unittest.main()
