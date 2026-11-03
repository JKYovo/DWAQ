"""CPU tests for the minimal task, including raw history and parent timing."""
from __future__ import annotations

import ast
import importlib.util
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


stillness = load_file("cpu_minimal_stillness", LAB / "envs/elf3/stillness.py")
helpers = load_file("cpu_minimal_feedback_helpers", ROOT / "tests/test_feedback_environment.py")


def actual_g1_method(name):
    source = ast.parse((LAB / "envs/g1/g1_dwaq_env.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "G1DwaqEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual G1 method>", "exec"), namespace)
    return namespace[name]


def load_actual_environment():
    parent = ModuleType("legged_lab.envs.elf3.elf3_feedback_env")
    parent.Elf3DwaqFeedbackEnv = helpers.FeedbackEnv
    with patch.dict(sys.modules, {"legged_lab.envs.elf3.elf3_feedback_env": parent,
                                "legged_lab.envs.elf3.stillness": stillness}):
        return load_file("cpu_minimal_still_env", LAB / "envs/elf3/still_env.py").Elf3DwaqStillEnv


Environment = load_actual_environment()


def inputs(batch=5):
    generator = torch.Generator().manual_seed(17)
    obs = torch.randn(batch, 100, generator=generator, dtype=torch.float64)
    hist = torch.randn(batch, 500, generator=generator, dtype=torch.float64)
    obs[:, 6:9] = 0.0
    return obs, hist


def environment():
    baseline = helpers.make_environment()
    env = Environment.__new__(Environment)
    env.__dict__.update(baseline.__dict__)
    env.command_generator.command[0] = 0.0
    env.command_generator.command[1] = torch.tensor([0.1, 0.0, 0.0])
    env.extras = {}
    env.raw_observations, env.raw_history = inputs(2)
    env.raw_observations[:, 6:9] = env.command_generator.command
    env.raw_critic = torch.randn(2, 142, dtype=torch.float64)
    env.prev_critic_obs = env.raw_critic.clone() + 1.0
    env.dwaq_obs_history_buffer = SimpleNamespace(buffer=env.raw_history.reshape(2, 5, 100))
    env.compute_observations = lambda: (env.raw_observations, env.raw_critic)
    return env


class InputTests(unittest.TestCase):
    def test_current_zero_replaces_only_phase_in_current_and_all_history(self):
        obs, hist = inputs()
        before_obs, before_hist = obs.clone(), hist.clone()
        prepared, prepared_hist = stillness.prepare_still_inputs(obs, hist)
        fixed = torch.tensor(stillness.FIXED_PHASE, dtype=obs.dtype)
        torch.testing.assert_close(prepared[:, 96:100], fixed.expand(5, 4))
        torch.testing.assert_close(prepared_hist.reshape(5, 5, 100)[:, :, 96:100], fixed.expand(5, 5, 4))
        self.assertTrue(torch.equal(prepared[:, :96], obs[:, :96]))
        self.assertTrue(torch.equal(prepared_hist.reshape(5, 5, 100)[:, :, :96],
                                    hist.reshape(5, 5, 100)[:, :, :96]))
        self.assertTrue(torch.equal(obs, before_obs))
        self.assertTrue(torch.equal(hist, before_hist))
        self.assertNotEqual(prepared.data_ptr(), obs.data_ptr())
        self.assertNotEqual(prepared_hist.data_ptr(), hist.data_ptr())

    def test_first_zero_frame_ignores_old_moving_history_and_has_no_state(self):
        obs, hist = inputs(1)
        hist.reshape(1, 5, 100)[:, :, 6:9] = 0.6
        obs[:, 6] = 0.6
        moving = stillness.prepare_still_inputs(obs, hist)
        self.assertTrue(torch.equal(moving[0], obs))
        self.assertTrue(torch.equal(moving[1], hist))
        obs[:, 6:9] = 0.0
        zero = stillness.prepare_still_inputs(obs, hist)
        self.assertEqual(zero[1].reshape(1, 5, 100)[0, :, 96].tolist(), [0.0] * 5)
        obs[:, 8] = 0.1
        resumed = stillness.prepare_still_inputs(obs, hist)
        self.assertTrue(torch.equal(resumed[0], obs))
        self.assertTrue(torch.equal(resumed[1], hist))

    def test_point_one_and_pure_yaw_are_bitwise_baseline(self):
        obs, hist = inputs(5)
        obs[:, 6:9] = torch.tensor([[.1, 0., 0.], [-.1, 0., 0.], [0., .1, 0.],
                                   [0., 0., .1], [0., 0., -.1]], dtype=obs.dtype)
        prepared, prepared_hist = stillness.prepare_still_inputs(obs, hist)
        self.assertTrue(torch.equal(prepared, obs))
        self.assertTrue(torch.equal(prepared_hist, hist))

    def test_epsilon_and_nonfinite_commands(self):
        obs, hist = inputs(5)
        obs[0, 6:9] = torch.tensor([1e-6, -1e-6, 1e-6], dtype=obs.dtype)
        obs[1, 6] = 1.00001e-6
        obs[2, 6], obs[3, 7], obs[4, 8] = float("nan"), float("inf"), -float("inf")
        prepared, prepared_hist = stillness.prepare_still_inputs(obs, hist)
        self.assertEqual(prepared[0, 96:100].tolist(), list(stillness.FIXED_PHASE))
        torch.testing.assert_close(prepared[1:], obs[1:], rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.equal(prepared_hist[1:], hist[1:]))
        for epsilon in (-1., float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                stillness.prepare_still_inputs(obs, hist, epsilon)

    def test_shape_contract_and_noncontiguous_inputs(self):
        obs, hist = inputs(2)
        for broken_obs, broken_history in ((obs[:, :99], hist), (obs, hist[:, :499]),
                                            (obs, hist[:1]), (obs.unsqueeze(0), hist)):
            with self.assertRaises(ValueError):
                stillness.prepare_still_inputs(broken_obs, broken_history)
        wide_obs = torch.stack([obs, obs], dim=-1)[:, :, 0]
        wide_hist = torch.stack([hist, hist], dim=-1)[:, :, 0]
        prepared = stillness.prepare_still_inputs(wide_obs, wide_hist)
        expected = stillness.prepare_still_inputs(obs, hist)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(prepared, expected)))

    def test_torchscript_has_same_input_transformation(self):
        compiled = torch.jit.script(stillness.prepare_still_inputs)
        obs, hist = inputs(3)
        obs[1, 6] = .1
        obs[2, 8] = -.5
        expected = stillness.prepare_still_inputs(obs, hist)
        actual = compiled(obs, hist)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))


class DenseMotionTests(unittest.TestCase):
    def test_dense_raw_l1_at_small_medium_and_large_speed(self):
        env = environment()
        cfg = SimpleNamespace(joint_ids=slice(None))
        for speed in (.05, .2, 5.):
            env.robot.data.joint_vel.fill_(speed)
            env.velocity_feedback_model.value.zero_()
            result = stillness.stand_joint_motion(env, cfg)
            self.assertAlmostEqual(float(result[0]), speed, places=12)
            self.assertEqual(float(result[1]), 0.)
            env.robot.data.joint_vel.fill_(-speed)
            self.assertTrue(torch.equal(stillness.stand_joint_motion(env, cfg), result))

    def test_resolved_simulator_joint_indices_not_native_policy_positions(self):
        env = environment()
        names = [helpers.contract.JOINT_NAMES[i] for i in (1, 3, 5, 15, 17)]
        ids, resolved = env.robot.find_joints(names, preserve_order=True)
        self.assertEqual(resolved, names)
        self.assertNotEqual(ids, [1, 3, 5, 15, 17])
        env.robot.data.joint_vel.fill_(10.)
        env.robot.data.joint_vel[:, ids] = .2
        result = stillness.stand_joint_motion(env, SimpleNamespace(joint_ids=ids))
        self.assertAlmostEqual(float(result[0]), .2, places=12)
        all_ids, all_names = env.robot.find_joints(list(helpers.contract.JOINT_NAMES), preserve_order=True)
        self.assertEqual(all_names, list(helpers.contract.JOINT_NAMES))
        self.assertEqual(len(all_ids), 29)
        torch.testing.assert_close(stillness.stand_joint_motion(env, SimpleNamespace(joint_ids=all_ids))[0],
                                   env.robot.data.joint_vel[0].abs().mean())

    def test_immediate_zero_only_and_no_contact_or_already_stable_condition(self):
        env = environment()
        cfg = SimpleNamespace(joint_ids=slice(None))
        env.robot.data.joint_vel.fill_(5.)
        del env.contact_sensor
        env.robot.data.projected_gravity_b.fill_(float("nan"))
        env.robot.data.root_lin_vel_b.fill_(100.)
        self.assertEqual(stillness.stand_joint_motion(env, cfg).tolist(), [5., 0.])
        env.command_generator.command[1] = 0.
        self.assertEqual(stillness.stand_joint_motion(env, cfg).tolist(), [5., 5.])
        for axis in range(3):
            env.command_generator.command.zero_()
            env.command_generator.command[0, axis] = .1
            self.assertEqual(stillness.stand_joint_motion(env, cfg).tolist(), [0., 5.])

    def test_invalid_commands_and_configuration(self):
        env = environment()
        env.robot.data.joint_vel.fill_(1.)
        cfg = SimpleNamespace(joint_ids=slice(None))
        for value in (float("nan"), float("inf"), -float("inf")):
            env.command_generator.command.zero_()
            env.command_generator.command[0, 0] = value
            self.assertEqual(stillness.stand_joint_motion(env, cfg).tolist(), [0., 1.])
        with self.assertRaises(ValueError):
            stillness.stand_joint_motion(env, SimpleNamespace(joint_ids=[]))
        for epsilon in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                stillness.stand_joint_motion(env, cfg, epsilon)


class EnvironmentTests(unittest.TestCase):
    def test_actual_parent_get_observations_preserves_raw_buffers_and_critic(self):
        env = environment()
        raw_obs, raw_history, raw_critic = env.raw_observations.clone(), env.raw_history.clone(), env.raw_critic.clone()
        phase = env.leg_phase.clone()
        with patch.object(helpers.TensorOnlyBase, "get_observations", actual_g1_method("get_observations"),
                          create=True):
            obs, hist = env.get_observations()
        self.assertEqual(obs[0, 96:100].tolist(), list(stillness.FIXED_PHASE))
        self.assertTrue(torch.equal(obs[1], raw_obs[1]))
        self.assertTrue(torch.equal(hist[1], raw_history[1]))
        self.assertTrue(torch.equal(env.raw_observations, raw_obs))
        self.assertTrue(torch.equal(env.dwaq_obs_history_buffer.buffer.reshape(2, 500), raw_history))
        self.assertTrue(torch.equal(env.extras["observations"]["critic"], raw_critic))
        self.assertTrue(torch.equal(env.extras["observations"]["obs_hist"], hist))
        self.assertTrue(torch.equal(env.leg_phase, phase))
        self.assertEqual(env.feedback_update_count, 0)

    def test_step_transforms_return_only_after_parent_and_preserves_nonactor_fields(self):
        env = environment()
        rewards = torch.tensor([1., 2.])
        dones = torch.tensor([False, True])
        actions = torch.randn(2, 29, dtype=torch.float64)
        seen_actions = []

        def parent_step(parent, simulator_actions):
            seen_actions.append(simulator_actions.clone())
            parent.raw_observations[:, 96:100] = .3  # identifies post-step frames
            parent.raw_history.reshape(2, 5, 100)[:, :, 96:100] = .4
            parent.extras = {"observations": {"critic": parent.raw_critic,
                                              "obs_hist": parent.raw_history,
                                              "prev_critic_obs": parent.prev_critic_obs},
                             "time_outs": dones, "log": {"baseline": 2.}}
            return parent.raw_observations, rewards, dones, parent.extras

        with patch.object(helpers.TensorOnlyBase, "step", parent_step, create=True):
            obs, actual_rewards, actual_dones, extras = env.step(actions)
        self.assertTrue(torch.equal(seen_actions[0], actions[:, env.sim_to_policy]))
        self.assertIs(actual_rewards, rewards)
        self.assertIs(actual_dones, dones)
        self.assertIs(extras["observations"]["critic"], env.raw_critic)
        self.assertIs(extras["observations"]["prev_critic_obs"], env.prev_critic_obs)
        self.assertEqual(extras["log"], {"baseline": 2.})
        self.assertEqual(obs[0, 96:100].tolist(), list(stillness.FIXED_PHASE))
        self.assertTrue(torch.equal(obs[1], env.raw_observations[1]))
        self.assertTrue(torch.equal(extras["observations"]["obs_hist"][1], env.raw_history[1]))
        self.assertTrue((env.raw_observations[:, 96:100] == .3).all())
        self.assertTrue((env.raw_history.reshape(2, 5, 100)[:, :, 96:100] == .4).all())

    def test_reset_and_nonzero_resume_have_no_phase_rewrite_leak(self):
        env = environment()
        with patch.object(helpers.TensorOnlyBase, "get_observations", actual_g1_method("get_observations"),
                          create=True):
            env.get_observations()
            env.command_generator.command[0] = torch.tensor([0., 0., .1])
            env.raw_observations[:, 6:9] = env.command_generator.command
            env.reset(torch.tensor([0]))
            obs, hist = env.get_observations()
        self.assertTrue(torch.equal(obs, env.raw_observations))
        self.assertTrue(torch.equal(hist, env.raw_history))
        self.assertTrue(torch.equal(env.velocity_feedback_model.value[0], env.robot.data.joint_vel[0]))
        self.assertIs(Environment.reset, helpers.FeedbackEnv.reset)
        self.assertIs(Environment.compute_current_observations, helpers.FeedbackEnv.compute_current_observations)
        self.assertIs(Environment._post_physics_step, helpers.FeedbackEnv._post_physics_step)

    def test_actual_parent_step_keeps_physics_feedback_and_command_timing(self):
        env = environment()
        events = []
        env.headless = True
        env.physics_dt, env.step_dt = .005, .02
        env.sim_step_counter = 0
        env.episode_length_buf = torch.zeros(2, dtype=torch.long)
        env.cfg.sim = SimpleNamespace(decimation=4)
        env.clip_actions = 100.
        raw_history = env.raw_history.clone()
        targets = []
        env._get_current_critic_obs_with_height_scan = lambda: env.raw_critic
        env.action_buffer.compute = lambda action: action
        env.robot.set_joint_position_target = lambda target: targets.append(target.clone())

        def scene_update(dt):
            self.assertEqual(dt, .005)
            events.append("physics")
            env.robot.data.joint_vel.add_(.1)

        env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=scene_update)
        env.sim = SimpleNamespace(step=lambda render: None)
        env._update_gait_phase = lambda: events.append("clock")
        env._compute_forward_obstacle_height = lambda: events.append("terrain")

        def command_compute(dt):
            events.append("command")
            self.assertEqual(dt, .02)
            # New request after physics affects the returned next observation,
            # never retroactively the action already sent to physics.
            env.command_generator.command[0] = 0.
            env.command_generator.command[1] = torch.tensor([0., 0., .1])
            env.raw_observations[:, 6:9] = env.command_generator.command

        env.command_generator.compute = command_compute
        env.event_manager = SimpleNamespace(available_modes=[])
        env.check_reset = lambda: (torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))

        def reward_compute(dt):
            events.append("reward")
            self.assertEqual(dt, .02)
            self.assertTrue(torch.equal(env.raw_history, raw_history))
            self.assertNotEqual(env.raw_observations[0, 96:100].tolist(), list(stillness.FIXED_PHASE))
            return torch.tensor([3., 4.])

        env.reward_manager = SimpleNamespace(compute=reward_compute)
        actions = torch.arange(58, dtype=torch.float64).reshape(2, 29) / 100
        with patch.object(helpers.TensorOnlyBase, "step", actual_g1_method("step"), create=True):
            obs, rewards, dones, extras = env.step(actions)
        self.assertEqual(events, ["physics"] * 4 + ["clock", "terrain", "command", "reward"])
        self.assertEqual(env.feedback_update_count, 4)
        self.assertEqual(env.sim_step_counter, 4)
        self.assertEqual(env.episode_length_buf.tolist(), [1, 1])
        expected_target = (actions[:, env.sim_to_policy] * env.action_scale
                           + env.robot.data.default_joint_pos)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(torch.equal(target, expected_target) for target in targets))
        self.assertEqual(obs[0, 96:100].tolist(), list(stillness.FIXED_PHASE))
        self.assertTrue(torch.equal(obs[1], env.raw_observations[1]))
        self.assertTrue(torch.equal(env.raw_history, raw_history))
        self.assertIs(extras["observations"]["critic"], env.raw_critic)
        self.assertEqual(rewards.tolist(), [3., 4.])
        self.assertEqual(dones.tolist(), [False, False])

    def test_old_feedback_class_does_not_gain_stillness_transform(self):
        self.assertNotIn("get_observations", helpers.FeedbackEnv.__dict__)
        self.assertNotIn("step", helpers.FeedbackEnv.__dict__)
        env = helpers.make_environment()
        obs, critic = env.compute_current_observations()
        self.assertFalse(torch.equal(obs[0, 96:100], torch.tensor(stillness.FIXED_PHASE, dtype=obs.dtype)))
        self.assertTrue(torch.equal(critic[:, :100], obs))


if __name__ == "__main__":
    unittest.main()
