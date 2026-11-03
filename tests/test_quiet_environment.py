"""CPU-only command-only eligibility, actual arm motion and environment tests."""
from __future__ import annotations

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


quiet = load_file("cpu_actual_quiet_rewards", LAB / "envs/elf3/quiet_rewards.py")
feedback_helpers = load_file("cpu_quiet_feedback_environment_helpers", ROOT / "tests/test_feedback_environment.py")


def settings():
    return {"version": "elf3_stand_v2_zero_command", "mode": "zero_command_upper_body_stillness",
            "command_epsilon": 1e-6,
            "activation": {"mode": "zero_command_only", "history_requirement": "none"},
            "auxiliary": {"phase_invariance_coeff": 1., "arm_pose_coeff": .1,
                          "phase_offsets": [math.pi / 2, math.pi, 3 * math.pi / 2]},
            "physical": {"velocity_std": .05}}


def load_actual_environment():
    parent = ModuleType("legged_lab.envs.elf3.elf3_feedback_env")
    parent.Elf3DwaqFeedbackEnv = feedback_helpers.FeedbackEnv
    with patch.dict(sys.modules, {"legged_lab.elf3_contract": feedback_helpers.contract,
                                "legged_lab.envs.elf3.elf3_feedback_env": parent,
                                "legged_lab.envs.elf3.quiet_rewards": quiet}):
        return load_file("cpu_actual_quiet_environment", LAB / "envs/elf3/quiet_env.py").Elf3DwaqStandV2Env


Environment = load_actual_environment()


def environment():
    existing = feedback_helpers.make_environment()
    env = Environment.__new__(Environment)
    env.__dict__.update(existing.__dict__)
    env.cfg.quiet_standing = settings()
    env.step_dt = .02
    env.sim_step_counter = 0
    env.command_generator.command.zero_()
    env.robot.data.root_lin_vel_b.zero_()
    env.robot.data.root_ang_vel_b.zero_()
    env.contact_sensor.data.net_forces_w = torch.zeros(2, 3, 3, dtype=torch.float64)
    env.contact_sensor.data.net_forces_w[:, [0, 2], 2] = 100.
    env.init_buffers()
    return env


class ZeroCommandTests(unittest.TestCase):
    def test_first_zero_command_frame_is_active_with_no_settling_wait(self):
        env = environment()
        self.assertEqual(env.sim_step_counter, 0)
        self.assertTrue(env.get_standing_mask().all())
        self.assertFalse(hasattr(env, "quiet_context"))
        self.assertFalse(hasattr(quiet, "StandingQuietContext"))
        self.assertTrue(torch.equal(env.get_standing_mask(), quiet.zero_command_mask(env)))

    def test_epsilon_all_axes_point_one_and_pure_yaw_are_not_standing(self):
        env = environment()
        for command in ((.1, 0, 0), (-.1, 0, 0), (0, .1, 0), (0, -.1, 0),
                        (0, 0, .1), (0, 0, -.1), (1.00001e-6, 0, 0)):
            with self.subTest(command=command):
                env.command_generator.command[0] = torch.tensor(command)
                self.assertFalse(env.get_standing_mask()[0])
                self.assertTrue(env.get_standing_mask()[1])
        env.command_generator.command[0] = torch.tensor([1e-6, -1e-6, 1e-6], dtype=torch.float64)
        self.assertTrue(env.get_standing_mask()[0])

    def test_moving_to_zero_immediate_even_without_new_sim_step(self):
        env = environment()
        env.command_generator.command[:, 0] = .6
        self.assertFalse(env.get_standing_mask().any())
        env.command_generator.command[0] = 0.
        self.assertEqual(env.get_standing_mask().tolist(), [True, False])
        env.command_generator.command[1] = 0.
        self.assertTrue(env.get_standing_mask().all())
        self.assertEqual(env.sim_step_counter, 0)

    def test_contact_tilt_and_base_motion_do_not_disable_zero_command_constraint(self):
        cases = {
            "one_foot": lambda e: e.contact_sensor.data.net_forces_w.__setitem__((0, 2, 2), 0.),
            "no_contacts": lambda e: e.contact_sensor.data.net_forces_w.zero_(),
            "downward_contact": lambda e: e.contact_sensor.data.net_forces_w.fill_(-100.),
            "upside_down": lambda e: e.robot.data.projected_gravity_b.__setitem__(
                0, torch.tensor([0., 0., 1.])),
            "invalid_gravity": lambda e: e.robot.data.projected_gravity_b.fill_(float("nan")),
            "linear_motion": lambda e: e.robot.data.root_lin_vel_b.fill_(10.),
            "angular_motion": lambda e: e.robot.data.root_ang_vel_b.fill_(10.),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                env = environment()
                env.robot.data.joint_vel.zero_()
                change(env)
                self.assertTrue(env.get_standing_mask().all())
                reward = quiet.upper_body_stillness(env, SimpleNamespace(joint_ids=env.quiet_arm_joint_ids))
                self.assertTrue(torch.equal(reward, torch.ones(2, dtype=torch.float64)))

    def test_eligibility_does_not_read_privileged_state(self):
        env = environment()
        del env.contact_sensor
        del env.robot
        self.assertTrue(env.get_standing_mask().all())

    def test_reset_follows_new_commands_and_has_no_timer_or_stale_mask(self):
        env = environment()
        env.command_generator.command[0, 0] = .5
        env.reset(torch.tensor([0]))
        self.assertEqual(env.get_standing_mask().tolist(), [False, True])
        self.assertTrue(torch.equal(env.velocity_feedback_model.value[0], env.robot.data.joint_vel[0]))
        env.command_generator.command[0] = 0.
        self.assertTrue(env.get_standing_mask().all())
        env.command_generator.command[1, 2] = -.5
        env.reset(torch.tensor([1]))
        self.assertEqual(env.get_standing_mask().tolist(), [True, False])
        env.reset([])
        self.assertEqual(env.get_standing_mask().tolist(), [True, False])

    def test_reads_are_state_free_and_return_independent_masks(self):
        env = environment()
        command = env.command_generator.command.clone()
        for _ in range(10):
            self.assertTrue(env.get_standing_mask().all())
            mask = env.get_standing_mask()
            mask.zero_()
        self.assertTrue(env.get_standing_mask().all())
        self.assertTrue(torch.equal(env.command_generator.command, command))
        self.assertEqual(env.sim_step_counter, 0)
        self.assertFalse(hasattr(env, "quiet_context"))

    def test_nonfinite_commands_disable_reward_and_bad_epsilon_is_rejected(self):
        env = environment()
        env.robot.data.joint_vel.zero_()
        for value in (float("nan"), float("inf"), -float("inf")):
            for axis in range(3):
                with self.subTest(value=value, axis=axis):
                    env.command_generator.command[0] = 0.
                    env.command_generator.command[0, axis] = value
                    self.assertEqual(env.get_standing_mask().tolist(), [False, True])
                    result = quiet.upper_body_stillness(env, SimpleNamespace(joint_ids=env.quiet_arm_joint_ids))
                    self.assertTrue(torch.equal(result, torch.tensor([0., 1.], dtype=torch.float64)))
        for epsilon in (-1, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                quiet.zero_command_mask(env, epsilon)
        env.command_generator.command = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            quiet.zero_command_mask(env)


class PhysicalRewardTests(unittest.TestCase):
    def test_resolved_nonidentity_indices_raw_dq_not_filtered_observation(self):
        env = environment()
        ids = env.quiet_arm_joint_ids
        self.assertNotEqual(ids.tolist(), list(range(15, 29)))
        env.robot.data.joint_vel.fill_(10.)
        env.robot.data.joint_vel[:, ids] = .025
        env.velocity_feedback_model.value.zero_()
        result = quiet.upper_body_stillness(env, SimpleNamespace(joint_ids=ids))
        expected = torch.full((2,), math.exp(-.25), dtype=torch.float64)
        torch.testing.assert_close(result, expected)
        env.robot.data.joint_vel[:, ids] = .1
        smaller = quiet.upper_body_stillness(env, SimpleNamespace(joint_ids=ids))
        torch.testing.assert_close(smaller, torch.full((2,), math.exp(-4), dtype=torch.float64))
        self.assertTrue(env.get_standing_mask().all())  # high arm speed cannot disable its constraint

    def test_reward_immediate_bounded_zero_only_and_velocity_std_validation(self):
        env = environment()
        ids = SimpleNamespace(joint_ids=env.quiet_arm_joint_ids)
        env.robot.data.joint_vel.zero_()
        self.assertTrue(torch.equal(quiet.upper_body_stillness(env, ids), torch.ones(2, dtype=torch.float64)))
        env.command_generator.command[0, 2] = .1
        result = quiet.upper_body_stillness(env, ids)
        self.assertEqual(float(result[0]), 0.)
        self.assertEqual(float(result[1]), 1.)
        for std in (0, -.1, float("nan"), float("inf")):
            with self.subTest(std=std), self.assertRaises(ValueError):
                quiet.upper_body_stillness(env, ids, velocity_std=std)


class EnvironmentTests(unittest.TestCase):
    def test_named_arm_mapping_and_parent_observation_feedback_reset_unchanged(self):
        env = environment()
        names = [env.robot.joint_names[index] for index in env.quiet_arm_joint_ids]
        self.assertEqual(names, list(feedback_helpers.contract.JOINT_NAMES[15:]))
        self.assertEqual(env._num_obs, 100)
        self.assertEqual(env.dwaq_obs_history_length, 5)
        self.assertIs(Environment.compute_current_observations,
                      feedback_helpers.FeedbackEnv.compute_current_observations)
        self.assertIs(Environment._post_physics_step, feedback_helpers.FeedbackEnv._post_physics_step)
        self.assertIs(Environment.reset, feedback_helpers.FeedbackEnv.reset)
        phase = env.leg_phase.clone()
        for _ in range(5): env.get_standing_mask()
        self.assertTrue(torch.equal(env.leg_phase, phase))
        self.assertEqual(env.feedback_update_count, 0)


if __name__ == "__main__":
    unittest.main()
