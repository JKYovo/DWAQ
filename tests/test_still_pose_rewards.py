"""CPU-only tests of the zero-command, actual four-joint pose objective."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TienKung-Lab/legged_lab/envs/elf3/still_pose_rewards.py"
SPEC = importlib.util.spec_from_file_location("cpu_still_pose_rewards", SOURCE)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)


def environment(batch=4):
    # Deliberately neither policy order nor an adjacent simulator slice.
    indices = [27, 4, 20, 8]
    home = torch.zeros(batch, 29, dtype=torch.float64)
    home[:, indices] = torch.tensor([-.3, .6, -.3, .6], dtype=home.dtype)
    data = SimpleNamespace(joint_pos=home.clone(), default_joint_pos=home,
                           joint_pos_target=home.clone(), joint_vel=torch.zeros_like(home))
    env = SimpleNamespace(num_envs=batch, command_generator=SimpleNamespace(
        command=torch.zeros(batch, 3, dtype=home.dtype)), robot=SimpleNamespace(data=data))
    cfg = SimpleNamespace(joint_ids=indices, joint_names=list(rewards.STAND_POSE_JOINT_NAMES))
    return env, cfg


class StillPoseRewardTests(unittest.TestCase):
    def test_explicit_names_and_four_actual_angle_l1_sum_not_mean(self):
        self.assertEqual(rewards.STAND_POSE_JOINT_NAMES,
                         ("l_hip_y_joint", "l_knee_y_joint", "r_hip_y_joint", "r_knee_y_joint"))
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += torch.tensor([.1, -.2, .3, -.4])
        result = rewards.stand_leg_pose_l1(env, cfg)
        torch.testing.assert_close(result, torch.ones(4, dtype=torch.float64))

    def test_default_pose_is_zero_and_assets_home_is_not_hard_coded(self):
        env, cfg = environment()
        self.assertEqual(rewards.stand_leg_pose_l1(env, cfg).tolist(), [0.] * 4)
        env.robot.data.default_joint_pos[:, cfg.joint_ids] += .25
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.ones(4, dtype=torch.float64))
        env.robot.data.joint_pos.copy_(env.robot.data.default_joint_pos)
        self.assertEqual(rewards.stand_leg_pose_l1(env, cfg).tolist(), [0.] * 4)

    def test_point_one_forward_lateral_and_pure_yaw_are_not_standing(self):
        env, cfg = environment(batch=7)
        env.robot.data.joint_pos[:, cfg.joint_ids] += .2
        env.command_generator.command[:] = torch.tensor([
            [0., 0., 0.], [.1, 0., 0.], [-.1, 0., 0.], [0., .1, 0.],
            [0., -.1, 0.], [0., 0., .1], [0., 0., -.1]], dtype=torch.float64)
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.tensor([.8, 0., 0., 0., 0., 0., 0.], dtype=torch.float64))

    def test_epsilon_inclusive_and_above_epsilon_off(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        env.command_generator.command[0] = torch.tensor([1e-6, -1e-6, 1e-6], dtype=torch.float64)
        env.command_generator.command[1, 0] = 1.00001e-6
        env.command_generator.command[2, 1] = -1.00001e-6
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.tensor([.4, 0., 0., .4], dtype=torch.float64))

    def test_only_selected_simulator_joints_not_policy_indices_are_used(self):
        env, cfg = environment()
        env.robot.data.joint_pos.fill_(10.)
        env.robot.data.joint_pos[:, cfg.joint_ids] = env.robot.data.default_joint_pos[:, cfg.joint_ids]
        self.assertEqual(rewards.stand_leg_pose_l1(env, cfg).tolist(), [0.] * 4)
        env.robot.data.joint_pos[:, cfg.joint_ids[1]] += .3
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.full((4,), .3, dtype=torch.float64))

    def test_actual_position_not_target_or_action_or_velocity(self):
        env, cfg = environment()
        env.action = torch.zeros(4, 29, dtype=torch.float64)
        env.robot.data.joint_pos_target.copy_(env.robot.data.default_joint_pos)
        env.robot.data.joint_pos[:, cfg.joint_ids] += .2
        expected = torch.full((4,), .8, dtype=torch.float64)
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg), expected)
        env.robot.data.joint_pos_target.fill_(-100.)
        env.action.fill_(100.)
        env.robot.data.joint_vel.fill_(100.)
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg), expected)

    def test_nonfinite_commands_are_safely_inactive(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        env.command_generator.command[0, 0] = float("nan")
        env.command_generator.command[1, 1] = float("inf")
        env.command_generator.command[2, 2] = -float("inf")
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.tensor([0., 0., 0., .4], dtype=torch.float64))

    def test_current_command_only_no_state_and_no_reset_leak(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        before_fields = (set(vars(env)), set(vars(env.robot.data)))
        before_q = env.robot.data.joint_pos.clone()
        before_home = env.robot.data.default_joint_pos.clone()
        for _ in range(3):
            torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                       torch.full((4,), .4, dtype=torch.float64))
        env.command_generator.command[0, 0] = .6
        self.assertEqual(float(rewards.stand_leg_pose_l1(env, cfg)[0]), 0.)
        # Simulate a partial reset: new current q/command must entirely decide
        # the next value, without any timer, accumulated pose, or old command.
        env.robot.data.joint_pos[0] = env.robot.data.default_joint_pos[0]
        env.command_generator.command[0] = 0.
        self.assertEqual(float(rewards.stand_leg_pose_l1(env, cfg)[0]), 0.)
        env.robot.data.joint_pos[0, cfg.joint_ids[0]] += .3
        self.assertAlmostEqual(float(rewards.stand_leg_pose_l1(env, cfg)[0]), .3)
        torch.testing.assert_close(env.robot.data.joint_pos[1:], before_q[1:])
        torch.testing.assert_close(env.robot.data.default_joint_pos, before_home)
        self.assertEqual((set(vars(env)), set(vars(env.robot.data))), before_fields)

    def test_no_contact_terrain_gravity_or_already_settled_inputs_are_needed(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        # This minimal fixture has no support, scan, tilt, base velocity, history
        # or episode buffers. Even large joint speed does not gate the pose cost.
        env.robot.data.joint_vel.fill_(100.)
        torch.testing.assert_close(rewards.stand_leg_pose_l1(env, cfg),
                                   torch.full((4,), .4, dtype=torch.float64))

    def test_increment_produces_total_point_one_only_at_zero(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        env.command_generator.command[1:, 0] = .1
        inherited = -.02 * (env.robot.data.joint_pos[:, cfg.joint_ids]
                              - env.robot.data.default_joint_pos[:, cfg.joint_ids]).abs().sum(-1)
        total = inherited - .08 * rewards.stand_leg_pose_l1(env, cfg)
        torch.testing.assert_close(total, torch.tensor([-.04, -.008, -.008, -.008], dtype=torch.float64))

    def test_invalid_configuration_is_rejected(self):
        env, cfg = environment()
        for epsilon in (-1., float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                rewards.stand_leg_pose_l1(env, cfg, epsilon)
        for indices in ([], [1, 2, 3], [1, 2, 3, 4, 5], slice(None)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                rewards.stand_leg_pose_l1(env, SimpleNamespace(joint_ids=indices))
        env.command_generator.command = torch.zeros(4, 2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            rewards.stand_leg_pose_l1(env, cfg)


if __name__ == "__main__":
    unittest.main()
