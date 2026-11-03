"""CPU tests of actual-state standing objectives; no Isaac, rollout or hardware."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_still_v2_rewards.py"
SPEC = importlib.util.spec_from_file_location("cpu_feedback_still_v2_rewards", SOURCE)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)


def environment(batch=8):
    dtype = torch.float64
    home = torch.arange(29, dtype=dtype).expand(batch, -1).clone() / 100.
    quat = torch.zeros(batch, 3, 4, dtype=dtype)
    quat[..., 0] = 1.
    data = SimpleNamespace(joint_pos=home.clone(), default_joint_pos=home,
                           joint_vel=torch.zeros_like(home), body_quat_w=quat,
                           root_lin_vel_b=torch.zeros(batch, 3, dtype=dtype),
                           root_ang_vel_b=torch.zeros(batch, 3, dtype=dtype))
    robot = SimpleNamespace(data=data)
    env = SimpleNamespace(num_envs=batch, robot=robot, scene={"robot": robot},
                          command_generator=SimpleNamespace(command=torch.zeros(batch, 3, dtype=dtype)))
    cfg = SimpleNamespace(name="robot", joint_ids=list(reversed(range(29))), body_ids=[1])
    return env, cfg


def euler_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor((cr * cp * cy + sr * sp * sy,
                         sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy), dtype=torch.float64)


def quaternion_product(left, right):
    w0, x0, y0, z0 = left.unbind(-1)
    w1, x1, y1, z1 = right.unbind(-1)
    return torch.stack((w0*w1-x0*x1-y0*y1-z0*z1,
                        w0*x1+x0*w1+y0*z1-z0*y1,
                        w0*y1-x0*z1+y0*w1+z0*x1,
                        w0*z1+x0*y1-y0*x1+z0*w1), -1)


class FeedbackStillV2RewardTests(unittest.TestCase):
    def test_strict_commands_include_boundary_and_pure_yaw_is_moving(self):
        command = torch.tensor([[0., 0., 0.], [1e-6, -1e-6, 1e-6],
                                [1.001e-6, 0., 0.], [.1, 0., 0.], [0., .1, 0.],
                                [0., 0., .1], [0., 0., -.1]], dtype=torch.float64)
        self.assertEqual(rewards.strict_zero_command(command).tolist(),
                         [True, True, False, False, False, False, False])

    def test_nonfinite_commands_activate_neither_standing_nor_moving(self):
        env, cfg = environment(batch=4)
        env.command_generator.command[:3, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        env.robot.data.joint_pos += .1
        env.robot.data.joint_vel += 1.
        for function in (rewards.full_body_default_pose_l1, rewards.stand_joint_motion,
                         rewards.stand_base_motion, rewards.stand_base_stillness,
                         rewards.torso_upright, rewards.joint_deviation_l1_when_moving):
            with self.subTest(function=function.__name__):
                self.assertEqual(function(env, cfg)[:3].tolist(), [0., 0., 0.])

    def test_invalid_command_shapes_and_epsilon_fail_closed(self):
        for command in (None, torch.zeros(3), torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.int64)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                rewards.strict_zero_command(command)
        for epsilon in (-1., float("nan"), float("inf"), True, "invalid"):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                rewards.strict_zero_command(torch.zeros(2, 3), epsilon)
        self.assertEqual(rewards.strict_zero_command(torch.tensor([[0., 0., 0.], [1e-9, 0., 0.]]), 0.).tolist(),
                         [True, False])

    def test_full_body_cost_covers_every_one_of_29_actual_joints(self):
        env, cfg = environment(batch=29)
        env.robot.data.joint_pos += torch.eye(29, dtype=torch.float64)
        torch.testing.assert_close(rewards.full_body_default_pose_l1(env, cfg), torch.ones(29, dtype=torch.float64))
        cfg.joint_ids = slice(None)
        torch.testing.assert_close(rewards.full_body_default_pose_l1(env, cfg), torch.ones(29, dtype=torch.float64))

    def test_pose_is_sum_not_mean_and_uses_asset_defaults(self):
        env, cfg = environment()
        env.robot.data.joint_pos += .2
        torch.testing.assert_close(rewards.full_body_default_pose_l1(env, cfg),
                                   torch.full((8,), 5.8, dtype=torch.float64))
        env.robot.data.default_joint_pos += .2
        torch.testing.assert_close(rewards.full_body_default_pose_l1(env, cfg), torch.zeros(8, dtype=torch.float64))

    def test_actual_pose_is_independent_of_targets_actions_torques_and_velocity(self):
        env, cfg = environment()
        env.robot.data.joint_pos += .1
        expected = rewards.full_body_default_pose_l1(env, cfg)
        env.action = torch.full((8, 29), 1000.)
        env.robot.data.joint_pos_target = torch.full((8, 29), -1000.)
        env.robot.data.applied_torque = torch.full((8, 29), 10000.)
        env.robot.data.joint_vel.fill_(500.)
        torch.testing.assert_close(rewards.full_body_default_pose_l1(env, cfg), expected)

    def test_moving_group_l1_reuses_resolved_simulator_indices(self):
        env, cfg = environment(batch=4)
        cfg.joint_ids = [27, 4, 20, 8]
        env.robot.data.joint_pos[:, cfg.joint_ids] += torch.tensor([.1, -.2, .3, -.4], dtype=torch.float64)
        env.command_generator.command[:] = torch.tensor([[0., 0., 0.], [.1, 0., 0.],
                                                        [0., -.1, 0.], [0., 0., .1]], dtype=torch.float64)
        torch.testing.assert_close(rewards.joint_deviation_l1_when_moving(env, cfg),
                                   torch.tensor([0., 1., 1., 1.], dtype=torch.float64))
        env.robot.data.joint_pos[:, 0] += 999.
        torch.testing.assert_close(rewards.joint_deviation_l1_when_moving(env, cfg),
                                   torch.tensor([0., 1., 1., 1.], dtype=torch.float64))

    def test_all_standing_terms_return_zero_immediately_on_moving_command(self):
        env, cfg = environment(batch=3)
        env.robot.data.joint_pos += .1
        env.robot.data.joint_vel += .2
        env.robot.data.root_lin_vel_b += .01
        env.command_generator.command[:] = torch.tensor([[.1, 0., 0.], [0., -.1, 0.], [0., 0., .1]])
        for function in (rewards.full_body_default_pose_l1, rewards.stand_joint_motion,
                         rewards.stand_base_motion, rewards.stand_base_stillness, rewards.torso_upright):
            with self.subTest(function=function.__name__):
                torch.testing.assert_close(function(env, cfg), torch.zeros(3, dtype=torch.float64))

    def test_full_body_indices_reject_subset_duplicates_out_of_range_and_bool(self):
        for indices in ([0], list(range(28)) + [0], list(range(28)) + [29], [], [True] * 29):
            env, cfg = environment()
            cfg.joint_ids = indices
            for function in (rewards.full_body_default_pose_l1, rewards.stand_joint_motion):
                with self.subTest(indices=indices, function=function.__name__), self.assertRaises(ValueError):
                    function(env, cfg)

    def test_joint_velocity_is_mean_actual_raw_and_not_filtered_observation(self):
        env, cfg = environment()
        env.robot.data.joint_vel[:] = torch.arange(-14., 15., dtype=torch.float64)
        env.velocity_feedback_model = SimpleNamespace(value=torch.zeros(8, 29))
        expected = torch.full((8,), float(torch.arange(-14., 15.).abs().mean()), dtype=torch.float64)
        torch.testing.assert_close(rewards.stand_joint_motion(env, cfg), expected)

    def test_upright_matches_clean_physical_roll_pitch_formula(self):
        env, cfg = environment(batch=1)
        roll, pitch, yaw = .11, -.07, 1.2
        env.robot.data.body_quat_w[:, 1] = euler_quaternion(roll, pitch, yaw)
        gravity_xy_norm = math.sqrt(math.sin(pitch)**2 + (math.sin(roll)*math.cos(pitch))**2)
        expected = .5 * (math.exp(-10*(abs(roll)+abs(pitch))) + math.exp(-20*gravity_xy_norm))
        self.assertAlmostEqual(float(rewards.torso_upright(env, cfg)[0]), expected, places=12)

    def test_upright_world_yaw_rotation_invariance_for_random_unit_quaternions(self):
        env, cfg = environment(batch=512)
        generator = torch.Generator().manual_seed(42)
        q = torch.randn(512, 4, dtype=torch.float64, generator=generator)
        q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        env.robot.data.body_quat_w[:, 1] = q
        expected = rewards.torso_upright(env, cfg)
        yaw = torch.randn(512, dtype=torch.float64, generator=generator) * 5.
        rotation = torch.zeros_like(q)
        rotation[:, 0], rotation[:, 3] = torch.cos(yaw/2), torch.sin(yaw/2)
        env.robot.data.body_quat_w[:, 1] = quaternion_product(rotation, q)
        torch.testing.assert_close(rewards.torso_upright(env, cfg), expected, atol=1e-12, rtol=1e-10)

    def test_upright_tilt_reduces_score_but_world_heading_does_not(self):
        env, cfg = environment(batch=5)
        angles = [0., .05, .2, .5, 1.]
        env.robot.data.body_quat_w[:, 1] = torch.stack([euler_quaternion(a, 0., .7) for a in angles])
        score = rewards.torso_upright(env, cfg)
        self.assertAlmostEqual(float(score[0]), 1., places=12)
        self.assertTrue(bool((score[:-1] > score[1:]).all()))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))

    def test_upright_quaternion_sign_normalization_and_selected_body(self):
        env, cfg = environment(batch=3)
        q = euler_quaternion(.2, -.1, .3)
        env.robot.data.body_quat_w[:, 1] = torch.stack([q, -q, q*2.])
        env.robot.data.body_quat_w[:, 0] = euler_quaternion(1.2, 0., 0.)
        score = rewards.torso_upright(env, cfg)
        torch.testing.assert_close(score, score[0].expand(3))
        self.assertTrue(bool((score < 1).all()))

    def test_invalid_torso_quaternions_return_zero_and_wrong_body_count_rejected(self):
        env, cfg = environment(batch=3)
        env.robot.data.body_quat_w[0, 1] = 0.
        env.robot.data.body_quat_w[1, 1, 0] = float("nan")
        env.robot.data.body_quat_w[2, 1, 0] = float("inf")
        self.assertEqual(rewards.torso_upright(env, cfg).tolist(), [0., 0., 0.])
        cfg.body_ids = [0, 1]
        with self.assertRaises(ValueError):
            rewards.torso_upright(env, cfg)

    def test_upright_gradients_are_finite_at_identity_regular_tilt_and_pitch_pole(self):
        env, cfg = environment(batch=4)
        env.robot.data.body_quat_w[:, 1] = torch.stack([euler_quaternion(0., 0., 0.),
                                                       euler_quaternion(.2, -.1, .7),
                                                       euler_quaternion(0., math.pi/2, 0.),
                                                       euler_quaternion(0., -math.pi/2, 0.)])
        env.robot.data.body_quat_w.requires_grad_(True)
        rewards.torso_upright(env, cfg).sum().backward()
        self.assertTrue(bool(torch.isfinite(env.robot.data.body_quat_w.grad).all()))

    def test_base_motion_reads_actual_translation_and_all_angular_axes(self):
        env, cfg = environment(batch=1)
        env.robot.data.root_lin_vel_b[:] = torch.tensor([1., 2., 3.])
        env.robot.data.root_ang_vel_b[:] = torch.tensor([4., 5., 6.])
        self.assertAlmostEqual(float(rewards.stand_base_motion(env, cfg)[0]), 14. + .25*77.)

    def test_bounded_base_stillness_catches_motion_with_all_joints_stationary(self):
        env, cfg = environment(batch=4)
        env.robot.data.root_lin_vel_b[1, 0] = .05
        env.robot.data.root_ang_vel_b[2, 2] = .10
        env.robot.data.root_lin_vel_b[3, 2] = 10.
        torch.testing.assert_close(rewards.stand_joint_motion(env, cfg), torch.zeros(4, dtype=torch.float64))
        score = rewards.stand_base_stillness(env, cfg)
        torch.testing.assert_close(score, torch.tensor([1., math.exp(-1), math.exp(-1), 0.], dtype=torch.float64))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))

    def test_base_stillness_is_rotation_invariant_and_uses_configured_widths(self):
        env, cfg = environment(batch=2)
        env.robot.data.root_lin_vel_b[:] = torch.tensor([.01, .02, .03])
        env.robot.data.root_ang_vel_b[:] = torch.tensor([.04, .05, .06])
        expected = rewards.stand_base_stillness(env, cfg)
        angle = 1.25
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                                  [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
        env.robot.data.root_lin_vel_b = env.robot.data.root_lin_vel_b @ rotation.T
        env.robot.data.root_ang_vel_b = env.robot.data.root_ang_vel_b @ rotation.T
        torch.testing.assert_close(rewards.stand_base_stillness(env, cfg), expected)
        wide = rewards.stand_base_stillness(env, cfg, linear_std=.1, angular_std=.2)
        self.assertTrue(bool((wide > expected).all()))
        for width in (0., -1., float("nan"), float("inf"), True):
            for name in ("linear_std", "angular_std"):
                with self.subTest(width=width, name=name), self.assertRaises(ValueError):
                    rewards.stand_base_stillness(env, cfg, **{name: width})

    def test_pose_velocity_and_base_gradients_are_finite(self):
        env, cfg = environment()
        data = env.robot.data
        data.joint_pos += .1
        data.joint_vel += .1
        data.root_lin_vel_b += .01
        data.root_ang_vel_b += .02
        for field in ("joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b"):
            getattr(data, field).requires_grad_(True)
        loss = (rewards.full_body_default_pose_l1(env, cfg).sum()
                + rewards.stand_joint_motion(env, cfg).sum()
                + rewards.stand_base_motion(env, cfg).sum()
                + rewards.stand_base_stillness(env, cfg).sum())
        loss.backward()
        for field in ("joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b"):
            self.assertTrue(bool(torch.isfinite(getattr(data, field).grad).all()))

    def test_reset_no_cached_state_and_no_contact_or_settling_requirements(self):
        env, cfg = environment(batch=2)
        data = env.robot.data
        data.joint_pos += .1
        data.joint_vel += .2
        functions = (rewards.full_body_default_pose_l1, rewards.torso_upright,
                     rewards.stand_joint_motion, rewards.stand_base_motion,
                     rewards.stand_base_stillness, rewards.joint_deviation_l1_when_moving)
        before_keys = set(vars(env)), set(vars(data))
        before = {name: value.clone() for name, value in vars(data).items()}
        results = [function(env, cfg) for function in functions]
        for function, expected in zip(functions, results):
            torch.testing.assert_close(function(env, cfg), expected)
        self.assertEqual((set(vars(env)), set(vars(data))), before_keys)
        for name, value in before.items():
            torch.testing.assert_close(getattr(data, name), value)
        env.command_generator.command[0, 2] = .1
        self.assertEqual(float(rewards.full_body_default_pose_l1(env, cfg)[0]), 0.)
        env.command_generator.command[0] = 0.
        data.joint_pos[0] = data.default_joint_pos[0]
        data.joint_vel[0] = 0.
        self.assertEqual(float(rewards.full_body_default_pose_l1(env, cfg)[0]), 0.)
        self.assertEqual(float(rewards.stand_joint_motion(env, cfg)[0]), 0.)
        self.assertGreater(float(rewards.full_body_default_pose_l1(env, cfg)[1]), 0.)

    def test_invalid_actual_state_shapes_are_rejected(self):
        env, cfg = environment()
        env.robot.data.default_joint_pos = torch.zeros(1, 29)
        with self.assertRaises(ValueError):
            rewards.full_body_default_pose_l1(env, cfg)
        env.robot.data.root_lin_vel_b = torch.zeros(8, 2)
        with self.assertRaises(ValueError):
            rewards.stand_base_stillness(env, cfg)


if __name__ == "__main__":
    unittest.main()
