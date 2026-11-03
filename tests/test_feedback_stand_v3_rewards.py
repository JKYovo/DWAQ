"""CPU tests of extra zero-command pose sum-L1 and raw joint-motion mean-L1."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as S
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_rewards.py"
SPEC = importlib.util.spec_from_file_location("cpu_feedback_stand_v3_minimal_rewards", SOURCE)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)


def environment(batch=8):
    home = torch.arange(29, dtype=torch.float64).expand(batch, -1).clone()/100.
    data = S(joint_pos=home.clone(), default_joint_pos=home)
    robot = S(data=data)
    env = S(num_envs=batch, scene={"robot": robot}, robot=robot,
            command_generator=S(command=torch.zeros(batch, 3, dtype=torch.float64)))
    return env, S(name="robot", joint_ids=list(reversed(range(29))))


class FeedbackStandV3MinimalRewardTests(unittest.TestCase):
    def test_actual_default_pose_cost_is_zero_without_any_other_state(self):
        env, cfg = environment()
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), torch.zeros(8, dtype=torch.float64))

    def test_every_one_of_29_joints_point123_cost_is_point123(self):
        env, cfg = environment(batch=29)
        env.robot.data.joint_pos += torch.eye(29, dtype=torch.float64)*.123
        expected = torch.full((29,), .123, dtype=torch.float64)
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected, rtol=0, atol=1e-15)
        cfg.joint_ids = slice(None)
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected, rtol=0, atol=1e-15)

    def test_random_native_orders_pair_actual_and_default_by_same_indices(self):
        contract = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
        names = contract["joint_names"]
        generator = torch.Generator().manual_seed(42)
        for _ in range(50):
            permutation = torch.randperm(29, generator=generator).tolist()
            native_names = [names[index] for index in permutation]
            ids = [native_names.index(name) for name in names]
            default = torch.tensor([[contract["default_joint_pos"][name] for name in native_names]], dtype=torch.float64)
            data = S(joint_pos=default.clone(), default_joint_pos=default)
            env = S(num_envs=1, scene={"robot": S(data=data)},
                    command_generator=S(command=torch.zeros(1, 3, dtype=torch.float64)))
            cfg = S(name="robot", joint_ids=ids)
            self.assertEqual(rewards.zero_command_default_pose_l1(env, cfg).item(), 0.)
            for index in range(29):
                data.joint_pos.copy_(default)
                data.joint_pos[0, index] += .123
                self.assertAlmostEqual(rewards.zero_command_default_pose_l1(env, cfg).item(), .123, places=14)

    def test_sum_l1_has_no_mean_ratio_or_saturation(self):
        env, cfg = environment(batch=3)
        env.robot.data.joint_pos[0] += .1
        env.robot.data.joint_pos[1] += 1.
        env.robot.data.joint_pos[2] += 10.
        expected = torch.tensor([2.9, 29., 290.], dtype=torch.float64)
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected)

    def test_knees_and_hip_roll_are_not_omitted_or_diluted(self):
        names = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())["joint_names"]
        env, cfg = environment(batch=3)
        knee = [names.index(name) for name in ("l_knee_y_joint", "r_knee_y_joint")]
        hip_roll = [names.index(name) for name in ("l_hip_x_joint", "r_hip_x_joint")]
        env.robot.data.joint_pos[0, knee] += 1.
        env.robot.data.joint_pos[1, hip_roll] += .5
        env.robot.data.joint_pos[2, knee+hip_roll] += 2.
        expected = torch.tensor([2., 1., 8.], dtype=torch.float64)
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected)
        torch.testing.assert_close(-.1*rewards.zero_command_default_pose_l1(env, cfg), -.1*expected)

    def test_finite_strict_zero_includes_boundary_not_point_one_or_pure_yaw(self):
        commands = torch.tensor([[0., 0., 0.], [1e-6, -1e-6, 1e-6], [1.001e-6, 0., 0.],
                                 [.1, 0., 0.], [0., -.1, 0.], [0., 0., .1]], dtype=torch.float64)
        self.assertEqual(rewards.strict_zero_command(commands).tolist(), [True, True, False, False, False, False])
        env, cfg = environment(batch=6)
        env.command_generator.command.copy_(commands)
        env.robot.data.joint_pos += .1
        expected = torch.tensor([2.9, 2.9, 0., 0., 0., 0.], dtype=torch.float64)
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected)

    def test_invalid_commands_do_not_activate_standing_cost(self):
        env, cfg = environment(batch=4)
        env.robot.data.joint_pos += .1
        env.command_generator.command[:3, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        cost = rewards.zero_command_default_pose_l1(env, cfg)
        self.assertEqual(cost[:3].tolist(), [0., 0., 0.])
        self.assertAlmostEqual(cost[3].item(), 2.9)

    def test_actual_default_used_not_targets_actions_velocities_or_torques(self):
        env, cfg = environment()
        env.robot.data.joint_pos += .1
        expected = rewards.zero_command_default_pose_l1(env, cfg)
        env.actions = torch.full((8, 29), 999.)
        env.robot.data.joint_pos_target = torch.full((8, 29), -999.)
        env.robot.data.joint_vel = torch.full((8, 29), 100.)
        env.robot.data.applied_torque = torch.full((8, 29), 999.)
        env.robot.data.body_quat_w = torch.full((8, 3, 4), float("nan"))
        env.velocity_feedback_model = S(value=torch.full((8, 29), -100.))
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected)
        env.robot.data.default_joint_pos += .1
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), torch.zeros_like(expected), atol=1e-14, rtol=0)

    def test_no_support_settling_terrain_or_reset_cache_and_no_mutations(self):
        env, cfg = environment()
        env.robot.data.joint_pos += .1
        before_actual = env.robot.data.joint_pos.clone()
        before_default = env.robot.data.default_joint_pos.clone()
        before_command = env.command_generator.command.clone()
        expected = rewards.zero_command_default_pose_l1(env, cfg)
        self.assertTrue(torch.equal(env.robot.data.joint_pos, before_actual))
        self.assertTrue(torch.equal(env.robot.data.default_joint_pos, before_default))
        self.assertTrue(torch.equal(env.command_generator.command, before_command))
        env.command_generator.command[:, 2] = .1
        self.assertEqual(rewards.zero_command_default_pose_l1(env, cfg).tolist(), [0.]*8)
        env.command_generator.command.zero_()
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(env, cfg), expected)
        fresh, fresh_cfg = environment()
        fresh.robot.data.joint_pos += .1
        torch.testing.assert_close(rewards.zero_command_default_pose_l1(fresh, fresh_cfg), expected)

    def test_negative_weight_gradient_points_toward_correct_actual_default(self):
        env, cfg = environment(batch=1)
        env.robot.data.joint_pos = (env.robot.data.default_joint_pos-.4).requires_grad_()
        (-.1*rewards.zero_command_default_pose_l1(env, cfg)).sum().backward()
        torch.testing.assert_close(env.robot.data.joint_pos.grad, torch.full((1, 29), .1, dtype=torch.float64))

    def test_bad_indices_fail_not_silently_select_subset_duplicate_or_wrap(self):
        for indices in ([0], [], list(range(28))+[0], list(range(28))+[29], list(range(28))+[-1],
                        [True]*29, torch.arange(29).float()):
            env, cfg = environment()
            cfg.joint_ids = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                rewards.zero_command_default_pose_l1(env, cfg)

    def test_bad_shapes_or_epsilon_rejected(self):
        for epsilon in (-1., float("nan"), float("inf"), True, "invalid"):
            env, cfg = environment()
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                rewards.zero_command_default_pose_l1(env, cfg, epsilon)
        for value in (torch.zeros(8, 28), torch.zeros(8, 29, dtype=torch.int64), torch.zeros(7, 29)):
            env, cfg = environment()
            env.robot.data.joint_pos = value
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                rewards.zero_command_default_pose_l1(env, cfg)

    def test_nonfinite_actual_at_zero_is_not_hidden_by_extra_state_gate(self):
        env, cfg = environment(batch=1)
        env.robot.data.joint_pos[0, 3] = float("nan")
        self.assertTrue(torch.isnan(rewards.zero_command_default_pose_l1(env, cfg)).item())

    def test_old_component_score_functions_are_absent(self):
        for name in ("COMPONENT_WEIGHTS", "stand_components", "zero_stand_score", "joint_deviation_l1_when_moving"):
            self.assertFalse(hasattr(rewards, name))

    def test_joint_motion_zero_and_every_joint_point29_gives_mean_point01(self):
        env, cfg = environment(batch=29)
        env.robot.data.joint_vel = torch.zeros(29, 29, dtype=torch.float64)
        expected = torch.zeros(29, dtype=torch.float64)
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(env, cfg), expected)
        env.robot.data.joint_vel += torch.eye(29, dtype=torch.float64)*.29
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(env, cfg),
                                   torch.full((29,), .01, dtype=torch.float64), atol=1e-15, rtol=0)

    def test_joint_motion_reads_raw_not_filter_targets_actions_or_pose(self):
        env, cfg = environment()
        env.robot.data.joint_vel = torch.full((8, 29), -.3, dtype=torch.float64)
        env.velocity_feedback_model = S(value=torch.zeros(8, 29))
        env.actions = torch.full((8, 29), 1000.)
        env.robot.data.joint_pos_target = torch.full((8, 29), -1000.)
        env.robot.data.joint_vel_target = torch.zeros(8, 29)
        del env.robot.data.joint_pos
        del env.robot.data.default_joint_pos
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(env, cfg),
                                   torch.full((8,), .3, dtype=torch.float64))
        env.velocity_feedback_model.value.fill_(999.)
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(env, cfg),
                                   torch.full((8,), .3, dtype=torch.float64))
        env.robot.data.joint_vel.zero_()
        self.assertEqual(rewards.zero_command_joint_motion_l1(env, cfg).tolist(), [0.]*8)

    def test_joint_motion_nonzero_pure_yaw_and_nonfinite_commands_are_off(self):
        env, cfg = environment(batch=4)
        env.robot.data.joint_vel = torch.ones(4, 29, dtype=torch.float64)
        env.command_generator.command[:] = torch.tensor([[0., 0., 0.], [.1, 0., 0.],
                                                        [0., 0., .1], [float("nan"), 0., 0.]])
        self.assertEqual(rewards.zero_command_joint_motion_l1(env, cfg).tolist(), [1., 0., 0., 0.])

    def test_joint_motion_no_gate_cache_mutation_or_cross_reset_state(self):
        env, cfg = environment()
        env.robot.data.joint_vel = torch.full((8, 29), .29, dtype=torch.float64)
        before = env.robot.data.joint_vel.clone()
        expected = rewards.zero_command_joint_motion_l1(env, cfg)
        self.assertTrue(torch.equal(env.robot.data.joint_vel, before))
        env.command_generator.command[:, 0] = .1
        self.assertEqual(rewards.zero_command_joint_motion_l1(env, cfg).tolist(), [0.]*8)
        env.command_generator.command.zero_()
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(env, cfg), expected)
        fresh, fresh_cfg = environment()
        fresh.robot.data.joint_vel = before.clone()
        torch.testing.assert_close(rewards.zero_command_joint_motion_l1(fresh, fresh_cfg), expected)
        fresh.robot.data.joint_vel.zero_()
        self.assertEqual(rewards.zero_command_joint_motion_l1(fresh, fresh_cfg).tolist(), [0.]*8)

    def test_joint_motion_validates_all29_indices_and_velocity_shape(self):
        for indices in ([0], list(range(28))+[0], list(range(28))+[29], [True]*29):
            env, cfg = environment()
            env.robot.data.joint_vel = torch.zeros(8, 29, dtype=torch.float64)
            cfg.joint_ids = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                rewards.zero_command_joint_motion_l1(env, cfg)
        for value in (torch.zeros(8, 28), torch.zeros(8, 29, dtype=torch.int64), torch.zeros(7, 29)):
            env, cfg = environment()
            env.robot.data.joint_vel = value
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                rewards.zero_command_joint_motion_l1(env, cfg)


if __name__ == "__main__":
    unittest.main()
