"""CPU checks of the independent still-v2 actual waist and inherited leg terms."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as S
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load(name):
    spec = importlib.util.spec_from_file_location("cpu_"+name, DIRECTORY/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


waist = load("still_v2_rewards")
leg = load("still_pose_rewards")
contract = load("still_v2_contract")


def environment(batch=8):
    home = torch.arange(29, dtype=torch.float64).expand(batch, -1).clone()/100.
    data = S(joint_pos=home.clone(), default_joint_pos=home)
    env = S(num_envs=batch, robot=S(data=data),
            command_generator=S(command=torch.zeros(batch, 3, dtype=torch.float64)))
    return env, S(joint_ids=[20, 4, 27])


class StillV2RewardTests(unittest.TestCase):
    def test_default_actual_pose_cost_zero(self):
        env, cfg = environment()
        self.assertEqual(waist.stand_waist_pose_l1(env, cfg).tolist(), [0.]*8)

    def test_each_waist_joint_uses_same_actual_default_index(self):
        env, cfg = environment(batch=3)
        for row, index in enumerate(cfg.joint_ids):
            env.robot.data.joint_pos[row, index] += .123
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg),
                                   torch.full((3,), .123, dtype=torch.float64), atol=1e-15, rtol=0)

    def test_only_three_waist_joints_and_sum_not_mean_or_ratio(self):
        env, cfg = environment(batch=3)
        env.robot.data.joint_pos[:, cfg.joint_ids] += torch.tensor([[.1, -.2, .3], [1., -2., 3.], [10., -20., 30.]])
        expected = torch.tensor([.6, 6., 60.], dtype=torch.float64)
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)
        env.robot.data.joint_pos[:, 0] += 999.
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)

    def test_index_order_and_nonzero_synthetic_default_do_not_cross_pair(self):
        env, cfg = environment(batch=1)
        env.robot.data.joint_pos[:, cfg.joint_ids] += torch.tensor([.1, -.2, .3], dtype=torch.float64)
        expected = waist.stand_waist_pose_l1(env, cfg)
        for indices in ([20, 27, 4], [4, 20, 27], [27, 4, 20]):
            cfg.joint_ids = indices
            torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)

    def test_strict_all_three_commands_include_zero_boundary_not_point_one_yaw(self):
        env, cfg = environment(batch=7)
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        env.command_generator.command[:] = torch.tensor([
            [0., 0., 0.], [1e-6, -1e-6, 1e-6], [1.001e-6, 0., 0.],
            [.1, 0., 0.], [0., -.1, 0.], [0., 0., .1], [0., 0., -.1]
        ], dtype=torch.float64)
        expected = torch.tensor([.3, .3, 0., 0., 0., 0., 0.], dtype=torch.float64)
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)

    def test_nonfinite_commands_off(self):
        env, cfg = environment(batch=3)
        env.robot.data.joint_pos += .5
        env.command_generator.command[:, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        self.assertEqual(waist.stand_waist_pose_l1(env, cfg).tolist(), [0.]*3)

    def test_no_target_velocity_phase_contact_or_settling_dependency(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        expected = waist.stand_waist_pose_l1(env, cfg)
        env.robot.data.joint_pos_target = torch.full((8, 29), -999.)
        env.robot.data.joint_vel = torch.full((8, 29), 1000.)
        env.robot.data.body_quat_w = torch.full((8, 3, 4), float("nan"))
        env.actions = torch.full((8, 29), 999.)
        env.leg_phase = torch.rand(8, 2)
        env.velocity_feedback_model = S(value=torch.zeros(8, 29))
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)
        env.robot.data.default_joint_pos[:, cfg.joint_ids] += .1
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), torch.zeros_like(expected), atol=1e-14, rtol=0)

    def test_no_state_mutation_or_reset_cache(self):
        env, cfg = environment()
        env.robot.data.joint_pos[:, cfg.joint_ids] += .1
        before = env.robot.data.joint_pos.clone()
        expected = waist.stand_waist_pose_l1(env, cfg)
        self.assertTrue(torch.equal(env.robot.data.joint_pos, before))
        env.command_generator.command[:, 0] = .3
        self.assertEqual(waist.stand_waist_pose_l1(env, cfg).tolist(), [0.]*8)
        env.command_generator.command.zero_()
        torch.testing.assert_close(waist.stand_waist_pose_l1(env, cfg), expected)
        fresh, fresh_cfg = environment()
        fresh.robot.data.joint_pos[:, fresh_cfg.joint_ids] += .1
        torch.testing.assert_close(waist.stand_waist_pose_l1(fresh, fresh_cfg), expected)

    def test_inherited_leg_and_new_waist_have_intended_total_coefficients(self):
        env, waist_cfg = environment(batch=3)
        leg_cfg = S(joint_ids=[3, 8, 16, 25])
        env.robot.data.joint_pos[:, leg_cfg.joint_ids] += .2
        env.robot.data.joint_pos[:, waist_cfg.joint_ids] += .3
        env.command_generator.command[:] = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [0., 0., .1]])
        leg_cost = (env.robot.data.joint_pos[:, leg_cfg.joint_ids]-env.robot.data.default_joint_pos[:, leg_cfg.joint_ids]).abs().sum(-1)
        waist_cost = (env.robot.data.joint_pos[:, waist_cfg.joint_ids]-env.robot.data.default_joint_pos[:, waist_cfg.joint_ids]).abs().sum(-1)
        total_leg = -.02*leg_cost-.13*leg.stand_leg_pose_l1(env, leg_cfg)
        total_waist = -.20*waist_cost-.10*waist.stand_waist_pose_l1(env, waist_cfg)
        torch.testing.assert_close(total_leg, leg_cost*leg_cost.new_tensor([-.15, -.02, -.02]))
        torch.testing.assert_close(total_waist, waist_cost*waist_cost.new_tensor([-.30, -.20, -.20]))

    def test_negative_incremental_reward_gradient_points_to_default(self):
        env, cfg = environment(batch=1)
        env.robot.data.joint_pos = (env.robot.data.default_joint_pos-.2).requires_grad_()
        (-.1*waist.stand_waist_pose_l1(env, cfg)).sum().backward()
        expected = torch.zeros(1, 29, dtype=torch.float64)
        expected[:, cfg.joint_ids] = .1
        torch.testing.assert_close(env.robot.data.joint_pos.grad, expected)

    def test_bad_selection_shape_and_epsilon_fail(self):
        for indices in ([0], [0, 1], [0, 0, 1], [0, 1, 29], [0, 1, -1], [True, False, True],
                        [], slice(None), torch.tensor([0., 1., 2.])):
            env, cfg = environment()
            cfg.joint_ids = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                waist.stand_waist_pose_l1(env, cfg)
        for epsilon in (-1., float("nan"), float("inf"), True):
            env, cfg = environment()
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                waist.stand_waist_pose_l1(env, cfg, epsilon)
        env, cfg = environment()
        env.robot.data.default_joint_pos = torch.zeros(7, 29)
        with self.assertRaises(ValueError):
            waist.stand_waist_pose_l1(env, cfg)

    def test_pose_settings_exact_parent_reference_and_additive_weights(self):
        settings = contract.pose_settings()
        self.assertEqual(settings["version"], "elf3_dwaq_still_v2")
        self.assertEqual(settings["parent_task"], "elf3_dwaq_still")
        self.assertEqual(settings["reference_task"], "elf3_dwaq_still_pose_v1")
        self.assertEqual(settings["leg_joint_names"], list(leg.STAND_POSE_JOINT_NAMES))
        self.assertEqual(settings["waist_joint_names"], list(waist.STAND_WAIST_JOINT_NAMES))
        self.assertAlmostEqual(settings["leg_baseline_weight"]+settings["leg_incremental_weight"], -.15)
        self.assertAlmostEqual(settings["waist_baseline_weight"]+settings["waist_incremental_weight"], -.30)
        settings["leg_joint_names"].clear()
        self.assertEqual(len(contract.pose_settings()["leg_joint_names"]), 4)

    def test_pose_contract_rejects_unsupported_settings_before_source_reads(self):
        settings = contract.pose_settings()
        settings["leg_incremental_weight"] = -.08
        with self.assertRaises(ValueError):
            contract.pose_experiment_contract(settings)


if __name__ == "__main__":
    unittest.main()
