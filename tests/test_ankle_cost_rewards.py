"""CPU-only checks for the isolated ankle-effort/stance-speed experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "TienKung-Lab/legged_lab/envs/elf3/ankle_cost_rewards.py"
SPEC = importlib.util.spec_from_file_location("isolated_ankle_cost_rewards", SOURCE)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)


def speed_env(phase=((.2, .7),), speed=((.1, .2),), command=((.3, 0., 0.),)):
    return SimpleNamespace(
        ankle_cost_feet_body_names=rewards.FOOT_BODY_NAMES,
        ankle_cost_feet_speed_mean=torch.tensor(speed, dtype=torch.float64),
        leg_phase=torch.tensor(phase, dtype=torch.float64),
        command_generator=SimpleNamespace(command=torch.tensor(command, dtype=torch.float64)),
    )


class TestAnkleTorque(unittest.TestCase):
    def setUp(self):
        self.names = ["l_knee_y_joint", "r_ankle_x_joint", "l_ankle_y_joint",
                      "waist_y_joint", "r_ankle_y_joint", "l_ankle_x_joint", "r_knee_y_joint"]
        self.torque = torch.tensor([[100., 1., 2., 200., 3., 4., 300.],
                                    [100., -2., 3., 200., -4., 5., 300.]], dtype=torch.float64, requires_grad=True)
        self.asset = SimpleNamespace(joint_names=self.names, data=SimpleNamespace(applied_torque=self.torque))
        self.env = SimpleNamespace(scene={"robot": self.asset}, command_generator=SimpleNamespace(command=torch.zeros(2, 3)))
        self.cfg = SimpleNamespace(name="robot", joint_ids=[2, 5, 4, 1])

    def test_named_selection_is_not_simulator_order(self):
        actual = rewards.ankle_torque_l2(self.env, self.cfg)
        torch.testing.assert_close(actual, torch.tensor([30., 54.], dtype=torch.float64))
        self.cfg.joint_ids = [1, 4, 5, 2]
        torch.testing.assert_close(rewards.ankle_torque_l2(self.env, self.cfg), actual)

    def test_gradients_only_four_ankles(self):
        rewards.ankle_torque_l2(self.env, self.cfg).sum().backward()
        expected = torch.zeros_like(self.torque)
        expected[:, self.cfg.joint_ids] = 2 * self.torque.detach()[:, self.cfg.joint_ids]
        torch.testing.assert_close(self.torque.grad, expected)

    def test_command_does_not_gate_torque_cost(self):
        first = rewards.ankle_torque_l2(self.env, self.cfg)
        self.env.command_generator.command[:] = torch.tensor([.5, -.1, .2])
        torch.testing.assert_close(rewards.ankle_torque_l2(self.env, self.cfg), first)

    def test_rejects_wrong_or_ambiguous_joint_ids(self):
        for indices in ([0, 1, 2, 4], [1, 1, 2, 4], [1, 2, 4], [1, 2, 4, 9], slice(None)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                rewards.ankle_torque_l2(self.env, SimpleNamespace(name="robot", joint_ids=indices))

    def test_nonfinite_torque_rejected(self):
        self.asset.data.applied_torque = self.torque.detach().clone()
        self.asset.data.applied_torque[0, 2] = float("nan")
        with self.assertRaises(RuntimeError):
            rewards.ankle_torque_l2(self.env, self.cfg)


class TestStanceSpeed(unittest.TestCase):
    def test_exact_formula_stance_vs_swing(self):
        env = speed_env()
        actual = rewards.stance_foot_speed_cost(env)
        expected = -torch.expm1(torch.tensor(-200 * .1**2, dtype=torch.float64))
        torch.testing.assert_close(actual[0], expected, rtol=1e-10, atol=1e-10)
        self.assertTrue(bool(((actual >= 0) & (actual <= 2)).all()))

    def test_all_zero_commands_off_nonzero_yaw_and_lateral_on(self):
        command = ((0., 0., 0.), (1e-6, -1e-6, 1e-6), (0., .1, 0.), (0., 0., .1))
        env = speed_env(phase=((.2, .7),) * 4, speed=((.1, .2),) * 4, command=command)
        value = rewards.stance_foot_speed_cost(env)
        torch.testing.assert_close(value[:2], torch.zeros(2, dtype=value.dtype))
        self.assertGreater(float(value[2]), 0)
        torch.testing.assert_close(value[2], value[3])

    def test_phase_boundary_and_wrap_are_continuous(self):
        for boundary in (0., .55, 1.):
            env = speed_env(phase=((boundary - 1e-7, boundary - 1e-7),
                                   (boundary + 1e-7, boundary + 1e-7)),
                            speed=((.1, .1),) * 2, command=((.3, 0., 0.),) * 2)
            value = rewards.stance_foot_speed_cost(env)
            self.assertLess(float((value[0] - value[1]).abs()), 2e-5)

    def test_phase_is_periodic(self):
        env = speed_env(phase=((.03, .54), (1.03, -.46)), speed=((.1, .2),) * 2,
                        command=((.3, 0., 0.),) * 2)
        value = rewards.stance_foot_speed_cost(env)
        torch.testing.assert_close(value[0], value[1], rtol=1e-10, atol=1e-10)

    def test_speed_and_phase_gradients_are_finite(self):
        env = speed_env(phase=((.545, .005),))
        env.ankle_cost_feet_speed_mean.requires_grad_(True)
        env.leg_phase.requires_grad_(True)
        rewards.stance_foot_speed_cost(env).sum().backward()
        self.assertTrue(bool(torch.isfinite(env.leg_phase.grad).all()))
        self.assertTrue(bool(torch.isfinite(env.ankle_cost_feet_speed_mean.grad).all()))
        self.assertTrue(bool((env.ankle_cost_feet_speed_mean.grad > 0).all()))

    def test_read_only_cache_and_no_double_updates(self):
        env = speed_env()
        snapshot = (env.ankle_cost_feet_speed_mean.clone(), env.leg_phase.clone(),
                    env.command_generator.command.clone(), set(vars(env)))
        first = rewards.stance_foot_speed_cost(env)
        second = rewards.stance_foot_speed_cost(env)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(snapshot[0], env.ankle_cost_feet_speed_mean)
        torch.testing.assert_close(snapshot[1], env.leg_phase)
        torch.testing.assert_close(snapshot[2], env.command_generator.command)
        self.assertEqual(snapshot[3], set(vars(env)))

    def test_zero_speed_zero_cost_and_extreme_finite_speed_bounded(self):
        env = speed_env(phase=((.2, .2), (.2, .2)), speed=((0., 0.), (1e20, 1e20)),
                        command=((.3, 0., 0.),) * 2)
        value = rewards.stance_foot_speed_cost(env)
        torch.testing.assert_close(value, torch.tensor([0., 2.], dtype=value.dtype))

    def test_missing_or_reversed_feet_layout_rejected(self):
        for layout in ((), tuple(reversed(rewards.FOOT_BODY_NAMES))):
            env = speed_env()
            env.ankle_cost_feet_body_names = layout
            with self.assertRaises(ValueError):
                rewards.stance_foot_speed_cost(env)

    def test_invalid_shapes_parameters_and_nonfinite_rejected(self):
        for key, bad in (("phase_sigma", 0.), ("phase_sigma", float("nan")),
                         ("stance_threshold", 1.), ("speed_scale", -1.), ("command_epsilon", -1.)):
            with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                rewards.stance_foot_speed_cost(speed_env(), **{key: bad})
        env = speed_env()
        env.ankle_cost_feet_speed_mean = torch.ones(1, 3)
        with self.assertRaises(ValueError):
            rewards.stance_foot_speed_cost(env)
        for bad in (-.1, float("nan"), float("inf")):
            env = speed_env()
            env.ankle_cost_feet_speed_mean[0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                rewards.stance_foot_speed_cost(env)


if __name__ == "__main__":
    unittest.main()
