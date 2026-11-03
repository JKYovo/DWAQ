"""CPU execution of actual periodic reward gates and their original functions."""
import ast
from pathlib import Path
from types import SimpleNamespace as S
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


class Entity:
    """Minimal resolved SceneEntityCfg stand-in; no simulator is imported."""
    def __init__(self, name, body_ids=slice(None)):
        self.name, self.body_ids = name, body_ids


def actual_functions():
    namespace = {"torch": torch, "SceneEntityCfg": Entity, "COMMAND_EPSILON": 1e-6}
    for source, names in (
        (ROOT / "TienKung-Lab/legged_lab/mdp/rewards.py", ("gait_phase_contact", "feet_swing_height")),
        (DIRECTORY / "standing_rewards.py", ("moving_command_mask", "gait_phase_contact_when_moving",
                                             "feet_swing_height_when_moving")),
    ):
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(functions) == len(names)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
    return namespace


def environment(batch=8):
    class Scene(dict):
        pass

    generator = torch.Generator().manual_seed(42)
    positions = torch.rand(batch, 5, 3, generator=generator, dtype=torch.float64)
    forces = torch.randn(batch, 5, 3, generator=generator, dtype=torch.float64)*2
    # At least one foot swings with a positive original height cost; the other
    # supports with >1 N, allowing genuine gait scores rather than trivial zeros.
    forces[:, 1] = torch.tensor([0., 0., 20.], dtype=torch.float64)
    forces[:, 3] = 0.
    positions[:, 2, 2], positions[:, 4, 2] = .04, .23
    scene = Scene(robot=S(data=S(body_pos_w=positions)))
    scene.sensors = {"contact_sensor": S(data=S(net_forces_w=forces))}
    env = S(num_envs=batch, scene=scene,
            command_generator=S(command=torch.zeros(batch, 3, dtype=torch.float64)),
            leg_phase=torch.rand(batch, 2, generator=generator, dtype=torch.float64))
    return env, Entity("contact_sensor", [1, 3]), Entity("robot", [2, 4])


class FeedbackStandV3PeriodicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = actual_functions()

    def test_zero_disables_nonzero_original_phase_and_swing_rewards(self):
        env, sensor, asset = environment()
        env.leg_phase.zero_()
        f = self.functions
        self.assertTrue(bool((f["gait_phase_contact"](env, sensor) > 0.).all()))
        self.assertTrue(bool((f["feet_swing_height"](env, sensor, asset) > 0.).all()))
        for name, kwargs in (("gait_phase_contact_when_moving", {"sensor_cfg": sensor}),
                             ("feet_swing_height_when_moving", {"sensor_cfg": sensor, "asset_cfg": asset})):
            with self.subTest(function=name):
                self.assertEqual(f[name](env, **kwargs).tolist(), [0.]*8)

    def test_boundary_strict_zero_and_all_nonzero_velocity_axes(self):
        env, sensor, asset = environment(batch=10)
        env.command_generator.command[:] = torch.tensor([
            [0., 0., 0.], [1e-6, -1e-6, 1e-6], [1.001e-6, 0., 0.], [.1, 0., 0.],
            [-.1, 0., 0.], [0., .1, 0.], [0., -.1, 0.], [0., 0., .1], [0., 0., -.1], [1., 0., 0.]
        ], dtype=torch.float64)
        moving = self.functions["moving_command_mask"](env)
        self.assertEqual(moving.tolist(), [False, False, True, True, True, True, True, True, True, True])
        for raw_name, gated_name, kwargs in (
            ("gait_phase_contact", "gait_phase_contact_when_moving", {"sensor_cfg": sensor}),
            ("feet_swing_height", "feet_swing_height_when_moving", {"sensor_cfg": sensor, "asset_cfg": asset}),
        ):
            original = self.functions[raw_name](env, **kwargs)
            gated = self.functions[gated_name](env, **kwargs)
            self.assertEqual(gated[:2].tolist(), [0., 0.])
            self.assertTrue(torch.equal(gated[2:], original[2:]))

    def test_moving_values_bitwise_equal_parent_at_point_three_one_and_yaw(self):
        env, sensor, asset = environment(batch=384)
        for command in ([.3, 0., 0.], [1., 0., 0.], [0., 0., .1], [0., -.1, -.2]):
            env.command_generator.command[:] = torch.tensor(command, dtype=torch.float64)
            for original_name, gated_name, kwargs in (
                ("gait_phase_contact", "gait_phase_contact_when_moving", {"sensor_cfg": sensor}),
                ("feet_swing_height", "feet_swing_height_when_moving", {"sensor_cfg": sensor, "asset_cfg": asset}),
            ):
                with self.subTest(command=command, function=gated_name):
                    original = self.functions[original_name](env, **kwargs)
                    gated = self.functions[gated_name](env, **kwargs)
                    self.assertTrue(torch.equal(gated, original))

    def test_nondefault_parameters_are_forwarded_without_changes(self):
        env, sensor, asset = environment()
        env.command_generator.command[:, 2] = .1
        f = self.functions
        self.assertTrue(torch.equal(f["gait_phase_contact"](env, sensor, stance_threshold=.23),
                                    f["gait_phase_contact_when_moving"](env, sensor, stance_threshold=.23)))
        self.assertTrue(torch.equal(f["feet_swing_height"](env, sensor, asset, target_height=.17),
                                    f["feet_swing_height_when_moving"](env, sensor, asset, target_height=.17)))

    def test_no_phase_clock_raw_state_mutation_or_reset_cache(self):
        env, sensor, asset = environment()
        f = self.functions
        phase = env.leg_phase.clone()
        positions = env.scene["robot"].data.body_pos_w.clone()
        forces = env.scene.sensors["contact_sensor"].data.net_forces_w.clone()
        for original_name, gated_name, kwargs in (
            ("gait_phase_contact", "gait_phase_contact_when_moving", {"sensor_cfg": sensor}),
            ("feet_swing_height", "feet_swing_height_when_moving", {"sensor_cfg": sensor, "asset_cfg": asset}),
        ):
            self.assertEqual(f[gated_name](env, **kwargs).tolist(), [0.]*8)
            env.command_generator.command[:, 0] = .3
            self.assertTrue(torch.equal(f[gated_name](env, **kwargs), f[original_name](env, **kwargs)))
            env.command_generator.command.zero_()
            self.assertEqual(f[gated_name](env, **kwargs).tolist(), [0.]*8)
        self.assertTrue(torch.equal(env.leg_phase, phase))
        self.assertTrue(torch.equal(env.scene["robot"].data.body_pos_w, positions))
        self.assertTrue(torch.equal(env.scene.sensors["contact_sensor"].data.net_forces_w, forces))
        fresh, fresh_sensor, fresh_asset = environment()
        self.assertEqual(f["gait_phase_contact_when_moving"](fresh, fresh_sensor).tolist(), [0.]*8)
        self.assertEqual(f["feet_swing_height_when_moving"](fresh, fresh_sensor, fresh_asset).tolist(), [0.]*8)


if __name__ == "__main__":
    unittest.main()
