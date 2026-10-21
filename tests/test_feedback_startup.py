"""Regression tests for configuration sentinels and bounded physics recording."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("startup_contract", "TienKung-Lab/legged_lab/envs/elf3/feedback_contract.py")
tools = load("startup_tools", "scripts/feedback_run_tools.py")


class StartupTests(unittest.TestCase):
    def test_legitimate_infinite_camera_constants_have_stable_distinct_hashes(self):
        plus = contract.canonical_sha256({"disabled_camera": {"far": float("inf")}})
        self.assertEqual(plus, contract.canonical_sha256({"disabled_camera": {"far": float("inf")}}))
        self.assertNotEqual(plus, contract.canonical_sha256({"disabled_camera": {"far": float("-inf")}}))
        self.assertNotEqual(plus, contract.canonical_sha256({"disabled_camera": {"far": "+inf"}}))

    def test_nan_config_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            contract.canonical_sha256({"bad": float("nan")})

    def test_physics_trace_is_bounded_physical_rate_and_does_not_change_state(self):
        count = 2
        q = torch.arange(58, dtype=torch.float).reshape(count, 29)
        data = SimpleNamespace(joint_pos=q, joint_vel=q + 1, root_ang_vel_b=torch.ones(count, 3))
        force = torch.ones(count, 3, 3)
        calls = []
        env = SimpleNamespace(physics_dt=.005, num_envs=count, device="cpu",
                              robot=SimpleNamespace(data=data), policy_joint_ids=torch.arange(28, -1, -1),
                              feet_cfg=SimpleNamespace(body_ids=[2, 0]),
                              contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=force)),
                              velocity_feedback_model=SimpleNamespace(value=q + 2),
                              episode_length_buf=torch.zeros(count, dtype=torch.long),
                              _post_physics_step=lambda: calls.append(1))
        trace = tools.PhysicsTrace(env, .01)
        before = q.clone()
        for _ in range(4):
            env._post_physics_step()
        self.assertEqual(len(calls), 4)
        self.assertEqual(trace.position, 2)
        torch.testing.assert_close(q, before)
        torch.testing.assert_close(trace.arrays["joint_pos"][0], q[:, env.policy_joint_ids])
        with tempfile.TemporaryDirectory() as folder:
            result = trace.finish(folder, [str(i) for i in range(29)])
            self.assertTrue(result["finite"])
            self.assertEqual(result["sampling_hz"], 200)
            self.assertTrue((Path(folder) / "physics_trace.npz").exists())


if __name__ == "__main__":
    unittest.main()
