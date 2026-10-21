"""Small-run diagnostics; recording never filters actuator targets or changes physics."""
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch


class PhysicsTrace:
    """Bounded GPU recorder sampled at physics rate, copied to CPU only on finish."""
    def __init__(self, env, seconds, max_envs=4):
        if not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Physics tracing must be bounded to (0, 120] seconds")
        self.dt = env.physics_dt
        self.capacity = math.ceil(seconds / self.dt)
        self.count = min(max_envs, env.num_envs)
        self.position = 0
        self.arrays = {
            "joint_pos": torch.empty(self.capacity, self.count, 29, device=env.device),
            "joint_vel": torch.empty(self.capacity, self.count, 29, device=env.device),
            "reported_joint_vel": torch.empty(self.capacity, self.count, 29, device=env.device),
            "gyro": torch.empty(self.capacity, self.count, 3, device=env.device),
            "feet_force": torch.empty(self.capacity, self.count, 2, 3, device=env.device),
            "episode_control_step": torch.empty(self.capacity, self.count, dtype=torch.long, device=env.device),
        }
        original = env._post_physics_step

        def observe(instance):
            original()
            if self.position >= self.capacity:
                return
            index = self.position
            ids = instance.policy_joint_ids
            data = instance.robot.data
            self.arrays["joint_pos"][index].copy_(data.joint_pos[:self.count, ids])
            self.arrays["joint_vel"][index].copy_(data.joint_vel[:self.count, ids])
            self.arrays["reported_joint_vel"][index].copy_(instance.velocity_feedback_model.value[:self.count, ids])
            self.arrays["gyro"][index].copy_(data.root_ang_vel_b[:self.count])
            self.arrays["feet_force"][index].copy_(
                instance.contact_sensor.data.net_forces_w[:self.count, instance.feet_cfg.body_ids])
            self.arrays["episode_control_step"][index].copy_(instance.episode_length_buf[:self.count])
            self.position += 1

        env._post_physics_step = MethodType(observe, env)

    def finish(self, directory, names):
        directory = Path(directory)
        arrays = {key: value[:self.position].detach().cpu().numpy() for key, value in self.arrays.items()}
        arrays.update(physics_dt=np.asarray(self.dt), joint_names=np.asarray(names))
        np.savez_compressed(directory / "physics_trace.npz", **arrays)
        summary = {"sampling_hz": 1 / self.dt, "samples": self.position,
                   "environments": self.count, "physical_state_not_action_targets": True,
                   "training_exploration_not_deterministic_policy_evaluation": True,
                   "finite": all(np.isfinite(value).all() for value in arrays.values()
                                 if np.issubdtype(value.dtype, np.number)),
                   "spectral_validation": "Not an acceptance test: random-policy short episodes may be insufficient. "
                                          "Use stable matched-speed episodes for vibration comparisons."}
        if not summary["finite"]:
            raise FloatingPointError("Nonfinite physics trace")
        (directory / "physics_trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
