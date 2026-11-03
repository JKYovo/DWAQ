"""Opt-in zero-command stillness; all feedback and actuator PD are inherited."""
from __future__ import annotations

import torch

from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.elf3_feedback_env import Elf3DwaqFeedbackEnv
from legged_lab.envs.elf3.quiet_rewards import zero_command_mask


class Elf3DwaqStandV2Env(Elf3DwaqFeedbackEnv):
    def init_buffers(self):
        super().init_buffers()
        names = tuple(JOINT_NAMES[15:])
        indices, resolved_names = self.robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved_names) != names or len(indices) != 14:
            raise ValueError(f"Quiet standing requires the fourteen named arm joints: {resolved_names}")
        self.quiet_arm_joint_ids = torch.tensor(indices, device=self.device, dtype=torch.long)

    def get_standing_mask(self):
        """Snapshot current zero command without timers or physical-state gates.

        PPO records this immediately beside obs_t/history_t. A moving-to-zero
        command change takes effect on this first read, independent of contact,
        current motion or episode age.
        """
        return zero_command_mask(self, epsilon=self.cfg.quiet_standing.get("command_epsilon", 1e-6))
