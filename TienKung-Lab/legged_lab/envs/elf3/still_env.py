"""Minimal zero-command ELF3 task; raw buffers, critic and physics stay intact."""
from __future__ import annotations

from legged_lab.envs.elf3.elf3_feedback_env import Elf3DwaqFeedbackEnv
from legged_lab.envs.elf3.stillness import COMMAND_EPSILON, prepare_still_inputs


class Elf3DwaqStillEnv(Elf3DwaqFeedbackEnv):
    def get_observations(self):
        observations, history = super().get_observations()
        observations, history = prepare_still_inputs(observations, history, COMMAND_EPSILON)
        self.extras["observations"]["obs_hist"] = history
        return observations, history

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        observations, history = prepare_still_inputs(
            observations, extras["observations"]["obs_hist"], COMMAND_EPSILON
        )
        extras["observations"]["obs_hist"] = history
        return observations, rewards, dones, extras
