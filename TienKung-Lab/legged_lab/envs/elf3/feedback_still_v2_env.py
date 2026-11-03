"""Only strict-zero actor phase handling differs from feedback_v1 physics."""
from legged_lab.envs.elf3.elf3_feedback_env import Elf3DwaqFeedbackEnv
from legged_lab.envs.elf3.stillness import prepare_still_inputs


class Elf3DwaqFeedbackStillV2Env(Elf3DwaqFeedbackEnv):
    def _prepare(self, observations, history):
        return prepare_still_inputs(observations, history, self.cfg.strict_standing_v2["command_epsilon"])

    def get_observations(self):
        observations, history = self._prepare(*super().get_observations())
        self.extras["observations"]["obs_hist"] = history
        return observations, history

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        observations, history = self._prepare(observations, extras["observations"]["obs_hist"])
        extras["observations"]["obs_hist"] = history
        return observations, rewards, dones, extras
