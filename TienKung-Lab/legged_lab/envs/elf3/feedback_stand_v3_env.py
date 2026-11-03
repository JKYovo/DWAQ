"""Zero-only Actor phase canonicalization; feedback-v1 physics/reset unchanged."""
from legged_lab.envs.elf3.elf3_feedback_env import Elf3DwaqFeedbackEnv
from legged_lab.envs.elf3.feedback_stand_v3_inputs import prepare_stand_inputs


class Elf3DwaqFeedbackStandV3Env(Elf3DwaqFeedbackEnv):
    def _prepare(self, observations, history):
        # Clone inputs, leaving raw history, critic and the gait clock untouched.
        return prepare_stand_inputs(observations, history, self.cfg.strict_standing_v3["command_epsilon"])

    def get_observations(self):
        observations, history = self._prepare(*super().get_observations())
        self.extras["observations"]["obs_hist"] = history
        return observations, history

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        observations, history = self._prepare(observations, extras["observations"]["obs_hist"])
        extras["observations"]["obs_hist"] = history
        return observations, rewards, dones, extras
