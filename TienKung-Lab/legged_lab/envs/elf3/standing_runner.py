"""Independent reward identity; no change to feedback PPO, Actor or deployment."""
from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.standing_contract import standing_experiment_contract
from legged_lab.envs.elf3.standing_rewards import (
    feet_swing_height_when_moving,
    gait_phase_contact_when_moving,
)


class Elf3DwaqStandRunner(Elf3DwaqFeedbackRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        identity = standing_experiment_contract(env.cfg.standing_experiment)
        expected = {
            "gait_phase_contact": gait_phase_contact_when_moving,
            "feet_swing_height": feet_swing_height_when_moving,
        }
        for name, function in expected.items():
            term = getattr(env.cfg.reward, name)
            if term.func is not function or term.params.get("command_epsilon") != identity["command_epsilon"]:
                raise ValueError(f"Standing task requires its private zero-command gate for {name}")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def contract(self):
        result = super().contract()
        result["standing_experiment"] = standing_experiment_contract(self.env.cfg.standing_experiment)
        return result
