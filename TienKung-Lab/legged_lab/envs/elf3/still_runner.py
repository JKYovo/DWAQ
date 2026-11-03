"""Minimal standing task: original PPO, no extra phase/pose auxiliary branch."""
from legged_lab.elf3_contract import JOINT_NAMES
from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.standing_rewards import feet_swing_height_when_moving, gait_phase_contact_when_moving
from legged_lab.envs.elf3.still_contract import still_experiment_contract
from legged_lab.envs.elf3.stillness import stand_joint_motion


class Elf3DwaqStillRunner(Elf3DwaqFeedbackRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        identity = still_experiment_contract(env.cfg.stillness)
        for name, function in (("gait_phase_contact", gait_phase_contact_when_moving),
                               ("feet_swing_height", feet_swing_height_when_moving)):
            term = getattr(env.cfg.reward, name)
            if term.func is not function or term.params.get("command_epsilon") != identity["command_epsilon"]:
                raise ValueError(f"Minimal standing requires its zero-command reward gate: {name}")
        term = env.cfg.reward.stand_joint_motion
        if (term.func is not stand_joint_motion or term.weight != identity["joint_motion_weight"]
                or tuple(term.params["asset_cfg"].joint_names) != tuple(JOINT_NAMES)
                or term.params.get("command_epsilon") != identity["command_epsilon"]):
            raise ValueError("Minimal standing requires the versioned raw all-joint motion cost")
        if train_cfg["algorithm"].get("class_name") != "DWAQPPO":
            raise ValueError("Minimal standing uses the original DWAQPPO, not the v2 auxiliary algorithm")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def contract(self):
        result = super().contract()
        result["standing_experiment"] = still_experiment_contract(self.env.cfg.stillness)
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        command = self.env.command_generator.command[:, :3]
        mask = command.isfinite().all(-1) & (command.abs().amax(-1) <= self.env.cfg.stillness["command_epsilon"])
        self.writer.add_scalar("Still/zero_command_fraction", mask.float().mean().item(), locs["it"])
        if mask.any():
            data = self.env.robot.data
            self.writer.add_scalar("Still/raw_joint_velocity_abs_mean",
                                   data.joint_vel[mask].abs().mean().item(), locs["it"])
            self.writer.add_scalar("Still/root_linear_speed_mean",
                                   data.root_lin_vel_b[mask].norm(dim=-1).mean().item(), locs["it"])
            self.writer.add_scalar("Still/root_angular_speed_mean",
                                   data.root_ang_vel_b[mask].norm(dim=-1).mean().item(), locs["it"])
