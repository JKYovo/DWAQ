"""Independent stand-v2 identity; all deployment and feedback-v1 paths remain unchanged."""
from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.quiet_contract import quiet_experiment_contract
from legged_lab.envs.elf3.quiet_learning import Elf3DwaqStandPPO
from legged_lab.envs.elf3.quiet_rewards import upper_body_stillness
from legged_lab.envs.elf3.standing_rewards import feet_swing_height_when_moving, gait_phase_contact_when_moving


class Elf3DwaqStandV2Runner(Elf3DwaqFeedbackRunner):
    algorithm_class = Elf3DwaqStandPPO

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        identity = quiet_experiment_contract(env.cfg.quiet_standing)
        for name, function in (("gait_phase_contact", gait_phase_contact_when_moving),
                               ("feet_swing_height", feet_swing_height_when_moving)):
            term = getattr(env.cfg.reward, name)
            if term.func is not function or term.params.get("command_epsilon") != identity["command_epsilon"]:
                raise ValueError(f"stand-v2 requires its zero-command gate: {name}")
        term = env.cfg.reward.upper_body_stillness
        if (term.func is not upper_body_stillness or term.weight != .3
                or term.params["velocity_std"] != identity["physical"]["velocity_std"]):
            raise ValueError("stand-v2 requires the versioned bounded physical arm-stillness reward")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.alg.configure_standing(env, env.cfg.quiet_standing)

    def contract(self):
        result = super().contract()
        result["standing_experiment"] = quiet_experiment_contract(self.env.cfg.quiet_standing)
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        for name, value in self.alg.last_standing_metrics.items():
            self.writer.add_scalar("Standing/" + name, value, locs["it"])
        data = self.env.robot.data
        arm_ids = self.env.quiet_arm_joint_ids
        mask = self.env.get_standing_mask()
        self.writer.add_scalar("Standing/zero_command_env_fraction", mask.float().mean().item(), locs["it"])
        if mask.any():
            self.writer.add_scalar("Standing/raw_arm_velocity_rms",
                                   data.joint_vel[mask][:, arm_ids].square().mean().sqrt().item(), locs["it"])
            self.writer.add_scalar("Standing/actual_arm_pose_error_rms",
                                   (data.joint_pos[mask][:, arm_ids] - data.default_joint_pos[mask][:, arm_ids])
                                   .square().mean().sqrt().item(), locs["it"])
