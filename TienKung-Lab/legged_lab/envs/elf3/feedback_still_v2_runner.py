"""Independent feedback-v1-based standing objective; original DWAQ PPO/VAE."""
from copy import deepcopy
import torch

from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.feedback_still_v2_contract import standing_experiment_contract


def validate_reward_identity(cfg):
    """Check private terms even through native train/play without its launcher."""
    from legged_lab.envs.elf3.elf3_feedback_still_v2_config import Elf3DwaqFeedbackStillV2EnvCfg
    expected = Elf3DwaqFeedbackStillV2EnvCfg()
    names = ("gait_phase_contact", "feet_swing_height", "joint_deviation_hip", "joint_deviation_ankle",
             "joint_deviation_arms", "joint_deviation_legs", "zero_full_body_pose", "zero_torso_upright",
             "zero_joint_motion", "zero_base_stillness")
    for name in names:
        actual, reference = getattr(cfg.reward, name), getattr(expected.reward, name)
        if actual.func is not reference.func or actual.weight != reference.weight:
            raise ValueError(f"Incorrect standing v2 reward function/weight: {name}")
        if set(actual.params) != set(reference.params):
            raise ValueError(f"Incorrect standing v2 reward parameters: {name}")
        for key, value in reference.params.items():
            if key in ("asset_cfg", "sensor_cfg"):
                # Resolved numeric ids legitimately appear after env creation.
                for field in ("name", "joint_names", "body_names", "preserve_order"):
                    if getattr(actual.params[key], field, None) != getattr(value, field, None):
                        raise ValueError(f"Incorrect standing v2 named entity: {name}.{field}")
            elif actual.params[key] != value:
                raise ValueError(f"Incorrect standing v2 parameter: {name}.{key}")


def validate_standing_configuration(cfg, train_cfg):
    from legged_lab.envs.elf3.elf3_feedback_still_v2_config import (
        Elf3DwaqFeedbackStillV2AgentCfg, Elf3DwaqFeedbackStillV2EnvCfg,
    )
    expected = Elf3DwaqFeedbackStillV2EnvCfg().to_dict()
    actual = deepcopy(cfg.to_dict())
    standing_experiment_contract(cfg.strict_standing_v2)
    for item in (actual, expected):
        item.pop("device", None)
        item["scene"].pop("num_envs", None)
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError("Unexpected standing v2 environment/reward/PD/terrain/command override")
    agent = Elf3DwaqFeedbackStillV2AgentCfg().to_dict()
    actual = deepcopy(train_cfg)
    for item in (actual, agent):
        for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
            item.pop(key, None)
    if canonical_sha256(actual) != canonical_sha256(agent):
        raise ValueError("Unexpected standing v2 learning recipe, save interval, or seed override")
    return {"parent_task": "elf3_dwaq_feedback_v1", "initialization": "from_scratch",
            "pose_mode": "replace_zero_group_terms_with_full_body_default_pose",
            "motion_clock_preserved": True, "asset_PD_feedback_commands_terrain_PPO_unchanged": True,
            "additional_hip_action_constraints": False, "world_yaw_locked": False}


class Elf3DwaqFeedbackStillV2Runner(Elf3DwaqFeedbackRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        standing_experiment_contract(env.cfg.strict_standing_v2)
        validate_reward_identity(env.cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._standing_joint_ids = env.policy_joint_ids
        self._standing_knee_ids, _ = env.robot.find_joints(["l_knee_y_joint", "r_knee_y_joint"], preserve_order=True)
        self._standing_torso_ids, _ = env.robot.find_bodies(["torso_link"], preserve_order=True)

    def contract(self):
        result = super().contract()
        result["standing_experiment"] = standing_experiment_contract(self.env.cfg.strict_standing_v2)
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        from legged_lab.envs.elf3.feedback_still_v2_rewards import strict_zero_command
        mask = strict_zero_command(self.env.command_generator.command[:, :3],
                                   self.env.cfg.strict_standing_v2["command_epsilon"])
        self.writer.add_scalar("StandingV2/zero_command_fraction", mask.float().mean().item(), locs["it"])
        if not mask.any():
            return
        data = self.env.robot.data
        ids = self._standing_joint_ids
        quat = data.body_quat_w[mask, self._standing_torso_ids[0]]
        w, x, y, z = quat.unbind(-1)
        roll = torch.atan2(2 * (w*x+y*z), 1-2*(x*x+y*y))
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
        metrics = {
            "default_pose_l1_sum": (data.joint_pos[mask][:, ids]-data.default_joint_pos[mask][:, ids]).abs().sum(-1).mean(),
            "raw_joint_speed_abs_mean": data.joint_vel[mask][:, ids].abs().mean(),
            "root_linear_speed_mean": data.root_lin_vel_b[mask].norm(dim=-1).mean(),
            "root_angular_speed_mean": data.root_ang_vel_b[mask].norm(dim=-1).mean(),
            "torso_roll_mean_deg": torch.rad2deg(roll).mean(),
            "torso_pitch_mean_deg": torch.rad2deg(pitch).mean(),
            "torso_tilt_abs_mean_deg": torch.rad2deg(roll.abs()+pitch.abs()).mean(),
            "left_knee_mean_deg": torch.rad2deg(data.joint_pos[mask][:, self._standing_knee_ids[0]]).mean(),
            "right_knee_mean_deg": torch.rad2deg(data.joint_pos[mask][:, self._standing_knee_ids[1]]).mean(),
        }
        # A single CPU transfer; no repeated GPU concatenations for statistics.
        values = torch.stack(list(metrics.values())).detach().cpu().tolist()
        for key, value in zip(metrics, values):
            self.writer.add_scalar("StandingV2/"+key, value, locs["it"])
