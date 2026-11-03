"""Minimal reward-identity guard and diagnostics, not an extra training algorithm."""
from copy import deepcopy

import torch

from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.feedback_stand_v3_contract import stand_experiment_contract


def validate_reward_identity(cfg):
    from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackEnvCfg
    from legged_lab.envs.elf3.elf3_feedback_stand_v3_config import Elf3DwaqFeedbackStandV3EnvCfg
    # Validate exactly the two approved wrappers, then undo only those changes
    # for comparison with the entire parent reward configuration.
    original = Elf3DwaqFeedbackEnvCfg().reward.to_dict()
    actual = deepcopy(cfg.reward.to_dict())
    actual.pop("zero_default_pose_l1", None)
    actual.pop("zero_joint_motion", None)
    from legged_lab.envs.elf3.standing_rewards import gait_phase_contact_when_moving, feet_swing_height_when_moving
    for name, wrapper in (("gait_phase_contact", gait_phase_contact_when_moving),
                          ("feet_swing_height", feet_swing_height_when_moving)):
        term = actual[name]
        # Isaac's to_dict serializes callables to names; check identity on the
        # live term, not on that serialized representation.
        if getattr(cfg.reward, name).func is not wrapper or term["params"].pop("command_epsilon", None) != 1e-6:
            raise ValueError(f"Incorrect zero-command periodic reward gate: {name}")
        term["func"] = original[name]["func"]
    if canonical_sha256(actual) != canonical_sha256(original):
        raise ValueError("Zero-phase v3 must preserve feedback-v1 rewards except the two approved zero-command gates")
    expected_cfg = Elf3DwaqFeedbackStandV3EnvCfg()
    for term_name in ("zero_default_pose_l1", "zero_joint_motion"):
        reference = getattr(expected_cfg.reward, term_name)
        term = getattr(cfg.reward, term_name, None)
        if term is None or term.func is not reference.func or term.weight != reference.weight:
            raise ValueError(f"Incorrect added function/weight: {term_name}")
        if set(term.params) != set(reference.params):
            raise ValueError(f"Incorrect added parameter keys: {term_name}")
        if term.params["command_epsilon"] != reference.params["command_epsilon"]:
            raise ValueError("Incorrect zero-command threshold")
        entity, expected = term.params["asset_cfg"], reference.params["asset_cfg"]
        for name in ("name", "joint_names", "preserve_order"):
            if getattr(entity, name, None) != getattr(expected, name, None):
                raise ValueError(f"Incorrect named joints: {term_name}")


def validate_stand_configuration(cfg, train_cfg):
    from legged_lab.envs.elf3.elf3_feedback_stand_v3_config import (
        Elf3DwaqFeedbackStandV3AgentCfg, Elf3DwaqFeedbackStandV3EnvCfg,
    )
    stand_experiment_contract(cfg.strict_standing_v3)
    expected, actual = Elf3DwaqFeedbackStandV3EnvCfg().to_dict(), deepcopy(cfg.to_dict())
    for item in (actual, expected):
        item.pop("device", None)
        item["scene"].pop("num_envs", None)
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError("Unexpected default-pose v3 environment/reward/physics/commands override")
    expected_agent, actual_agent = Elf3DwaqFeedbackStandV3AgentCfg().to_dict(), deepcopy(train_cfg)
    for item in (expected_agent, actual_agent):
        for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
            item.pop(key, None)
    if canonical_sha256(actual_agent) != canonical_sha256(expected_agent):
        raise ValueError("Unexpected default-pose v3 policy/PPO/learning recipe override")
    return {"parent_task": "elf3_dwaq_feedback_v1", "initialization": "from_scratch",
            "only_added_rewards": ["zero_default_pose_l1", "zero_joint_motion"],
            "baseline_pose_rewards_preserved": True,
            "zero_only_periodic_reward_gates": ["gait_phase_contact", "feet_swing_height"],
            "moving_actor_inputs_and_history_unchanged": True,
            "phase_preprocessing": "strict zero only; current and all history phases [0,0,1,1]; raw buffers untouched",
            "asset_PD_feedback_commands_terrain_PPO_unchanged": True}


class Elf3DwaqFeedbackStandV3Runner(Elf3DwaqFeedbackRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        stand_experiment_contract(env.cfg.strict_standing_v3)
        validate_reward_identity(env.cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._pose_entity = env.reward_manager.get_term_cfg("zero_default_pose_l1").params["asset_cfg"]
        self._pose_knee_ids, _ = env.robot.find_joints(["l_knee_y_joint", "r_knee_y_joint"], preserve_order=True)

    def contract(self):
        result = super().contract()
        result["standing_experiment"] = stand_experiment_contract(self.env.cfg.strict_standing_v3)
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        from legged_lab.envs.elf3.feedback_stand_v3_rewards import strict_zero_command
        mask = strict_zero_command(self.env.command_generator.command,
                                   self.env.cfg.strict_standing_v3["command_epsilon"])
        self.writer.add_scalar("StandV3/zero_command_snapshot_fraction", mask.float().mean().item(), locs["it"])
        if not mask.any():
            return
        data, ids = self.env.robot.data, self._pose_entity.joint_ids
        q, home = data.joint_pos[mask][:, ids], data.default_joint_pos[mask][:, ids]
        gravity = data.projected_gravity_b[mask]
        metrics = {
            "snapshot_default_pose_l1_sum": (q-home).abs().sum(-1).mean(),
            "snapshot_raw_joint_velocity_rms": data.joint_vel[mask][:, ids].square().mean().sqrt(),
            "snapshot_root_linear_speed": data.root_lin_vel_b[mask].norm(dim=-1).mean(),
            "snapshot_root_angular_speed": data.root_ang_vel_b[mask].norm(dim=-1).mean(),
            "snapshot_body_tilt_deg": torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1))).mean(),
            "snapshot_left_knee_deg": torch.rad2deg(data.joint_pos[mask][:, self._pose_knee_ids[0]]).mean(),
            "snapshot_right_knee_deg": torch.rad2deg(data.joint_pos[mask][:, self._pose_knee_ids[1]]).mean(),
        }
        values = torch.stack(list(metrics.values())).detach().cpu().tolist()
        for key, value in zip(metrics, values):
            self.writer.add_scalar("StandV3/"+key, value, locs["it"])
