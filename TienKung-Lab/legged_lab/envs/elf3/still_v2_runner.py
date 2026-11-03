"""Fresh still-v2 identity: retain proven still-v1 and add two named pose costs."""
from copy import deepcopy

import torch

from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.still_runner import Elf3DwaqStillRunner
from legged_lab.envs.elf3.still_v2_contract import pose_experiment_contract


def validate_reward_identity(cfg):
    """Validate live callbacks as well as serialized parent reward values."""
    from legged_lab.envs.elf3.elf3_still_config import Elf3DwaqStillEnvCfg
    from legged_lab.envs.elf3.elf3_still_v2_config import Elf3DwaqStillV2EnvCfg

    parent = Elf3DwaqStillEnvCfg().reward.to_dict()
    actual = deepcopy(cfg.reward.to_dict())
    for name in ("stand_leg_pose", "stand_waist_pose"):
        actual.pop(name, None)
    if canonical_sha256(actual) != canonical_sha256(parent):
        raise ValueError("Still-v2 must preserve every original still-v1 reward")
    expected = Elf3DwaqStillV2EnvCfg()
    for name in ("stand_leg_pose", "stand_waist_pose"):
        term, reference = getattr(cfg.reward, name, None), getattr(expected.reward, name)
        if term is None or term.func is not reference.func or term.weight != reference.weight:
            raise ValueError(f"Incorrect still-v2 named pose function/weight: {name}")
        if set(term.params) != set(reference.params):
            raise ValueError(f"Incorrect still-v2 pose parameter keys: {name}")
        if term.params["command_epsilon"] != reference.params["command_epsilon"]:
            raise ValueError(f"Incorrect still-v2 zero threshold: {name}")
        for field in ("name", "joint_names", "preserve_order"):
            actual_value = getattr(term.params["asset_cfg"], field, None)
            expected_value = getattr(reference.params["asset_cfg"], field, None)
            if actual_value != expected_value:
                raise ValueError(f"Incorrect still-v2 pose entity: {name}.{field}")


def validate_still_v2_configuration(cfg, train_cfg):
    """Require the complete registered task; allow only launch bookkeeping."""
    from legged_lab.envs.elf3.elf3_still_v2_config import Elf3DwaqStillV2AgentCfg, Elf3DwaqStillV2EnvCfg

    pose_experiment_contract(cfg.still_v2_pose)
    validate_reward_identity(cfg)
    actual, expected = deepcopy(cfg.to_dict()), Elf3DwaqStillV2EnvCfg().to_dict()
    for item in (actual, expected):
        item.pop("device", None)
        item["scene"].pop("num_envs", None)
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError("Unexpected still-v2 environment/physics/terrain/command override")
    actual_agent, expected_agent = deepcopy(train_cfg), Elf3DwaqStillV2AgentCfg().to_dict()
    for item in (actual_agent, expected_agent):
        for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
            item.pop(key, None)
    if canonical_sha256(actual_agent) != canonical_sha256(expected_agent):
        raise ValueError("Unexpected still-v2 policy/PPO/learning/seed override")
    return {
        "parent_task": "elf3_dwaq_still", "reference_task": "elf3_dwaq_still_pose_v1",
        "initialization": "from_scratch", "only_added_rewards": ["stand_leg_pose", "stand_waist_pose"],
        "leg_incremental_weight": -.13, "leg_total_zero_weight": -.15,
        "waist_incremental_weight": -.10, "waist_total_zero_weight": -.30,
        "original_still_phase_rewards_motion_asset_PD_feedback_commands_terrain_PPO_unchanged": True,
        "new_actor_inputs_or_auxiliary_losses": False,
    }


class Elf3DwaqStillV2Runner(Elf3DwaqStillRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        identity = pose_experiment_contract(env.cfg.still_v2_pose)
        validate_reward_identity(env.cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._still_v2_leg_ids, leg_names = env.robot.find_joints(identity["leg_joint_names"], preserve_order=True)
        self._still_v2_waist_ids, waist_names = env.robot.find_joints(identity["waist_joint_names"], preserve_order=True)
        if tuple(leg_names) != tuple(identity["leg_joint_names"]) or tuple(waist_names) != tuple(identity["waist_joint_names"]):
            raise ValueError("Still-v2 diagnostic joint mapping disagrees with named reward targets")

    def contract(self):
        result = super().contract()
        # Do not alter the parent's standing_experiment or its five source hashes.
        result["still_v2_experiment"] = pose_experiment_contract(self.env.cfg.still_v2_pose)
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        command = self.env.command_generator.command[:, :3]
        mask = command.isfinite().all(-1) & (command.abs().amax(-1) <= self.env.cfg.still_v2_pose["command_epsilon"])
        if not mask.any():
            return
        data = self.env.robot.data
        settings = self.env.cfg.still_v2_pose
        leg = data.joint_pos[mask][:, self._still_v2_leg_ids]
        leg_home = data.default_joint_pos[mask][:, self._still_v2_leg_ids]
        waist = data.joint_pos[mask][:, self._still_v2_waist_ids]
        waist_home = data.default_joint_pos[mask][:, self._still_v2_waist_ids]
        gravity = data.projected_gravity_b[mask]
        metrics = {
            "snapshot_leg_default_pose_l1_sum": (leg-leg_home).abs().sum(-1).mean(),
            "snapshot_waist_default_pose_l1_sum": (waist-waist_home).abs().sum(-1).mean(),
            "snapshot_torso_tilt_deg": torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1))).mean(),
            "snapshot_raw_joint_velocity_abs_mean": data.joint_vel[mask].abs().mean(),
        }
        for names, values, home in ((settings["leg_joint_names"], leg, leg_home),
                                     (settings["waist_joint_names"], waist, waist_home)):
            for index, name in enumerate(names):
                metrics[f"snapshot/{name}/actual_rad"] = values[:, index].mean()
                metrics[f"snapshot/{name}/default_error_rad"] = (values[:, index]-home[:, index]).mean()
        # One transfer for the added diagnostics, not repeated GPU scalar concatenation.
        values = torch.stack(list(metrics.values())).detach().cpu().tolist()
        for key, value in zip(metrics, values):
            self.writer.add_scalar("StillV2/" + key, value, locs["it"])
