"""Strict new-task resume plus one explicitly validated legacy initialization."""
from copy import copy, deepcopy
import hashlib

import torch
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.elf3_still_config import Elf3DwaqStillAgentCfg, Elf3DwaqStillEnvCfg
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.still_pose_contract import (
    PARENT_ITERATION, PARENT_SHA256, pose_experiment_contract,
)
from legged_lab.envs.elf3.still_pose_rewards import STAND_POSE_JOINT_NAMES, stand_leg_pose_l1
from legged_lab.envs.elf3.still_runner import Elf3DwaqStillRunner


def validate_pose_configuration(cfg, train_cfg):
    """Allow only the new term/settings, task identity and launch bookkeeping."""
    baseline_env = Elf3DwaqStillEnvCfg()
    expected = baseline_env.to_dict()
    actual = deepcopy(cfg.to_dict())
    pose_experiment_contract(actual.pop("zero_command_pose"))
    actual["reward"].pop("stand_leg_pose")
    # A reduced environment count is allowed for a separately recorded smoke test.
    actual["scene"]["num_envs"] = expected["scene"]["num_envs"]
    actual["device"] = expected["device"]
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError("Unexpected environment change besides zero-command leg pose")
    baseline_agent = Elf3DwaqStillAgentCfg().to_dict()
    candidate = deepcopy(train_cfg)
    bookkeeping = ("runner_class_name", "experiment_name", "wandb_project", "run_name",
                   "max_iterations", "resume", "load_run", "load_checkpoint", "device")
    for key in bookkeeping:
        if key in baseline_agent:
            candidate[key] = deepcopy(baseline_agent[key])
    if canonical_sha256(candidate) != canonical_sha256(baseline_agent):
        raise ValueError("Unexpected algorithm, policy, seed, sampling or save configuration change")
    return {"only_reward_change": "zero-only four-joint L1 incremental weight -0.08",
            "zero_total_weight": -0.10, "moving_weight": -0.02,
            "original_algorithm_policy_PD_asset_commands_and_terrain": True,
            "smoke_num_envs_override": cfg.scene.num_envs != baseline_env.scene.num_envs}


class Elf3DwaqStillPoseRunner(Elf3DwaqStillRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Full training-recipe audit belongs to explicit training initialization.
        # Native play legitimately changes terrain/noise/commands before creating
        # the runner; its inference-specific checkpoint validator handles that.
        self.configuration_audit = None
        identity = pose_experiment_contract(env.cfg.zero_command_pose)
        term = env.cfg.reward.stand_leg_pose
        baseline = env.cfg.reward.joint_deviation_legs
        names = tuple(term.params["asset_cfg"].joint_names)
        if (term.func is not stand_leg_pose_l1 or term.weight != identity["incremental_weight"]
                or names != STAND_POSE_JOINT_NAMES
                or term.params.get("command_epsilon") != identity["command_epsilon"]
                or baseline.weight != identity["baseline_weight"]):
            raise ValueError("Incorrect zero-command pose objective")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._pose_joint_ids, names = env.robot.find_joints(list(STAND_POSE_JOINT_NAMES), preserve_order=True)
        if tuple(names) != STAND_POSE_JOINT_NAMES:
            raise ValueError("Pose logging joint mapping differs from the reward")

    def contract(self):
        result = super().contract()
        # Keep the old inference contract intact, including its five source hashes.
        result["zero_command_pose_experiment"] = pose_experiment_contract(self.env.cfg.zero_command_pose)
        return result

    def _legacy_contract(self):
        # Reconstruct the original registered 4096-env configuration, not a
        # contract with the saved hashes blindly copied over the new reward.
        legacy = copy(self)
        legacy.env = copy(self.env)
        legacy.env.cfg = Elf3DwaqStillEnvCfg()
        return Elf3DwaqStillRunner.contract(legacy)

    def initialize_from_still(self, path):
        """One-time migration, only from the user's pinned final100000 checkpoint.

        Normal new-task resume continues to use the inherited strict load().
        There is no force flag and no global weakening of checkpoint validation.
        """
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Legacy initialization requires a fresh runner")
        self.configuration_audit = validate_pose_configuration(self.env.cfg, self.cfg)
        resolved = self._resolve_checkpoint_path(path)
        with open(resolved, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        if digest != PARENT_SHA256:
            raise ValueError("Source is not the pinned model_100000.pt")
        saved = torch.load(resolved, map_location="cpu", weights_only=False)
        if saved.get("iter") != PARENT_ITERATION:
            raise ValueError("Unexpected source checkpoint iteration")
        validate_checkpoint_contract(saved, self._legacy_contract())
        old_recipe = saved["infos"]["training_recipe"]
        for key in ("algorithm", "policy", "seed", "num_steps_per_env", "empirical_normalization",
                    "upper_body_mirror_loss_coeff", "upper_body_pose_loss_coeff", "save_interval"):
            if old_recipe.get(key) != self.cfg.get(key):
                raise ValueError(f"Source training parameter changed: {key}")
        DWAQOnPolicyRunner.load(self, resolved, load_optimizer=True)
        self.current_learning_iteration = PARENT_ITERATION + 1
        restored = self.alg.policy.state_dict()
        for name, value in saved["model_state_dict"].items():
            if not torch.equal(value, restored[name].detach().cpu()):
                raise ValueError(f"Weights not exactly restored: {name}")
        optimizer = self.alg.optimizer.state_dict()
        source_optimizer = saved["optimizer_state_dict"]
        if optimizer["param_groups"] != source_optimizer["param_groups"]:
            raise ValueError("Optimizer parameter groups changed during initialization")
        if set(optimizer["state"]) != set(source_optimizer["state"]):
            raise ValueError("Optimizer state entries were not restored")
        for index, state in source_optimizer["state"].items():
            for name, value in state.items():
                current = optimizer["state"][index][name]
                if torch.is_tensor(value):
                    if not torch.equal(value, current.detach().cpu()):
                        raise ValueError(f"Optimizer tensor not exactly restored: {index}/{name}")
                elif current != value:
                    raise ValueError(f"Optimizer value changed: {index}/{name}")
        return {"checkpoint_sha256": digest, "checkpoint_iteration": PARENT_ITERATION,
                "next_iteration": self.current_learning_iteration,
                "weights_exactly_restored": True, "optimizer_exactly_restored": True,
                "optimizer_state_entries": len(optimizer["state"]),
                "learning_rate_restored": self.alg.learning_rate,
                "environment_curriculum_buffers_and_RNG_restored": False}

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        command = self.env.command_generator.command[:, :3]
        active = command.isfinite().all(-1) & (command.abs().amax(-1) <= 1e-6)
        if active.any():
            data = self.env.robot.data
            q = data.joint_pos[active][:, self._pose_joint_ids]
            target = data.joint_pos_target[active][:, self._pose_joint_ids]
            default = data.default_joint_pos[active][:, self._pose_joint_ids]
            # One compact CPU transfer rather than repeated tiny GPU concatenations.
            values = torch.cat((q.mean(0), target.mean(0), (q - default).abs().sum(-1).mean().view(1))).detach().cpu().tolist()
            labels = [f"{name}/actual_rad" for name in STAND_POSE_JOINT_NAMES]
            labels += [f"{name}/target_rad" for name in STAND_POSE_JOINT_NAMES]
            labels += ["default_pose_l1_sum"]
            for label, value in zip(labels, values):
                self.writer.add_scalar(f"StillPose/{label}", value, locs["it"])
