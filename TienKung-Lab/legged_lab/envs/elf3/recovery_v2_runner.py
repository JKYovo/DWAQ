"""Explicit curriculum-parent migration; full policy/Adam preserved, recovery AMP is new."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path

import torch

from rsl_rl.runners import DWAQOnPolicyRunner
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.reward_curriculum import (
    Elf3DwaqRewardCurriculumRunner, validate_configuration as validate_parent,
)
from legged_lab.envs.elf3.recovery_v2_core import SETTINGS, recovery_contract
from legged_lab.envs.elf3.recovery_v2_env import make_configs, projected_parent_cfg, projected_parent_recipe
from legged_lab.envs.elf3.recovery_v2_amp import DWAQPPO, RecoveryAMP


def validate_configuration(cfg, recipe):
    expected, _ = make_configs(cfg.recovery["data_directory"], cfg.scene.num_envs, cfg.device)
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError("Unexpected recovery environment/physics/reward override")
    if recipe.get("runner_class_name") != "Elf3DwaqRecoveryRunner" or recipe.get("recovery") != SETTINGS:
        raise ValueError("Use the explicit40% recovery runner and fixed settings")
    parent = validate_parent(projected_parent_cfg(cfg), projected_parent_recipe(recipe))
    return {"recovery_fraction": .4, "fall_delay_steps": 250, "recovery_only_pose_masks": True,
            "parent_configuration_audit": parent}


class Elf3DwaqRecoveryRunner(Elf3DwaqRewardCurriculumRunner):
    algorithm_class = DWAQPPO

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        parent_env = copy(env)
        parent_env.cfg = projected_parent_cfg(env.cfg)
        super().__init__(parent_env, projected_parent_recipe(train_cfg), log_dir, device, checkpoint_dir)
        self.env, self.cfg = env, deepcopy(train_cfg)
        self.train_cfg, self.configuration_audit = self.cfg, audit
        # The parent audit constructs a shallow environment projection. Its
        # curriculum attributes must also be installed on the live environment.
        env.configure_reward_curriculum(self.reward_curriculum_settings, 0)
        self.amp = RecoveryAMP(env.motion_library.expert_pairs, device, env.motion_library.expert_clip_lengths)
        env.recovery_v2_amp = self.amp
        self.alg.configure_recovery(env, self.amp)

    def contract(self):
        result = super().contract()
        result["recovery_experiment"] = recovery_contract(self.env.motion_library.identity())
        return result

    def parent_contract(self, num_envs=16384):
        view = copy(self)
        view.env, view.cfg = copy(self.env), projected_parent_recipe(self.cfg)
        view.env.cfg = projected_parent_cfg(self.env.cfg)
        view.env.cfg.scene.num_envs = num_envs
        return Elf3DwaqRewardCurriculumRunner.contract(view)

    def initialize_from_curriculum(self, path, expected_sha256):
        path = Path(path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Pinned50000 checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["iter"] != 50000 or saved["infos"].get("initialization_only"):
            raise ValueError("Require the completed curriculum model50000")
        validate_checkpoint_contract(saved, self.parent_contract())
        if saved["infos"].get("reward_curriculum_state") != self.curriculum_state(50000):
            raise ValueError("Parent must have completed its exact20k/30k reward schedule")
        previous, current = deepcopy(saved["infos"]["training_recipe"]), projected_parent_recipe(self.cfg)
        for recipe in (previous, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                recipe.pop(key, None)
        if canonical_sha256(previous) != canonical_sha256(current):
            raise ValueError("Parent PPO/Actor/CENet/auxiliary/LR settings differ")
        if self.alg.optimizer.state or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Explicit migration requires fresh target Adam and trained parent Adam")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = 50001
        self.env.set_training_iteration(50001)
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        return {"source_iteration": 50000, "next_iteration": 50001,
                "model_and_Adam_exactly_restored": True, "learning_rate": self.alg.learning_rate,
                "recovery_discriminator_initialization": "fresh; no AMP policy weights loaded"}

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["recovery_training_state"] = self.amp.state_dict()
        return super().save(path, infos)

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.contract())
        if saved["infos"].get("initialization_only"):
            raise ValueError("Initialization-only snapshot cannot resume")
        if saved["infos"].get("reward_curriculum_state") != self.curriculum_state(int(saved["iter"])):
            raise ValueError("Recovery continuation curriculum state mismatch")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer)
        self.amp.load_state_dict(saved["infos"]["recovery_training_state"])
        self.current_learning_iteration = int(saved["iter"])+1
        self.env.set_training_iteration(self.current_learning_iteration)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        env = self.env
        metrics = {
            "assigned_recovery_v2_envs": env.recovery_assigned.sum(),
            "active_fraction": env.recovery_active.float().mean(),
            "recovery_sampled_total": env.recovery_sampled_total,
            "fallen_reference_resets_total": env.recovery_fallen_total,
            "upright_recovered_total": env.recovery_success_total,
            "sampled_fallen_success_ratio": env.recovery_success_total/env.recovery_fallen_total.clamp(min=1),
            "deep_lying_resets_total": env.recovery_deep_total,
            "deep_lying_stable_recoveries_total": env.recovery_deep_success_total,
            "deep_lying_success_ratio": env.recovery_deep_success_total/env.recovery_deep_total.clamp(min=1),
            "amp_style_mean": env.recovery_v2_amp_style_mean,
        }
        values = torch.stack([v.float() for v in metrics.values()]).detach().cpu().tolist()
        for key, value in zip(metrics, values):
            self.writer.add_scalar("Recovery/"+key, value, locs["it"])
        for key, value in self.amp.last_metrics.items():
            self.writer.add_scalar("RecoveryAMP/"+key, value, locs["it"])
