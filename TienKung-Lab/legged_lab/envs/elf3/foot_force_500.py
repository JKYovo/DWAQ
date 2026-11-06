"""One-parameter continuation of the current force-peak stage, not a restart."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path
import torch
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.foot_force_peak import (
    Elf3DwaqFootForcePeakEnv, Elf3DwaqFootForcePeakRunner,
    make_configs as make_parent_configs, validate_configuration as validate_parent,
)
from legged_lab.envs.elf3.foot_force_500_contract import threshold_contract


RUNNER_CLASS = "Elf3DwaqFootForce500Runner"
Elf3DwaqFootForce500Env = Elf3DwaqFootForcePeakEnv


def make_configs(num_envs=16384, device="cuda:0", run_name="force_peak_500N"):
    cfg, agent = make_parent_configs(num_envs, device, run_name)
    cfg.reward.feet_force.params["threshold"] = 500
    agent.runner_class_name = RUNNER_CLASS
    return cfg, agent


def projected_parent_cfg(cfg):
    result = deepcopy(cfg)
    result.reward.feet_force.params["threshold"] = 400
    return result


def validate_configuration(cfg, recipe):
    if recipe.get("runner_class_name") != RUNNER_CLASS:
        raise ValueError("Require the explicit 500 N continuation runner")
    reference, _ = make_configs(cfg.scene.num_envs, cfg.device)
    if canonical_sha256(cfg.reward.feet_force.to_dict()) != canonical_sha256(reference.reward.feet_force.to_dict()):
        raise ValueError("Only change force threshold to500; weight, function, cap and feet unchanged")
    parent = deepcopy(recipe)
    parent["runner_class_name"] = "Elf3DwaqFootForcePeakRunner"
    return {"only_environment_change": "reward.feet_force.params.threshold:400->500",
            "parent_audit": validate_parent(projected_parent_cfg(cfg), parent)}


class Elf3DwaqFootForce500Runner(Elf3DwaqFootForcePeakRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        projected_env, projected_recipe = copy(env), deepcopy(train_cfg)
        projected_env.cfg = projected_parent_cfg(env.cfg)
        projected_recipe["runner_class_name"] = "Elf3DwaqFootForcePeakRunner"
        super().__init__(projected_env, projected_recipe, log_dir, device, checkpoint_dir)
        self.env, self.cfg, self.configuration_audit = env, deepcopy(train_cfg), audit

    def contract(self):
        result = super().contract()
        # Keep inherited reward-stage identities as origin records, and describe
        # the effective force threshold explicitly in the newest contract.
        result["foot_force_threshold_experiment"] = threshold_contract()
        return result

    def parent_contract(self, num_envs=16384):
        projected = copy(self)
        projected.env, projected.cfg = copy(self.env), deepcopy(self.cfg)
        projected.env.cfg = projected_parent_cfg(self.env.cfg)
        projected.env.cfg.scene.num_envs = num_envs
        projected.cfg["runner_class_name"] = "Elf3DwaqFootForcePeakRunner"
        return Elf3DwaqFootForcePeakRunner.contract(projected)

    def initialize_from_force_peak(self, path, expected_sha256):
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Promotion requires a fresh runner before restoring Adam")
        path = Path(path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Pinned current-checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.parent_contract())
        old, current = deepcopy(saved["infos"]["training_recipe"]), deepcopy(self.cfg)
        current["runner_class_name"] = "Elf3DwaqFootForcePeakRunner"
        for recipe in (old, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                recipe.pop(key, None)
        if canonical_sha256(old) != canonical_sha256(current) or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Keep the full saved policy/PPO/auxiliary/seed and trained Adam")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = int(saved["iter"]) + 1
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
