"""Strict model60000 migration with complete policy/Adam/AMP state restoration."""
from copy import copy, deepcopy
from pathlib import Path

import torch

from rsl_rl.runners import DWAQOnPolicyRunner
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.flat_recovery_env import make_configs as flat_configs
from legged_lab.envs.elf3.flat_recovery_runner import Elf3DwaqFlatRecoveryRunner, validate_configuration as flat_audit
from legged_lab.envs.elf3.recovery_init_env import make_configs, projected_flat_cfg, projected_flat_recipe
from legged_lab.envs.elf3.recovery_init_core import SETTINGS, experiment_contract, sha256


def validate_configuration(cfg, recipe):
    expected, _ = make_configs(cfg.recovery["data_directory"], cfg.recovery_initialization["bank_path"],
                              cfg.scene.num_envs, cfg.device)
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError("Unexpected override outside recovery initialization")
    if recipe.get("recovery_initialization") != SETTINGS or recipe.get("runner_class_name") != "Elf3DwaqRecoveryInitRunner":
        raise ValueError("Use the independent recovery initialization version")
    return {"initialization_only_change": True,
            "parent": flat_audit(projected_flat_cfg(cfg), projected_flat_recipe(recipe))}


class Elf3DwaqRecoveryInitRunner(Elf3DwaqFlatRecoveryRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        view = copy(env)
        view.cfg = projected_flat_cfg(env.cfg)
        super().__init__(view, projected_flat_recipe(train_cfg), log_dir, device, checkpoint_dir)
        self.env, self.cfg = env, deepcopy(train_cfg)
        self.train_cfg, self.configuration_audit = self.cfg, audit
        env.configure_reward_curriculum(self.reward_curriculum_settings, 0)
        env.recovery_v2_amp = self.amp
        self.alg.configure_recovery(env, self.amp)

    def flat_contract(self, num_envs=None):
        view = copy(self)
        view.env = copy(self.env)
        view.cfg = projected_flat_recipe(self.cfg)
        view.env.cfg = projected_flat_cfg(self.env.cfg)
        if num_envs is not None:
            view.env.cfg, _ = flat_configs(self.env.cfg.recovery["data_directory"], num_envs, self.env.cfg.device)
        return Elf3DwaqFlatRecoveryRunner.contract(view)

    def contract(self):
        result = self.flat_contract()
        result["recovery_initialization_experiment"] = experiment_contract(self.env.initialization_bank.identity())
        return result

    def _restore_training_state(self, path, saved):
        if saved["infos"].get("initialization_only") or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Require a completed model with trained Adam")
        if saved["infos"].get("reward_curriculum_state") != self.curriculum_state(int(saved["iter"])):
            raise ValueError("Curriculum progress differs from the source checkpoint")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.amp.load_state_dict(saved["infos"]["recovery_training_state"])
        self.current_learning_iteration = int(saved["iter"])+1
        self.env.set_training_iteration(self.current_learning_iteration)
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "policy")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        self._assert_tree_equal(self.amp.state_dict(), saved["infos"]["recovery_training_state"], "AMP")
        return {"source_iteration": int(saved["iter"]), "next_iteration": self.current_learning_iteration,
                "policy_Adam_discriminator_normalizer_replay_exactly_restored": True,
                "learning_rate": self.alg.learning_rate}

    def initialize_from_flat(self, path, expected_sha256):
        path = Path(path).resolve(strict=True)
        if sha256(path) != expected_sha256:
            raise ValueError("Source model60000 SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["iter"] != 60000 or self.alg.optimizer.state:
            raise ValueError("Migration requires original completed model60000 and a fresh runner")
        validate_checkpoint_contract(saved, self.flat_contract(16384))
        previous, current = deepcopy(saved["infos"]["training_recipe"]), projected_flat_recipe(self.cfg)
        for recipe in (previous, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                recipe.pop(key, None)
        if canonical_sha256(previous) != canonical_sha256(current):
            raise ValueError("PPO/Actor/VAE/reward curriculum settings must match the source")
        return self._restore_training_state(path, saved)

    def load(self, path, load_optimizer=True):
        if not load_optimizer:
            raise ValueError("Continuation must restore Adam and AMP state")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.contract())
        return self._restore_training_state(path, saved)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        env = self.env
        values = torch.stack((env.initialization_sampled, env.initialization_fallen,
                              env.initialization_recovered)).detach().cpu().tolist()
        for pose in range(3):
            for velocity in range(3):
                group = pose*3+velocity
                tag = f"RecoveryInit/{('reference','supine_reference','supine_procedural')[pose]}/{('zero','small','larger')[velocity]}"
                for label, row in zip(("sampled", "fallen", "recovered"), values):
                    self.writer.add_scalar(tag+"/"+label, row[group], locs["it"])
