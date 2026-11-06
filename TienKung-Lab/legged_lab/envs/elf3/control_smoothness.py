"""Exactly one added reward and one 1.2x coefficient change to height-force v2."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path

import torch
from isaaclab.managers import RewardTermCfg
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.control_smoothness_contract import SETTINGS, smoothness_contract
from legged_lab.envs.elf3.control_smoothness_rewards import action_smooth, action_smooth_parts
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.height_force_strong import (
    Elf3DwaqHeightForceStrongEnv, Elf3DwaqHeightForceStrongRunner,
    make_configs as make_parent_configs, validate_height_force_strong_configuration,
)


RUNNER_CLASS = "Elf3DwaqControlSmoothnessRunner"
Elf3DwaqControlSmoothnessEnv = Elf3DwaqHeightForceStrongEnv


def make_configs(num_envs=16384, device="cuda:0", run_name="height_force_smooth_v1"):
    cfg, agent = make_parent_configs(num_envs, device, run_name)
    if cfg.reward.dof_acc_l2.weight != SETTINGS["previous_dof_acc_weight"]:
        raise ValueError("Actual parent acceleration coefficient changed")
    if cfg.reward.action_rate_l2.weight != SETTINGS["parent_action_rate_weight"]:
        raise ValueError("Keep the original action-rate coefficient")
    cfg.reward.action_smooth = RewardTermCfg(func=action_smooth, weight=SETTINGS["action_smooth_weight"])
    cfg.reward.dof_acc_l2.weight = SETTINGS["dof_acc_weight"]
    agent.runner_class_name = RUNNER_CLASS
    return cfg, agent


def projected_parent_cfg(cfg):
    result = deepcopy(cfg)
    delattr(result.reward, "action_smooth")
    result.reward.dof_acc_l2.weight = SETTINGS["previous_dof_acc_weight"]
    return result


def validate_configuration(cfg, recipe):
    if recipe.get("runner_class_name") != RUNNER_CLASS:
        raise ValueError("Require the explicit smoothness-stage runner")
    reference, _ = make_configs(cfg.scene.num_envs, cfg.device)
    for name in ("action_smooth", "dof_acc_l2"):
        actual, expected = getattr(cfg.reward, name, None), getattr(reference.reward, name)
        if (actual is None or actual.func is not expected.func
                or canonical_sha256(actual.to_dict()) != canonical_sha256(expected.to_dict())):
            raise ValueError(f"Wrong smoothness stage reward: {name}")
    if abs(cfg.reward.action_smooth.weight) > abs(cfg.reward.action_rate_l2.weight) / 5:
        raise ValueError("Action smoothness must not exceed one fifth of action rate")
    parent_recipe = deepcopy(recipe)
    parent_recipe["runner_class_name"] = "Elf3DwaqHeightForceStrongRunner"
    audit = validate_height_force_strong_configuration(projected_parent_cfg(cfg), parent_recipe)
    return {"only_added_reward": "action_smooth", "action_smooth_weight": -.001,
            "dof_acc_old_weight": -2.5e-7, "dof_acc_new_weight": -3e-7,
            "dof_acc_multiplier": 1.2, "other_configuration_unchanged": True, "parent_audit": audit}


class Elf3DwaqControlSmoothnessRunner(Elf3DwaqHeightForceStrongRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        parent_env, parent_recipe = copy(env), deepcopy(train_cfg)
        parent_env.cfg = projected_parent_cfg(env.cfg)
        parent_recipe["runner_class_name"] = "Elf3DwaqHeightForceStrongRunner"
        super().__init__(parent_env, parent_recipe, log_dir, device, checkpoint_dir)
        self.env, self.cfg, self.configuration_audit = env, deepcopy(train_cfg), audit

    def contract(self):
        result = super().contract()
        result["control_smoothness_experiment"] = smoothness_contract()
        return result

    def parent_contract(self, num_envs=16384):
        projected = copy(self)
        projected.env, projected.cfg = copy(self.env), deepcopy(self.cfg)
        projected.env.cfg = projected_parent_cfg(self.env.cfg)
        projected.env.cfg.scene.num_envs = num_envs
        projected.cfg["runner_class_name"] = "Elf3DwaqHeightForceStrongRunner"
        return Elf3DwaqHeightForceStrongRunner.contract(projected)

    def initialize_from_height_parent(self, path, expected_sha256):
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Reward promotion requires a fresh runner before restoring Adam")
        path = Path(path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Pinned parent checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.parent_contract())
        old, current = deepcopy(saved["infos"]["training_recipe"]), deepcopy(self.cfg)
        current["runner_class_name"] = "Elf3DwaqHeightForceStrongRunner"
        for item in (old, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                item.pop(key, None)
        if canonical_sha256(old) != canonical_sha256(current) or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Keep the full PPO/policy/learning recipe and trained Adam")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = int(saved["iter"]) + 1
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        command = self.env.command_generator.command
        velocity = self.env.robot.data.root_lin_vel_b[:, :2]
        requested = command[:, :2].norm(dim=-1)
        moving = requested > .05
        count = moving.sum().clamp(min=1)
        def mean_moving(value):
            return torch.where(moving, value, 0.).sum() / count
        first, second, amplitude = action_smooth_parts(self.env.action_buffer._circular_buffer.buffer)
        aligned_ratio = (velocity * command[:, :2]).sum(-1) / requested.square().clamp(min=.05**2)
        values = {
            "action_first_difference_l2": first.mean(),
            "action_second_difference_l2": second.mean(),
            "action_amplitude_cost": amplitude.mean(),
            "moving_command_speed_m_s": mean_moving(requested),
            "moving_actual_speed_m_s": mean_moving(velocity.norm(dim=-1)),
            "moving_velocity_error_m_s": mean_moving((velocity-command[:, :2]).norm(dim=-1)),
            "moving_aligned_speed_ratio": mean_moving(aligned_ratio),
        }
        # One transfer for all added logging, no simulator-name lookups or GPU
        # tiny-tensor concatenation inside a reward callback.
        numbers = torch.stack(list(values.values())).detach().cpu().tolist()
        for key, number in zip(values, numbers):
            self.writer.add_scalar("ControlSmoothness/" + key, number, locs["it"])
