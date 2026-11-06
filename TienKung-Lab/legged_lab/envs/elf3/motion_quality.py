"""Two isolated reward edits; no change to the parent simulator or policy."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path

import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.ankle_cost import (
    Elf3DwaqAnkleCostEnv, Elf3DwaqAnkleCostRunner,
    make_configs as make_parent_configs, validate_ankle_cost_configuration,
)
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.motion_quality_contract import SETTINGS, motion_quality_contract
from legged_lab.envs.elf3.motion_quality_rewards import (
    FOOT_BODY_NAMES, KNEE_JOINT_NAMES, support_knee_excess_l2,
)


# Exact same environment class: no extra physics hook, state, observation,
# action processing, contact history update, or reset logic is introduced.
Elf3DwaqMotionQualityEnv = Elf3DwaqAnkleCostEnv
NEW_REWARD = "support_knee_excess_l2"


def make_configs(num_envs=16384, device="cuda:0", run_name="still_v2_motion_quality_v1"):
    cfg, agent = make_parent_configs(num_envs, device, run_name)
    if cfg.reward.action_rate_l2.weight != SETTINGS["action_rate_previous_weight"]:
        raise ValueError("Original action-rate baseline changed; review before applying this recipe")
    cfg.reward.action_rate_l2.weight = SETTINGS["action_rate_weight"]
    cfg.reward.support_knee_excess_l2 = RewardTermCfg(
        func=support_knee_excess_l2, weight=SETTINGS["knee_weight"], params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_JOINT_NAMES), preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "threshold_deg": SETTINGS["knee_threshold_deg"],
            "support_force_threshold": SETTINGS["support_force_threshold_n"],
            "command_epsilon": SETTINGS["command_epsilon"],
        })
    agent.runner_class_name = "Elf3DwaqMotionQualityRunner"
    return cfg, agent


def projected_ankle_cfg(cfg):
    result = cfg.copy()
    delattr(result.reward, NEW_REWARD)
    result.reward.action_rate_l2.weight = SETTINGS["action_rate_previous_weight"]
    return result


def validate_motion_quality_configuration(cfg, train_cfg):
    if train_cfg.get("runner_class_name") != "Elf3DwaqMotionQualityRunner":
        raise ValueError("Use the explicit motion-quality runner")
    if any(hasattr(cfg.reward, name) for name in SETTINGS["excluded_soft_step_rewards"]):
        raise ValueError("The two discarded soft-step rewards must not be present")
    expected, _ = make_configs(cfg.scene.num_envs, cfg.device)
    for name in ("action_rate_l2", NEW_REWARD):
        term, reference = getattr(cfg.reward, name, None), getattr(expected.reward, name)
        if (term is None or term.func is not reference.func
                or canonical_sha256(term.to_dict()) != canonical_sha256(reference.to_dict())):
            raise ValueError(f"Wrong motion-quality callback/weight/parameters: {name}")
    parent_agent = deepcopy(train_cfg)
    parent_agent["runner_class_name"] = "Elf3DwaqAnkleCostRunner"
    audit = validate_ankle_cost_configuration(projected_ankle_cfg(cfg), parent_agent)
    return {"version": SETTINGS["version"], "modified_parent_rewards": ["action_rate_l2"],
            "added_rewards": [NEW_REWARD], "removed_soft_step_rewards": SETTINGS["excluded_soft_step_rewards"],
            "action_rate_weight": SETTINGS["action_rate_weight"], "knee_weight": SETTINGS["knee_weight"],
            "knee_threshold_deg": SETTINGS["knee_threshold_deg"], "parent_audit": audit,
            "all_other_parent_configuration_preserved": True}


class Elf3DwaqMotionQualityRunner(Elf3DwaqAnkleCostRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_motion_quality_configuration(env.cfg, train_cfg)
        parent_env, parent_agent = copy(env), deepcopy(train_cfg)
        parent_env.cfg = projected_ankle_cfg(env.cfg)
        parent_agent["runner_class_name"] = "Elf3DwaqAnkleCostRunner"
        super().__init__(parent_env, parent_agent, log_dir, device, checkpoint_dir)
        self.env, self.cfg, self.configuration_audit = env, deepcopy(train_cfg), audit

    def contract(self):
        result = super().contract()
        result["motion_quality_experiment"] = motion_quality_contract()
        return result

    def _ankle_parent_contract(self, num_envs=None):
        parent = copy(self)
        parent.env, parent.cfg = copy(self.env), deepcopy(self.cfg)
        parent.env.cfg = projected_ankle_cfg(self.env.cfg)
        if num_envs is not None:
            parent.env.cfg.scene.num_envs = num_envs
        parent.cfg["runner_class_name"] = "Elf3DwaqAnkleCostRunner"
        return Elf3DwaqAnkleCostRunner.contract(parent)

    def initialize_from_ankle_cost(self, path, expected_sha256):
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Explicit reward promotion requires a fresh runner")
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Pinned ankle-cost parent checkpoint SHA256 changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        # No soft-step checkpoint may pass as an unchanged ankle-cost parent.
        try:
            validate_checkpoint_contract(saved, self._ankle_parent_contract())
        except ValueError:
            validate_checkpoint_contract(saved, self._ankle_parent_contract(num_envs=16384))
        old, current = deepcopy(saved["infos"]["training_recipe"]), deepcopy(self.cfg)
        current["runner_class_name"] = "Elf3DwaqAnkleCostRunner"
        for recipe in (old, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                recipe.pop(key, None)
        if canonical_sha256(old) != canonical_sha256(current) or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Parent must preserve the complete PPO recipe and nonempty Adam state")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = int(saved["iter"]) + 1
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        return {"checkpoint": str(path), "checkpoint_sha256": digest,
                "completed_iteration": int(saved["iter"]), "next_iteration": self.current_learning_iteration,
                "model_and_Adam_exactly_restored": True, "learning_rate": self.alg.learning_rate,
                "environment_curriculum_RNG_and_histories_restored": False}

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        term = self.env.reward_manager.get_term_cfg(NEW_REWARD)
        cost = support_knee_excess_l2(self.env, **term.params).mean()
        # Snapshot metric, not an episode mean. One GPU->CPU transfer for the
        # added diagnostics; no extra contact updates or GPU scalar loops.
        values = torch.stack([cost, cost * term.weight]).detach().cpu().tolist()
        for name, value in zip(("snapshot_support_knee_excess_rad2", "snapshot_knee_weighted_rate"), values):
            self.writer.add_scalar("MotionQuality/" + name, value, locs["it"])
        self.writer.add_scalar("MotionQuality/action_rate_weight", SETTINGS["action_rate_weight"], locs["it"])
