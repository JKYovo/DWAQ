"""Height-force v2: only three explicit scalar changes from v1; no physics edits."""
from copy import copy, deepcopy

from legged_lab.envs.elf3.ankle_cost import Elf3DwaqAnkleCostRunner
from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.height_force import (
    Elf3DwaqHeightForceEnv, Elf3DwaqHeightForceRunner,
    make_configs as make_parent_configs, projected_motion_cfg, validate_height_force_configuration,
)
from legged_lab.envs.elf3.height_force_strong_contract import SETTINGS, height_force_strong_contract
from legged_lab.envs.elf3.motion_quality import Elf3DwaqMotionQualityRunner


RUNNER_CLASS = "Elf3DwaqHeightForceStrongRunner"
Elf3DwaqHeightForceStrongEnv = Elf3DwaqHeightForceEnv


def make_configs(num_envs=16384, device="cuda:0", run_name="height_force_v2"):
    cfg, agent = make_parent_configs(num_envs, device, run_name)
    cfg.reward.base_height.weight = SETTINGS["height_weight"]
    cfg.reward.feet_force.weight = SETTINGS["feet_force_weight"]
    cfg.reward.feet_force.params["threshold"] = SETTINGS["feet_force_threshold_n"]
    agent.runner_class_name = RUNNER_CLASS
    return cfg, agent


def projected_height_cfg(cfg):
    result = deepcopy(cfg)
    reference, _ = make_parent_configs(cfg.scene.num_envs, cfg.device)
    result.reward.base_height = deepcopy(reference.reward.base_height)
    result.reward.feet_force = deepcopy(reference.reward.feet_force)
    return result


def validate_height_force_strong_configuration(cfg, train_cfg):
    if train_cfg.get("runner_class_name") != RUNNER_CLASS:
        raise ValueError("Require the isolated height-force-v2 runner")
    reference, _ = make_configs(cfg.scene.num_envs, cfg.device)
    for name in ("base_height", "feet_force"):
        term, expected = getattr(cfg.reward, name, None), getattr(reference.reward, name)
        if (term is None or term.func is not expected.func
                or canonical_sha256(term.to_dict()) != canonical_sha256(expected.to_dict())):
            raise ValueError(f"Wrong v2 reward function/weight/parameters: {name}")
    projected = deepcopy(train_cfg)
    projected["runner_class_name"] = "Elf3DwaqHeightForceRunner"
    previous = validate_height_force_configuration(projected_height_cfg(cfg), projected)
    return {"version": SETTINGS["version"], "height_weight": SETTINGS["height_weight"],
        "feet_force_weight": SETTINGS["feet_force_weight"], "feet_force_threshold_n": SETTINGS["feet_force_threshold_n"],
        "only_three_scalar_changes_from_v1": True, "height_gates_unchanged": True,
        "zero_command_rewards_unchanged": True, "automatic_weight_escalation": False, "parent_audit": previous}


class Elf3DwaqHeightForceStrongRunner(Elf3DwaqHeightForceRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_height_force_strong_configuration(env.cfg, train_cfg)
        projected_env, projected_agent = copy(env), deepcopy(train_cfg)
        projected_env.cfg = projected_height_cfg(env.cfg)
        projected_agent["runner_class_name"] = "Elf3DwaqHeightForceRunner"
        super().__init__(projected_env, projected_agent, log_dir, device, checkpoint_dir)
        self.env, self.cfg, self.configuration_audit = env, deepcopy(train_cfg), audit

    def contract(self):
        result = Elf3DwaqAnkleCostRunner.contract(self)
        result["height_force_experiment"] = height_force_strong_contract()
        return result

    def initialize_from_clean_ankle(self, path, expected_sha256):
        validate_height_force_strong_configuration(self.env.cfg, self.cfg)
        projected = copy(self)
        projected.env, projected.cfg = copy(self.env), deepcopy(self.cfg)
        # Config-only projection for strict checkpoint validation, NEVER the
        # real reward manager. The shared policy/Adam are loaded in place.
        projected.env.cfg = projected_motion_cfg(projected_height_cfg(self.env.cfg))
        projected.cfg["runner_class_name"] = "Elf3DwaqMotionQualityRunner"
        result = Elf3DwaqMotionQualityRunner.initialize_from_ankle_cost(projected, path, expected_sha256)
        if projected.alg is not self.alg:
            raise ValueError("Parent initialization must share the actual policy and Adam")
        self.current_learning_iteration = projected.current_learning_iteration
        return result
