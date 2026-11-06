"""Sampling-only foot-force repair layered over the unchanged smoothing stage."""
from copy import copy, deepcopy
import torch

from legged_lab.envs.elf3.control_smoothness import (
    Elf3DwaqControlSmoothnessEnv, Elf3DwaqControlSmoothnessRunner,
    make_configs as make_parent_configs, validate_configuration as validate_parent,
)
from legged_lab.envs.elf3.ankle_cost_rewards import FOOT_BODY_NAMES
from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.foot_force_peak_contract import peak_contract
from legged_lab.envs.elf3.foot_force_peak_rewards import body_force_control_peak, accumulate_peak


RUNNER_CLASS = "Elf3DwaqFootForcePeakRunner"


def make_configs(num_envs=16384, device="cuda:0", run_name="force_peak_from57000"):
    cfg, agent = make_parent_configs(num_envs, device, run_name)
    cfg.reward.feet_force.func = body_force_control_peak
    agent.runner_class_name = RUNNER_CLASS
    return cfg, agent


def projected_parent_cfg(cfg):
    result = deepcopy(cfg)
    reference, _ = make_parent_configs(cfg.scene.num_envs, cfg.device)
    result.reward.feet_force.func = reference.reward.feet_force.func
    return result


def validate_configuration(cfg, recipe):
    if recipe.get("runner_class_name") != RUNNER_CLASS:
        raise ValueError("Require the force-peak repair Runner")
    expected, _ = make_configs(cfg.scene.num_envs, cfg.device)
    if canonical_sha256(cfg.reward.feet_force.to_dict()) != canonical_sha256(expected.reward.feet_force.to_dict()):
        raise ValueError("Force sampling repair must keep the original weight, threshold, cap and feet")
    parent = deepcopy(recipe)
    parent["runner_class_name"] = "Elf3DwaqControlSmoothnessRunner"
    return {"only_change_from_smooth_stage": "complete control-window force peak",
            "parent_audit": validate_parent(projected_parent_cfg(cfg), parent)}


class Elf3DwaqFootForcePeakEnv(Elf3DwaqControlSmoothnessEnv):
    def init_buffers(self):
        super().init_buffers()
        self._force_peak_sensor_ids, names = self.contact_sensor.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)
        if tuple(names) != FOOT_BODY_NAMES:
            raise ValueError("Verify explicit left/right contact-foot mapping")
        self.force_peak_metric = torch.zeros(self.num_envs, device=self.device)
        self.force_peak_each_foot = torch.zeros((self.num_envs, 2), device=self.device)
        self.force_peak_last_metric = torch.zeros_like(self.force_peak_metric)
        self._force_peak_substeps = 0

    def step(self, actions):
        self.force_peak_metric.zero_()
        self.force_peak_each_foot.zero_()
        self.force_peak_last_metric.zero_()
        self._force_peak_substeps = 0
        result = super().step(actions)
        if self._force_peak_substeps != self.cfg.sim.decimation:
            raise RuntimeError("Every physics substep must update the force cache exactly once")
        return result

    def _post_physics_step(self):
        super()._post_physics_step()  # keep 200Hz velocity feedback and stance speed sampling
        force = self.contact_sensor.data.net_forces_w[:, self._force_peak_sensor_ids, :]
        metric = force[..., 2].norm(dim=-1)
        self.force_peak_last_metric.copy_(metric)
        self.force_peak_metric.copy_(torch.maximum(self.force_peak_metric, metric))
        accumulate_peak(self.force_peak_each_foot, force)
        self._force_peak_substeps += 1

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self, "force_peak_metric"):
            self.force_peak_metric[env_ids] = 0.
            self.force_peak_each_foot[env_ids] = 0.
            self.force_peak_last_metric[env_ids] = 0.


class Elf3DwaqFootForcePeakRunner(Elf3DwaqControlSmoothnessRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        parent_env, parent_recipe = copy(env), deepcopy(train_cfg)
        parent_env.cfg = projected_parent_cfg(env.cfg)
        parent_recipe["runner_class_name"] = "Elf3DwaqControlSmoothnessRunner"
        super().__init__(parent_env, parent_recipe, log_dir, device, checkpoint_dir)
        self.env, self.cfg, self.configuration_audit = env, deepcopy(train_cfg), audit

    def contract(self):
        result = super().contract()
        result["foot_force_sampling_experiment"] = peak_contract()
        return result

    def initialize_from_height_parent(self, path, expected_sha256):
        projected = copy(self)
        projected.env, projected.cfg = copy(self.env), deepcopy(self.cfg)
        projected.env.cfg = projected_parent_cfg(self.env.cfg)
        projected.cfg["runner_class_name"] = "Elf3DwaqControlSmoothnessRunner"
        Elf3DwaqControlSmoothnessRunner.initialize_from_height_parent(projected, path, expected_sha256)
        if projected.alg is not self.alg:
            raise ValueError("Promotion must restore the actual policy and Adam")
        self.current_learning_iteration = projected.current_learning_iteration

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        values = torch.stack((self.env.force_peak_metric.mean(), self.env.force_peak_last_metric.mean(),
                              self.env.force_peak_each_foot[:, 0].mean(), self.env.force_peak_each_foot[:, 1].mean()))
        for key, value in zip(("control_peak_norm_N", "last_substep_norm_N", "left_control_peak_N", "right_control_peak_N"),
                              values.detach().cpu().tolist()):
            self.writer.add_scalar("FootForceSampling/snapshot_" + key, value, locs["it"])
