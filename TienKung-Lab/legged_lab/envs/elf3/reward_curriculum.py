"""Still-v2 -> final500N quality rewards, without modifying historical tasks."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path

import torch
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.elf3_still_v2_config import Elf3DwaqStillV2AgentCfg, Elf3DwaqStillV2EnvCfg
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.foot_force_500 import (
    Elf3DwaqFootForce500Env, Elf3DwaqFootForce500Runner,
    make_configs as make_final_configs, validate_configuration as validate_final,
)
from legged_lab.envs.elf3.reward_curriculum_contract import (
    TASK, WEIGHT_ENDPOINTS, RolloutClock, apply_weights, curriculum_alpha,
    curriculum_contract, effective_weights, schedule_settings, validate_settings,
)
from legged_lab.envs.elf3.still_v2_runner import Elf3DwaqStillV2Runner


RUNNER_CLASS = "Elf3DwaqRewardCurriculumRunner"


def make_configs(num_envs=16384, device="cuda:0", run_name="still_v2_curriculum",
                 ramp_start=20000, ramp_end=30000):
    cfg, agent = make_final_configs(num_envs, device, run_name)
    agent.runner_class_name = RUNNER_CLASS
    agent.experiment_name = agent.wandb_project = TASK
    agent.max_iterations = 50001
    agent.reward_curriculum = schedule_settings(ramp_start, ramp_end)
    return cfg, agent


def projected_final_recipe(recipe):
    result = deepcopy(recipe)
    result.pop("reward_curriculum", None)
    _, reference = make_final_configs()
    for key in ("runner_class_name", "experiment_name", "wandb_project"):
        result[key] = getattr(reference, key)
    return result


def validate_configuration(cfg, recipe):
    if recipe.get("runner_class_name") != RUNNER_CLASS:
        raise ValueError("Require the explicit reward-curriculum runner")
    settings = validate_settings(recipe.get("reward_curriculum"))
    if recipe.get("experiment_name") != TASK or recipe.get("wandb_project") != TASK:
        raise ValueError("Curriculum requires its independent experiment namespace")
    reference, _ = make_final_configs(cfg.scene.num_envs, cfg.device)
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(reference.to_dict()):
        raise ValueError("Keep the exact final-reference environment, assets and reward functions")
    audit = validate_final(cfg, projected_final_recipe(recipe))
    return {"schedule": settings, "global_peak_sampling_and_500N_from_stage1": True,
            "weight_endpoints": {k: list(v) for k, v in WEIGHT_ENDPOINTS.items()},
            "final_reference_audit": audit}


class Elf3DwaqRewardCurriculumEnv(Elf3DwaqFootForce500Env):
    def configure_reward_curriculum(self, settings, iteration):
        self.reward_curriculum_settings = validate_settings(settings)
        self.reward_curriculum_clock = RolloutClock()
        self.set_training_iteration(iteration)

    def set_training_iteration(self, iteration):
        self.reward_curriculum_weights = apply_weights(
            self.reward_manager, iteration, self.reward_curriculum_settings)
        self.reward_curriculum_iteration = iteration
        self.reward_curriculum_alpha = curriculum_alpha(iteration, self.reward_curriculum_settings)

    def step(self, actions):
        clock = getattr(self, "reward_curriculum_clock", None)
        if clock is not None and clock.active and clock.iteration != self.reward_curriculum_iteration:
            self.set_training_iteration(clock.iteration)
        result = super().step(actions)
        if clock is not None:
            clock.advance()
        return result


class Elf3DwaqRewardCurriculumRunner(Elf3DwaqFootForce500Runner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        audit = validate_configuration(env.cfg, train_cfg)
        self.reward_curriculum_settings = validate_settings(train_cfg["reward_curriculum"])
        super().__init__(env, projected_final_recipe(train_cfg), log_dir, device, checkpoint_dir)
        self.cfg = self.train_cfg = deepcopy(train_cfg)
        self.configuration_audit = audit
        env.configure_reward_curriculum(self.reward_curriculum_settings, 0)

    def contract(self):
        result = super().contract()
        result["reward_curriculum_experiment"] = curriculum_contract(self.reward_curriculum_settings)
        return result

    def original_still_contract(self, num_envs):
        original = copy(self)
        original.env = copy(self.env)
        original.env.cfg = Elf3DwaqStillV2EnvCfg()
        original.env.cfg.scene.num_envs = num_envs
        original.cfg = Elf3DwaqStillV2AgentCfg().to_dict()
        return Elf3DwaqStillV2Runner.contract(original)

    def initialize_from_still(self, path, expected_sha256, source_num_envs):
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Promotion requires a fresh runner")
        path = Path(path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Read-only original still checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("infos") or {}).get("initialization_only"):
            raise ValueError("Initialization snapshots cannot be training parents")
        validate_checkpoint_contract(saved, self.original_still_contract(source_num_envs))
        old, current = deepcopy(saved["infos"]["training_recipe"]), deepcopy(self.cfg)
        current.pop("reward_curriculum", None)
        for key in ("runner_class_name", "experiment_name", "wandb_project"):
            current[key] = getattr(Elf3DwaqStillV2AgentCfg(), key)
        for item in (old, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                item.pop(key, None)
        if canonical_sha256(old) != canonical_sha256(current) or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Keep the complete original policy/PPO/auxiliary settings and trained Adam")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = int(saved["iter"]) + 1
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        self.env.set_training_iteration(self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("infos") or {}).get("initialization_only"):
            raise ValueError("Initialization snapshots cannot resume")
        validate_checkpoint_contract(saved, self.contract())
        state = saved["infos"].get("reward_curriculum_state")
        if state != self.curriculum_state(int(saved["iter"])):
            raise ValueError("Checkpoint curriculum progress/weights do not match its completed iteration")
        result = DWAQOnPolicyRunner.load(self, path, load_optimizer=load_optimizer)
        self.current_learning_iteration = int(saved["iter"]) + 1
        self.env.set_training_iteration(self.current_learning_iteration if load_optimizer else int(saved["iter"]))
        return result

    def curriculum_state(self, iteration):
        return {"completed_iteration": iteration, "next_iteration": iteration+1,
                "settings": deepcopy(self.reward_curriculum_settings),
                "alpha": curriculum_alpha(iteration, self.reward_curriculum_settings),
                "weights": effective_weights(iteration, self.reward_curriculum_settings)}

    def save(self, path, infos=None):
        infos = dict(infos or {})
        if self.env.reward_curriculum_iteration != self.current_learning_iteration:
            raise ValueError("Never save a model with misaligned rollout/curriculum iteration")
        infos["reward_curriculum_state"] = self.curriculum_state(self.current_learning_iteration)
        super().save(path, infos)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.env.set_training_iteration(self.current_learning_iteration)
        clock = self.env.reward_curriculum_clock
        clock.begin(self.current_learning_iteration, self.num_steps_per_env)
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            clock.end()

    def log(self, locs, width=80, pad=35):
        if self.env.reward_curriculum_iteration != locs["it"]:
            raise ValueError("Reward schedule must stay fixed throughout each PPO rollout")
        super().log(locs, width, pad)
        self.writer.add_scalar("RewardCurriculum/alpha", self.env.reward_curriculum_alpha, locs["it"])
        self.writer.add_scalar("RewardCurriculum/iteration", locs["it"], locs["it"])
        for name, weight in self.env.reward_curriculum_weights.items():
            self.writer.add_scalar("RewardCurriculum/weight_"+name, weight, locs["it"])
        self.writer.add_scalar("RewardCurriculum/feet_force_threshold_N", 500., locs["it"])
