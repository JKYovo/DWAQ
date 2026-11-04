"""Only two reward additions to the original ELF3 still-v2 continuation.

No registry/global monkeypatch or old-task source change is needed. Physics,
actor observations and feedback remain the original implementation. New
checkpoint files live in their stage directory; TensorBoard can stay in the
original run directory for the existing SwanLab ACK stream.
"""
from copy import copy, deepcopy
from pathlib import Path

import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.ankle_cost_contract import SETTINGS, reward_stage_contract
from legged_lab.envs.elf3.ankle_cost_rewards import ANKLE_JOINT_NAMES, FOOT_BODY_NAMES, ankle_torque_l2, stance_foot_speed_cost
from legged_lab.envs.elf3.elf3_still_v2_config import Elf3DwaqStillV2AgentCfg, Elf3DwaqStillV2EnvCfg
from legged_lab.envs.elf3.feedback_contract import canonical_sha256, validate_checkpoint_contract
from legged_lab.envs.elf3.still_env import Elf3DwaqStillEnv
from legged_lab.envs.elf3.still_runner import Elf3DwaqStillRunner
from legged_lab.envs.elf3.still_v2_runner import Elf3DwaqStillV2Runner, validate_still_v2_configuration


NEW_REWARDS = ("ankle_torque_l2", "stance_foot_speed_cost")


def make_configs(num_envs=16384, device="cuda:0", run_name="still_v2_ankle_cost_v1"):
    cfg, agent = Elf3DwaqStillV2EnvCfg(), Elf3DwaqStillV2AgentCfg()
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = num_envs, 42, device
    agent.device, agent.run_name, agent.logger = device, run_name, "tensorboard"
    agent.runner_class_name = "Elf3DwaqAnkleCostRunner"
    agent.max_iterations, agent.save_interval = 100001, 500
    cfg.reward.ankle_torque_l2 = RewardTermCfg(
        func=ankle_torque_l2, weight=SETTINGS["ankle_torque_weight"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ANKLE_JOINT_NAMES), preserve_order=True)})
    cfg.reward.stance_foot_speed_cost = RewardTermCfg(
        func=stance_foot_speed_cost, weight=SETTINGS["stance_speed_weight"],
        params={name: SETTINGS[name] for name in ("stance_threshold", "phase_sigma", "speed_scale", "command_epsilon")})
    return cfg, agent


def projected_parent_cfg(cfg):
    result = cfg.copy()
    for name in NEW_REWARDS:
        if not hasattr(result.reward, name):
            raise ValueError(f"Missing added ankle-cost reward: {name}")
        delattr(result.reward, name)
    return result


def validate_ankle_cost_configuration(cfg, train_cfg):
    _, expected_agent = make_configs(cfg.scene.num_envs, cfg.device, train_cfg.get("run_name", ""))
    expected, _ = make_configs(cfg.scene.num_envs, cfg.device)
    for name in NEW_REWARDS:
        term, reference = getattr(cfg.reward, name, None), getattr(expected.reward, name)
        if term is None or term.func is not reference.func or term.weight != reference.weight:
            raise ValueError(f"Incorrect ankle-cost reward callback or weight: {name}")
        if canonical_sha256(term.to_dict()) != canonical_sha256(reference.to_dict()):
            raise ValueError(f"Incorrect ankle-cost reward parameters/mapping: {name}")
    if train_cfg.get("runner_class_name") != "Elf3DwaqAnkleCostRunner":
        raise ValueError("Use the explicit ankle-cost stage runner")
    if train_cfg.get("save_interval") != 500 or train_cfg.get("seed") != 42:
        raise ValueError("Keep save500 and seed42")
    # Existing strict parent validator catches every other reward/asset/PD/
    # observation/PPO/learning/sampling modification. Do not weaken that code.
    parent_agent = deepcopy(train_cfg)
    parent_agent["runner_class_name"] = "Elf3DwaqStillV2Runner"
    validate_still_v2_configuration(projected_parent_cfg(cfg), parent_agent)
    if canonical_sha256(expected_agent.policy.to_dict()) != canonical_sha256(train_cfg["policy"]):
        raise ValueError("Actor/CENet recipe changed")
    return {"only_added_rewards": list(NEW_REWARDS), "weights": [-.0005, -1.],
            "parent_configuration_preserved": True, "recipe_version": SETTINGS["version"],
            "num_envs": cfg.scene.num_envs, "seed": 42}


class Elf3DwaqAnkleCostEnv(Elf3DwaqStillEnv):
    def init_buffers(self):
        super().init_buffers()
        self._ankle_cost_feet_ids, names = self.robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)
        if tuple(names) != FOOT_BODY_NAMES:
            raise ValueError("Ankle-cost foot bodies must be explicitly left/right ankle-x")
        self.ankle_cost_feet_body_names = tuple(names)
        self._ankle_cost_speed_sum = torch.zeros((self.num_envs, 2), device=self.device)
        self.ankle_cost_feet_speed_mean = torch.zeros_like(self._ankle_cost_speed_sum)
        self._ankle_cost_substeps = 0

    def step(self, actions):
        self._ankle_cost_speed_sum.zero_()
        self.ankle_cost_feet_speed_mean.zero_()
        self._ankle_cost_substeps = 0
        result = super().step(actions)
        if self._ankle_cost_substeps != self.cfg.sim.decimation:
            raise RuntimeError("Foot-speed reward must sample every physical substep exactly once")
        return result

    def _post_physics_step(self):
        super()._post_physics_step()  # preserves the original 200Hz dq feedback update
        speed = self.robot.data.body_lin_vel_w[:, self._ankle_cost_feet_ids].norm(dim=-1)
        self._ankle_cost_speed_sum.add_(speed)
        self._ankle_cost_substeps += 1
        self.ankle_cost_feet_speed_mean.copy_(self._ankle_cost_speed_sum / self._ankle_cost_substeps)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self, "_ankle_cost_speed_sum"):
            self._ankle_cost_speed_sum[env_ids] = 0.
            self.ankle_cost_feet_speed_mean[env_ids] = 0.


class Elf3DwaqAnkleCostRunner(Elf3DwaqStillV2Runner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_dir=None):
        self.configuration_audit = validate_ankle_cost_configuration(env.cfg, train_cfg)
        # Skip only the old v2 constructor's exact no-extra-reward assertion;
        # the strict projected-parent audit above has already checked it.
        Elf3DwaqStillRunner.__init__(self, env, train_cfg, log_dir=log_dir, device=device)
        settings = env.cfg.still_v2_pose
        self._still_v2_leg_ids, names = env.robot.find_joints(settings["leg_joint_names"], preserve_order=True)
        if list(names) != settings["leg_joint_names"]:
            raise ValueError("Original still-v2 leg diagnostics mapping changed")
        self._still_v2_waist_ids, names = env.robot.find_joints(settings["waist_joint_names"], preserve_order=True)
        if list(names) != settings["waist_joint_names"]:
            raise ValueError("Original still-v2 waist diagnostics mapping changed")
        self.checkpoint_dir = Path(checkpoint_dir or log_dir) if (checkpoint_dir or log_dir) else None
        self._ankle_diagnostic_ids, names = env.robot.find_joints(list(ANKLE_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ANKLE_JOINT_NAMES:
            raise ValueError("Ankle diagnostic mapping changed")

    def contract(self):
        result = super().contract()
        result["reward_stage_experiment"] = reward_stage_contract()
        return result

    def _parent_contract(self):
        legacy = copy(self)
        legacy.env = copy(self.env)
        legacy.env.cfg = projected_parent_cfg(self.env.cfg)
        # The original parent fingerprint includes its 16384 environments.
        # A smoke run changes only the new simulator batch, not its source CP.
        legacy.env.cfg.scene.num_envs = 16384
        legacy.cfg = deepcopy(self.cfg)
        legacy.cfg["runner_class_name"] = "Elf3DwaqStillV2Runner"
        return Elf3DwaqStillV2Runner.contract(legacy)

    def initialize_from_parent(self, path, expected_sha256):
        import hashlib
        if self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError("Parent migration requires a fresh runner, not an already trained optimizer")
        path = Path(path).resolve(strict=True)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError("Pinned original checkpoint SHA256 changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("infos") or {}).get("initialization_only"):
            raise ValueError("Cannot continue from initialization-only weights")
        validate_checkpoint_contract(saved, self._parent_contract())
        old_recipe = saved["infos"]["training_recipe"]
        for key in ("algorithm", "policy", "seed", "num_steps_per_env", "empirical_normalization",
                    "upper_body_mirror_loss_coeff", "upper_body_pose_loss_coeff", "save_interval"):
            if old_recipe.get(key) != self.cfg.get(key):
                raise ValueError(f"Parent PPO/Actor/learning setting changed: {key}")
        if not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Parent continuation must restore nonempty Adam state")
        DWAQOnPolicyRunner.load(self, str(path), load_optimizer=True)
        self.current_learning_iteration = int(saved["iter"]) + 1
        # Exact model, Adam and LR equality is also verified by the launcher;
        # this comparison makes the explicit promotion safe in other callers.
        self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        return {"checkpoint": str(path), "checkpoint_sha256": actual_sha,
                "completed_iteration": int(saved["iter"]), "next_iteration": self.current_learning_iteration,
                "weights_exactly_restored": True, "optimizer_exactly_restored": True,
                "learning_rate_restored": self.alg.learning_rate,
                "environment_curriculum_RNG_and_histories_restored": False}

    @staticmethod
    def _assert_tree_equal(actual, expected, label):
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                raise ValueError(f"Restored tensor mismatch: {label}")
        elif isinstance(expected, dict):
            if set(actual) != set(expected):
                raise ValueError(f"Restored keys mismatch: {label}")
            for key in expected:
                Elf3DwaqAnkleCostRunner._assert_tree_equal(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                raise ValueError(f"Restored sequence length mismatch: {label}")
            for index, (a, b) in enumerate(zip(actual, expected)):
                Elf3DwaqAnkleCostRunner._assert_tree_equal(a, b, f"{label}.{index}")
        elif actual != expected:
            raise ValueError(f"Restored value mismatch: {label}")

    def load(self, path, load_optimizer=True):
        # Full new contract, including both reward weights and source hashes.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.contract())
        result = DWAQOnPolicyRunner.load(self, str(path), load_optimizer=load_optimizer)
        self.current_learning_iteration = int(saved["iter"]) + 1
        return result

    def save(self, path, infos=None):
        if self.checkpoint_dir is None:
            raise ValueError("Ankle-cost checkpoint directory is required")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = self.checkpoint_dir / Path(path).name
        if target.exists():
            # The upstream runner saves the last update both in its periodic
            # branch and on normal learn() exit. Allow only that same-state
            # idempotent save, without overwriting an existing file.
            saved = torch.load(target, map_location="cpu", weights_only=False)
            if int(saved["iter"]) == self.current_learning_iteration:
                validate_checkpoint_contract(saved, self.contract())
                self._assert_tree_equal(self.alg.policy.state_dict(), saved["model_state_dict"], "saved model")
                self._assert_tree_equal(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "saved Adam")
                return
            raise FileExistsError(f"Never overwrite a completed stage checkpoint: {target}")
        temporary = target.with_suffix(".pt.tmp")
        super().save(str(temporary), infos)
        temporary.replace(target)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        data = self.env.robot.data
        values = torch.stack([
            data.applied_torque[:, self._ankle_diagnostic_ids].square().mean().sqrt(),
            self.env.ankle_cost_feet_speed_mean.mean(),
            self.env.command_generator.command.abs().amax(-1).gt(1e-6).float().mean(),
        ]).detach().cpu().tolist()
        for key, value in zip(("estimated_ankle_torque_rms", "mean_substep_foot_speed", "moving_command_fraction"), values):
            self.writer.add_scalar("AnkleCost/" + key, value, locs["it"])
        self.writer.add_scalar("AnkleCost/reward_stage_active", 1., locs["it"])
