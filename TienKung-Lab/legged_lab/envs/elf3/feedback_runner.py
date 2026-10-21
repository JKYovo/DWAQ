"""Independent checkpoint identity and CPU scalar logging for the feedback trial."""
import math

import torch

from legged_lab.envs.elf3.feedback_contract import (
    canonical_sha256,
    experiment_contract,
    validate_checkpoint_contract,
)
from legged_lab.envs.elf3.runner import Elf3DwaqUpperBodySymmetryRunner


class Elf3DwaqFeedbackRunner(Elf3DwaqUpperBodySymmetryRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        experiment_contract(env.cfg.velocity_feedback, env.physics_dt, env.step_dt,
                            train_cfg["policy"].get("context_mode"))
        if train_cfg.get("empirical_normalization", False):
            raise ValueError("Keep the existing unnormalized deployment interface")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if self.alg.policy.context_mode != "mean":
            raise ValueError("Runner did not construct a mean-context policy")

    def contract(self):
        result = super().contract()
        result["feedback_experiment"] = experiment_contract(
            self.env.cfg.velocity_feedback, self.env.physics_dt, self.env.step_dt,
            self.cfg["policy"].get("context_mode"),
        )
        result["feedback_experiment"]["unchanged_reward_sha256"] = canonical_sha256(
            self.env.cfg.reward.to_dict()
        )
        env_recipe = self.env.cfg.to_dict()
        env_recipe.pop("device", None)
        env_recipe["scene"]["robot"]["spawn"]["usd_path"] = "<asset-contract-checked>"
        result["feedback_experiment"]["training_recipe_sha256"] = canonical_sha256({
            "environment": env_recipe,
            "algorithm": self.cfg["algorithm"],
            "policy": self.cfg["policy"],
            "seed": self.cfg["seed"],
            "num_steps_per_env": self.cfg["num_steps_per_env"],
        })
        return result

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["training_recipe"] = self.cfg
        super().save(path, infos)

    def load(self, path, load_optimizer=True):
        resolved = self._resolve_checkpoint_path(path)
        saved = torch.load(resolved, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(saved, self.contract())
        result = super().load(resolved, load_optimizer=load_optimizer)
        # The saved index denotes the last completed update, not the next one.
        self.current_learning_iteration = int(saved["iter"]) + 1
        return result

    def log(self, locs, width=80, pad=35):
        values = {}
        for frame in locs["ep_infos"]:
            for key, value in frame.items():
                numbers = (value.detach().cpu().reshape(-1).tolist()
                           if isinstance(value, torch.Tensor) else [float(value)])
                if not all(math.isfinite(number) for number in numbers):
                    raise FloatingPointError(f"Nonfinite episode metric: {key}")
                values.setdefault(key, []).extend(numbers)
        for key in ("mean_value_loss", "mean_surrogate_loss", "mean_autoenc_loss",
                    "mean_symmetry_loss", "mean_pose_loss"):
            if locs.get(key) is not None and not math.isfinite(float(locs[key])):
                raise FloatingPointError(f"Nonfinite loss: {key}")
        super().log(dict(locs, ep_infos=[]), width, pad)
        for key, numbers in values.items():
            if numbers:
                self.writer.add_scalar(key if "/" in key else "Episode/" + key,
                                       math.fsum(numbers) / len(numbers), locs["it"])
        raw = self.env.robot.data.joint_vel
        reported = self.env.velocity_feedback_model.value
        self.writer.add_scalar("Feedback/raw_velocity_rms", raw.square().mean().sqrt().item(), locs["it"])
        self.writer.add_scalar("Feedback/reporting_error_rms",
                               (raw - reported).square().mean().sqrt().item(), locs["it"])
