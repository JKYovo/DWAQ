"""Upstream training with an ELF3 observation/action contract on checkpoints."""
import hashlib
import json

import torch
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.assets.elf3 import CONTRACT
from legged_lab.envs.elf3.symmetry import ARM_ACTION_IDS, mirror_actions, mirror_history, mirror_observations


class Elf3DwaqOnPolicyRunner(DWAQOnPolicyRunner):
    def contract(self):
        return {
            "robot": "elf3",
            "asset_contract_sha256": hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest(),
            "joint_names": CONTRACT["joint_names"],
            "num_obs": self.env.num_obs,
            "num_privileged_obs": self.env.num_privileged_obs,
            "history_length": self.env.num_obs_hist,
            "control_dt": self.env.step_dt,
            "gait_period": self.env.cfg.robot.gait_phase.period,
            "gait_offset": self.env.cfg.robot.gait_phase.offset,
        }

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["elf3_contract"] = self.contract()
        super().save(path, infos)

    def load(self, path, load_optimizer=True):
        resolved = self._resolve_checkpoint_path(path)
        saved = torch.load(resolved, map_location="cpu", weights_only=False)
        if (saved.get("infos") or {}).get("elf3_contract") != self.contract():
            raise ValueError("Checkpoint is not compatible with this ELF3 DWAQ contract. "
                             "G1 DWAQ and ELF3 AMP weights cannot be loaded by matching dimensions alone.")
        return super().load(resolved, load_optimizer=load_optimizer)


class Elf3DwaqUpperBodySymmetryRunner(Elf3DwaqOnPolicyRunner):
    """ELF3 test runner with a mirror loss on arm actions only."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        coefficient = float(train_cfg["upper_body_mirror_loss_coeff"])
        self.alg.configure_upper_body_symmetry(
            mirror_observations=mirror_observations,
            mirror_history=mirror_history,
            mirror_actions=mirror_actions,
            action_ids=ARM_ACTION_IDS,
            coefficient=coefficient,
        )

    def contract(self):
        contract = super().contract()
        contract["upper_body_symmetry"] = {
            "action_joint_names": CONTRACT["joint_names"][15:],
            "mirror_loss_coeff": float(self.cfg["upper_body_mirror_loss_coeff"]),
            "observation_mirror": "sagittal_full_history",
        }
        return contract
