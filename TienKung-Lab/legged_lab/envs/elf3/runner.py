"""Upstream training with an ELF3 observation/action contract on checkpoints."""
import hashlib
import json

import torch
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.assets.elf3 import CONTRACT


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
            "shoulder_x_max_deviation": self.env.cfg.shoulder_x_max_deviation,
            "policy_action_clip": self.env.cfg.policy_action_clip,
        }

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["elf3_contract"] = self.contract()
        super().save(path, infos)

    def load(self, path, load_optimizer=True):
        resolved = self._resolve_checkpoint_path(path)
        saved = torch.load(resolved, map_location="cpu", weights_only=False)
        checkpoint_contract = (saved.get("infos") or {}).get("elf3_contract")
        expected_contract = self.contract()
        legacy_contract = dict(expected_contract)
        legacy_contract.pop("shoulder_x_max_deviation")
        legacy_contract.pop("policy_action_clip")
        if checkpoint_contract not in (expected_contract, legacy_contract):
            raise ValueError("Checkpoint is not compatible with this ELF3 DWAQ contract. "
                             "G1 DWAQ and ELF3 AMP weights cannot be loaded by matching dimensions alone.")
        return super().load(resolved, load_optimizer=load_optimizer)
