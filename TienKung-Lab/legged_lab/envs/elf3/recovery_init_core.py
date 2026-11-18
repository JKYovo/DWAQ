"""Recovery reset sampling only; no change to rewards, Actor inputs or physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TASK = "elf3_dwaq_recovery40_init_v4"
SETTINGS = {
    "version": TASK,
    "augmented_pose_fraction": .5,
    "augmented_bank_mix": "half perturbed supine reference frames, half procedural supine poses",
    "velocity_probabilities": [.75, .20, .05],
    "root_linear_bounds_mps": [[0., 0., 0.], [.4, .4, .3], [.8, .8, .5]],
    "root_angular_bounds_radps": [0., 2., 4.],
    "joint_velocity_bounds_radps": [0., 3., 6.],
    "velocity_frame": "world root-link linear/angular velocity; native joint order",
    "normal_walking_initialization": "unchanged WalkandRun sampler and reference velocities",
    "recovery_commands": "zero; existing separate flat pad",
    "recovery_exit": "unchanged immediate height>=0.8nominal (0.84m)",
    "runtime_fall_sampling": False,
    "reward_physics_policy_and_AMP_expert_unchanged": True,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InitializationBank:
    def __init__(self, path, joint_names, device="cpu"):
        self.path = Path(path).resolve(strict=True)
        metadata = json.loads(self.path.with_suffix(".json").read_text())
        if metadata["version"] != TASK or metadata["bank_sha256"] != sha256(self.path):
            raise ValueError("Initialization bank identity mismatch")
        contract_path = Path(__file__).parents[2]/"configs/elf3/contract.json"
        contract = json.loads(contract_path.read_text())
        if metadata["asset_contract_sha256"] != hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest():
            raise ValueError("Initialization bank was generated for different robot assets")
        with np.load(self.path, allow_pickle=False) as source:
            names = source["joint_names"].tolist()
            order = [names.index(n) for n in joint_names]
            arrays = {"joint_pos": source["joint_pos"][:, order],
                      "root_pos": source["root_pos"], "root_quat": source["root_quat"],
                      "pose_kind": source["pose_kind"]}
        count = len(arrays["root_pos"])
        if (arrays["joint_pos"].shape != (count, 29) or arrays["root_pos"].shape != (count, 3)
                or arrays["root_quat"].shape != (count, 4) or arrays["pose_kind"].shape != (count,)):
            raise ValueError("Malformed recovery initialization bank")
        if count < 256 or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Initialization bank must have enough finite poses")
        q = arrays["root_quat"]
        if (not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-5)
                or not ((arrays["root_pos"][:, 2] > 0) & (arrays["root_pos"][:, 2] < .55)).all()
                or not (2*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2]) > .65).all()
                or not np.isin(arrays["pose_kind"], [1, 2]).all()):
            raise ValueError("Bank must contain normalized, low-height supine poses")
        self.frames = {key: torch.as_tensor(value, device=device,
                         dtype=torch.long if key == "pose_kind" else torch.float32)
                       for key, value in arrays.items()}
        self.count = count

    def identity(self):
        return {"path": str(self.path), "sha256": sha256(self.path),
                "metadata_sha256": sha256(self.path.with_suffix(".json")), "poses": self.count}

    def sample(self, reference):
        """Only recovery rows enter here; reference tensors are never mutated."""
        frame = {key: value.clone() for key, value in reference.items()}
        n, device = len(frame["root_pos"]), frame["root_pos"].device
        augmented = torch.rand(n, device=device) < SETTINGS["augmented_pose_fraction"]
        ids = torch.randint(self.count, (n,), device=device)
        kinds = torch.zeros(n, dtype=torch.long, device=device)
        for key in ("root_pos", "root_quat", "joint_pos"):
            frame[key][augmented] = self.frames[key][ids[augmented]]
        kinds[augmented] = self.frames["pose_kind"][ids[augmented]]
        draw = torch.rand(n, device=device)
        p = SETTINGS["velocity_probabilities"]
        velocity_kind = (draw >= p[0]).long()+(draw >= p[0]+p[1]).long()
        linear = torch.tensor(SETTINGS["root_linear_bounds_mps"], device=device)[velocity_kind]
        angular = torch.tensor(SETTINGS["root_angular_bounds_radps"], device=device)[velocity_kind]
        joint = torch.tensor(SETTINGS["joint_velocity_bounds_radps"], device=device)[velocity_kind]
        frame["root_lin_vel"] = (2*torch.rand(n, 3, device=device)-1)*linear
        frame["root_ang_vel"] = (2*torch.rand(n, 3, device=device)-1)*angular[:, None]
        frame["joint_vel"] = (2*torch.rand(n, 29, device=device)-1)*joint[:, None]
        return frame, kinds, velocity_kind


def experiment_contract(bank_identity):
    directory = Path(__file__).parent
    return {**deepcopy(SETTINGS), "bank": deepcopy(bank_identity), "sources_sha256": {
        name: sha256(directory/name) for name in
        ("recovery_init_core.py", "recovery_init_env.py", "recovery_init_runner.py")}}
