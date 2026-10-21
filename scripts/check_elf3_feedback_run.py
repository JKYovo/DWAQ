"""Validate an isolated completed smoke run, not locomotion/hardware performance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    init = json.loads((run / "initialization.json").read_text())
    final = json.loads((run / "training_exit.json").read_text())
    assert init["from_scratch"] and init["checkpoint_loaded"] is None
    assert init["optimizer_state_entries"] == 0 and init["initial_iteration"] == 0
    assert init["num_envs"] == args.num_envs and init["seed"] == 42
    assert final["success"] and final["last_iteration"] == args.iterations - 1
    assert not final["auto_restart"]
    for relative, expected in init["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    env_cfg = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    assert agent["policy"]["context_mode"] == "mean"
    assert agent["runner_class_name"] == "Elf3DwaqFeedbackRunner"
    assert float(agent["upper_body_mirror_loss_coeff"]) == .1
    assert float(agent["upper_body_pose_loss_coeff"]) == .02
    assert float(agent["algorithm"]["min_learning_rate"]) == 1e-5
    assert float(agent["algorithm"]["max_learning_rate"]) == 3e-4
    assert all(float(group["armature"]) == .01 for group in env_cfg["scene"]["robot"]["actuators"].values())
    checkpoint = torch.load(run / f"model_{args.iterations - 1}.pt", map_location="cpu", weights_only=False)
    assert checkpoint["infos"]["elf3_contract"] == init["experiment_contract"]
    tensors = []
    def visit(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(checkpoint)
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    for tag in events.Tags()["scalars"]:
        assert all(math.isfinite(point.value) for point in events.Scalars(tag)), tag
    for tag in ("Loss/value_function", "Loss/surrogate", "Loss/autoencoder",
                "Loss/upper_body_symmetry", "Loss/upper_body_pose", "Feedback/reporting_error_rms"):
        assert [point.step for point in events.Scalars(tag)] == list(range(args.iterations)), tag
    assert all(1e-5 - 1e-10 <= point.value <= 3e-4 + 1e-10
               for point in events.Scalars("Loss/learning_rate"))
    trace_path = run / "physics_trace.npz"
    trace_summary = None
    if trace_path.exists():
        with np.load(trace_path, allow_pickle=False) as trace:
            assert float(trace["physics_dt"]) == .005
            assert trace["joint_vel"].shape == trace["reported_joint_vel"].shape
            assert trace["joint_pos"].shape[0] == args.iterations * 24 * 4
            for name in ("joint_pos", "joint_vel", "reported_joint_vel", "gyro", "feet_force"):
                assert np.isfinite(trace[name]).all(), name
            error = float(np.sqrt(np.mean((trace["joint_vel"] - trace["reported_joint_vel"]) ** 2)))
            if env_cfg["velocity_feedback"]["mode"] == "lpf20":
                assert error > 0
            trace_summary = {"samples": trace["joint_pos"].shape[0], "sample_hz": 200,
                             "raw_minus_reported_velocity_rms": error}
    report = {"passed": True, "gate": "technical_smoke_only_not_gait_or_hardware_acceptance",
              "run": str(run), "num_envs": args.num_envs, "iterations": args.iterations,
              "finite_checkpoint_tensors": len(tensors), "source_hashes_verified": len(init["source_sha256"]),
              "trace": trace_summary, "test_weights_used_for_formal_training": False}
    (run / "smoke_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
