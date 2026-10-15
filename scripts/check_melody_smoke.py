"""Validate a completed isolated 0918 smoke run; its weights are never reused."""
import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    initial = json.loads((run / "initialization.json").read_text())
    final = json.loads((run / "training_exit.json").read_text())
    last = args.iterations - 1
    assert initial["from_scratch"] and initial["checkpoint_loaded"] is None
    assert initial["optimizer_state_entries"] == 0 and initial["num_envs"] == args.num_envs
    assert final["success"] and final["last_iteration"] == last
    for key in ("effective_armature_min", "effective_armature_max", "physx_armature_min", "physx_armature_max"):
        assert abs(initial["runtime_robot_validation"][key] - .01) < 1e-8, key

    def finite(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    checkpoint = torch.load(run / f"model_{last}.pt", map_location="cpu", weights_only=False)
    finite(checkpoint)
    assert initial["recipe"] == "elf3_0918_sympose_armature_0p01_v3"
    assert initial["task"] == "elf3_dwaq_upper_symmetry_pose"
    validation = initial["registered_task_validation"]
    assert validation["exact_registered_runner_class"] == "Elf3DwaqUpperBodySymmetryRunner"
    symmetry = checkpoint["infos"]["elf3_contract"]["upper_body_symmetry"]
    assert symmetry["mirror_loss_coeff"] == .1 and symmetry["pose_loss_coeff"] == .02
    audit = json.loads((run / "launch_override_audit.json").read_text())
    assert audit["algorithm_policy_reward_overrides"] == []
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    points = [point for tag in events.Tags()["scalars"] for point in events.Scalars(tag)]
    assert points and all(math.isfinite(point.value) for point in points)
    # Upstream also writes /time series whose X coordinate is elapsed seconds.
    # Only iteration-indexed loss series must have exactly 0..last as steps.
    assert max(point.step for point in points) >= last
    for tag in ("Loss/upper_body_symmetry", "Loss/upper_body_pose"):
        losses = events.Scalars(tag)
        assert len(losses) == args.iterations and any(point.value > 0 for point in losses), tag
        assert [point.step for point in losses] == list(range(args.iterations)), tag
    assert all(1e-5 - 1e-10 <= point.value <= 3e-4 + 1e-10 for point in events.Scalars("Loss/learning_rate"))
    print(json.dumps({"passed": True, "run": str(run), "num_envs": args.num_envs,
                      "iterations": args.iterations, "weights_reused": False}, indent=2))


if __name__ == "__main__":
    main()
