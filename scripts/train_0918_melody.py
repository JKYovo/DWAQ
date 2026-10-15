"""Registered 2362a3a symmetry/pose task from scratch, retaining current assets."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=100001)
parser.add_argument("--run-name", default="armature_0p01_registered_symmetry_pose_scratch_4096_seed42")
parser.add_argument("--swanlab-project", default=None)
parser.add_argument("--verify-config-only", action="store_true", help="Validate and compare configs without creating environments or training")
parser.add_argument("--reference-run", type=Path, help="Optional saved run whose params must match; no weights are loaded")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import inspect
import json
import math
import os
import signal
import subprocess
from datetime import datetime
from types import MethodType
import torch
from isaaclab.utils.io import dump_yaml
from legged_lab.envs import *
from legged_lab.utils import task_registry
from legged_lab.elf3_contract import CONTRACT
from legged_lab.envs.elf3.runner import Elf3DwaqUpperBodySymmetryRunner
from release_0918_checks import check_source_manifest, compare_saved_run
from melody_control_checks import check_configuration, check_runtime, assert_finite_weights
from melody_launch_audit import audit_overrides, check_saved_configuration, check_override_guard

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2362a3ade4f12846ee26c2da0d33a6634cad5349"
TASK = "elf3_dwaq_upper_symmetry_pose"
RECIPE = "elf3_0918_sympose_armature_0p01_v3"
# Identical to this commit's upstream legged_lab/scripts/train.py settings.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def cpu_episode_log(self, locs, width=80, pad=35):
    """Logging-only adapter; the exact registered runner initializes/learns/saves."""
    values = {}
    for frame in locs["ep_infos"]:
        for key, value in frame.items():
            numbers = value.detach().cpu().reshape(-1).tolist() if isinstance(value, torch.Tensor) else [float(value)]
            if not all(math.isfinite(v) for v in numbers):
                raise FloatingPointError(f"Nonfinite episode metric: {key}")
            values.setdefault(key, []).extend(numbers)
    for key in ("mean_value_loss", "mean_surrogate_loss", "mean_autoenc_loss", "mean_symmetry_loss", "mean_pose_loss"):
        if not math.isfinite(float(locs[key])):
            raise FloatingPointError(f"Nonfinite loss: {key}")
    Elf3DwaqUpperBodySymmetryRunner.log(self, dict(locs, ep_infos=[]), width, pad)
    for key, numbers in values.items():
        if numbers:
            self.writer.add_scalar(key if "/" in key else "Episode/" + key, math.fsum(numbers) / len(numbers), locs["it"])


def main():
    if args.num_envs <= 0 or args.iterations <= 0:
        raise ValueError("Positive counts required")
    source_validation = check_source_manifest(ROOT)
    registered_env, registered_agent = task_registry.get_cfgs(TASK)
    # Never manufacture this task by adding coefficients to the base Agent.
    cfg, agent = registered_env.copy(), registered_agent.copy()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = 42
    agent.seed = 42
    agent.resume = False
    agent.max_iterations = args.iterations
    agent.save_interval = 500
    agent.logger = "tensorboard"
    agent.run_name = args.run_name
    override_audit = audit_overrides(registered_env, registered_agent, cfg, agent)
    override_audit["negative_guard_tests_passed"] = check_override_guard(registered_env, registered_agent)
    check_configuration(cfg, agent, ROOT)
    reference_validation = compare_saved_run(cfg, agent, args.reference_run) if args.reference_run else None
    if args.verify_config_only:
        print(json.dumps({"source_validation": source_validation, "reference_validation": reference_validation,
                          "override_audit": override_audit}, indent=2), flush=True)
        return
    directory = ROOT / "logs" / agent.experiment_name / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    directory.mkdir(parents=True, exist_ok=False)
    env = None; runner = None; completed = False
    try:
        dump_yaml(str(directory / "params/registered_task_env.yaml"), registered_env)
        dump_yaml(str(directory / "params/registered_task_agent.yaml"), registered_agent)
        override_audit["source_validation"] = source_validation
        override_audit["comparison_to_reference_run"] = reference_validation
        env = task_registry.get_task_class(TASK)(cfg, headless=args.headless)
        runtime = check_runtime(env, exercise_mapping=args.iterations < 100001)
        runner_class = globals()[agent.runner_class_name]
        assert runner_class is Elf3DwaqUpperBodySymmetryRunner
        runner = runner_class(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        # Keep exact registered class, __init__, learn, save, PPO and auxiliary losses.
        runner.log = MethodType(cpu_episode_log, runner)
        modules = {"environment": inspect.getfile(type(env)), "policy": inspect.getfile(type(runner.alg.policy)),
            "ppo": inspect.getfile(type(runner.alg)), "source_runner": inspect.getfile(type(runner))}
        if not all(Path(p).resolve().is_relative_to(ROOT) for p in modules.values()):
            raise ValueError(f"Foreign training code imported: {modules}")
        assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        saved_validation = check_saved_configuration(directory, runner, agent)
        override_audit["post_dump_validation"] = saved_validation
        override_audit["non_configuration_adapters"] = {
            "runner_class": type(runner).__name__,
            "methods_unchanged": ["__init__", "learn", "save", "load", "contract"],
            "log_only": "CPU aggregation of episode scalars; upstream logging of all five losses",
            "swanlab": "isolated sidecar; no training imports or optimization changes",
            "torch_backend_flags": "identical to pinned upstream scripts/train.py",
            "preflight_action_probe": args.iterations < 100001,
            "formal_initialization": "no diagnostic env.step/reset; original runner initialization",
        }
        (directory / "launch_override_audit.json").write_text(json.dumps(override_audit, indent=2) + "\n")
        with (directory / "source_changes.patch").open("w") as changes:
            subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, stdout=changes, check=True)
        sources = [ROOT / "scripts" / name for name in ["train_0918_melody.py", "melody_control_checks.py", "melody_launch_audit.py", "check_melody_smoke.py", "run_host_python.sh", "train_0918_melody.sh", "release_0918_checks.py"]]
        sources += [ROOT / name for name in ["TienKung-Lab/legged_lab/assets/elf3/__init__.py",
            "TienKung-Lab/legged_lab/assets/elf3/elf3.py", "TienKung-Lab/legged_lab/elf3_contract.py",
            "TienKung-Lab/legged_lab/configs/elf3/contract.json", "scripts/references/melody_elf3.py"]]
        initialization = {"recipe": RECIPE, "from_scratch": True, "checkpoint_loaded": None,
            "optimizer_state_entries": 0, "iteration": 0, "training_pid": os.getpid(), "source_commit": COMMIT,
            "seed": 42, "num_envs": env.num_envs, "target_iteration": args.iterations - 1,
            "save_interval": 500, "task": TASK, "source_modules": modules,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "runtime_robot_validation": runtime, "registered_task_validation": saved_validation,
            "automatic_training_restart": False, "checkpoint_resume_old_recipe_allowed": False}
        (directory / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
        assert_finite_weights(runner)
        runner.save(str(directory / "initial_random_model.pt"), {"launch_recipe": RECIPE})
        print(f"[INFO] Exact registered symmetry/pose runner, saved Agent config and retained assets verified. Log directory: {directory}", flush=True)
        print(f"[INFO] Agent/runner verification: {json.dumps(saved_validation)}", flush=True)
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(json.dumps({"project": args.swanlab_project,
                "name": "ELF3_0918_SymmetryPose_Armature0p01_Scratch_4096_seed42",
                "config": initialization | {"learning_rate": agent.algorithm.learning_rate,
                    "armature_all_joints": .01,
                    "min_learning_rate": agent.algorithm.min_learning_rate,
                    "max_learning_rate": agent.algorithm.max_learning_rate,
                    "upper_body_mirror_loss_coeff": agent.upper_body_mirror_loss_coeff,
                    "upper_body_pose_loss_coeff": agent.upper_body_pose_loss_coeff,
                    "steps_per_env": agent.num_steps_per_env, "mini_batches": agent.algorithm.num_mini_batches,
                    "ppo_epochs": agent.algorithm.num_learning_epochs, "override_audit": override_audit}}, indent=2) + "\n")
            worker = ROOT / "scripts/pose_scratch_swanlab.py"
            with (directory / "swanlab_supervisor.log").open("a") as output:
                subprocess.Popen([os.sys.executable, str(worker), "--training-pid", str(os.getpid()), "--log-dir", str(directory)],
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        assert_finite_weights(runner); completed = True
    finally:
        (directory / "training_exit.json").write_text(json.dumps({"success": completed,
            "last_iteration": runner.current_learning_iteration if runner else None, "auto_restart": False}, indent=2) + "\n")
        if runner is not None and runner.writer is not None: runner.writer.close()
        if env is not None: env.close()


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Training explicitly stopped by signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
