"""Launch the registered standing-fix task unchanged, with an isolated SwanLab sidecar.

The child is the native training entry point. This parent only records launch
provenance, starts logging and records completion; it never restarts or stops
training based on resource use, losses or progress.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from release_0918_checks import check_source_manifest
from train_elf3_feedback import source_fingerprints


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_stand_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(TASK, "elf3_dwaq_stand_v2", "elf3_dwaq_still"), default=TASK)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="stand_scratch_4096_seed42")
    parser.add_argument("--swanlab-project", default="DWAQ")
    args = parser.parse_args()
    task = args.task
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("Positive environment/iteration counts required")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("run-name must be one filename component")
    integrity = check_source_manifest(ROOT)
    # Unique suffix makes discovering the native timestamped log unambiguous.
    suffix = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    log_root = ROOT / "logs" / task
    command = [sys.executable, str(ROOT / "TienKung-Lab/legged_lab/scripts/train.py"),
               "--task", task, "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
               "--seed", str(args.seed), "--run_name", suffix, "--logger", "tensorboard", "--headless"]
    fingerprints = source_fingerprints()
    fingerprints[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    print("[LAUNCH] " + json.dumps({"command": command, "from_scratch": True,
                                    "automatic_stop_or_restart": False}), flush=True)
    child = subprocess.Popen(command, cwd=ROOT)
    directory = None
    try:
        # Startup-only discovery, not a training-health monitor. Once params
        # exist the only operation left on the training child is wait().
        while True:
            status = child.poll()
            matches = list(log_root.glob(f"*_{suffix}"))
            if len(matches) > 1:
                raise RuntimeError("Ambiguous native training log directory")
            # Native train closes both YAML files before it can save model_0.
            # Wait for that ordering barrier, not merely an opened YAML file.
            # If a short run already exited, discover its completed files too.
            if (matches and (matches[0] / "params/agent.yaml").is_file()
                    and ((matches[0] / "model_0.pt").exists() or status is not None)):
                directory = matches[0]
                import yaml
                agent = yaml.safe_load((directory / "params/agent.yaml").read_text())
                env = yaml.load((directory / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
                runner_name = {TASK: "Elf3DwaqStandRunner", "elf3_dwaq_stand_v2": "Elf3DwaqStandV2Runner",
                               "elf3_dwaq_still": "Elf3DwaqStillRunner"}[task]
                expected_agent = {"runner_class_name": runner_name, "experiment_name": task,
                                  "seed": args.seed, "max_iterations": args.iterations,
                                  "save_interval": 500, "resume": False, "logger": "tensorboard",
                                  "upper_body_mirror_loss_coeff": .1, "upper_body_pose_loss_coeff": .02}
                if any(agent.get(key) != value for key, value in expected_agent.items()):
                    raise ValueError("Unexpected saved agent configuration")
                if agent["policy"]["context_mode"] != "mean" or int(env["scene"]["num_envs"]) != args.num_envs:
                    raise ValueError("Wrong context mode or environment count")
                if env["velocity_feedback"]["mode"] != "lpf20":
                    raise ValueError("The standing task must retain the 20 Hz feedback model")
                init = {"task": task, "from_scratch": True, "checkpoint_loaded": None,
                        "initial_iteration": 0, "target_iteration": args.iterations - 1,
                        "num_envs": args.num_envs, "seed": args.seed, "save_interval": 500,
                        "training_pid": child.pid, "launcher_pid": os.getpid(), "command": command,
                        "automatic_training_restart": False, "automatic_training_stop": False,
                        "source_integrity": integrity, "source_sha256": fingerprints,
                        "standing_experiment": env.get("stillness", env.get("quiet_standing", env.get("standing_experiment"))),
                        "velocity_feedback": env["velocity_feedback"],
                        "policy": agent["policy"], "algorithm": agent["algorithm"]}
                (directory / "initialization.json").write_text(json.dumps(init, indent=2) + "\n")
                if args.swanlab_project:
                    (directory / "swanlab_request.json").write_text(json.dumps({
                        "project": args.swanlab_project, "name": f"{task}_{suffix}", "config": init,
                    }, indent=2) + "\n")
                    with (directory / "swanlab_supervisor.log").open("a") as output:
                        subprocess.Popen([sys.executable, str(ROOT / "scripts/pose_scratch_swanlab.py"),
                                          "--training-pid", str(child.pid), "--log-dir", str(directory)],
                                         cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                print(f"[READY] Training PID {child.pid}; log directory: {directory}", flush=True)
                break
            if status is not None:
                break
            time.sleep(1)
        result = child.wait()
        if directory is None:
            raise RuntimeError(f"Training ended before configuration was saved (exit {result})")
        target = directory / f"model_{args.iterations - 1}.pt"
        success = result == 0 and target.is_file()
        (directory / "training_exit.json").write_text(json.dumps({
            "success": success, "child_returncode": result, "target_checkpoint": str(target),
            "target_checkpoint_exists": target.is_file(), "auto_restart": False,
        }, indent=2) + "\n")
        if not success:
            raise RuntimeError(f"Training stopped before target checkpoint; exit {result}. No restart.")
    except BaseException:
        # In particular, no kill/restart is hidden in an exception handler.
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if directory is not None:
            (directory / "launcher_failure.txt").write_text(detail)
        raise


if __name__ == "__main__":
    main()
