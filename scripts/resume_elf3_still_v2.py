"""Explicitly resume the original ELF3 still-v2 run and its existing SwanLab run.

No fresh weights, changed training recipe, automatic training restart, or stop
monitor is introduced. Original params/initialization and source checkpoint are
read-only; a separate resume journal describes the restored optimizer and the
unserialized simulator/RNG state. ``--verify-config-only`` is CPU-only preflight.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_still_v2"
DEFAULT_RUN = ROOT / "logs/elf3_dwaq_still_v2/2026-11-02_19-58-27_still_v2_leg_waist_scratch_16384_seed42_3791567"
SWAN_RUN = "Rainbowow/DWAQ/vckg3ugg"
TIME_TAGS = ("Train/mean_reward/time", "Train/mean_episode_length/time")
# Previously reviewed cache-path portability change; no physical/training file
# is exempted. ARROW_SHA256 and its process-local display alias remain checked.
DISPLAY_SOURCE_CHANGE = (
    "b1c17e78882d8337acc549cbbebded885555063b5b34cb7a51b39898e34c380d",
    "b7bb592c654035e574cad0c6d44c330effc7fa4aed31377cdc26692729f17f38",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Completed model_N.pt; default is latest numbered checkpoint in log-dir.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--target-iteration", type=int, default=100000)
    parser.add_argument("--use-cached-visual", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-config-only", action="store_true",
                        help="CPU metadata/source/weight/Adam preflight; no Isaac app or environment.")
    parser.add_argument("--swanlab-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--training-pid", type=int, help=argparse.SUPPRESS)
    return parser


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def updates_to_target(saved_iteration, target_iteration):
    if saved_iteration < 0 or target_iteration <= saved_iteration:
        raise ValueError("Target must exceed the completed checkpoint iteration")
    return target_iteration - saved_iteration


def source_audit(initialization):
    checked, approved_display = [], []
    for relative, expected in initialization["source_sha256"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual == expected:
            checked.append(relative)
        elif (relative == "scripts/train_elf3_cached_visuals.py"
              and (expected, actual) == DISPLAY_SOURCE_CHANGE):
            approved_display.append({"path": relative, "initial_sha256": expected,
                                     "current_sha256": actual, "display_only": True})
        else:
            raise ValueError(f"Original still-v2 training-source fingerprint changed: {relative}")
    return {"identical_files": len(checked), "approved_display_only_changes": approved_display,
            "all_training_dependencies_unchanged": True}


def require_finite_tree(value, label="checkpoint"):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Nonfinite {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite {label}")


def assert_restored_equal(actual, expected, label):
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype:
            raise ValueError(f"Restored tensor type/dtype mismatch: {label}")
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise ValueError(f"Restored tensor mismatch: {label}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Restored mapping mismatch: {label}")
        for key in expected:
            assert_restored_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ValueError(f"Restored sequence mismatch: {label}")
        for index, (first, second) in enumerate(zip(actual, expected)):
            assert_restored_equal(first, second, f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"Restored value mismatch: {label}")


def preflight(args):
    import torch
    import yaml
    directory = args.log_dir.resolve(strict=True)
    initialization = json.loads((directory / "initialization.json").read_text())
    if (initialization.get("task"), initialization.get("num_envs"), initialization.get("seed"),
            initialization.get("target_iteration"), initialization.get("save_interval")) != (
                TASK, 16384, 42, 100000, 500):
        raise ValueError("Expected the original 16384-env seed42 still-v2 run")
    if args.target_iteration != initialization["target_iteration"]:
        raise ValueError("Keep the original model_100000 target; do not silently shorten/extend training")
    agent = yaml.safe_load((directory / "params/agent.yaml").read_text())
    if (agent["seed"], agent["max_iterations"], agent["save_interval"], agent["experiment_name"],
            agent["runner_class_name"], agent["policy"]["context_mode"]) != (
                42, 100001, 500, TASK, "Elf3DwaqStillV2Runner", "mean"):
        raise ValueError("Saved Agent recipe differs from the expected original still-v2 recipe")
    if args.device != agent["device"]:
        raise ValueError("This explicit recovery preserves the original device configuration")
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [(int(match.group(1)), path) for path in directory.glob("model_*.pt")
                      if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))]
        if not candidates:
            raise FileNotFoundError("No completed numbered checkpoint found")
        checkpoint = max(candidates, key=lambda entry: entry[0])[1]
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != directory or not re.fullmatch(r"model_\d+\.pt", checkpoint.name):
        raise ValueError("Resume the completed original-run checkpoint directly, without copies")
    source = source_audit(initialization)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved.get("infos") or {}).get("initialization_only"):
        raise ValueError("Initialization-only weights cannot resume completed training")
    if (saved.get("infos") or {}).get("elf3_contract") != initialization["experiment_contract"]:
        raise ValueError("Checkpoint identity differs from the original still-v2 initialization")
    iteration = int(saved["iter"])
    if checkpoint.name != f"model_{iteration}.pt":
        raise ValueError("Checkpoint filename and internal iteration disagree")
    remaining = updates_to_target(iteration, args.target_iteration)
    if not saved["optimizer_state_dict"]["state"]:
        raise ValueError("The original trained Adam state must be restored, not reset")
    require_finite_tree(saved["model_state_dict"], "model")
    require_finite_tree(saved["optimizer_state_dict"], "optimizer")
    run = json.loads((directory / "swanlab_run.json").read_text())
    cursor = json.loads((directory / "swanlab_scalar_repair_state.json").read_text())
    if run.get("id") != "vckg3ugg" or cursor.get("run_path") != SWAN_RUN:
        raise ValueError("The existing SwanLab run does not match the requested original run")
    if (cursor.get("event_log_dir", str(directory)) != str(directory)
            or cursor.get("time_offset_seconds", 0) != 0):
        raise ValueError("Do not repoint or rewrite the existing acknowledged SwanLab source")
    protected = {str(path): sha256(path) for path in (
        directory / "initialization.json", directory / "params/env.yaml",
        directory / "params/agent.yaml", checkpoint,
    )}
    report = {
        "task": TASK, "original_log_dir": str(directory), "checkpoint": str(checkpoint),
        "checkpoint_sha256": protected[str(checkpoint)], "completed_iteration": iteration,
        "next_iteration": iteration + 1, "remaining_updates": remaining,
        "target_iteration": args.target_iteration, "num_envs": 16384, "seed": 42,
        "save_interval": 500, "model_tensors": len(saved["model_state_dict"]),
        "optimizer_entries": len(saved["optimizer_state_dict"]["state"]),
        "loaded_learning_rates": [group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]],
        "source_audit": source, "swanlab_run_path": SWAN_RUN,
        "cloud_ack_iteration_before_resume": cursor.get("last_iteration"),
        "cloud_recomputed_prefix_policy": "retain existing ACK; do not overwrite already uploaded iteration points",
        "elapsed_time_log_offset": max((cursor.get("cursors", {}).get(tag, 0) for tag in TIME_TAGS), default=0) + 1,
        "elapsed_time_offset_is_logging_only": True,
        "environment_resume_note": "Checkpoint stores weights/Adam, not terrain curriculum, physics, RNG or history/filter buffers; reinitialize those with original recipe and seed42.",
        "from_scratch": False, "new_swanlab_run": False,
        "automatic_training_restart": False, "automatic_training_stop": False,
    }
    return directory, initialization, agent, saved, protected, report


def swanlab_supervisor(directory, training_pid):
    """Retry the existing-run scalar uploader only; never launch training."""
    import psutil
    process = psutil.Process(training_pid)
    created = process.create_time()

    def alive():
        try:
            return (process.is_running() and process.create_time() == created
                    and process.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return False

    command = [sys.executable, str(ROOT / "scripts/repair_swanlab_scalars.py"),
               "--log-dir", str(directory), "--training-pid", str(training_pid),
               "--run-path", SWAN_RUN, "--restore-running", "--workers", "4",
               "--batch-size", "5000", "--finish-on-training-exit"]
    with (directory / "swanlab_resume_repair.log").open("a") as output:
        while True:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode == 0 or not alive():
                return
            print(f"Existing-run uploader exited {result.returncode}; retry logging only in 5s", flush=True)
            time.sleep(5)


def launch_existing_swanlab(directory):
    with (directory / "swanlab_resume_supervisor.log").open("a") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--swanlab-only",
             "--log-dir", str(directory), "--training-pid", str(os.getpid())],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    return process.pid


def run(args, prepared):
    directory, original, saved_agent, checkpoint, protected, report = prepared
    lock = (directory / "still_v2_resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless, device=args.device).app
    visual_context = ExitStack()
    env = runner = None
    success = False
    entered_training = False
    journal = directory / f"resume_{report['completed_iteration']}_{os.getpid()}.json"
    try:
        import torch
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs import Elf3DwaqStillV2Runner
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from legged_lab.envs.elf3.still_v2_runner import validate_still_v2_configuration
        from legged_lab.utils import task_registry
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest, compare_saved_run
        from train_elf3_still_v2 import (
            cached_visual_alias, validate_parent_recipe, validate_runtime_overrides,
        )

        visual_cache = visual_context.enter_context(cached_visual_alias(args.use_cached_visual))
        integrity, assets = check_source_manifest(ROOT), verify_retained_assets()
        registered_cfg, registered_agent = task_registry.get_cfgs(TASK)
        cfg, agent = registered_cfg.copy(), registered_agent.copy()
        cfg.scene.num_envs, cfg.scene.seed, cfg.device = 16384, 42, args.device
        agent.seed, agent.device = 42, args.device
        agent.max_iterations, agent.save_interval = 100001, 500
        agent.run_name, agent.logger, agent.resume = saved_agent["run_name"], "tensorboard", False
        runtime = validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent)
        parent_cfg, parent_agent = task_registry.get_cfgs("elf3_dwaq_still")
        parent = validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)
        exact = validate_still_v2_configuration(cfg, agent.to_dict())
        compare_saved_run(cfg, agent, directory)
        if canonical_sha256(cfg.to_dict()) != original["configuration_sha256"]:
            raise ValueError("Restored live environment differs from the original configuration SHA")
        if canonical_sha256(agent.to_dict()) != original["agent_configuration_sha256"]:
            raise ValueError("Restored live Agent differs from the original configuration SHA")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = task_registry.get_task_class(TASK)(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Original 5x100/29 interface changed")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("Original physical implicit PD changed")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqStillV2Runner(env, agent.to_dict(), log_dir=str(directory), device=args.device)
        runner.configuration_audit = exact
        runner.load(report["checkpoint"], load_optimizer=True)
        if runner.current_learning_iteration != report["next_iteration"]:
            raise ValueError("Runner did not advance from completed checkpoint to its next update")
        if not runner.alg.optimizer.state:
            raise ValueError("Loaded Adam state is unexpectedly empty")
        assert_restored_equal(runner.alg.policy.state_dict(), checkpoint["model_state_dict"], "model")
        assert_restored_equal(runner.alg.optimizer.state_dict(), checkpoint["optimizer_state_dict"], "Adam")
        if runner.alg.learning_rate != report["loaded_learning_rates"][0]:
            raise ValueError("Loaded algorithm LR differs from original Adam LR")
        assert_finite_weights(runner)
        runner.tot_time = float(report["elapsed_time_log_offset"])
        runner.tot_timesteps = runner.current_learning_iteration * agent.num_steps_per_env * env.num_envs
        report.update({
            "training_pid": os.getpid(), "restored_weights_and_Adam_exact": True,
            "restored_next_iteration": runner.current_learning_iteration,
            "configuration_audit": {"runtime": runtime, "parent": parent, "exact": exact},
            "experiment_contract": runner.contract(), "source_integrity": integrity,
            "assets": assets, "display_only_visual_cache": visual_cache,
            "resume_launcher_sha256": sha256(Path(__file__).resolve()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "original_files_readonly_sha256": protected,
        })
        save_json(journal, report)
        report["swanlab_supervisor_pid"] = launch_existing_swanlab(directory)
        save_json(journal, report)
        print("[READY] " + json.dumps(report, default=str), flush=True)
        entered_training = True
        runner.learn(num_learning_iterations=report["remaining_updates"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory / f"model_{args.target_iteration}.pt").is_file():
            raise RuntimeError("Training ended without model_100000.pt")
        success = True
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        (directory / f"resume_failure_{os.getpid()}.txt").write_text(detail)
        raise
    finally:
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
            for path, expected in protected.items():
                if sha256(path) != expected:
                    success = False
                    raise ValueError(f"Protected original file was modified: {path}")
        finally:
            if entered_training:
                save_json(directory / "training_exit.json", {
                    "success": success, "last_iteration": runner.current_learning_iteration,
                    "target_iteration": args.target_iteration, "resume_journal": journal.name,
                    "automatic_training_restart": False, "automatic_training_stop": False,
                })
            try:
                if env is not None:
                    env.close()
            finally:
                try:
                    visual_context.close()
                finally:
                    app.close()
                    lock.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.swanlab_only:
        if args.training_pid is None:
            raise ValueError("--swanlab-only requires --training-pid")
        swanlab_supervisor(args.log_dir.resolve(strict=True), args.training_pid)
        return
    prepared = preflight(args)
    if args.verify_config_only:
        print(json.dumps(prepared[-1], indent=2, ensure_ascii=False))
        return
    run(args, prepared)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit stop signal {signum}; training will not auto-restart")

    signal.signal(signal.SIGTERM, stop)
    main()
