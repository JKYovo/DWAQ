"""Continue the pinned still100000 model with only zero-command leg-pose strengthening.

Separate logs and SwanLab sidecar; never auto-restarts or resource-monitors
training. The target is an inclusive checkpoint index, not an update count.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_still_pose_v1"


def updates_to_target(next_iteration, target_iteration):
    if next_iteration < 0 or target_iteration < next_iteration:
        raise ValueError("Target checkpoint must be at or above the next update")
    return target_iteration - next_iteration + 1


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-iteration", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="zero_pose_resume100000")
    parser.add_argument("--swanlab-project", default="DWAQ")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if args.num_envs < 1 or args.seed != 42:
        parser.error("Positive env count and the original seed42 are required")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("run-name must be one filename component")
    remaining = updates_to_target(100001, args.target_iteration)
    app = AppLauncher(args).app
    env = runner = directory = None
    success = False
    try:
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs import Elf3DwaqStillPoseRunner
        from legged_lab.utils import task_registry
        from legged_lab.envs.elf3.still_pose_contract import PARENT_SHA256
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest
        from train_elf3_feedback import source_fingerprints

        integrity = check_source_manifest(ROOT)
        assets = verify_retained_assets()
        source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if source_sha != PARENT_SHA256:
            raise ValueError("Expected the pinned original still model_100000.pt")
        registered_cfg, registered_agent = task_registry.get_cfgs(TASK)
        cfg, agent = registered_cfg.copy(), registered_agent.copy()
        cfg.scene.num_envs, cfg.scene.seed, cfg.device = args.num_envs, args.seed, args.device
        agent.seed, agent.device = args.seed, args.device
        agent.max_iterations = remaining
        agent.run_name, agent.logger, agent.resume = args.run_name, "tensorboard", True
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT / "logs" / TASK / f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        imports = directory / "imports"
        imports.mkdir()
        imported = imports / "model_100000.pt"
        shutil.copy2(args.checkpoint, imported)
        if hashlib.sha256(imported.read_bytes()).hexdigest() != source_sha:
            raise ValueError("Imported checkpoint copy failed SHA256 validation")
        imported.chmod(0o444)
        agent.load_run, agent.load_checkpoint = str(imports), imported.name
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = task_registry.get_task_class(TASK)(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Observation/action dimensions changed")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("The physical PD implementation must remain unchanged")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqStillPoseRunner(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        restored = runner.initialize_from_still(str(imported))
        if runner.current_learning_iteration != 100001:
            raise ValueError("Wrong continuation index")
        assert_finite_weights(runner)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        fingerprints = source_fingerprints()
        fingerprints[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        initialization = {
            "task": TASK, "from_scratch": False, "checkpoint_source": str(args.checkpoint),
            "checkpoint_loaded": str(imported), **restored,
            "initial_iteration": runner.current_learning_iteration,
            "target_iteration": args.target_iteration, "learning_updates": remaining,
            "num_envs": args.num_envs, "seed": args.seed, "save_interval": agent.save_interval,
            "training_pid": os.getpid(), "automatic_training_restart": False,
            "automatic_training_stop": False, "configuration_audit": runner.configuration_audit,
            "experiment_contract": runner.contract(), "source_sha256": fingerprints,
            "source_integrity": integrity, "assets": assets,
            "environment_resume_note": "Original checkpoint has no curriculum/physical/RNG/buffer states; reinitialize with original configuration and seed42.",
            "initial_max_terrain_level": cfg.scene.max_init_terrain_level,
            "policy": agent.policy.to_dict(), "algorithm": agent.algorithm.to_dict(),
        }
        (directory / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
        (directory / "configuration_audit.json").write_text(json.dumps(runner.configuration_audit, indent=2) + "\n")
        # An explicitly marked pre-update snapshot is evidence, never a resume point.
        runner.save(str(directory / "imported_initialization_only.pt"), {"initialization_only": True})
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(json.dumps({
                "project": args.swanlab_project, "name": f"{TASK}_{stamp}_{args.run_name}",
                "config": initialization,
            }, indent=2) + "\n")
            with (directory / "swanlab_supervisor.log").open("a") as output:
                subprocess.Popen([sys.executable, str(ROOT / "scripts/pose_scratch_swanlab.py"),
                                  "--training-pid", str(os.getpid()), "--log-dir", str(directory)],
                                 cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print("[READY] " + json.dumps({"directory": str(directory), "restored": restored,
                                        "updates": remaining, "target": args.target_iteration}), flush=True)
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != source_sha:
            raise ValueError("Original source checkpoint changed")
        if hashlib.sha256(imported.read_bytes()).hexdigest() != source_sha:
            raise ValueError("Imported read-only checkpoint changed")
        if not (directory / f"model_{args.target_iteration}.pt").is_file():
            raise RuntimeError("Training ended without the requested target checkpoint")
        success = True
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if directory is not None:
            (directory / "failure_traceback.txt").write_text(detail)
        raise
    finally:
        if directory is not None:
            (directory / "training_exit.json").write_text(json.dumps({
                "success": success, "last_iteration": runner.current_learning_iteration if runner else None,
                "target_iteration": args.target_iteration, "auto_restart": False,
            }, indent=2) + "\n")
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit stop signal {signum}; no automatic restart")
    signal.signal(signal.SIGTERM, stop)
    main()
