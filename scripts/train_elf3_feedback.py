"""Isolated DWAQ consistency/velocity-feedback experiment; never auto-restarts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("elf3_dwaq_feedback_v1",)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    # The launcher is safe to try without accidentally starting a long run.
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Total update count, from zero; 100001 ends at model_100000.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="feedback_check")
    parser.add_argument("--resume", type=Path, help="Only an identical experiment contract is allowed")
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--verify-config-only", action="store_true")
    parser.add_argument("--trace-seconds", type=float, default=0,
                        help="Optional bounded 200 Hz physical-state trace, at most 120 simulated seconds")
    return parser


def validate_effective_config(cfg, agent, baseline_env, baseline_agent):
    """Fail closed: no incidental reward, asset, PD, task, LR or sampling edits."""
    from melody_launch_audit import differences
    from legged_lab.envs.elf3.feedback_contract import experiment_contract
    allowed_env = {"velocity_feedback", "scene.num_envs", "scene.seed", "device"}
    env_delta = differences(baseline_env.to_dict(), cfg.to_dict())
    # A new mapping is reported once by differences(), not recursively.
    for item in env_delta:
        if item["path"] not in allowed_env:
            raise ValueError(f"Unexpected environment override: {item}")
    allowed_agent = {"policy.context_mode", "runner_class_name", "experiment_name", "wandb_project",
                     "max_iterations", "save_interval", "run_name", "resume", "seed", "device", "logger"}
    agent_delta = differences(baseline_agent.to_dict(), agent.to_dict())
    for item in agent_delta:
        if item["path"] not in allowed_agent:
            raise ValueError(f"Unexpected agent override: {item}")
    if agent.runner_class_name != "Elf3DwaqFeedbackRunner" or agent.policy.context_mode != "mean":
        raise ValueError("Expected the isolated mean-context feedback runner")
    if agent.algorithm.to_dict() != baseline_agent.algorithm.to_dict():
        raise ValueError("Algorithm parameters changed")
    if cfg.reward.to_dict() != baseline_env.reward.to_dict():
        raise ValueError("Reward configuration changed")
    if agent.upper_body_mirror_loss_coeff != .1 or agent.upper_body_pose_loss_coeff != .02:
        raise ValueError("Upper-body auxiliary losses changed")
    if agent.resume or agent.save_interval != 500:
        raise ValueError("Fresh initialization flag and 500-update save interval required")
    exp = experiment_contract(cfg.velocity_feedback, cfg.sim.dt,
                              cfg.sim.dt * cfg.sim.decimation, agent.policy.context_mode)
    return {"environment_changes": env_delta, "agent_changes": agent_delta,
            "reward_asset_PD_commands_terrain_and_algorithm_unchanged": True,
            "experiment": exp}


def source_fingerprints():
    roots = [ROOT / "TienKung-Lab/legged_lab/envs/elf3",
             ROOT / "TienKung-Lab/rsl_rl/rsl_rl"]
    paths = {path for root in roots for path in root.rglob("*.py")}
    paths.update(ROOT / name for name in (
        "scripts/train_elf3_feedback.py", "scripts/feedback_run_tools.py",
        "scripts/feedback_collision_audit.py", "scripts/release_0918_checks.py",
        "TienKung-Lab/legged_lab/envs/g1/g1_dwaq_env.py",
        "TienKung-Lab/legged_lab/envs/__init__.py",
        "TienKung-Lab/legged_lab/assets/elf3/elf3.py",
        "TienKung-Lab/legged_lab/configs/elf3/contract.json",
    ))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    from isaaclab.app import AppLauncher
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("Positive environment and iteration counts required")
    if not math.isfinite(args.trace_seconds) or not 0 <= args.trace_seconds <= 120:
        parser.error("--trace-seconds must be between 0 and 120")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be one filename component")
    app = AppLauncher(args).app
    env = runner = trace = None
    directory = None
    success = False
    try:
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.envs import Elf3DwaqFeedbackRunner
        from legged_lab.utils import task_registry
        from legged_lab.elf3_contract import JOINT_NAMES, verify_retained_assets
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest

        integrity = check_source_manifest(ROOT)
        assets = verify_retained_assets()
        registered_env, registered_agent = task_registry.get_cfgs(args.task)
        cfg, agent = registered_env.copy(), registered_agent.copy()
        cfg.scene.num_envs, cfg.scene.seed, cfg.device = args.num_envs, args.seed, args.device
        agent.seed, agent.device = args.seed, args.device
        agent.max_iterations, agent.save_interval = args.iterations, 500
        agent.run_name, agent.logger, agent.resume = args.run_name, "tensorboard", False
        # Historical baseline remains an internal validator, not a public task.
        from legged_lab.envs.elf3.elf3_dwaq_config import (
            Elf3DwaqEnvCfg, Elf3DwaqUpperBodySymmetryPoseAgentCfg,
        )
        baseline_env, baseline_agent = Elf3DwaqEnvCfg(), Elf3DwaqUpperBodySymmetryPoseAgentCfg()
        audit = validate_effective_config(cfg, agent, baseline_env, baseline_agent)
        if args.verify_config_only:
            print(json.dumps({"configuration": audit, "source_integrity": integrity,
                              "retained_assets": assets}, indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT / "logs" / agent.experiment_name / f"{stamp}_{args.run_name}"
        directory.mkdir(parents=True, exist_ok=False)
        print(f"[INFO] Feedback experiment directory: {directory}", flush=True)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        (directory / "configuration_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = task_registry.get_task_class(args.task)(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Observation/action dimensions changed")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("The physical PD implementation must remain unchanged")
        actual_armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(actual_armature, torch.full_like(actual_armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqFeedbackRunner(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        if runner.alg.optimizer.state:
            raise ValueError("Expected a fresh optimizer before explicit resume")
        if args.resume:
            runner.load(str(args.resume))
        remaining = args.iterations - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError("Target iteration count is not above the checkpoint's completed update")
        from feedback_collision_audit import capture_contact_audit
        capture_contact_audit(env, directory / "contact_runtime_audit.json")
        if args.trace_seconds:
            from feedback_run_tools import PhysicsTrace
            trace = PhysicsTrace(env, args.trace_seconds)
        original_checkpoint_hash = (hashlib.sha256(args.resume.read_bytes()).hexdigest()
                                    if args.resume else None)
        init = {"recipe": "elf3_feedback_v1", "task": args.task,
                "from_scratch": args.resume is None, "checkpoint_loaded": str(args.resume) if args.resume else None,
                "checkpoint_loaded_sha256": original_checkpoint_hash,
                "optimizer_state_entries": len(runner.alg.optimizer.state),
                "initial_iteration": runner.current_learning_iteration,
                "target_iteration": args.iterations - 1, "num_envs": args.num_envs, "seed": args.seed,
                "save_interval": 500, "training_pid": os.getpid(), "automatic_training_restart": False,
                "experiment_contract": runner.contract(), "configuration_sha256": canonical_sha256(cfg.to_dict()),
                "source_sha256": source_fingerprints(), "source_integrity": integrity,
                "feedback_model_status": "20 Hz observation-model hypothesis; NOT identified servo/D-term dynamics",
                "extra_deployment_filter": False}
        (directory / "initialization.json").write_text(json.dumps(init, indent=2) + "\n")
        assert_finite_weights(runner)
        if args.resume is None:
            runner.save(str(directory / "initial_random_model.pt"), {"initialization_only": True})
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(json.dumps({
                "project": args.swanlab_project, "name": f"{args.task}_{stamp}_{args.run_name}",
                "config": init | {"learning_rate": agent.algorithm.learning_rate,
                                  "min_learning_rate": agent.algorithm.min_learning_rate,
                                  "max_learning_rate": agent.algorithm.max_learning_rate}}, indent=2) + "\n")
            with (directory / "swanlab_supervisor.log").open("a") as output:
                subprocess.Popen([sys.executable, str(ROOT / "scripts/pose_scratch_swanlab.py"),
                                  "--training-pid", str(os.getpid()), "--log-dir", str(directory)],
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if args.resume and hashlib.sha256(args.resume.read_bytes()).hexdigest() != original_checkpoint_hash:
            raise RuntimeError("Source checkpoint changed")
        success = True
    except BaseException:
        # Kit shutdown may terminate the interpreter before an outer traceback
        # is printed. Persist the original error before closing the application.
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if directory is not None:
            (directory / "failure_traceback.txt").write_text(detail)
        raise
    finally:
        trace_error = None
        try:
            if directory is not None:
                if trace is not None:
                    try:
                        trace.finish(directory, JOINT_NAMES)
                    except Exception as error:
                        trace_error = error
                        success = False
                (directory / "training_exit.json").write_text(json.dumps({
                    "success": success, "last_iteration": runner.current_learning_iteration if runner else None,
                    "trace_error": str(trace_error) if trace_error else None,
                    "auto_restart": False}, indent=2) + "\n")
        finally:
            try:
                if runner is not None and runner.writer is not None:
                    runner.writer.close()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    app.close()
        if trace_error is not None:
            raise RuntimeError("Physics trace finalization failed") from trace_error


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit stop signal {signum}; training will not auto-restart")
    signal.signal(signal.SIGTERM, stop)
    main()
