"""Continue the original still-v2 run with its isolated two-reward stage.

The original checkpoint, params and initialization stay read-only. Formal
TensorBoard/SwanLab continue the existing run; new checkpoints and stage params
are written below ``source_run/stages``. Smoke weights cannot initialize a
formal stage. There is no automatic training restart or stopping monitor.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "logs/elf3_dwaq_still_v2/2026-11-02_19-58-27_still_v2_leg_waist_scratch_16384_seed42_3791567"
TARGET_ITERATION = 100000
FORMAL_ENVS = 16384
SAVE_INTERVAL = 500
SEED = 42
STAGE_PREFIX = "ankle_cost_v1"
SOURCE_DIR = ROOT / "TienKung-Lab/legged_lab/envs/elf3"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_resume():
    # This module imports only standard-library modules until preflight/run.
    return load_file(ROOT / "scripts/resume_elf3_still_v2.py", "ankle_cost_original_resume")


def stage_contract_module():
    return load_file(SOURCE_DIR / "ankle_cost_contract.py", "ankle_cost_launcher_contract")


def canonical_sha256(value):
    helper = load_file(SOURCE_DIR / "feedback_contract.py", "ankle_cost_launcher_fingerprint")
    return helper.canonical_sha256(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Read-only completed checkpoint from the original still-v2 run.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage-dir", type=Path,
                        help="New directory below source-run/stages; smoke uses artifacts/ankle_cost_smoke_*.")
    parser.add_argument("--resume-stage", type=Path,
                        help="Resume this new reward stage's checkpoint, not a smoke or unrelated model.")
    parser.add_argument("--smoke", action="store_true", help="Separate 64-env, 20-update test; no SwanLab.")
    parser.add_argument("--num-envs", type=int, help="Smoke-only environment count override.")
    parser.add_argument("--iterations", type=int, help="Smoke-only update count override.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-cached-visual", action="store_true")
    parser.add_argument("--verify-config-only", action="store_true",
                        help="CPU provenance/contract/Adam checks; no Isaac app, files or training.")
    return parser


def resolve_args(args):
    if not args.smoke and (args.num_envs is not None or args.iterations is not None):
        raise ValueError("Formal continuation keeps 16384 envs and target 100000; overrides are smoke-only")
    args.num_envs = args.num_envs if args.num_envs is not None else (64 if args.smoke else FORMAL_ENVS)
    args.iterations = args.iterations if args.iterations is not None else (20 if args.smoke else None)
    if args.num_envs < 1 or (args.iterations is not None and args.iterations < 1):
        raise ValueError("Smoke environment count and update count must be positive")
    if args.smoke and args.resume_stage is not None:
        raise ValueError("Smoke always restores the original parent into a new isolated test stage")
    return args


def choose_stage_directory(args, source_run, iteration):
    if args.resume_stage is not None:
        checkpoint = args.resume_stage.resolve(strict=True)
        directory = args.stage_dir.resolve() if args.stage_dir else checkpoint.parent
        if directory != checkpoint.parent:
            raise ValueError("resume-stage must be a checkpoint directly in stage-dir")
        base = source_run / "stages"
        if not directory.is_relative_to(base) or directory == base:
            raise ValueError("Formal stage checkpoints must remain below the original run's stages directory")
        return directory
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        base = ROOT / "artifacts"
        default = base / f"ankle_cost_smoke_from{iteration}_{stamp}_{os.getpid()}"
    else:
        base = source_run / "stages"
        default = base / f"{STAGE_PREFIX}_from{iteration}_{stamp}_{os.getpid()}"
    directory = args.stage_dir.resolve() if args.stage_dir else default
    if directory == base or not directory.is_relative_to(base):
        raise ValueError("Keep stage output inside its isolated stages/artifacts directory")
    if args.smoke and not directory.name.startswith("ankle_cost_smoke_"):
        raise ValueError("Smoke directory names must begin ankle_cost_smoke_")
    if directory.exists():
        raise FileExistsError("New stages never overwrite existing directories; use --resume-stage for formal recovery")
    return directory


def validate_stage_checkpoint(saved, initialization, parent_report, args, contract_helper):
    """CPU checks before a new-stage resume is allowed to start Isaac."""
    if initialization.get("smoke") is not False or initialization.get("num_envs") != FORMAL_ENVS:
        raise ValueError("A smoke or non-16384-env checkpoint cannot initialize formal training")
    if (initialization.get("seed"), initialization.get("target_iteration"),
            initialization.get("save_interval"), initialization.get("device")) != (
                SEED, TARGET_ITERATION, SAVE_INTERVAL, args.device):
        raise ValueError("Stage seed, target, save cadence or device changed")
    if (initialization.get("source_checkpoint_sha256"), initialization.get("source_run")) != (
            parent_report["checkpoint_sha256"], parent_report["original_log_dir"]):
        raise ValueError("Stage provenance differs from the explicitly supplied original parent")
    if (saved.get("infos") or {}).get("initialization_only"):
        raise ValueError("An initialization snapshot is not a completed stage checkpoint")
    metadata = (saved.get("infos") or {}).get("elf3_contract")
    if not isinstance(metadata, dict):
        raise ValueError("Stage checkpoint is missing its experiment contract")
    contract_helper.validate_reward_stage_contract(metadata.get("reward_stage_experiment"))
    if metadata != initialization.get("experiment_contract"):
        raise ValueError("Stage checkpoint differs from the recorded complete experiment contract")
    recipe = (saved.get("infos") or {}).get("training_recipe")
    if canonical_sha256(recipe) != initialization.get("agent_configuration_sha256"):
        raise ValueError("Stage checkpoint Agent recipe differs from stage initialization")
    iteration = saved.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ValueError("Invalid completed checkpoint iteration")
    if iteration < parent_report["completed_iteration"] or iteration >= TARGET_ITERATION:
        raise ValueError("Stage checkpoint iteration is outside the authorized continuation")
    if not saved.get("optimizer_state_dict", {}).get("state"):
        raise ValueError("Stage Adam state must be present and restored")
    return iteration


def preflight(args):
    args = resolve_args(args)
    original = original_resume()
    baseline_args = SimpleNamespace(log_dir=args.source_run, checkpoint=args.checkpoint,
                                    target_iteration=TARGET_ITERATION, device=args.device)
    baseline = original.preflight(baseline_args)
    directory, parent_init, parent_agent, parent_saved, protected, parent_report = baseline
    protected = dict(protected)
    helper = stage_contract_module()
    stage_identity = helper.reward_stage_contract()
    stage_dir = choose_stage_directory(args, directory, parent_report["completed_iteration"])
    stage_initialization, stage_saved = None, None
    selected_checkpoint = Path(parent_report["checkpoint"])
    completed = parent_report["completed_iteration"]
    if args.resume_stage is not None:
        import torch
        selected_checkpoint = args.resume_stage.resolve(strict=True)
        if not re.fullmatch(r"model_\d+\.pt", selected_checkpoint.name):
            raise ValueError("Expected a numbered completed stage checkpoint")
        init_path = stage_dir / "initialization.json"
        stage_initialization = json.loads(init_path.read_text())
        stage_saved = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        completed = validate_stage_checkpoint(stage_saved, stage_initialization, parent_report, args, helper)
        if selected_checkpoint.name != f"model_{completed}.pt":
            raise ValueError("Stage checkpoint filename and saved iteration disagree")
        original.require_finite_tree(stage_saved["model_state_dict"], "stage.model")
        original.require_finite_tree(stage_saved["optimizer_state_dict"], "stage.Adam")
        for path in (selected_checkpoint, init_path, stage_dir / "params/env.yaml", stage_dir / "params/agent.yaml"):
            protected[str(path)] = original.sha256(path)
    updates = args.iterations if args.smoke else TARGET_ITERATION - completed
    if updates < 1:
        raise ValueError("No authorized updates remain")
    report = {
        "task": "elf3_dwaq_still_v2", "stage": STAGE_PREFIX, "smoke": args.smoke,
        "source_run": str(directory), "source_checkpoint": parent_report["checkpoint"],
        "source_checkpoint_sha256": parent_report["checkpoint_sha256"],
        "stage_dir": str(stage_dir), "checkpoint_directory": str(stage_dir),
        "selected_checkpoint": str(selected_checkpoint), "selected_checkpoint_sha256": original.sha256(selected_checkpoint),
        "resuming_stage": args.resume_stage is not None,
        "completed_iteration": completed, "next_iteration": completed + 1,
        "updates_requested": updates, "expected_last_iteration": completed + updates,
        "target_iteration": TARGET_ITERATION, "num_envs": args.num_envs,
        "seed": SEED, "save_interval": SAVE_INTERVAL, "device": args.device,
        "reward_stage_experiment": stage_identity,
        "from_scratch": False, "restore_model_Adam_and_learning_rate": True,
        "swanlab_run_path": parent_report["swanlab_run_path"],
        "swanlab_enabled": not args.smoke, "new_swanlab_run": False,
        "cloud_ack_iteration_before_resume": parent_report["cloud_ack_iteration_before_resume"],
        "elapsed_time_log_offset": parent_report["elapsed_time_log_offset"],
        "cloud_recomputed_prefix_policy": parent_report["cloud_recomputed_prefix_policy"],
        "environment_resume_note": parent_report["environment_resume_note"],
        "original_source_audit": parent_report["source_audit"],
        "automatic_training_restart": False, "automatic_training_stop": False,
        "smoke_weights_allowed_for_formal_training": False,
    }
    return {"original": original, "baseline": baseline, "stage_dir": stage_dir,
            "stage_initialization": stage_initialization, "stage_saved": stage_saved,
            "protected": protected, "report": report}


def run(args, prepared):
    """Launch only after CPU preflight; this function is never called by tests."""
    original, report = prepared["original"], prepared["report"]
    directory = Path(report["source_run"])
    stage_dir = prepared["stage_dir"]
    protected = prepared["protected"]
    app = env = runner = lock = None
    success = entered_training = False
    detail = None
    visual_context = ExitStack()
    if args.smoke:
        stage_dir.mkdir(parents=True, exist_ok=False)
        lock_path = stage_dir / "smoke.lock"
    else:
        lock_path = directory / "still_v2_resume.lock"
    journal = stage_dir / f"resume_{report['completed_iteration']}_{os.getpid()}.json"
    try:
        lock = lock_path.open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The original still-v2 run is still locked by another training process") from error
        if not args.smoke and not args.resume_stage:
            stage_dir.mkdir(parents=True, exist_ok=False)
        # Recheck protected bytes after taking the formal run's shared lock.
        for path, digest in protected.items():
            if original.sha256(path) != digest:
                raise ValueError(f"Protected original/stage file changed after preflight: {path}")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless, device=args.device).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs.elf3.ankle_cost import (
            Elf3DwaqAnkleCostEnv, Elf3DwaqAnkleCostRunner, make_configs,
            validate_ankle_cost_configuration,
        )
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest
        from train_elf3_still_v2 import cached_visual_alias

        visual = visual_context.enter_context(cached_visual_alias(args.use_cached_visual))
        assets, integrity = verify_retained_assets(), check_source_manifest(ROOT)
        cfg, agent = make_configs(num_envs=args.num_envs, device=args.device,
                                  run_name="still_v2_ankle_cost_v1")
        audit = validate_ankle_cost_configuration(cfg, agent.to_dict())
        cfg_sha, agent_sha = canonical_sha256(cfg.to_dict()), canonical_sha256(agent.to_dict())
        if args.resume_stage:
            initialization = prepared["stage_initialization"]
            if (cfg_sha, agent_sha) != (initialization.get("configuration_sha256"),
                                        initialization.get("agent_configuration_sha256")):
                raise ValueError("The restored stage configuration differs from its immutable initialization")
        else:
            dump_yaml(str(stage_dir / "params/env.yaml"), cfg)
            dump_yaml(str(stage_dir / "params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = Elf3DwaqAnkleCostEnv(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Keep the original 5x100 observation / 29 action interface")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("Keep the original physical implicit PD")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqAnkleCostRunner(env, agent.to_dict(),
                    log_dir=str(stage_dir if args.smoke else directory), device=args.device,
                    checkpoint_dir=str(stage_dir))
        if args.resume_stage:
            runner.load(report["selected_checkpoint"], load_optimizer=True)
            saved = prepared["stage_saved"]
            restoration = {"resumed_existing_stage": True}
        else:
            restoration = runner.initialize_from_parent(report["source_checkpoint"], report["source_checkpoint_sha256"])
            saved = prepared["baseline"][3]
        if runner.current_learning_iteration != report["next_iteration"]:
            raise ValueError("Continuation must start after the checkpoint's last completed update")
        original.assert_restored_equal(runner.alg.policy.state_dict(), saved["model_state_dict"], "model")
        original.assert_restored_equal(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
        saved_lrs = [group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]]
        if (not saved_lrs or runner.alg.learning_rate != saved_lrs[0]
                or [group["lr"] for group in runner.alg.optimizer.param_groups] != saved_lrs):
            raise ValueError("Restored learning rate differs from saved Adam")
        assert_finite_weights(runner)
        runner.tot_time = 0. if args.smoke else float(report["elapsed_time_log_offset"])
        runner.tot_timesteps = runner.current_learning_iteration * agent.num_steps_per_env * env.num_envs
        report.update({"training_pid": os.getpid(), "restoration": restoration,
            "restored_weights_and_Adam_exact": True, "loaded_learning_rates": saved_lrs,
            "configuration_sha256": cfg_sha, "agent_configuration_sha256": agent_sha,
            "experiment_contract": runner.contract(), "configuration_audit": audit,
            "assets": assets, "source_integrity": integrity, "display_only_visual_cache": visual,
            "launcher_sha256": original.sha256(Path(__file__).resolve()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "original_files_readonly_sha256": protected})
        if not args.resume_stage:
            original.save_json(stage_dir / "initialization.json", report)
        original.save_json(journal, report)
        # One persistent marker makes the objective change visible on the same
        # TensorBoard/SwanLab timeline; it changes no learning calculation.
        runner._init_logger()
        runner.writer.add_scalar("AnkleCost/reward_stage_active", 1., report["next_iteration"])
        runner.writer.flush()
        print("[READY] " + json.dumps(report, default=str), flush=True)
        if not args.smoke:
            report["swanlab_supervisor_pid"] = original.launch_existing_swanlab(directory)
            original.save_json(journal, report)
        entered_training = True
        runner.learn(num_learning_iterations=report["updates_requested"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        expected = stage_dir / f"model_{report['expected_last_iteration']}.pt"
        if not expected.is_file():
            raise RuntimeError(f"Training returned without its expected stage checkpoint: {expected}")
        success = True
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if stage_dir.exists():
            (stage_dir / f"failure_{os.getpid()}.txt").write_text(detail)
        raise
    finally:
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
            for path, digest in protected.items():
                if original.sha256(path) != digest:
                    success = False
                    raise ValueError(f"Protected original/stage file modified: {path}")
        finally:
            try:
                if stage_dir.exists():
                    exit_record = {"success": success, "entered_training": entered_training,
                        "last_iteration": getattr(runner, "current_learning_iteration", None),
                        "target_iteration": TARGET_ITERATION, "stage_dir": str(stage_dir),
                        "smoke": args.smoke, "resume_journal": journal.name,
                        "automatic_training_restart": False, "automatic_training_stop": False,
                        "error": detail}
                    original.save_json(stage_dir / "training_exit.json", exit_record)
                    if not args.smoke and entered_training:
                        # Existing uploader reads this only after the training
                        # PID exits. Keep that lifecycle file in its old place.
                        original.save_json(directory / "training_exit.json", exit_record)
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    try:
                        visual_context.close()
                    finally:
                        if app is not None:
                            app.close()
                        if lock is not None:
                            lock.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    prepared = preflight(args)
    if args.verify_config_only:
        print(json.dumps(prepared["report"], indent=2, ensure_ascii=False))
        return
    run(args, prepared)


if __name__ == "__main__":
    def explicit_stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal {signum}; no automatic training restart")
    signal.signal(signal.SIGTERM, explicit_stop)
    main()
