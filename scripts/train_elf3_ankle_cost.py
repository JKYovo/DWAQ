"""Portable ELF3 still-v2 ankle-cost training, without private logs or accounts.

Defaults are a new 64-environment / 20-update technical check. For the full
recipe use --num-envs 16384 --iterations 100001. ``--iterations`` always means
the total indexed update count: N ends at model_(N-1), also when continuing.

--resume restores a completed ankle-cost checkpoint and Adam with the complete
stage contract. --parent-checkpoint explicitly promotes a compatible original
still-v2 checkpoint, requiring its SHA256. Neither mode restores simulator,
terrain-curriculum, RNG, observation-history or feedback-filter state. Every
invocation writes a NEW log/checkpoint directory, never the source run.

--verify-config-only needs Isaac Sim startup but creates no environment and
performs no learning. Importing this module or using --help starts no simulator.
Only load trusted checkpoints: torch checkpoints are not an untrusted format.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_still_v2_ankle_cost_v1"
RUNNER_CLASS = "Elf3DwaqAnkleCostRunner"
SEED = 42
SAVE_INTERVAL = 500
SOURCE_DIR = ROOT / "TienKung-Lab/legged_lab/envs/elf3"
BOOKKEEPING_KEYS = ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint")


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_helpers():
    return load_file(ROOT / "scripts/train_elf3_still_v2.py", "public_ankle_cost_still_helpers")


def contract_helpers():
    return load_file(SOURCE_DIR / "feedback_contract.py", "public_ankle_cost_contract_helpers")


def stage_contract_helpers():
    return load_file(SOURCE_DIR / "ankle_cost_contract.py", "public_ankle_cost_stage_identity")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Total update count N, not additional updates; final checkpoint is model_(N-1).pt")
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    parser.add_argument("--run-name", default="ankle_cost_check")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--swanlab-project", default=None,
                        help="Optional project in your own authenticated account; creates a NEW run")
    parser.add_argument("--use-cached-visual", action="store_true")
    parser.add_argument("--verify-config-only", action="store_true",
                        help="Starts Isaac app, validates configs/CPU checkpoint metadata, then exits before creating env")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path, help="Completed checkpoint of this exact ankle-cost stage")
    checkpoint.add_argument("--parent-checkpoint", type=Path,
                            help="Original still-v2 checkpoint; explicit promotion, not generic/AMP/G1 import")
    parser.add_argument("--parent-sha256", help="Required SHA256 with --parent-checkpoint")
    return parser


def validate_args(args):
    if args.seed != SEED:
        raise ValueError("This published recipe keeps seed 42")
    if args.num_envs < 1 or args.iterations < 1:
        raise ValueError("Positive environment and total update counts are required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
        raise ValueError("run-name must be one ASCII filename component, beginning with a letter/digit")
    if args.resume is not None and args.parent_checkpoint is not None:
        raise ValueError("Choose --resume OR --parent-checkpoint, never both")
    if args.parent_checkpoint is not None:
        if not args.parent_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", args.parent_sha256):
            raise ValueError("--parent-checkpoint requires an explicit 64-digit --parent-sha256")
        args.parent_sha256 = args.parent_sha256.lower()
    elif args.parent_sha256 is not None:
        raise ValueError("--parent-sha256 is only valid with --parent-checkpoint")
    if not args.device or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", args.device):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    if args.swanlab_project is not None:
        if not args.swanlab_project.strip() or args.swanlab_project.strip() != args.swanlab_project:
            raise ValueError("swanlab-project must be a nonempty project name without surrounding whitespace")
    return args


def update_budget(iterations, completed_iteration=None):
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations is a positive total update count")
    if completed_iteration is None:
        start = 0
    else:
        if isinstance(completed_iteration, bool) or not isinstance(completed_iteration, int) or completed_iteration < 0:
            raise ValueError("Completed checkpoint iteration must be a nonnegative integer")
        start = completed_iteration + 1
    remaining = iterations - start
    if remaining < 1:
        raise ValueError("--iterations must exceed the checkpoint iteration + 1; it is total, not additional updates")
    return {"start_iteration": start, "learning_updates": remaining, "target_iteration": iterations - 1}


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


def validate_checkpoint_header(saved, mode, iterations):
    """CPU-only fail-closed checks; live runner later checks the FULL contract."""
    if mode not in ("resume", "parent"):
        raise ValueError("Checkpoint mode must be resume or parent")
    if not isinstance(saved, dict) or (saved.get("infos") or {}).get("initialization_only"):
        raise ValueError("An initialization-only snapshot is not a completed training checkpoint")
    infos = saved.get("infos") or {}
    metadata = infos.get("elf3_contract")
    if not isinstance(metadata, dict):
        raise ValueError("Missing ELF3 checkpoint contract")
    contract = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
    expected_asset = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (metadata.get("asset_contract_sha256") != expected_asset
            or metadata.get("joint_names") != contract["joint_names"]
            or metadata.get("num_obs") != 100 or metadata.get("history_length") != 5):
        raise ValueError("Checkpoint asset/joint order/5x100 layout differs from the published ELF3 recipe")
    if (metadata.get("standing_experiment") or {}).get("version") != "elf3_dwaq_still_v1":
        raise ValueError("Checkpoint is not the original minimal still policy")
    if (metadata.get("still_v2_experiment") or {}).get("version") != "elf3_dwaq_still_v2":
        raise ValueError("Checkpoint is not the independent still-v2 pose task")
    feedback = metadata.get("feedback_experiment") or {}
    if (feedback.get("version") != "elf3_feedback_v1"
            or feedback.get("velocity_feedback", {}).get("mode") != "lpf20"
            or feedback.get("velocity_feedback", {}).get("cutoff_hz") != 20.0):
        raise ValueError("Keep the original 20 Hz velocity-observation model")
    if mode == "resume":
        stage_contract_helpers().validate_reward_stage_contract(metadata.get("reward_stage_experiment"))
        expected_runner = RUNNER_CLASS
    else:
        if "reward_stage_experiment" in metadata:
            raise ValueError("An ankle-cost model must use --resume, not --parent-checkpoint")
        expected_runner = "Elf3DwaqStillV2Runner"
    recipe = infos.get("training_recipe")
    if (not isinstance(recipe, dict) or recipe.get("runner_class_name") != expected_runner
            or recipe.get("seed") != SEED or recipe.get("save_interval") != SAVE_INTERVAL):
        raise ValueError("Wrong saved Runner, seed, save cadence or missing training recipe")
    if not isinstance(saved.get("model_state_dict"), dict) or not saved["model_state_dict"]:
        raise ValueError("Missing trained model state")
    optimizer = saved.get("optimizer_state_dict") or {}
    if not isinstance(optimizer.get("state"), dict) or not optimizer["state"]:
        raise ValueError("Nonempty trained Adam state is required; optimizer resets are not supported")
    groups = optimizer.get("param_groups") or []
    lrs = [group.get("lr") for group in groups]
    if (not lrs or any(isinstance(lr, bool) or not isinstance(lr, (int, float))
                       or not math.isfinite(lr) or lr <= 0 for lr in lrs)
            or len(set(lrs)) != 1):
        raise ValueError("Expected finite positive and equal saved Adam learning rates")
    require_finite_tree(saved["model_state_dict"], "model")
    require_finite_tree(optimizer, "Adam")
    budget = update_budget(iterations, saved.get("iter"))
    return {**budget, "completed_iteration": saved["iter"], "loaded_learning_rates": lrs}


def prepare_checkpoint(args):
    args = validate_args(args)
    path = args.resume or args.parent_checkpoint
    if path is None:
        return {"mode": "fresh", "saved": None, "path": None, "sha256": None,
                **update_budget(args.iterations)}
    import torch
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("An explicit checkpoint file is required")
    before = sha256(path)
    mode = "resume" if args.resume is not None else "parent"
    if mode == "parent" and before != args.parent_sha256:
        raise ValueError("Parent checkpoint SHA256 does not match the explicitly supplied digest")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    header = validate_checkpoint_header(saved, mode, args.iterations)
    if sha256(path) != before:
        raise ValueError("Source checkpoint changed while reading")
    return {"mode": mode, "saved": saved, "path": path, "sha256": before, **header}


def validate_saved_recipe(saved, current_recipe, mode):
    old, expected = deepcopy(saved["infos"]["training_recipe"]), deepcopy(current_recipe)
    if mode == "parent":
        expected["runner_class_name"] = "Elf3DwaqStillV2Runner"
    for value in (old, expected):
        for key in BOOKKEEPING_KEYS:
            value.pop(key, None)
    if contract_helpers().canonical_sha256(old) != contract_helpers().canonical_sha256(expected):
        raise ValueError("Saved policy/PPO/auxiliary/seed/rollout recipe differs; only launch bookkeeping may change")
    return {"full_training_recipe_matched": True, "ignored_bookkeeping": list(BOOKKEEPING_KEYS)}


def verify_restored_state(runner, saved):
    runner._assert_tree_equal(runner.alg.policy.state_dict(), saved["model_state_dict"], "model")
    runner._assert_tree_equal(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
    expected = [group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]]
    actual = [group["lr"] for group in runner.alg.optimizer.param_groups]
    if not expected or actual != expected or runner.alg.learning_rate != expected[0]:
        raise ValueError("Restored Adam/algorithm learning rate differs from saved state")
    if runner.current_learning_iteration != saved["iter"] + 1:
        raise ValueError("Continuation must start at completed iteration + 1")
    return {"model_and_Adam_exactly_restored": True, "learning_rates": expected,
            "next_iteration": runner.current_learning_iteration}


def choose_run_directory(args, stamp=None, pid=None):
    validate_args(args)
    stamp = stamp or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid() if pid is None else pid
    base = (ROOT / "logs" / TASK).resolve()
    directory = base / f"{stamp}_{args.run_name}_{pid}"
    if directory.parent != base or directory.exists():
        raise FileExistsError("A new, uniquely named public run directory is required")
    return directory


def source_fingerprints():
    helper = load_file(ROOT / "scripts/train_elf3_feedback.py", "public_ankle_cost_source_fingerprints")
    result = helper.source_fingerprints()
    extra = (*public_helpers().EXPERIMENT_SOURCE_PATHS, "scripts/train_elf3_ankle_cost.py",
             "TienKung-Lab/legged_lab/envs/elf3/ankle_cost.py",
             "TienKung-Lab/legged_lab/envs/elf3/ankle_cost_contract.py",
             "TienKung-Lab/legged_lab/envs/elf3/ankle_cost_rewards.py")
    result.update({relative: sha256(ROOT / relative) for relative in extra})
    return result


def launch_new_swanlab(directory):
    """Start only the existing bounded uploader, always with a new user run."""
    with (directory / "swanlab_supervisor.log").open("a") as output:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/pose_scratch_swanlab.py"),
             "--training-pid", str(os.getpid()), "--log-dir", str(directory)],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return process.pid


def run(args, prepared):
    helpers = public_helpers()
    app = env = runner = directory = None
    visual_context = ExitStack()
    success = entered_training = False
    detail = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless, device=args.device).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs.elf3.ankle_cost import (
            Elf3DwaqAnkleCostEnv, Elf3DwaqAnkleCostRunner, make_configs,
            validate_ankle_cost_configuration,
        )
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest

        visual = visual_context.enter_context(helpers.cached_visual_alias(args.use_cached_visual))
        assets, integrity = verify_retained_assets(), check_source_manifest(ROOT)
        cfg, agent = make_configs(args.num_envs, args.device, args.run_name)
        agent.max_iterations, agent.resume = args.iterations, False
        audit = validate_ankle_cost_configuration(cfg, agent.to_dict())
        if prepared["saved"] is not None:
            audit["checkpoint_recipe"] = validate_saved_recipe(prepared["saved"], agent.to_dict(), prepared["mode"])
        if args.verify_config_only:
            print(helpers.report_json({"task": TASK, "mode": prepared["mode"],
                "configuration_audit": audit, "environment": cfg.to_dict(), "agent": agent.to_dict(),
                "configuration_sha256": canonical_sha256(cfg.to_dict()),
                "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
                "source_integrity": integrity, "assets": assets,
                "starts_Isaac_app": True, "creates_environment": False, "learning_updates": 0,
                "checkpoint_full_live_contract_check": "Deferred until actual environment/runner creation; config-only checks CPU metadata and Agent recipe" if prepared["saved"] is not None else "No checkpoint selected"}, indent=2))
            return
        directory = choose_run_directory(args)
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = Elf3DwaqAnkleCostEnv(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Keep the original 5x100 observation / 29-action interface")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("Physical implicit PD must remain unchanged")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0.)
        runner = Elf3DwaqAnkleCostRunner(env, agent.to_dict(), log_dir=str(directory),
                                      device=args.device, checkpoint_dir=str(directory))
        if prepared["mode"] == "fresh":
            helpers.assert_fresh_runner(runner)
            restoration = {"from_scratch": True, "checkpoint_loaded": None, "optimizer_state_entries": 0}
        else:
            if sha256(prepared["path"]) != prepared["sha256"]:
                raise ValueError("Source checkpoint changed after CPU preflight")
            if prepared["mode"] == "resume":
                runner.load(str(prepared["path"]), load_optimizer=True)
            else:
                runner.initialize_from_parent(str(prepared["path"]), args.parent_sha256)
            restoration = verify_restored_state(runner, prepared["saved"])
        if runner.current_learning_iteration != prepared["start_iteration"]:
            raise ValueError("Runner start iteration differs from the requested total update budget")
        assert_finite_weights(runner)
        initialization = {
            "recipe": TASK, "task": TASK, "entrypoint": "scripts/train_elf3_ankle_cost.py",
            "mode": prepared["mode"], "from_scratch": prepared["mode"] == "fresh",
            "source_checkpoint": str(prepared["path"]) if prepared["path"] else None,
            "source_checkpoint_sha256": prepared["sha256"], "source_checkpoint_read_only": True,
            "initial_iteration": prepared["start_iteration"], "target_iteration": prepared["target_iteration"],
            "total_update_count": args.iterations, "learning_updates": prepared["learning_updates"],
            "num_envs": args.num_envs, "seed": SEED, "save_interval": SAVE_INTERVAL,
            "training_pid": os.getpid(), "restoration": restoration,
            "configuration_audit": audit, "experiment_contract": runner.contract(),
            "configuration_sha256": canonical_sha256(cfg.to_dict()),
            "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
            "source_sha256": source_fingerprints(), "source_integrity": integrity,
            "assets": assets, "git": helpers.git_provenance(), "display_only_visual_cache": visual,
            "environment_resume_note": "Physics, terrain curriculum, RNG, observation/action histories and feedback filter state are reinitialized, not recovered from checkpoint.",
            "same_recipe_from_scratch_is_not_the_same_training_history_as_parent_promotion": True,
            "feedback_model_status": "Retained 20Hz joint-speed observation model; physical PD unfiltered",
            "extra_deployment_filter": False, "new_swanlab_run": bool(args.swanlab_project),
            "private_run_or_account_required": False, "automatic_training_restart": False,
            "automatic_training_stop": False,
        }
        (directory / "initialization.json").write_text(helpers.report_json(initialization, indent=2) + "\n")
        (directory / "configuration_audit.json").write_text(helpers.report_json(audit, indent=2) + "\n")
        snapshot = "initial_random_model.pt" if prepared["mode"] == "fresh" else "initial_loaded_model.pt"
        runner.save(str(directory / snapshot), {"initialization_only": True, "origin": prepared["mode"]})
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(helpers.report_json({
                "project": args.swanlab_project, "name": directory.name,
                "config": initialization}, indent=2) + "\n")
            launch_new_swanlab(directory)
        print("[READY] " + helpers.report_json({"directory": str(directory), "task": TASK,
            "mode": prepared["mode"], "start_iteration": prepared["start_iteration"],
            "learning_updates": prepared["learning_updates"], "target_iteration": prepared["target_iteration"],
            "new_swanlab_run": bool(args.swanlab_project)}), flush=True)
        entered_training = True
        runner.learn(num_learning_iterations=prepared["learning_updates"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory / f"model_{prepared['target_iteration']}.pt").is_file():
            raise RuntimeError("Training returned without the requested final checkpoint")
        success = True
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        if directory is not None:
            (directory / "failure_traceback.txt").write_text(detail)
        raise
    finally:
        try:
            if prepared["path"] is not None and sha256(prepared["path"]) != prepared["sha256"]:
                success = False
                raise ValueError("Read-only source checkpoint changed during this invocation")
        finally:
            if directory is not None:
                (directory / "training_exit.json").write_text(helpers.report_json({
                    "success": success, "entered_training": entered_training,
                    "last_iteration": getattr(runner, "current_learning_iteration", None),
                    "target_iteration": prepared["target_iteration"], "error": detail,
                    "automatic_training_restart": False, "automatic_training_stop": False}, indent=2) + "\n")
            try:
                if runner is not None and runner.writer is not None:
                    runner.writer.close()
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


def main(argv=None):
    args = validate_args(build_parser().parse_args(argv))
    prepared = prepare_checkpoint(args)
    run(args, prepared)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal {signum}; no automatic training restart")
    signal.signal(signal.SIGTERM, stop)
    main()
