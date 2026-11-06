"""Portable current ELF3 force500 recipe: scratch or strict same-stage resume.

No local training logs/parent checkpoints/accounts are required. Safe defaults
are 64 envs /20 updates. Formal:16384 envs, seed42,100001 updates,save500.
Starting all final rewards from scratch is NOT the historical staged run.
--verify-config-only starts Kit but creates no environment and does no learning.
"""
from contextlib import ExitStack
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

import train_elf3_ankle_cost as shared


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_still_v2_force500_v1"
RUNNER_CLASS = "Elf3DwaqFootForce500Runner"
RUNTIME_FILES = (
    "control_smoothness.py", "control_smoothness_contract.py", "control_smoothness_rewards.py",
    "foot_force_peak.py", "foot_force_peak_contract.py", "foot_force_peak_rewards.py",
    "foot_force_500.py", "foot_force_500_contract.py", "height_force.py", "height_force_contract.py",
    "height_force_reference.py", "height_force_rewards.py", "height_force_strong.py",
    "height_force_strong_contract.py", "motion_quality.py", "motion_quality_contract.py",
    "motion_quality_rewards.py", "soft_step_rewards.py",
)


def build_parser():
    import argparse
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--num-envs", type=int, default=64)
    result.add_argument("--iterations", type=int, default=20,
                        help="Total indexed update count N; ends at model_(N-1), not additional updates")
    result.add_argument("--seed", type=int, choices=(42,), default=42)
    result.add_argument("--run-name", default="force500_scratch")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--headless", action="store_true")
    result.add_argument("--resume", type=Path, help="Trusted completed checkpoint of this exact500N recipe")
    result.add_argument("--swanlab-project", default=None, help="Optional NEW run in your authenticated account")
    result.add_argument("--verify-config-only", action="store_true")
    result.add_argument("--use-cached-visual", action="store_true")
    result.add_argument("--reference-run", type=Path, help="Optional read-only comparison to saved env/Agent params")
    result.set_defaults(parent_checkpoint=None, parent_sha256=None)
    return result


def prepare_checkpoint(args):
    shared.validate_args(args)
    if args.resume is None:
        return {"path": None, "saved": None, "sha256": None, "mode": "fresh",
                **shared.update_budget(args.iterations)}
    import torch
    path = args.resume.resolve(strict=True)
    digest = shared.sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    identity = shared.load_file(
        shared.SOURCE_DIR / "foot_force_500_contract.py", "published_force500_identity")
    metadata = (saved.get("infos") or {}).get("elf3_contract") or {}
    if metadata.get("foot_force_threshold_experiment") != identity.threshold_contract():
        raise ValueError("Only this exact500N reward stage/source version may resume")
    if saved["infos"]["training_recipe"]["runner_class_name"] != RUNNER_CLASS:
        raise ValueError("Wrong saved500N runner")
    projected = shared.deepcopy(saved)
    projected["infos"]["training_recipe"]["runner_class_name"] = shared.RUNNER_CLASS
    header = shared.validate_checkpoint_header(projected, "resume", args.iterations)
    if shared.sha256(path) != digest:
        raise ValueError("Read-only checkpoint changed during CPU preflight")
    return {"path": path, "saved": saved, "sha256": digest, "mode": "resume", **header}


def run(args, prepared):
    helpers = shared.public_helpers()
    app = env = runner = directory = None
    visuals = ExitStack()
    success = entered = False
    error = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless, device=args.device).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs.elf3.foot_force_500 import (
            Elf3DwaqFootForce500Env, Elf3DwaqFootForce500Runner, make_configs, validate_configuration,
        )
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest, compare_saved_run

        visual = visuals.enter_context(helpers.cached_visual_alias(args.use_cached_visual))
        cfg, agent = make_configs(args.num_envs, args.device, args.run_name)
        agent.max_iterations, agent.resume = args.iterations, False
        audit = validate_configuration(cfg, agent.to_dict())
        if args.reference_run is not None:
            audit["saved_configuration_reference"] = compare_saved_run(cfg, agent, args.reference_run)
        if prepared["saved"] is not None:
            audit["saved_Agent_recipe"] = shared.validate_saved_recipe(
                prepared["saved"], agent.to_dict(), "resume")
        report = {
            "task": TASK, "recipe": TASK, "entrypoint": "scripts/train_elf3_force500.py",
            "mode": prepared["mode"], "from_scratch": prepared["mode"] == "fresh",
            "source_checkpoint": str(prepared["path"]) if prepared["path"] else None,
            "source_checkpoint_sha256": prepared["sha256"],
            "initial_iteration": prepared["start_iteration"], "target_iteration": prepared["target_iteration"],
            "learning_updates": prepared["learning_updates"], "num_envs": args.num_envs,
            "seed": 42, "save_interval": 500, "configuration_audit": audit,
            "configuration_sha256": canonical_sha256(cfg.to_dict()),
            "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
            "source_integrity": check_source_manifest(ROOT), "assets": verify_retained_assets(),
            "display_only_visual_cache": visual, "new_swanlab_run": bool(args.swanlab_project),
            "private_logs_or_parent_checkpoint_required": False,
            "automatic_training_restart": False, "automatic_training_stop": False,
            "environment_resume_note": "Physics/curriculum/RNG/history/filter reset; resume restores model/Adam/LR/iteration.",
            "scratch_final_recipe_is_not_historical_staged_training": True,
        }
        if args.verify_config_only:
            print(helpers.report_json({**report, "environment": cfg.to_dict(), "agent": agent.to_dict(),
                  "creates_environment": False, "learning_updates": 0}, indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT / "logs" / TASK / f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
        env = Elf3DwaqFootForce500Env(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Keep5x100 ->29 deployment interface")
        if not all(a.is_implicit_model for a in env.robot.actuators.values()):
            raise ValueError("Keep implicit physical PD")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0.)
        runner = Elf3DwaqFootForce500Runner(env, agent.to_dict(), str(directory), args.device, str(directory))
        if prepared["path"] is None:
            helpers.assert_fresh_runner(runner)
            restoration = {"from_scratch": True, "checkpoint_loaded": None, "optimizer_state_entries": 0}
        else:
            runner.load(str(prepared["path"]), load_optimizer=True)
            restoration = shared.verify_restored_state(runner, prepared["saved"])
        if runner.current_learning_iteration != prepared["start_iteration"]:
            raise ValueError("Runner start differs from requested update budget")
        assert_finite_weights(runner)
        fingerprints = shared.source_fingerprints()
        for name in RUNTIME_FILES:
            relative = "TienKung-Lab/legged_lab/envs/elf3/" + name
            fingerprints[relative] = shared.sha256(ROOT / relative)
        fingerprints["scripts/train_elf3_force500.py"] = shared.sha256(Path(__file__))
        report.update(training_pid=os.getpid(), restoration=restoration, experiment_contract=runner.contract(),
                      source_sha256=fingerprints, git=helpers.git_provenance())
        (directory / "initialization.json").write_text(helpers.report_json(report, indent=2) + "\n")
        runner.save(str(directory / ("initial_random_model.pt" if prepared["path"] is None else "initial_loaded_model.pt")),
                    {"initialization_only": True, "origin": prepared["mode"]})
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(helpers.report_json({
                "project": args.swanlab_project, "name": directory.name, "config": report}, indent=2) + "\n")
            shared.launch_new_swanlab(directory)
        print("[READY] " + helpers.report_json({"directory": str(directory), "task": TASK,
              "mode": prepared["mode"], "start_iteration": prepared["start_iteration"],
              "target_iteration": prepared["target_iteration"], "num_envs": args.num_envs}), flush=True)
        entered = True
        runner.learn(prepared["learning_updates"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory / f"model_{prepared['target_iteration']}.pt").is_file():
            raise RuntimeError("Missing requested final checkpoint")
        success = True
    except BaseException:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if directory is not None:
            (directory / "failure_traceback.txt").write_text(error)
        raise
    finally:
        try:
            if prepared["path"] is not None and shared.sha256(prepared["path"]) != prepared["sha256"]:
                raise ValueError("Read-only resume checkpoint changed")
            if directory is not None:
                (directory / "training_exit.json").write_text(helpers.report_json({
                    "success": success, "entered_training": entered, "error": error,
                    "last_iteration": getattr(runner, "current_learning_iteration", None),
                    "automatic_training_restart": False, "automatic_training_stop": False}, indent=2) + "\n")
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                try:
                    visuals.close()
                finally:
                    if app is not None:
                        app.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        shared.validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    run(args, prepare_checkpoint(args))


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal {signum}; no automatic restart")
    signal.signal(signal.SIGTERM, stop)
    main()
