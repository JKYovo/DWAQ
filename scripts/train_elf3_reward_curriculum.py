"""Still-v2 reward curriculum: scratch, explicit original-parent promotion, or strict resume.

Safe defaults:64 envs /20 updates, no cloud/active-run changes. Full scratch:
--num-envs16384 --iterations50001 --ramp-start20000 --ramp-end30000.
--iterations is TOTAL indexed update count N: final model_(N-1), not additional.
Only trusted checkpoints. Original parents require explicit SHA256 and batch
size. A fresh log/SwanLab run is always used, never a source log directory.
"""
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

import train_elf3_ankle_cost as shared


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_still_v2_curriculum_v1"
RUNNER_CLASS = "Elf3DwaqRewardCurriculumRunner"
RUNTIME_FILES = (
    "reward_curriculum.py", "reward_curriculum_contract.py",
    "control_smoothness.py", "control_smoothness_contract.py", "control_smoothness_rewards.py",
    "foot_force_peak.py", "foot_force_peak_contract.py", "foot_force_peak_rewards.py",
    "foot_force_500.py", "foot_force_500_contract.py", "height_force.py", "height_force_contract.py",
    "height_force_reference.py", "height_force_rewards.py", "height_force_strong.py",
    "height_force_strong_contract.py", "motion_quality.py", "motion_quality_contract.py",
    "motion_quality_rewards.py", "soft_step_rewards.py",
)


def schedule_helpers():
    return shared.load_file(shared.SOURCE_DIR/"reward_curriculum_contract.py", "curriculum_launch_identity")


def build_parser():
    parser = shared.build_parser()
    parser.description = __doc__
    parser.set_defaults(run_name="reward_curriculum_check")
    parser.add_argument("--ramp-start", type=int, default=20000)
    parser.add_argument("--ramp-end", type=int, default=30000)
    parser.add_argument("--parent-num-envs", type=int, default=16384,
                        help="Actual original-parent training batch; used by full checkpoint contract validation")
    parser.add_argument("--activate", action="store_true",
                        help="Update local active-run pointer, only if the previous training PID is stopped")
    parser.add_argument("--stdout-log", type=Path)
    return parser


def validate_args(args):
    shared.validate_args(args)
    schedule_helpers().schedule_settings(args.ramp_start, args.ramp_end)
    if args.parent_num_envs < 1:
        raise ValueError("Positive parent-num-envs required")
    return args


def validate_saved_recipe(saved, recipe, mode):
    old, current = shared.deepcopy(saved["infos"]["training_recipe"]), shared.deepcopy(recipe)
    if mode == "parent":
        current.pop("reward_curriculum", None)
        current.update(runner_class_name="Elf3DwaqStillV2Runner",
                       experiment_name="elf3_dwaq_still_v2", wandb_project="elf3_dwaq_still_v2")
    for item in (old, current):
        for key in shared.BOOKKEEPING_KEYS:
            item.pop(key, None)
    if shared.contract_helpers().canonical_sha256(old) != shared.contract_helpers().canonical_sha256(current):
        raise ValueError("Saved full PPO/Actor/auxiliary recipe or curriculum differs")
    return {"full_training_recipe_matched": True, "mode": mode}


def prepare_checkpoint(args):
    validate_args(args)
    if args.resume is None:
        return shared.prepare_checkpoint(args)
    import torch
    path = args.resume.resolve(strict=True)
    digest = shared.sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = (saved.get("infos") or {}).get("elf3_contract") or {}
    expected = schedule_helpers().curriculum_contract(schedule_helpers().schedule_settings(args.ramp_start, args.ramp_end))
    if metadata.get("reward_curriculum_experiment") != expected:
        raise ValueError("Resume requires identical curriculum schedule, source version and force sampling")
    if saved["infos"]["training_recipe"]["runner_class_name"] != RUNNER_CLASS:
        raise ValueError("Wrong curriculum runner")
    projected = shared.deepcopy(saved)
    projected["infos"]["training_recipe"]["runner_class_name"] = shared.RUNNER_CLASS
    header = shared.validate_checkpoint_header(projected, "resume", args.iterations)
    if shared.sha256(path) != digest:
        raise ValueError("Read-only checkpoint changed during preflight")
    return {"mode": "resume", "saved": saved, "path": path, "sha256": digest, **header}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(shared.public_helpers().report_json(value, indent=2)+"\n")
    temporary.replace(path)


def run(args, prepared):
    helpers = shared.public_helpers()
    app = env = runner = directory = lock = None
    visuals = ExitStack()
    success = entered = False
    error = None
    previous = None
    try:
        if args.activate and not args.verify_config_only:
            import psutil
            lock = (ROOT/"logs/elf3_reward_curriculum.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pointer = ROOT/"logs/active_elf3_still_v2.json"
            previous = json.loads(pointer.read_text()) if pointer.exists() else None
            if previous and psutil.pid_exists(previous["training_pid"]):
                raise ValueError("Previous confirmed training PID is still alive; never stop it implicitly")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless, device=args.device).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs.elf3.reward_curriculum import (
            Elf3DwaqRewardCurriculumEnv, Elf3DwaqRewardCurriculumRunner, make_configs, validate_configuration,
        )
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest

        visual = visuals.enter_context(helpers.cached_visual_alias(args.use_cached_visual))
        cfg, agent = make_configs(args.num_envs, args.device, args.run_name, args.ramp_start, args.ramp_end)
        agent.max_iterations, agent.resume = args.iterations, False
        audit = validate_configuration(cfg, agent.to_dict())
        if prepared["saved"] is not None:
            audit["saved_recipe"] = validate_saved_recipe(prepared["saved"], agent.to_dict(), prepared["mode"])
        identity = schedule_helpers()
        settings = identity.schedule_settings(args.ramp_start, args.ramp_end)
        report = {
            "task": TASK, "recipe": TASK, "entrypoint": "scripts/train_elf3_reward_curriculum.py",
            "mode": prepared["mode"], "from_scratch": prepared["mode"] == "fresh",
            "source_checkpoint": str(prepared["path"]) if prepared["path"] else None,
            "source_checkpoint_sha256": prepared["sha256"], "source_checkpoint_read_only": True,
            "parent_num_envs": args.parent_num_envs,
            "initial_iteration": prepared["start_iteration"], "target_iteration": prepared["target_iteration"],
            "learning_updates": prepared["learning_updates"], "num_envs": args.num_envs,
            "seed": 42, "save_interval": 500, "reward_curriculum": settings,
            "initial_alpha": identity.curriculum_alpha(prepared["start_iteration"], settings),
            "initial_effective_weights": identity.effective_weights(prepared["start_iteration"], settings),
            "configuration_audit": audit,
            "configuration_sha256": canonical_sha256(cfg.to_dict()),
            "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
            "params_env_weights_are_final_reference_not_initial_live_weights": True,
            "source_integrity": check_source_manifest(ROOT), "assets": verify_retained_assets(),
            "display_only_visual_cache": visual, "new_swanlab_run": bool(args.swanlab_project),
            "automatic_training_restart": False, "automatic_training_stop": False,
            "environment_resume_note": "Model/Adam/LR/iteration restored; physics/terrain/RNG/history/filter reset.",
            "historical_parent_used_last_substep_force_sampling": prepared["mode"] == "parent",
            "new_course_uses_peak500_sampling_in_all_stages": True,
        }
        if args.verify_config_only:
            print(helpers.report_json({**report, "environment": cfg.to_dict(), "agent": agent.to_dict(),
                  "creates_environment": False, "learning_updates": 0}, indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT/"logs"/TASK/f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory/"params/env.yaml"), cfg)
        dump_yaml(str(directory/"params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
        env = Elf3DwaqRewardCurriculumEnv(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("Keep original 5x100/29 interface")
        if not all(a.is_implicit_model for a in env.robot.actuators.values()):
            raise ValueError("Keep implicit physical PD")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0.)
        runner = Elf3DwaqRewardCurriculumRunner(env, agent.to_dict(), str(directory), args.device, str(directory))
        if prepared["mode"] == "fresh":
            helpers.assert_fresh_runner(runner)
            restoration = {"from_scratch": True, "optimizer_state_entries": 0}
        else:
            if prepared["mode"] == "parent":
                runner.initialize_from_still(prepared["path"], prepared["sha256"], args.parent_num_envs)
            else:
                runner.load(str(prepared["path"]), load_optimizer=True)
            restoration = shared.verify_restored_state(runner, prepared["saved"])
        if runner.current_learning_iteration != prepared["start_iteration"]:
            raise ValueError("Runner start differs from total update budget")
        assert_finite_weights(runner)
        fingerprints = shared.source_fingerprints()
        for name in RUNTIME_FILES:
            relative = "TienKung-Lab/legged_lab/envs/elf3/"+name
            fingerprints[relative] = shared.sha256(ROOT/relative)
        fingerprints["scripts/train_elf3_reward_curriculum.py"] = shared.sha256(Path(__file__))
        report.update(training_pid=os.getpid(), restoration=restoration, experiment_contract=runner.contract(),
                      source_sha256=fingerprints, git=helpers.git_provenance())
        write_json(directory/"initialization.json", report)
        write_json(directory/"configuration_audit.json", audit)
        runner.save(str(directory/("initial_random_model.pt" if prepared["path"] is None else "initial_loaded_model.pt")),
                    {"initialization_only": True, "origin": prepared["mode"]})
        if args.swanlab_project:
            write_json(directory/"swanlab_request.json", {"project": args.swanlab_project,
                       "name": directory.name, "config": report})
            report["swanlab_supervisor_pid"] = shared.launch_new_swanlab(directory)
        if args.activate:
            if previous:
                write_json(directory/"previous_active.json", previous)
            write_json(ROOT/"logs/active_elf3_still_v2.json", {**report,
                       "runner_class_name": RUNNER_CLASS, "runtime_runner_class": RUNNER_CLASS,
                       "checkpoint_dir": str(directory), "tensorboard_log_dir": str(directory),
                       "start_iteration": prepared["start_iteration"],
                       "stage_started_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                       "stdout_log": str(args.stdout_log.resolve()) if args.stdout_log else None})
        write_json(directory/"launch.json", report)
        print("[READY] "+helpers.report_json({"directory": str(directory), "task": TASK,
              "mode": prepared["mode"], "start_iteration": prepared["start_iteration"],
              "target_iteration": prepared["target_iteration"], "num_envs": args.num_envs,
              "schedule": settings, "initial_alpha": report["initial_alpha"], "restoration": restoration}), flush=True)
        runner.git_status_repos = []
        entered = True
        runner.learn(prepared["learning_updates"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory/f"model_{prepared['target_iteration']}.pt").is_file():
            raise RuntimeError("Missing requested final checkpoint")
        success = True
    except BaseException:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if directory is not None:
            (directory/"failure_traceback.txt").write_text(error)
        raise
    finally:
        try:
            if prepared["path"] is not None and shared.sha256(prepared["path"]) != prepared["sha256"]:
                raise ValueError("Read-only original checkpoint changed")
            if directory is not None:
                write_json(directory/"training_exit.json", {"success": success, "entered_training": entered,
                           "last_iteration": getattr(runner, "current_learning_iteration", None), "error": error,
                           "automatic_training_restart": False, "automatic_training_stop": False})
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
                    if lock is not None:
                        lock.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args, prepare_checkpoint(args))


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal {signum}; no automatic restart")
    signal.signal(signal.SIGTERM, stop)
    main()
