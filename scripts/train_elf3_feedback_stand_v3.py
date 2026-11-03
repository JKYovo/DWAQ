"""Fresh feedback-v1 with zero-only pose/motion terms and periodic-drive removal.

No checkpoint/resume CLI, automatic restart, or automatic training-stop monitor
is provided. 100001 updates produce the inclusive model_100000.pt checkpoint.
"""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_feedback_stand_v3"
PARENT_TASK = "elf3_dwaq_feedback_v1"
RUNNER_CLASS = "Elf3DwaqFeedbackStandV3Runner"
PARENT_REWARD_PATHS = {
    "reward.zero_default_pose_l1", "reward.zero_joint_motion",
    "reward.gait_phase_contact.func", "reward.gait_phase_contact.params.command_epsilon",
    "reward.feet_swing_height.func", "reward.feet_swing_height.params.command_epsilon",
}


EXPERIMENT_SOURCE_PATHS = (
    "scripts/train_elf3_feedback_stand_v3.py",
    "scripts/train_elf3_cached_visuals.py",
    "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_contract.py",
    "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_rewards.py",
    "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py",
    "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_env.py",
    "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_runner.py",
    "TienKung-Lab/legged_lab/envs/elf3/elf3_feedback_stand_v3_config.py",
    "TienKung-Lab/legged_lab/envs/elf3/standing_rewards.py",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Update count from zero; 100001 ends at model_100000.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="feedback_stand_v3_check")
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--verify-config-only", action="store_true")
    parser.add_argument("--use-cached-visual", action="store_true",
                        help="Use the already-inspected local display-arrow cache, process-locally only")
    return parser


def differences(before, after, prefix=""):
    """Pure configuration comparison; importing this module never starts Isaac."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                result.append({"path": path, "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], path))
        return result
    return [] if before == after else [{"path": prefix, "before": before, "after": after}]


def report_default(value):
    """Explicit report-only encoding; never changes config/contract hashing."""
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if callable(value):
        module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
        if isinstance(module, str) and isinstance(name, str):
            return f"{module}.{name}"
        raise TypeError(f"Unsupported callable in report: {type(value).__name__}")
    convert = getattr(value, "to_dict", None)
    if callable(convert):
        return convert()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def report_json(value, **options):
    return json.dumps(value, default=report_default, **options)


def validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent):
    """The CLI cannot silently modify the registered experiment recipe."""
    allowed_env = {"scene.num_envs", "scene.seed", "device"}
    allowed_agent = {"seed", "device", "max_iterations", "save_interval", "run_name", "logger", "resume"}
    env_delta = differences(registered_cfg.to_dict(), cfg.to_dict())
    agent_delta = differences(registered_agent.to_dict(), agent.to_dict())
    for item in env_delta:
        if item["path"] not in allowed_env:
            raise ValueError(f"Unexpected runtime environment override: {item}")
    for item in agent_delta:
        if item["path"] not in allowed_agent:
            raise ValueError(f"Unexpected runtime agent override: {item}")
    if agent.runner_class_name != RUNNER_CLASS or agent.experiment_name != TASK:
        raise ValueError("The complete registered feedback-stand-v3 task/runner is required")
    if agent.policy.to_dict() != registered_agent.policy.to_dict():
        raise ValueError("Policy parameters must match the registered experiment")
    if agent.algorithm.to_dict() != registered_agent.algorithm.to_dict():
        raise ValueError("PPO parameters must match the registered experiment")
    if agent.upper_body_mirror_loss_coeff != .1 or agent.upper_body_pose_loss_coeff != .02:
        raise ValueError("The recommended task's upper-body auxiliary coefficients changed")
    if agent.resume or agent.save_interval != 500 or agent.logger != "tensorboard":
        raise ValueError("Fresh initialization, TensorBoard, and a 500-update save interval are required")
    feedback = cfg.velocity_feedback
    if feedback["mode"] != "lpf20" or feedback["cutoff_hz"] != 20.0:
        raise ValueError("The existing 20 Hz joint-speed observation model must be retained")
    return {"registered_task": TASK, "parent_task": PARENT_TASK,
            "runtime_environment_changes": env_delta, "runtime_agent_changes": agent_delta,
            "registered_recipe_preserved": True, "from_scratch": True,
            "checkpoint_load_supported": False}


def validate_parent_recipe(cfg, agent, parent_cfg, parent_agent):
    """Only the explicitly requested integration may differ from feedback-v1."""
    allowed_env = PARENT_REWARD_PATHS | {"strict_standing_v3", "scene.num_envs", "scene.seed", "device"}
    allowed_agent = {"runner_class_name", "experiment_name", "wandb_project", "run_name", "logger",
                     "resume", "max_iterations", "save_interval", "seed", "device"}
    env_delta = differences(parent_cfg.to_dict(), cfg.to_dict())
    agent_delta = differences(parent_agent.to_dict(), agent.to_dict())
    for item in env_delta:
        if item["path"] not in allowed_env:
            raise ValueError(f"Unexpected change from recommended feedback-v1 environment: {item}")
    for item in agent_delta:
        if item["path"] not in allowed_agent:
            raise ValueError(f"Unexpected change from recommended feedback-v1 agent: {item}")
    if agent.policy.to_dict() != parent_agent.policy.to_dict():
        raise ValueError("Policy must be exactly the recommended feedback-v1 policy")
    if agent.algorithm.to_dict() != parent_agent.algorithm.to_dict():
        raise ValueError("PPO must be exactly the recommended feedback-v1 algorithm")
    return {"parent_task": PARENT_TASK, "environment_changes": env_delta, "agent_changes": agent_delta,
            "allowed_reward_paths": sorted(PARENT_REWARD_PATHS),
            "asset_PD_commands_terrain_randomization_noise_policy_and_algorithm_unchanged": True}


def assert_fresh_runner(runner):
    if runner.current_learning_iteration != 0:
        raise ValueError("Fresh training must begin at iteration zero")
    if runner.alg.optimizer.state:
        raise ValueError("Fresh training must have an empty optimizer state")


def git_provenance():
    """Record tracked and untracked state without changing the repository."""
    def git(*arguments):
        completed = subprocess.run(["git", "-C", str(ROOT), *arguments],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout
    try:
        return {"head": git("rev-parse", "HEAD").decode().strip(),
                "status_porcelain": git("status", "--porcelain=v1").decode(errors="replace"),
                "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD", "--")).hexdigest()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error),
                "note": "Content fingerprints remain authoritative in a non-git source export."}


@contextmanager
def cached_visual_alias(enabled):
    """Optional existing headless-startup workaround; never alters robot assets."""
    if not enabled:
        yield None
        return
    from train_elf3_cached_visuals import ARROW, ARROW_SHA256, URL_PATH
    import omni.client

    if not ARROW.is_file() or hashlib.sha256(ARROW.read_bytes()).hexdigest() != ARROW_SHA256:
        raise ValueError("The inspected display-only arrow cache is unavailable or changed")
    if not omni.client.initialize():
        raise RuntimeError("Omni Client initialization failed")
    aliases = [f"{scheme}://{URL_PATH}" for scheme in ("http", "https")]
    try:
        for url in aliases:
            omni.client.set_alias(url, str(ARROW))
            result, local_path = omni.client.get_local_file(url)
            if result != omni.client.Result.OK or Path(local_path).resolve() != ARROW.resolve():
                raise RuntimeError(f"Display-only cache alias did not resolve: {url}")
        metadata = {"file": str(ARROW), "sha256": ARROW_SHA256, "aliases": aliases,
                    "global_config_changed": False, "training_config_changed": False}
        print("[INFO] Process-local visual cache: " + report_json(metadata), flush=True)
        yield metadata
    finally:
        for url in aliases:
            omni.client.set_alias(url, None)
        omni.client.shutdown()


def run(args):
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    visual_context = ExitStack()
    env = runner = directory = None
    success = False
    try:
        # omni.client is supplied by the launched Kit application, not the
        # Python environment. Install this optional alias only after startup.
        visual_cache = visual_context.enter_context(cached_visual_alias(args.use_cached_visual))
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs import Elf3DwaqFeedbackStandV3Runner
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from legged_lab.envs.elf3.feedback_stand_v3_runner import validate_stand_configuration
        from legged_lab.utils import task_registry
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest
        from train_elf3_feedback import source_fingerprints

        integrity = check_source_manifest(ROOT)
        assets = verify_retained_assets()
        registered_cfg, registered_agent = task_registry.get_cfgs(TASK)
        cfg, agent = registered_cfg.copy(), registered_agent.copy()
        cfg.scene.num_envs, cfg.scene.seed, cfg.device = args.num_envs, args.seed, args.device
        agent.seed, agent.device = args.seed, args.device
        agent.max_iterations, agent.save_interval = args.iterations, 500
        agent.run_name, agent.logger, agent.resume = args.run_name, "tensorboard", False
        runtime_audit = validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent)
        parent_cfg, parent_agent = task_registry.get_cfgs(PARENT_TASK)
        parent_audit = validate_parent_recipe(cfg, agent, parent_cfg, parent_agent)
        exact_recipe_audit = validate_stand_configuration(cfg, agent.to_dict())
        if args.verify_config_only:
            print(report_json({"runtime_configuration": runtime_audit,
                              "parent_task_comparison": parent_audit,
                              "exact_standing_recipe": exact_recipe_audit,
                              "registered_environment": cfg.to_dict(),
                              "registered_agent": agent.to_dict(),
                              "source_integrity": integrity, "assets": assets},
                             indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT / "logs" / TASK / f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env = task_registry.get_task_class(TASK)(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("The 100 x 5 observation / 29 action interface changed")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("The existing implicit physical PD implementation changed")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqFeedbackStandV3Runner(env, agent.to_dict(),
                                             log_dir=str(directory), device=agent.device)
        runner.configuration_audit = exact_recipe_audit
        assert_fresh_runner(runner)
        assert_finite_weights(runner)
        fingerprints = source_fingerprints()
        for relative in EXPERIMENT_SOURCE_PATHS:
            fingerprints[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        audit = {"runtime": runtime_audit, "parent_task_comparison": parent_audit,
                 "runner_configuration_audit": runner.configuration_audit}
        initialization = {
            "recipe": TASK, "task": TASK, "parent_task": PARENT_TASK,
            "from_scratch": True, "checkpoint_loaded": None, "optimizer_state_entries": 0,
            "initial_iteration": 0, "target_iteration": args.iterations - 1,
            "learning_updates": args.iterations, "num_envs": args.num_envs, "seed": args.seed,
            "save_interval": 500, "training_pid": os.getpid(), "automatic_training_restart": False,
            "automatic_training_stop": False, "experiment_contract": runner.contract(),
            "configuration_audit": audit, "configuration_sha256": canonical_sha256(cfg.to_dict()),
            "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
            "source_sha256": fingerprints, "source_integrity": integrity, "assets": assets,
            "git": git_provenance(), "display_only_visual_cache": visual_cache,
            "initial_max_terrain_level": cfg.scene.max_init_terrain_level,
            "policy": agent.policy.to_dict(), "algorithm": agent.algorithm.to_dict(),
            "feedback_model_status": "Retained 20 Hz joint-speed observation model; physical PD unfiltered.",
            "extra_deployment_filter": False,
        }
        (directory / "initialization.json").write_text(report_json(initialization, indent=2) + "\n")
        (directory / "configuration_audit.json").write_text(report_json(audit, indent=2) + "\n")
        runner.save(str(directory / "initial_random_model.pt"), {"initialization_only": True})
        if args.swanlab_project:
            (directory / "swanlab_request.json").write_text(report_json({
                "project": args.swanlab_project, "name": f"{TASK}_{stamp}_{args.run_name}",
                "config": initialization,
            }, indent=2) + "\n")
            with (directory / "swanlab_supervisor.log").open("a") as output:
                subprocess.Popen([sys.executable, str(ROOT / "scripts/pose_scratch_swanlab.py"),
                                  "--training-pid", str(os.getpid()), "--log-dir", str(directory)],
                                 cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print("[READY] " + report_json({"directory": str(directory), "task": TASK,
                                        "from_scratch": True, "initial_iteration": 0,
                                        "updates": args.iterations, "target": args.iterations - 1}), flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory / f"model_{args.iterations - 1}.pt").is_file():
            raise RuntimeError("Training ended without the requested final checkpoint")
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
            (directory / "training_exit.json").write_text(report_json({
                "success": success, "last_iteration": runner.current_learning_iteration if runner else None,
                "target_iteration": args.iterations - 1, "auto_restart": False,
                "automatic_training_stop": False,
            }, indent=2) + "\n")
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
                    app.close()


def main():
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("Positive environment and iteration counts are required")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("--run-name must be one filename component, not '.' or '..'")
    run(args)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit stop signal {signum}; no automatic restart")
    signal.signal(signal.SIGTERM, stop)
    main()
