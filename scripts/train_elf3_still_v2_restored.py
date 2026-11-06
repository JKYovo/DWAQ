"""Original still_v2 scratch stage, restored without changing historical sources.

Safe defaults: 64 environments / 20 updates. Historical first-stage endpoint:
--num-envs 16384 --iterations 18501 --seed 42 (ends at model_18500.pt).
No checkpoint loading, active-run replacement, automatic stop or restart.
--verify-config-only starts Kit, but creates no environment and does no learning.
"""
from contextlib import ExitStack
from datetime import datetime
import hashlib
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

from train_elf3_ankle_cost import launch_new_swanlab
import train_elf3_still_v2 as original


ROOT = Path(__file__).resolve().parents[1]
TASK = original.TASK


def build_parser():
    parser = original.build_parser()
    parser.description = __doc__
    parser.set_defaults(run_name="still_v2_restored")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-run", type=Path,
                        help="Optional read-only exact env/Agent comparison with saved historical params")
    return parser


def validate_args(args):
    if args.num_envs <= 0 or args.iterations <= 0 or args.seed != 42:
        raise ValueError("Positive counts and the historical seed 42 are required")
    if (not args.run_name or Path(args.run_name).name != args.run_name
            or args.run_name in {".", ".."}):
        raise ValueError("run-name must be one filename component")
    if args.swanlab_project is not None and not args.swanlab_project.strip():
        raise ValueError("swanlab-project cannot be blank")
    return args


def configure(args, registry, parent_cfg, parent_agent):
    """Use the complete original registered recipe, not a recreated reward list."""
    registered_cfg, registered_agent = registry.get_cfgs(TASK)
    cfg, agent = registered_cfg.copy(), registered_agent.copy()
    cfg.scene.num_envs, cfg.scene.seed, cfg.device = args.num_envs, args.seed, args.device
    agent.seed, agent.device = args.seed, args.device
    agent.max_iterations, agent.save_interval = args.iterations, 500
    agent.run_name, agent.logger, agent.resume = args.run_name, "tensorboard", False
    audit = {
        "runtime": original.validate_runtime_overrides(cfg, agent, registered_cfg, registered_agent),
        # The retired parent is constructed privately, never re-registered.
        "parent_task_comparison": original.validate_parent_recipe(cfg, agent, parent_cfg, parent_agent),
    }
    return cfg, agent, audit


def run(args):
    app = env = runner = directory = None
    visuals = ExitStack()
    success = False
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless, device=args.device).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs import Elf3DwaqStillV2Runner
        from legged_lab.envs.elf3.elf3_still_config import Elf3DwaqStillEnvCfg, Elf3DwaqStillAgentCfg
        from legged_lab.envs.elf3.feedback_contract import canonical_sha256
        from legged_lab.envs.elf3.still_v2_runner import validate_still_v2_configuration
        from legged_lab.utils import task_registry
        from melody_control_checks import assert_finite_weights
        from release_0918_checks import check_source_manifest
        from train_elf3_feedback import source_fingerprints

        visual = visuals.enter_context(original.cached_visual_alias(args.use_cached_visual))
        integrity, assets = check_source_manifest(ROOT), verify_retained_assets()
        cfg, agent, audit = configure(args, task_registry, Elf3DwaqStillEnvCfg(), Elf3DwaqStillAgentCfg())
        audit["exact_standing_recipe"] = validate_still_v2_configuration(cfg, agent.to_dict())
        if args.reference_run is not None:
            from release_0918_checks import compare_saved_run
            audit["historical_reference"] = compare_saved_run(cfg, agent, args.reference_run)
        report = {
            "task": TASK, "from_scratch": True, "checkpoint_loaded": None,
            "initial_iteration": 0, "target_iteration": args.iterations - 1,
            "learning_updates": args.iterations, "num_envs": args.num_envs, "seed": args.seed,
            "save_interval": 500, "configuration_audit": audit,
            "configuration_sha256": canonical_sha256(cfg.to_dict()),
            "agent_configuration_sha256": canonical_sha256(agent.to_dict()),
            "source_integrity": integrity, "assets": assets,
            "automatic_training_restart": False, "automatic_training_stop": False,
            "active_run_pointer_modified": False, "display_only_visual_cache": visual,
        }
        if args.verify_config_only:
            print(original.report_json({**report, "environment": cfg.to_dict(), "agent": agent.to_dict(),
                  "creates_environment": False, "learning_updates": 0}, indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT / "logs" / TASK / f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
        env = task_registry.get_task_class(TASK)(cfg, headless=args.headless)
        if (env.num_obs, env.num_obs_hist, env.num_actions) != (100, 5, 29):
            raise ValueError("The historical 5x100 / 29 interface changed")
        if not all(actuator.is_implicit_model for actuator in env.robot.actuators.values()):
            raise ValueError("Keep the historical implicit physical PD")
        armature = env.robot.root_physx_view.get_dof_armatures()
        torch.testing.assert_close(armature, torch.full_like(armature, .01), atol=1e-8, rtol=0)
        runner = Elf3DwaqStillV2Runner(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        original.assert_fresh_runner(runner)
        assert_finite_weights(runner)
        fingerprints = source_fingerprints()
        for relative in (*original.EXPERIMENT_SOURCE_PATHS, "scripts/train_elf3_still_v2_restored.py"):
            fingerprints[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        report.update(recipe=TASK, parent_task=original.PARENT_TASK, optimizer_state_entries=0,
                      training_pid=os.getpid(), experiment_contract=runner.contract(),
                      source_sha256=fingerprints, git=original.git_provenance(),
                      initial_max_terrain_level=cfg.scene.max_init_terrain_level,
                      policy=agent.policy.to_dict(), algorithm=agent.algorithm.to_dict())
        (directory / "initialization.json").write_text(original.report_json(report, indent=2) + "\n")
        (directory / "configuration_audit.json").write_text(original.report_json(audit, indent=2) + "\n")
        runner.save(str(directory / "initial_random_model.pt"), {"initialization_only": True})
        if args.swanlab_project:
            request = {"project": args.swanlab_project, "name": f"{TASK}_{stamp}_{args.run_name}", "config": report}
            (directory / "swanlab_request.json").write_text(original.report_json(request, indent=2) + "\n")
            launch_new_swanlab(directory)
        print("[READY] " + original.report_json({"directory": str(directory), "task": TASK,
              "from_scratch": True, "target_iteration": args.iterations - 1}), flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        assert_finite_weights(runner)
        if not (directory / f"model_{args.iterations - 1}.pt").is_file():
            raise RuntimeError("Missing requested final checkpoint")
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
            (directory / "training_exit.json").write_text(original.report_json({
                "success": success, "target_iteration": args.iterations - 1,
                "last_iteration": runner.current_learning_iteration if runner else None,
                "auto_restart": False, "automatic_training_stop": False,
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
                    visuals.close()
                finally:
                    if app is not None:
                        app.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit stop signal {signum}; no automatic restart")
    signal.signal(signal.SIGTERM, stop)
    main()
