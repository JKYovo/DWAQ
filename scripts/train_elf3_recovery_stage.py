"""Continue curriculum50000 with the current flat40% recovery/init-v4 recipe.

Default final checkpoint: model_80000.pt. --parent-checkpoint accepts the
completed curriculum50000 or historical flat-v3 model60000, with its SHA256.
--resume restores completed init-v4 policy/Adam/AMP, including replay and moments.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

import train_elf3_ankle_cost as shared

ROOT = Path(__file__).resolve().parents[1]
TASK = "elf3_dwaq_recovery40_init_v4"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--iterations", type=int, default=80001)
    p.add_argument("--seed", type=int, choices=(42,), default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--run-name", default="recovery40_from50000_target80000")
    p.add_argument("--motion-dir", type=Path, default=ROOT/"data/elf3_amp_v4")
    p.add_argument("--bank", type=Path, default=ROOT/"data/elf3_recovery_init_v4/supine_4096_seed20261117.npz")
    p.add_argument("--swanlab-project", default=None)
    p.add_argument("--use-cached-visual", action="store_true")
    p.add_argument("--preflight-steps", type=int, default=0)
    p.add_argument("--verify-config-only", action="store_true")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--parent-checkpoint", type=Path)
    g.add_argument("--resume", type=Path)
    p.add_argument("--parent-sha256")
    p.add_argument("--parent-num-envs", type=int, default=16384,
                   help="Actual curriculum50000 batch; full course passes its own environment count")
    return p


def prepare(args):
    import re
    import torch
    if (args.num_envs < 4 or args.parent_num_envs < 4 or args.preflight_steps < 0 or args.seed != 42
            or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", args.device)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name)):
        raise ValueError("Invalid environment count, preflight steps or run name")
    args.bank.resolve(strict=True)
    source = (args.parent_checkpoint or args.resume).resolve(strict=True)
    digest = shared.sha256(source)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    from recovery_curriculum_plan import checkpoint_stage
    mode = checkpoint_stage(saved)
    if args.parent_checkpoint:
        if not args.parent_sha256 or digest != args.parent_sha256.lower():
            raise ValueError("Parent requires its explicit SHA256")
        if mode not in ("curriculum_boundary", "historical_flat60000"):
            raise ValueError("Parent must be curriculum50000 or historical flat-v3 model60000")
    elif mode != "recovery":
        raise ValueError("--resume requires an init-v4 checkpoint; use --parent-checkpoint for curriculum50000")
    if args.parent_sha256 and not args.parent_checkpoint:
        raise ValueError("--parent-sha256 requires --parent-checkpoint")
    if saved["infos"].get("initialization_only") or not saved["optimizer_state_dict"]["state"]:
        raise ValueError("Require a completed checkpoint with trained Adam")
    start = int(saved["iter"])+1
    if args.iterations <= start:
        raise ValueError("Target must include at least one new learning update")
    return {"source": source, "sha256": digest, "start": start, "mode": mode,
            "target": args.iterations-1, "updates": args.iterations-start}


def write(path, value):
    Path(path).write_text(shared.public_helpers().report_json(value, indent=2)+"\n")


def preflight(env, runner, steps):
    """Verify reset velocities, untouched walking rows, and live reward routing."""
    import torch
    from unittest.mock import patch
    from legged_lab.envs.elf3.recovery_v2_core import MUTED_TERMS
    all_ids = torch.arange(env.num_envs, device=env.device)
    walking = all_ids[~env.recovery_assigned]
    recovery = all_ids[env.recovery_assigned]
    original_sample = env.motion_library.sample
    captured = {}

    def capture(category, count):
        frame = original_sample(category, count)
        captured[category] = frame
        return frame

    for _ in range(4):
        with patch.object(env.motion_library, "sample", side_effect=capture):
            env.reset(all_ids)
        raw = captured["WalkandRun"]
        torch.testing.assert_close(env.robot.data.joint_vel[walking], raw["joint_vel"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(env.robot.data.root_link_vel_w[walking],
            torch.cat((raw["root_lin_vel"], raw["root_ang_vel"]), -1), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(env.robot.data.root_link_vel_w[recovery],
            env.initialization_requested_root_velocity[recovery], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(env.robot.data.joint_vel[recovery],
            env.initialization_requested_joint_velocity[recovery], rtol=1e-5, atol=1e-5)
        zero = recovery[env.initialization_velocity_kind[recovery] == 0]
        assert (env.initialization_requested_root_velocity[zero] == 0).all()
        assert (env.initialization_requested_joint_velocity[zero] == 0).all()
        augmented = recovery[env.initialization_pose_kind[recovery] > 0]
        heights = env.robot.data.root_pos_w[augmented, 2]-env.scene.env_origins[augmented, 2]
        assert ((heights > 0) & (heights < .55)).all()
        assert (-env.robot.data.projected_gravity_b[augmented, 0] > .65).all()
        assert (env.command_generator.command[recovery] == 0).all()
    obs, _ = env.get_observations()
    extras = env.extras
    with torch.inference_mode(), patch.object(env.sim, "forward", wraps=env.sim.forward) as forward, \
            patch.object(env, "reset", wraps=env.reset) as resets, \
            patch.object(env.robot.root_physx_view, "get_coms", wraps=env.robot.root_physx_view.get_coms) as com_reads:
        for _ in range(steps):
            action = runner.alg.policy.act_inference(obs, extras["observations"]["obs_hist"])
            obs, reward, _, extras = env.step(action)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert (env.command_generator.command[recovery] == 0).all()
        count = sum(bool(c.args[0].numel()) for c in resets.call_args_list)
        assert forward.call_count == count and com_reads.call_count == 0
    original_mask = env.recovery_reward_mask.clone()
    try:
        env.recovery_reward_mask.copy_(torch.arange(env.num_envs, device=env.device)%2 == 0)
        for name in MUTED_TERMS:
            term = env.reward_manager.get_term_cfg(name)
            source = term.func.__wrapped__(env, **term.params)
            actual = term.func(env, **term.params)
            expected = torch.where(env.recovery_reward_mask, torch.zeros_like(source), source)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
    finally:
        env.recovery_reward_mask.copy_(original_mask)
    return {"steps": steps, "finite": True, "walking_reference_velocities_exact": True,
            "recovery_requested_velocities_exact": True, "zero_rows_exactly_zero": True,
            "augmented_poses_supine_and_below_exit_height": True, "unchanged_live_reward_routing": list(MUTED_TERMS),
            "nonempty_resets": count, "forward_calls": forward.call_count, "repeated_COM_reads": com_reads.call_count,
            "sampled_pose_by_velocity_counts": env.initialization_sampled.detach().cpu().reshape(3, 3).tolist()}


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
        from legged_lab.envs.elf3.recovery_init_env import make_configs, Elf3DwaqRecoveryInitEnv
        from legged_lab.envs.elf3.recovery_init_runner import validate_configuration
        from legged_lab.envs.elf3.recovery_course import Elf3DwaqRecoveryCourseRunner
        from legged_lab.envs.elf3.recovery_init_core import SETTINGS
        from release_0918_checks import check_source_manifest
        from melody_control_checks import assert_finite_weights
        visual = visuals.enter_context(helpers.cached_visual_alias(args.use_cached_visual))
        cfg, agent = make_configs(args.motion_dir.resolve(), args.bank.resolve(), args.num_envs, args.device, args.run_name)
        agent.max_iterations = args.iterations
        audit = validate_configuration(cfg, agent.to_dict())
        assets, baseline = verify_retained_assets(), check_source_manifest(ROOT)
        if args.verify_config_only:
            from legged_lab.envs.elf3.recovery_v2_core import MotionLibrary
            from legged_lab.envs.elf3.recovery_init_core import InitializationBank
            from legged_lab.elf3_contract import JOINT_NAMES
            print(helpers.report_json({"task": TASK, "mode": prepared["mode"],
                "initial_iteration": prepared["start"], "target_iteration": prepared["target"],
                "configuration_audit": audit, "assets": assets, "source_integrity": baseline,
                "environment": cfg.to_dict(), "agent": agent.to_dict(),
                "motion_data": MotionLibrary(args.motion_dir, JOINT_NAMES).identity(),
                "bank": InitializationBank(args.bank, JOINT_NAMES).identity(),
                "creates_environment": False, "learning_updates": 0}, indent=2), flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        directory = ROOT/"logs"/TASK/f"{stamp}_{args.run_name}_{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(directory/"params/env.yaml"), cfg)
        dump_yaml(str(directory/"params/agent.yaml"), agent)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
        env = Elf3DwaqRecoveryInitEnv(cfg, headless=args.headless)
        assert (env.num_obs, env.num_obs_hist, env.num_actions) == (100, 5, 29)
        torch.testing.assert_close(env.robot.root_physx_view.get_dof_armatures(),
            torch.full_like(env.robot.root_physx_view.get_dof_armatures(), .01), rtol=0, atol=1e-8)
        assert int(env.recovery_assigned.sum()) == int(args.num_envs*.4)
        runner = Elf3DwaqRecoveryCourseRunner(env, agent.to_dict(), str(directory), args.device, str(directory))
        if args.parent_checkpoint:
            if prepared["mode"] == "curriculum_boundary":
                restoration = runner.initialize_from_curriculum(prepared["source"], prepared["sha256"], args.parent_num_envs)
            else:
                restoration = runner.initialize_from_flat(prepared["source"], prepared["sha256"])
        else:
            restoration = runner.load(str(prepared["source"]))
        assert runner.current_learning_iteration == prepared["start"]
        assert_finite_weights(runner)
        report = {"task": TASK, "training_pid": os.getpid(), "from_scratch": False,
            "source_checkpoint": str(prepared["source"]), "source_checkpoint_sha256": prepared["sha256"],
            "initial_iteration": prepared["start"], "target_iteration": prepared["target"],
            "learning_updates": prepared["updates"], "num_envs": args.num_envs, "seed": 42,
            "initialization_settings": SETTINGS, "configuration_audit": audit, "mode": prepared["mode"],
            "assets": assets, "source_integrity": baseline, "display_only_visual_cache": visual,
            "restoration": restoration, "experiment_contract": runner.contract(),
            "actual_recovery_envs": int(env.recovery_assigned.sum()), "actual_normal_envs": int((~env.recovery_assigned).sum()),
            "runtime_fall_sampling": False, "physics_history_RNG_terrain_and_recovery_state_reinitialized": True,
            "automatic_training_restart": False, "automatic_training_stop": False, "git": helpers.git_provenance()}
        write(directory/"initialization.json", report)
        runner.save(str(directory/"initial_loaded_model.pt"), {"initialization_only": True})
        if args.preflight_steps:
            write(directory/"preflight.json", preflight(env, runner, args.preflight_steps))
        if args.swanlab_project:
            write(directory/"swanlab_request.json", {"project": args.swanlab_project, "name": directory.name, "config": report})
            shared.launch_new_swanlab(directory)
        print("[READY] "+helpers.report_json({"directory": str(directory), "start": prepared["start"],
            "target": prepared["target"], "num_envs": args.num_envs, "restoration": restoration}), flush=True)
        runner.git_status_repos = []
        entered = True
        runner.learn(prepared["updates"], init_at_random_ep_len=True)
        assert_finite_weights(runner)
        assert (directory/f"model_{prepared['target']}.pt").exists()
        success = True
    except BaseException:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if directory is not None:
            (directory/"failure_traceback.txt").write_text(error)
        raise
    finally:
        assert shared.sha256(prepared["source"]) == prepared["sha256"]
        if directory is not None:
            write(directory/"training_exit.json", {"success": success, "entered_training": entered,
                "last_iteration": getattr(runner, "current_learning_iteration", None), "error": error})
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        try:
            if env is not None:
                env.close()
        finally:
            visuals.close()
            if app is not None:
                app.close()


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    args = parser().parse_args()
    run(args, prepare(args))
