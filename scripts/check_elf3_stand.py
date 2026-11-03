"""Bounded Isaac check/play for a fixed command, including genuine zero-command standing.

Examples (from the repository root):
  bash elf3.sh scripts/check_elf3_stand.py --headless --num_envs 64 --steps 100
  bash elf3.sh scripts/check_elf3_stand.py --checkpoint /path/model_500.pt --num_envs 20

Without a checkpoint this sends zero normalized actions. That checks environment
plumbing, not the standing ability of a learned policy. It never trains or saves
weights. Each invocation creates its own environment/model.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("elf3_dwaq_stand_v1", "elf3_dwaq_stand_v2", "elf3_dwaq_feedback_v1", "elf3_dwaq_still"),
                        default="elf3_dwaq_stand_v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--command", nargs=3, type=float, default=(0., 0., 0.), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON report; existing files are not overwritten.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num_envs and steps must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve(strict=True)

    app = AppLauncher(args).app
    env = None
    try:
        import torch

        from legged_lab.elf3_contract import verify_retained_assets
        from legged_lab.envs import Elf3DwaqFeedbackRunner, Elf3DwaqStandRunner, Elf3DwaqStandV2Runner, Elf3DwaqStillRunner
        from legged_lab.envs.elf3.standing_rewards import (
            feet_swing_height_when_moving, gait_phase_contact_when_moving,
        )
        from legged_lab.mdp.rewards import feet_swing_height, gait_phase_contact
        from legged_lab.utils import task_registry
        from legged_lab.utils.elf3_play_contract import validate_play_checkpoint
        from rsl_rl.runners import DWAQOnPolicyRunner

        verify_retained_assets()
        env_cfg, agent_cfg = (deepcopy(cfg) for cfg in task_registry.get_cfgs(args.task))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.seed = agent_cfg.seed = args.seed
        env_cfg.device = agent_cfg.device = args.device
        # Explicit evaluation-only overrides, never written back to the registry.
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.terrain_generator = None
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.events.push_robot = None
        env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
        env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1., 1.)
        env_cfg.commands.debug_vis = False
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_heading_envs = 0.
        env_cfg.commands.rel_standing_envs = 0.
        for key, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.command):
            setattr(env_cfg.commands.ranges, key, (value, value))
        env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
        runner = None
        if args.checkpoint:
            runner_type = {"elf3_dwaq_stand_v1": Elf3DwaqStandRunner,
                           "elf3_dwaq_stand_v2": Elf3DwaqStandV2Runner,
                           "elf3_dwaq_still": Elf3DwaqStillRunner,
                           "elf3_dwaq_feedback_v1": Elf3DwaqFeedbackRunner}[args.task]
            runner = runner_type(env, agent_cfg.to_dict(), device=agent_cfg.device)
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            validate_play_checkpoint(saved, runner.contract())
            # Inference may change env count/terrain; training resume remains strict.
            DWAQOnPolicyRunner.load(runner, str(args.checkpoint), load_optimizer=False)
            runner.eval_mode()
        else:
            print("[CHECK ONLY] Zero normalized actions, not a learned-policy evaluation.", flush=True)

        obs, history = env.get_observations()
        if obs.shape != (args.num_envs, 100) or history.shape != (args.num_envs, 500):
            raise AssertionError(f"Observation contract changed: {obs.shape}, {history.shape}")
        if not env_cfg.robot.gait_phase.enable or env_cfg.robot.gait_phase.period != .8:
            raise AssertionError("The original phase clock must remain enabled")
        fixed_command = torch.tensor(args.command, device=env.device).expand(args.num_envs, -1)
        totals = torch.zeros(7, dtype=torch.float64, device=env.device)
        resets = torch.zeros((), dtype=torch.long, device=env.device)
        gate_checks = 0
        completed_steps = 0
        with torch.inference_mode():
            for index in range(args.steps):
                if not app.is_running():
                    break
                if not torch.equal(env.command_generator.command[:, :3], fixed_command):
                    raise AssertionError("Heading/resampling changed the requested fixed command")
                actions = (runner.alg.policy.act_inference(obs, history) if runner else
                           torch.zeros((args.num_envs, 29), device=env.device))
                obs, rewards, dones, extras = env.step(actions)
                history = extras["observations"]["obs_hist"]
                if not torch.equal(env.command_generator.command[:, :3], fixed_command):
                    raise AssertionError("Step/reset changed the requested fixed command")
                for name, value in (("obs", obs), ("history", history), ("rewards", rewards),
                                    ("actions", actions), ("raw_velocity", env.robot.data.joint_vel)):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite {name} at step {index}")
                for name, original, gated in (
                    ("gait_phase_contact", gait_phase_contact, gait_phase_contact_when_moving),
                    ("feet_swing_height", feet_swing_height, feet_swing_height_when_moving),
                ):
                    # RewardManager resolves body names on its own copied
                    # configuration; env_cfg still has unresolved slice(None).
                    params = deepcopy(env.reward_manager.get_term_cfg(name).params)
                    params.pop("command_epsilon", None)
                    raw = original(env, **params)
                    value = gated(env, **params)
                    moving = fixed_command.abs().amax(dim=-1) > 1e-6
                    expected = torch.where(moving, raw, torch.zeros_like(raw))
                    if not torch.equal(value, expected):
                        raise AssertionError(f"Reward gate failed: {name}")
                    gate_checks += 1
                # Exclude resets and first two seconds after each reset. These
                # are physical states, NOT target angles or filtered actor dq.
                valid = (~dones.bool()) & (env.episode_length_buf * env.step_dt >= 2.)
                data = env.robot.data
                gravity = data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=-1))
                contact = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2] > 1.
                samples = torch.stack((data.root_ang_vel_b[:, :2].square().mean(-1),
                                       data.root_lin_vel_b[:, :2].square().mean(-1),
                                       data.joint_vel.square().mean(-1), roll.square(), pitch.square(),
                                       contact.all(-1).float(), torch.ones_like(roll)), dim=-1)
                totals += samples[valid].sum(dim=0).to(torch.float64)
                resets += dones.long().sum()
                completed_steps += 1
                if (index + 1) % 100 == 0:
                    print(f"[CHECK] {index + 1}/{args.steps} steps", flush=True)
        values = totals.cpu().tolist()
        count = values[-1]
        report = {
            "task": args.task, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "learned_policy_evaluated": runner is not None, "command": args.command,
            "num_envs": args.num_envs, "completed_steps": completed_steps,
            "sim_duration_per_env_s": completed_steps * env.step_dt,
            "post_reset_warmup_s": 2., "metrics_are_diagnostic_only": True,
            "observation_shape": list(obs.shape), "history_shape": list(history.shape),
            "action_shape": [args.num_envs, 29], "phase_clock_enabled": True,
            "reward_gate_checks": gate_checks, "reset_events": resets.item(),
            "measured_env_frames_after_settling": count,
            "evaluation_overrides": "flat, noise/push off, zero reset velocity, default joint pose, fixed command",
            "other_randomization": "all other training randomization retained, including friction, mass, actuator gains and reset position/yaw",
            "physical_metrics": ({
                "root_ang_vel_xy_rms_rad_s": (values[0] / count) ** .5,
                "root_lin_vel_xy_rms_m_s": (values[1] / count) ** .5,
                "joint_vel_rms_rad_s": (values[2] / count) ** .5,
                "roll_rms_rad": (values[3] / count) ** .5,
                "pitch_rms_rad": (values[4] / count) ** .5,
                "double_support_fraction": values[5] / count,
            } if count else None),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        # Some Kit shutdown paths terminate before Python prints an unhandled
        # exception. Print it before closing, so a check cannot look successful.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
