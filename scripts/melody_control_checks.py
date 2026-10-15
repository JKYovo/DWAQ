"""Fail closed on configuration drift; preserve the retained physical asset."""
import hashlib
from pathlib import Path

import torch
from isaaclab.actuators import ImplicitActuator

from legged_lab.assets.elf3 import ASSET_DIR, ELF3_CFG
from legged_lab.elf3_contract import CONTRACT, JOINT_NAMES, verify_retained_assets
from references.melody_elf3 import ELF3LITE_CFG


def check_configuration(cfg, agent, root):
    assert ELF3_CFG is not ELF3LITE_CFG
    expected = ELF3LITE_CFG.copy()
    expected.spawn.usd_path = str(ASSET_DIR / "usd/elf3.usd")
    for actuator in expected.actuators.values():
        actuator.armature = .01
    assert cfg.scene.robot.to_dict() == expected.to_dict(), "Unexpected change beyond retained USD/uniform armature"
    assert cfg.scene.robot.spawn.usd_path == str(ASSET_DIR / "usd/elf3.usd")
    assert set(CONTRACT["armature"]) == set(JOINT_NAMES)
    assert set(CONTRACT["armature"].values()) == {.01}
    for actuator in cfg.scene.robot.actuators.values():
        assert actuator.armature == .01, "Every actuator group must use explicit armature 0.01"
        assert issubclass(actuator.class_type, ImplicitActuator)
    vendor = root / "scripts/references/melody_elf3.py"
    assert hashlib.sha256(vendor.read_bytes()).hexdigest() == CONTRACT["configuration_source"]["sha256"]
    verify_retained_assets()
    assert (cfg.sim.dt, cfg.sim.decimation) == (.005, 4)
    assert agent.num_steps_per_env == 24
    assert (agent.algorithm.num_mini_batches, agent.algorithm.num_learning_epochs) == (4, 5)
    assert agent.runner_class_name == "Elf3DwaqUpperBodySymmetryRunner"
    assert agent.upper_body_mirror_loss_coeff == .1 and agent.upper_body_pose_loss_coeff == .02
    assert agent.algorithm.learning_rate == 1e-4 and agent.algorithm.schedule == "adaptive"
    assert agent.algorithm.min_learning_rate == 1e-5 and agent.algorithm.max_learning_rate == 3e-4
    assert not cfg.domain_rand.action_delay.enable, "Use the requested 0918 recipe, not the later delay task"
    for term in (cfg.reward.track_lin_vel_xy_exp, cfg.reward.track_ang_vel_z_exp):
        assert term.weight == 2. and term.params["std"] == .5


def check_runtime(env, exercise_mapping=False):
    robot = env.robot
    ids = env.policy_joint_ids
    assert tuple(robot.joint_names[i] for i in ids.tolist()) == tuple(JOINT_NAMES)
    assert (env.num_obs, env.num_obs_hist, env.num_actions) == (100, 5, 29)
    assert all(act.is_implicit_model for act in robot.actuators.values())
    assert set(robot.actuators) == set(ELF3LITE_CFG.actuators)
    expected_effort = torch.tensor([CONTRACT["effort_limit"][n] for n in JOINT_NAMES], device=env.device, dtype=robot.data.joint_effort_limits.dtype)
    torch.testing.assert_close(robot.data.joint_effort_limits[:, ids], expected_effort.expand(env.num_envs, -1))
    torch.testing.assert_close(robot.data.joint_vel_limits[:, ids], torch.full((env.num_envs, 29), 20., device=env.device))
    expected_armature = torch.full((env.num_envs, 29), .01, device=env.device)
    torch.testing.assert_close(robot.data.joint_armature[:, ids], expected_armature, atol=1e-8, rtol=0)
    physx_armature = robot.root_physx_view.get_dof_armatures().to(env.device)[:, ids]
    torch.testing.assert_close(physx_armature, expected_armature, atol=1e-8, rtol=0)
    # 0918 retains its startup Kp/Kd domain randomization. Compare to the
    # original nominal gains and require every realized gain to stay in range.
    for field, key in (("joint_stiffness", "stiffness"), ("joint_damping", "damping")):
        expected = torch.tensor([CONTRACT[key][n] for n in JOINT_NAMES], device=env.device)
        ratio = getattr(robot.data, field)[:, ids] / expected
        assert torch.isfinite(ratio).all() and ((ratio >= .7999) & (ratio <= 1.2001)).all(), field
    masses = robot.data.default_mass.sum(dim=1)
    torch.testing.assert_close(masses, torch.full_like(masses, 43.2225077), atol=1e-4, rtol=0)
    scale = torch.tensor([CONTRACT["action_scale"][n] for n in robot.joint_names], device=env.device)
    torch.testing.assert_close(env.action_scale, scale)
    # Verify named actions and proprioception through the actual articulation.
    if exercise_mapping:
        actions = torch.linspace(-.1, .1, 29, device=env.device).repeat(env.num_envs, 1)
        obs, reward, done, extras = env.step(actions)
        native_scale = torch.tensor([CONTRACT["action_scale"][n] for n in JOINT_NAMES], device=env.device)
        torch.testing.assert_close(robot.data.joint_pos_target[:, ids], robot.data.default_joint_pos[:, ids] + actions * native_scale)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert torch.isfinite(extras["observations"]["obs_hist"]).all()
    actor, critic = env.compute_current_observations()
    torch.testing.assert_close(actor[:, 9:38], (robot.data.joint_pos - robot.data.default_joint_pos)[:, ids])
    torch.testing.assert_close(actor[:, 38:67], robot.data.joint_vel[:, ids])
    if exercise_mapping:
        torch.testing.assert_close(actor[:, 67:96], actions)
    for value in (actor, critic, robot.data.joint_pos, robot.data.joint_vel):
        assert torch.isfinite(value).all()
    if exercise_mapping:
        env.reset(torch.arange(env.num_envs, device=env.device))
    return {"robot_config_matches_original_except_usd_path_and_armature": True,
            "retained_usd_total_mass_kg": float(masses[0].item()),
            "control_type": "ImplicitActuator", "explicit_armature_override": True,
            "all_29_joint_armatures_verified": True,
            "effective_armature_min": float(robot.data.joint_armature.min().item()),
            "effective_armature_max": float(robot.data.joint_armature.abs().max().item()),
            "physx_armature_min": float(physx_armature.min().item()),
            "physx_armature_max": float(physx_armature.max().item()),
            "velocity_limit_rad_s": 20., "native_action_and_observation_mapping": True,
            "startup_gain_randomization": [0.8, 1.2],
            "effort_limits_native": expected_effort.cpu().tolist(),
            "policy_joint_ids": ids.cpu().tolist()}


def assert_finite_weights(runner):
    for name, value in runner.alg.policy.named_parameters():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite parameter: {name}")
