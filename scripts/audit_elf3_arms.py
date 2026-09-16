"""Audit live PhysX arm mapping, reward coverage, FK and symmetric PD response."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import numpy as np
import torch
import mujoco
import hashlib
from legged_lab.assets.elf3 import ASSET_DIR, CONTRACT, JOINT_NAMES
from legged_lab.envs.elf3.elf3_dwaq_config import Elf3DwaqFlatEnvCfg
from legged_lab.envs.elf3.elf3_dwaq_env import Elf3DwaqEnv

env = None
report = {}
try:
    cfg = Elf3DwaqFlatEnvCfg()
    cfg.scene.num_envs = 4
    cfg.commands.debug_vis = False
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    for name in vars(cfg.domain_rand.events):
        if not name.startswith("_"):
            setattr(cfg.domain_rand.events, name, None)
    env = Elf3DwaqEnv(cfg, headless=True)
    robot = env.robot
    device = env.device
    zero_vel = torch.zeros_like(robot.data.joint_vel)
    root = robot.data.default_root_state[:, :7].clone()
    root[:, :3] += env.scene.env_origins
    root[:, 2] += 1.5  # Keep the body suspended for isolated controller checks.
    base = robot.data.default_joint_pos.clone()
    ids = env.policy_joint_ids
    arm_names = [n for n in JOINT_NAMES if any(x in n for x in ('shoulder', 'elbow', 'wrist'))]
    report['joint_mapping'] = {n: {'policy_index': JOINT_NAMES.index(n),
                                  'physx_index': robot.joint_names.index(n)} for n in arm_names}
    term = env.reward_manager.get_term_cfg('joint_deviation_arms')
    selected = term.params['asset_cfg']
    selected_names = [robot.joint_names[i] for i in selected.joint_ids]
    assert set(selected_names) == set(arm_names + ['waist_y_joint', 'waist_x_joint', 'waist_z_joint'])
    report['reward_joint_names'] = selected_names
    report['reward_weight'] = term.weight
    shoulder_term = env.reward_manager.get_term_cfg('shoulder_pose_l2')
    shoulder_selected = shoulder_term.params['asset_cfg']
    shoulder_names = [robot.joint_names[i] for i in shoulder_selected.joint_ids]
    expected_shoulders = [n for n in arm_names if 'shoulder' in n]
    assert set(shoulder_names) == set(expected_shoulders)
    report['shoulder_pose_l2'] = {
        'joint_names': shoulder_names,
        'weight': shoulder_term.weight,
    }

    def set_pose(q):
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=device))
        robot.write_joint_state_to_sim(q, zero_vel)
        robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)

    # Each arm joint must independently change the actual reward by the same amount.
    contributions = {}
    for name in arm_names:
        q = base.clone()
        q[:, robot.joint_names.index(name)] += 0.1
        set_pose(q)
        value = term.func(env, **term.params)
        torch.testing.assert_close(value, torch.full_like(value, 0.1), atol=2e-6, rtol=0)
        contributions[name] = float(value[0])
    report['reward_for_each_0.1_rad_perturbation'] = contributions

    # Unique position/velocity sentinels catch permutations and sign errors in observations.
    q = base + torch.arange(29, device=device)[None] * 0.002
    v = torch.arange(29, device=device)[None].repeat(env.num_envs, 1) * 0.003
    set_pose(q)
    robot.write_joint_state_to_sim(q, v)
    env.sim.forward()
    env.scene.update(env.physics_dt)
    obs, _ = env.compute_current_observations()
    torch.testing.assert_close(obs[:, 9:38], (q-base)[:, ids] * env.obs_scales.joint_pos)
    torch.testing.assert_close(obs[:, 38:67], v[:, ids] * env.obs_scales.joint_vel)
    report['position_velocity_observation_mapping'] = 'passed'

    # Single-joint actions must actuate only that named joint target, including right arm.
    action_checks = {}
    for name in arm_names:
        set_pose(base)
        actions = torch.zeros((env.num_envs, 29), device=device)
        actions[:, JOINT_NAMES.index(name)] = 0.2
        env.step(actions)
        target = robot.data.joint_pos_target - base
        expected = torch.zeros_like(target)
        expected[:, robot.joint_names.index(name)] = 0.2 * CONTRACT['action_scale'][name]
        torch.testing.assert_close(target, expected, atol=1e-6, rtol=0)
        obs, _ = env.compute_current_observations()
        torch.testing.assert_close(obs[:, 67:96], actions * env.obs_scales.actions)
        action_checks[name] = float(target[0, robot.joint_names.index(name)])
    report['single_joint_target_offsets'] = action_checks

    # Large policy outputs must not lift either shoulder_x beyond the symmetric
    # safety envelope.  The constrained action is also what enters history.
    set_pose(base)
    actions = torch.zeros((env.num_envs, 29), device=device)
    left_policy = JOINT_NAMES.index('l_shoulder_x_joint')
    right_policy = JOINT_NAMES.index('r_shoulder_x_joint')
    actions[:, left_policy] = 100.0
    actions[:, right_policy] = -100.0
    obs, _, _, _ = env.step(actions)
    target = robot.data.joint_pos_target - base
    limit = env.cfg.shoulder_x_max_deviation
    left_sim = robot.joint_names.index('l_shoulder_x_joint')
    right_sim = robot.joint_names.index('r_shoulder_x_joint')
    torch.testing.assert_close(target[:, left_sim], torch.full_like(target[:, left_sim], limit), atol=1e-6, rtol=0)
    torch.testing.assert_close(target[:, right_sim], torch.full_like(target[:, right_sim], -limit), atol=1e-6, rtol=0)
    expected_action = limit / CONTRACT['action_scale']['l_shoulder_x_joint']
    torch.testing.assert_close(obs[:, 67 + left_policy], torch.full_like(obs[:, 67 + left_policy], expected_action))
    torch.testing.assert_close(obs[:, 67 + right_policy], torch.full_like(obs[:, 67 + right_policy], -expected_action))
    report['shoulder_x_safety_bound'] = {
        'max_deviation_rad': limit,
        'normalized_action_limit': expected_action,
    }
    assert float(env.action_buffer._circular_buffer.buffer.abs().max()) <= env.cfg.policy_action_clip
    report['policy_action_clip'] = env.cfg.policy_action_clip

    # Compare complete body transforms at multiple arm angles, not just link origins at home.
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / 'xmls/elf3.xml'))
    data = mujoco.MjData(model)
    max_position_error = max_rotation_error = 0.
    for amount in (0., 0.25, -0.15):
        q = base.clone()
        for name in arm_names:
            axis = name.split('_')[-2]
            mirror = -1 if name.startswith('r_') and axis in ('x', 'z') else 1
            q[:, robot.joint_names.index(name)] += amount * mirror
        set_pose(q)
        data.qpos[:3] = root[0, :3].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
        data.qpos[3:7] = [1, 0, 0, 0]
        for name in JOINT_NAMES:
            data.qpos[model.joint(name).qposadr[0]] = float(q[0, robot.joint_names.index(name)])
            np.testing.assert_allclose(robot.data.joint_pos_limits[0, robot.joint_names.index(name)].cpu(),
                                       model.jnt_range[model.joint(name).id], atol=2e-6)
        mujoco.mj_forward(model, data)
        for name in robot.body_names:
            i = robot.body_names.index(name)
            bid = model.body(name).id
            position = (robot.data.body_pos_w[0, i] - env.scene.env_origins[0]).cpu().numpy()
            quat = robot.data.body_quat_w[0, i].cpu().numpy()
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, quat.astype(float))
            max_position_error = max(max_position_error, float(np.abs(position-data.xpos[bid]).max()))
            max_rotation_error = max(max_rotation_error, float(np.abs(matrix-data.xmat[bid]).max()))
    assert max_position_error < 1e-5 and max_rotation_error < 1e-5
    report['multi_pose_fk'] = {'max_position_error_m': max_position_error,
                              'max_rotation_matrix_error': max_rotation_error, 'joint_limits': 'passed'}

    # With no policy, hold mirrored arms for 2 seconds, keeping the root suspended.
    set_pose(base)
    samples = []
    for step in range(100):
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=device))
        env.step(torch.zeros((env.num_envs, 29), device=device))
        samples.append(robot.data.joint_pos[0].cpu().numpy().copy())
    arr = np.array(samples)
    report['zero_action_suspended_2s'] = {
        n: {'final_deg': float(np.rad2deg(arr[-1, robot.joint_names.index(n)])),
            'default_deg': float(np.rad2deg(CONTRACT['default_joint_pos'][n]))}
        for n in arm_names}
    report['physics_arm_gains'] = {
        n: {'kp': float(robot.actuators['native_pd'].stiffness[0, robot.joint_names.index(n)]),
            'kd': float(robot.actuators['native_pd'].damping[0, robot.joint_names.index(n)])}
        for n in arm_names}

    # Actual motion, not just target-buffer checks: mirrored shoulder pulses.
    q = base.clone()
    left = robot.joint_names.index('l_shoulder_x_joint')
    right = robot.joint_names.index('r_shoulder_x_joint')
    q[:, left] += 0.3
    q[:, right] -= 0.3
    set_pose(q)
    motion = []
    for _ in range(75):
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=device))
        env.step(torch.zeros((env.num_envs, 29), device=device))
        motion.append(robot.data.joint_pos[0, [left, right]].cpu().numpy().copy())
    motion = np.array(motion)
    assert abs(motion[-1, 0]-0.2) < 0.02 and abs(motion[-1, 1]+0.2) < 0.02
    report['mirrored_shoulder_recovery'] = {
        'start_rad': [0.5, -0.5], 'final_rad': motion[-1].tolist(),
        'maximum_mirror_error_rad': float(np.abs(motion[:, 0]+motion[:, 1]).max())}

    if args.checkpoint:
        from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ
        before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        assert saved['infos']['elf3_contract']['joint_names'] == list(JOINT_NAMES)
        policy = ActorCritic_DWAQ(119, 311, 29, 500, 19, 100).to(device).eval()
        policy.load_state_dict(saved['model_state_dict'])
        root[:, 2] -= 1.5
        all_ids = torch.arange(env.num_envs, device=device)

        def reset_walk():
            env.reset(all_ids)
            set_pose(base)
            env.command_generator.vel_command_b[:] = torch.tensor([1., 0., 0.], device=device)
            env.command_generator.is_heading_env[:] = False
            env.command_generator.is_standing_env[:] = False
            return env.get_observations()[0]

        obs = reset_walk()
        targets, angles, heights = [], [], []
        resets = 0
        for _ in range(300):
            hist = env.dwaq_obs_history_buffer.buffer.reshape(env.num_envs, -1)
            with torch.inference_mode():
                actions = policy.act_inference(obs, hist)
            obs, _, dones, _ = env.step(actions)
            targets.append(robot.data.joint_pos_target[:, [left, right]].cpu().numpy().copy())
            angles.append(robot.data.joint_pos[:, [left, right]].cpu().numpy().copy())
            heights.append(robot.data.root_pos_w[:, 2].cpu().numpy().copy())
            if dones.any():
                resets += 1
                obs = reset_walk()
        report['policy_flat_walk'] = {
            'checkpoint': str(args.checkpoint), 'steps': 300, 'num_envs': env.num_envs,
            'reset_all_count': resets, 'command': [1., 0., 0.],
            'median_shoulder_targets_deg': np.rad2deg(np.median(targets, axis=(0, 1))).tolist(),
            'median_shoulder_actual_deg': np.rad2deg(np.median(angles, axis=(0, 1))).tolist(),
            'mean_root_height_m': float(np.mean(heights)),
            'conditions': 'flat, no domain randomization/noise, reset all upon any fall',
        }
        assert before == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report['status'] = 'passed'
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if env is not None:
        env.close()
    app.close()
