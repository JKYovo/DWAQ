"""Check the actual IsaacLab ELF3 asset, contracts and an upstream PPO update."""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--terrain", choices=["flat", "rough"], default="flat")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--output", type=Path, default=Path("artifacts/elf3_check.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import json
import subprocess
import numpy as np
import torch
import mujoco
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from legged_lab.assets.elf3 import ASSET_DIR, CONTRACT, JOINT_NAMES
from legged_lab.envs.elf3.elf3_dwaq_config import Elf3DwaqEnvCfg, Elf3DwaqFlatEnvCfg, Elf3DwaqAgentCfg
from legged_lab.envs.elf3.elf3_dwaq_env import Elf3DwaqEnv
from legged_lab.envs.elf3.runner import Elf3DwaqOnPolicyRunner
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqEnvCfg, G1DwaqAgentCfg

env = None
runner = None


def main():
    global env, runner
    project = Path(__file__).resolve().parents[1]
    cfg = Elf3DwaqFlatEnvCfg() if args.terrain == "flat" else Elf3DwaqEnvCfg()
    original = G1DwaqEnvCfg()
    weights = {}
    for name, source in vars(original.reward).items():
        if not isinstance(source, RewardTermCfg):
            continue
        adapted = getattr(cfg.reward, name)
        assert adapted.weight == source.weight and adapted.func is source.func, name
        for key, value in source.params.items():
            if not isinstance(value, SceneEntityCfg):
                assert adapted.params[key] == value, (name, key)
        weights[name] = adapted.weight
    agent = Elf3DwaqAgentCfg()
    original_agent = G1DwaqAgentCfg()
    assert agent.policy.to_dict() == original_agent.policy.to_dict()
    assert agent.algorithm.to_dict() == original_agent.algorithm.to_dict()
    # Network, PPO, storage, runner and reward implementation remain byte-identical.
    source_hashes = {}
    paths = ["TienKung-Lab/rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py",
             "TienKung-Lab/rsl_rl/rsl_rl/algorithms/dwaq_ppo.py",
             "TienKung-Lab/rsl_rl/rsl_rl/storage/rollout_storage_dwaq.py",
             "TienKung-Lab/rsl_rl/rsl_rl/runners/dwaq_on_policy_runner.py",
             "TienKung-Lab/legged_lab/envs/g1/g1_dwaq_config.py",
             "TienKung-Lab/legged_lab/mdp/rewards.py"]
    for path in paths:
        data = (project / path).read_bytes()
        assert data == subprocess.check_output(["git", "show", f"origin/main:{path}"], cwd=project)
        source_hashes[path] = hashlib.sha256(data).hexdigest()

    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "xmls/elf3.xml"))
    stage = Usd.Stage.Open(str(ASSET_DIR / "usd/elf3.usd"))
    usd_masses = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            name = prim.GetName()
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert body_id > 0, name
            np.testing.assert_allclose(mass.GetMassAttr().Get(), model.body_mass[body_id], rtol=1e-5)
            np.testing.assert_allclose(mass.GetCenterOfMassAttr().Get(), model.body_ipos[body_id], atol=1e-6)
            # Eigenvalues are invariant to how the importer chooses principal axes.
            np.testing.assert_allclose(sorted(mass.GetDiagonalInertiaAttr().Get()),
                                       sorted(model.body_inertia[body_id]), rtol=2e-4, atol=1e-7)
            axes = mass.GetPrincipalAxesAttr().Get()
            usd_rotation = Rotation.from_quat([*axes.GetImaginary(), axes.GetReal()]).as_matrix()
            mj_quat = model.body_iquat[body_id]
            mj_rotation = Rotation.from_quat(mj_quat[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(
                usd_rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ usd_rotation.T,
                mj_rotation @ np.diag(model.body_inertia[body_id]) @ mj_rotation.T,
                rtol=2e-4, atol=1e-7)
            usd_masses[name] = mass.GetMassAttr().Get()
    assert len(usd_masses) == model.nbody - 1

    cfg.scene.num_envs = args.num_envs
    cfg.commands.debug_vis = False
    # Only this geometry/indexing test uses deterministic defaults. Training
    # commands use the original randomization and full terrain grid.
    for name in vars(cfg.domain_rand.events):
        if not name.startswith("_"):
            setattr(cfg.domain_rand.events, name, None)
    cfg.noise.add_noise = False
    if cfg.scene.terrain_generator is not None:
        cfg.scene.terrain_generator.num_rows = 2
        cfg.scene.terrain_generator.num_cols = 5
        cfg.scene.max_init_terrain_level = 1
    env = Elf3DwaqEnv(cfg, headless=args.headless)
    ids = env.policy_joint_ids
    robot = env.robot
    # Verify IsaacLab body transforms at the default pose against MuJoCo FK.
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7] + torch.cat(
        [env.scene.env_origins, torch.zeros(env.num_envs, 4, device=env.device)], dim=1))
    robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.joint_vel))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    data = mujoco.MjData(model)
    data.qpos[:3] = CONTRACT["initial_position"]
    data.qpos[3:7] = [1, 0, 0, 0]
    for name in JOINT_NAMES:
        index = model.joint(name).qposadr[0]
        data.qpos[index] = CONTRACT["default_joint_pos"][name]
    mujoco.mj_forward(model, data)
    fk_error = 0.0
    for i, name in enumerate(robot.body_names):
        actual = (robot.data.body_pos_w[0, i] - env.scene.env_origins[0]).cpu().numpy()
        expected = data.xpos[model.body(name).id]
        fk_error = max(fk_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-5)
    for key, tensor in [("stiffness", robot.data.joint_stiffness), ("damping", robot.data.joint_damping),
                        ("armature", robot.data.joint_armature)]:
        # Explicit PD gains live in the actuator; simulator gains are zero.
        if key in ("stiffness", "damping"):
            tensor = getattr(robot.actuators["native_pd"], key)
        expected = torch.tensor([CONTRACT[key][n] for n in robot.joint_names], device=env.device)
        torch.testing.assert_close(tensor[0], expected)
    assert (env.num_obs, env.num_privileged_obs, env.num_obs_hist, env.num_actions) == (100, 311, 5, 29)
    actions = torch.linspace(-0.1, 0.1, 29, device=env.device).repeat(env.num_envs, 1)
    for _ in range(64):
        obs, reward, done, extras = env.step(actions)
        for tensor in [obs, reward, extras["observations"]["critic"], extras["observations"]["obs_hist"],
                       robot.data.joint_pos, robot.data.joint_vel]:
            assert torch.isfinite(tensor).all()
    actor, critic = env.compute_current_observations()
    torch.testing.assert_close(actor[:, 9:38], (robot.data.joint_pos - robot.data.default_joint_pos)[:, ids])
    torch.testing.assert_close(actor[:, 38:67], robot.data.joint_vel[:, ids])
    torch.testing.assert_close(actor[:, 67:96], actions)
    torch.testing.assert_close(critic[:, 100:103], robot.data.root_lin_vel_b)
    expected_feet = robot.data.body_pos_w[:, env.feet_body_ids] - robot.data.root_pos_w[:, None]
    expected_feet = quat_apply_inverse(robot.data.root_quat_w[:, None].expand(-1, 2, -1), expected_feet)
    torch.testing.assert_close(env.feet_pos_in_body, expected_feet)
    targets = robot.data.joint_pos_target[:, ids] - robot.data.default_joint_pos[:, ids]
    scales = torch.tensor([CONTRACT["action_scale"][n] for n in JOINT_NAMES], device=env.device)
    torch.testing.assert_close(targets, actions * scales)
    env.reset(torch.tensor([0], device=env.device))
    env.get_observations()
    history = env.dwaq_obs_history_buffer.buffer[0]
    torch.testing.assert_close(history, history[-1:].expand_as(history))
    torch.testing.assert_close(env.leg_phase[0], torch.tensor([0.0, 0.5], device=env.device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent / (args.output.stem + "_ppo")
    if run_dir.exists():
        raise FileExistsError(f"Use a fresh --output path: {run_dir}")
    runner = Elf3DwaqOnPolicyRunner(env, agent.to_dict(), log_dir=str(run_dir), device=env.device)
    before = {name: value.clone() for name, value in runner.alg.policy.named_parameters()}
    runner.learn(2)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    assert any(not torch.equal(before[n], p) for n, p in runner.alg.policy.named_parameters())
    runner.load(str(run_dir / "model_1.pt"))
    foreign = torch.load(run_dir / "model_1.pt", map_location="cpu", weights_only=False)
    foreign["infos"]["elf3_contract"]["robot"] = "g1"
    foreign_path = run_dir / "foreign_contract_test.pt"
    torch.save(foreign, foreign_path)
    try:
        runner.load(str(foreign_path))
    except ValueError:
        pass
    else:
        raise AssertionError("A same-size checkpoint with a foreign robot contract was accepted")
    foreign_path.unlink()
    report = {"passed": True, "terrain": args.terrain, "num_envs": env.num_envs,
              "usd_body_count": len(usd_masses), "total_mass_kg": sum(usd_masses.values()),
              "max_default_pose_fk_error_m": fk_error, "actor_dim": 100, "critic_dim": 311,
              "history_dim": 500, "action_dim": 29, "weights": weights,
              "upstream_sha256": source_hashes, "ppo_iterations": 2,
              "checkpoint_roundtrip": True, "foreign_contract_rejected": True,
              "checkpoint": str((run_dir / "model_1.pt").resolve()),
              "policy_joint_ids": ids.tolist(), "feet_articulation_ids": env.feet_body_ids,
              "feet_contact_ids": env.feet_cfg.body_ids}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ELF3 CHECK PASSED:", args.output)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    if runner is not None and runner.writer is not None:
        runner.writer.close()
    if env is not None:
        env.close()
    app.close()
