"""Play exported ELF3 recovery policies on flat ground, without Isaac Sim."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "TienKung-Lab/legged_lab/assets/elf3"
CONTRACT_PATH = ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json"
MOTION = ROOT / "data/elf3_amp_v4/Recovery/fallAndGetUp1_subject1.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported ONNX with matching .json sidecar")
    parser.add_argument("--motion", type=Path, default=MOTION)
    parser.add_argument("--frame", type=int, help="Fixed reference frame; otherwise sample deep lying frames")
    parser.add_argument("--start", choices=["lying", "standing"], default="lying")
    parser.add_argument("--command", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--reset-seconds", type=float, default=10., help="Reset interval, 0 disables repeated resets")
    parser.add_argument("--steps", type=int, default=30000, help="50 Hz control steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.reset_seconds) or args.reset_seconds < 0:
        parser.error("--steps must be positive and --reset-seconds finite and nonnegative")
    if not np.isfinite(args.command).all():
        parser.error("--command must be finite")

    contract = json.loads(CONTRACT_PATH.read_text())
    metadata = json.loads(args.policy.with_suffix(".json").read_text())
    if metadata.get("contract") != contract or metadata.get("policy_input_dim") != 500:
        raise ValueError("Policy metadata does not match the local 100 x 5 ELF3 contract")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ort.InferenceSession(str(args.policy), sess_options=options, providers=["CPUExecutionProvider"])
    if ([x.name for x in policy.get_inputs()] != ["history"]
            or [x.name for x in policy.get_outputs()] != ["actions"]):
        raise ValueError("Expected history -> actions ONNX interface")

    # Match the existing sim2sim entrypoint; no asset file is rewritten.
    xml = ET.parse(ASSETS / "mjcf/elf3.xml")
    xml.find("compiler").set("meshdir", str(ASSETS / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    model.opt.timestep = contract["physics_dt"]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    names = contract["joint_names"]
    joints = [model.joint(n).id for n in names]
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    actuators = [model.actuator(n).id for n in names]
    model.dof_armature[vids] = [contract["armature"][n] for n in names]
    default, kp, kd, effort, scale = [
        np.array([contract[key][n] for n in names])
        for key in ("default_joint_pos", "stiffness", "damping", "effort_limit", "action_scale")
    ]
    limits = model.jnt_range[joints]
    midpoint = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) * contract["soft_joint_pos_limit_factor"] / 2
    root_id = model.body(contract["physical_root"]).id
    data = mujoco.MjData(model)
    dt = model.opt.timestep * contract["control_decimation"]
    reset_steps = max(1, round(args.reset_seconds / dt)) if args.reset_seconds else 0
    rng = np.random.default_rng(args.seed)

    motion = None
    if args.start == "lying":
        with np.load(args.motion, allow_pickle=False) as archive:
            motion_names = archive["joint_names"].tolist()
            body_names = archive["body_names"].tolist()
            root = body_names.index(contract["physical_root"])
            order = [motion_names.index(n) for n in names]
            motion = {
                "q": archive["joint_pos"][:, order],
                "pos": archive["body_pos_w"][:, root],
                "quat": archive["body_quat_w"][:, root],
            }
        quats = motion["quat"]
        up_cos = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
        candidates = np.flatnonzero((motion["pos"][:, 2] < .55) & (up_cos < .5))
        if not candidates.size:
            raise ValueError("Motion has no deep lying poses")
        if args.frame is not None and not 0 <= args.frame < len(quats):
            parser.error("--frame is outside the motion clip")

    previous_action = np.zeros(29)
    history = None
    episode_steps = 0
    resets = 0
    peak_height = 0.

    def reset():
        nonlocal history, episode_steps, resets, peak_height
        mujoco.mj_resetData(model, data)
        selected = None
        if motion is None:
            data.qpos[:3] = contract["initial_position"]
            data.qpos[3:7] = [1., 0., 0., 0.]
            data.qpos[qids] = default
        else:
            selected = args.frame if args.frame is not None else int(rng.choice(candidates))
            data.qpos[:3] = [0., 0., motion["pos"][selected, 2]]
            quat = motion["quat"][selected]
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            data.qpos[qids] = np.clip(motion["q"][selected], midpoint - half, midpoint + half)
        # Test recovery from rest, rather than injecting reference motion velocity.
        data.qvel[:] = 0.
        previous_action.fill(0.)
        history, episode_steps = None, 0
        mujoco.mj_forward(model, data)
        peak_height = float(data.xpos[root_id, 2])
        resets += 1
        print(json.dumps({"reset": resets, "frame": selected,
                          "initial_root_height": peak_height, "initial_velocity": "zero"}), flush=True)

    reset()
    viewer = None
    if not args.headless:
        from mujoco import viewer as viewer_module
        viewer = viewer_module.launch_passive(model, data)
        viewer.cam.distance, viewer.cam.elevation = 3., -18.
    try:
        for step in range(args.steps):
            started = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            if reset_steps and episode_steps >= reset_steps:
                reset()
            rotation = data.xmat[root_id].reshape(3, 3)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_id, velocity, 1)
            phase = (episode_steps * dt / metadata["gait_period"]) % 1.
            phases = np.array([phase, (phase + metadata["gait_offset"]) % 1.])
            frame = np.concatenate([
                velocity[:3], rotation.T @ [0., 0., -1.], args.command,
                data.qpos[qids] - default, data.qvel[vids], previous_action,
                np.sin(2 * np.pi * phases), np.cos(2 * np.pi * phases),
            ]).astype(np.float32)
            history = np.tile(frame, (5, 1)) if history is None else np.concatenate([history[1:], frame[None]])
            action = policy.run(["actions"], {"history": np.clip(history.reshape(1, 500), -100, 100)})[0][0]
            if action.shape != (29,) or not np.isfinite(action).all():
                raise FloatingPointError("Invalid policy output")
            target = default + np.clip(action, -100, 100) * scale
            for _ in range(contract["control_decimation"]):
                torque = kp * (target - data.qpos[qids]) - kd * data.qvel[vids]
                data.ctrl[actuators] = np.clip(torque, -effort, effort)
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("Nonfinite MuJoCo state")
            previous_action[:] = action
            episode_steps += 1
            peak_height = max(peak_height, float(data.xpos[root_id, 2]))
            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[root_id]
                viewer.sync()
                time.sleep(max(0., dt - (time.monotonic() - started)))
        print(json.dumps({"control_steps": step + 1, "resets": resets,
                          "final_root_height": float(data.xpos[root_id, 2]),
                          "last_episode_peak_height": peak_height, "finite": True}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
