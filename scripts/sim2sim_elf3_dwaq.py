"""Run the exported ELF3 DWAQ policy in the unchanged Amp_mjlab MJCF."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ASSET_DIR = Path(__file__).resolve().parents[1] / "TienKung-Lab/legged_lab/assets/elf3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported policy.pt")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=3000, help="Number of 50 Hz control steps")
    parser.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    parser.add_argument("--scene", choices=["flat", "stairs"], default="flat")
    parser.add_argument("--step-height", type=float, default=0.05)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    metadata = json.loads(args.policy.with_suffix(".json").read_text())
    contract = json.loads((ASSET_DIR / "contract.json").read_text())
    if metadata.get("contract") != contract:
        raise ValueError("Policy metadata does not match the local ELF3 asset")
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    xml = ET.parse(ASSET_DIR / "xmls/elf3.xml")
    xml.find("compiler").set("meshdir", str(ASSET_DIR / "xmls/robot_meshes"))
    if args.scene == "stairs":
        if not 0 < args.step_height <= 0.3:
            raise ValueError("--step-height must be in (0, 0.3]")
        for i in range(4):
            height = (i + 1) * args.step_height
            ET.SubElement(xml.find("worldbody"), "geom", name=f"stair_{i}", type="box",
                          pos=f"{1.5 + i * .4} 0 {height / 2}", size=f".2 1.5 {height / 2}",
                          rgba=".4 .4 .4 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    model.opt.timestep = contract["physics_dt"]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    names = contract["joint_names"]
    joint_ids = [model.joint(name).id for name in names]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = [model.actuator(name).id for name in names]
    model.dof_armature[dof_ids] = [contract["armature"][n] for n in names]
    default = np.array([contract["default_joint_pos"][n] for n in names])
    kp = np.array([contract["stiffness"][n] for n in names])
    kd = np.array([contract["damping"][n] for n in names])
    effort = np.array([contract["effort_limit"][n] for n in names])
    scale = np.array([contract["action_scale"][n] for n in names])
    data = mujoco.MjData(model)
    data.qpos[:3] = contract["initial_position"]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qpos_ids] = default
    mujoco.mj_forward(model, data)
    root_id = model.body(contract["physical_root"]).id
    previous_action = np.zeros(29)
    history = None
    dt = contract["physics_dt"] * contract["control_decimation"]
    viewer = None
    if not args.headless:
        from mujoco import viewer as viewer_module
        viewer = viewer_module.launch_passive(model, data)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
    try:
        for step in range(args.steps):
            started = time.monotonic()
            rotation = data.xmat[root_id].reshape(3, 3)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_id, velocity, 1)
            phase = (step * dt % metadata["gait_period"]) / metadata["gait_period"]
            phases = np.array([phase, (phase + metadata["gait_offset"]) % 1.0])
            frame = np.concatenate([velocity[:3], rotation.T @ [0, 0, -1], args.command,
                                    data.qpos[qpos_ids] - default, data.qvel[dof_ids], previous_action,
                                    np.sin(2 * np.pi * phases), np.cos(2 * np.pi * phases)]).astype(np.float32)
            history = np.tile(frame, (5, 1)) if history is None else np.concatenate([history[1:], frame[None]])
            with torch.inference_mode():
                action = policy(torch.from_numpy(history.reshape(1, 500))).numpy()[0]
            if not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite policy output")
            target = default + np.clip(action, -100, 100) * scale
            for _ in range(contract["control_decimation"]):
                torque = kp * (target - data.qpos[qpos_ids]) - kd * data.qvel[dof_ids]
                data.ctrl[actuator_ids] = np.clip(torque, -effort, effort)
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("Nonfinite MuJoCo state")
            previous_action = action.copy()
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = data.xpos[root_id]
                viewer.sync()
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
        print(json.dumps({"control_steps": step + 1, "sim_time": data.time,
                          "final_root_height": float(data.qpos[2]), "finite": True}))
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
