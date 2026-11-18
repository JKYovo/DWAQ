"""Build additional reset poses with offline FK; never rewrite robot assets."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/"TienKung-Lab/legged_lab/assets/elf3"
MOTION = ROOT/"data/elf3_amp_v4/Recovery/fallAndGetUp1_subject1.npz"


def quaternion(axis, angle):
    value = np.zeros(4)
    value[0], value[axis+1] = np.cos(angle/2), np.sin(angle/2)
    return value


def multiply(a, b):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, a, b)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--poses", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20261117)
    args = p.parse_args()
    if args.poses < 256 or args.poses % 2 or args.output.exists() or args.output.with_suffix(".json").exists():
        p.error("Use an even pose count>=256 and a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ROOT/"TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
    names = contract["joint_names"]
    xml = ET.parse(ASSETS/"mjcf/elf3.xml")
    xml.find("compiler").set("meshdir", str(ASSETS/"meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    data = mujoco.MjData(model)
    joints = [model.joint(n).id for n in names]
    qids = model.jnt_qposadr[joints]
    limits = model.jnt_range[joints]
    mid, half = limits.mean(-1), np.diff(limits, axis=-1).ravel()*contract["soft_joint_pos_limit_factor"]*.5
    colliders = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("This bank builder expects the retained mesh collision asset")
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        colliders.append((g, model.mesh_vert[start:start+count]))
    with np.load(MOTION, allow_pickle=False) as motion:
        order = [motion["joint_names"].tolist().index(n) for n in names]
        body = motion["body_names"].tolist().index("torso_link")
        quats = motion["body_quat_w"][:, body].astype(np.float64)
        source_q = motion["joint_pos"][:, order].astype(np.float64)
        w, x, y, z = quats.T
        supine = np.flatnonzero((motion["body_pos_w"][:, body, 2] < .55)
                               & (1-2*(x*x+y*y) < .5) & (2*(x*z-w*y) > .7))
    rng = np.random.default_rng(args.seed)
    rows, attempted = [], 0
    rejected = {"self_penetration": 0, "orientation_or_height": 0}
    index = {n: i for i, n in enumerate(names)}
    legs = [(0., 0., 0.), (-.3, .6, -.3), (-.6, 1.2, -.6), (-1., 1.9, -.6)]
    noise = np.array([.04 if "waist" in n or "wrist" in n else .10 for n in names])
    while len(rows) < args.poses:
        attempted += 1
        if attempted > args.poses*80:
            raise RuntimeError(f"Not enough feasible reset poses: {len(rows)} accepted; {rejected}")
        kind = 1 if len(rows) < args.poses//2 else 2
        if kind == 1:
            frame = int(rng.choice(supine))
            q = source_q[frame].copy()+rng.uniform(-1, 1, 29)*noise
            quat = multiply(quaternion(0, rng.uniform(-.20, .20)), quats[frame])
        else:
            q = np.array([contract["default_joint_pos"][n] for n in names])
            for side in ("l", "r"):
                hip, knee, ankle = legs[int(rng.integers(len(legs)))]
                for part, value in (("hip_y", hip), ("knee_y", knee), ("ankle_y", ankle)):
                    q[index[f"{side}_{part}_joint"]] = value
                q[index[f"{side}_hip_x_joint"]] = rng.uniform(-.10, .10)
                q[index[f"{side}_hip_z_joint"]] = rng.uniform(-.15, .15)
                arm = int(rng.integers(4))
                pitch, roll, elbow = [(0., .05, 0.), (.2, .2, .6), (-.2, .4, 1.2), (.1, .9, .4)][arm]
                for part, value in (("shoulder_y", pitch), ("shoulder_x", roll if side=="l" else -roll), ("elbow_y", elbow)):
                    q[index[f"{side}_{part}_joint"]] = value
            q += rng.uniform(-1, 1, 29)*noise
            quat = multiply(quaternion(0, rng.uniform(-np.pi/6, np.pi/6)),
                            quaternion(1, -np.pi/2+rng.uniform(-np.pi/12, np.pi/12)))
        quat = multiply(quaternion(2, rng.uniform(-np.pi, np.pi)), quat)
        quat /= np.linalg.norm(quat)
        q = np.clip(q, mid-half, mid+half)
        mujoco.mj_resetData(model, data)
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = [0., 0., 2.], quat, q
        mujoco.mj_forward(model, data)
        if any(c.dist < -.01 and model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0
               for c in data.contact):
            rejected["self_penetration"] += 1
            continue
        lowest = min(float((vertices@data.geom_xmat[g].reshape(3, 3)[2]+data.geom_xpos[g, 2]).min())
                     for g, vertices in colliders)
        clearance = float(rng.uniform(.005, .02))
        height = 2.+clearance-lowest
        rotation = data.xmat[model.body("torso_link").id].reshape(3, 3)
        if not (.05 < height < .55 and rotation[2, 0] > .65 and rotation[2, 2] < .5):
            rejected["orientation_or_height"] += 1
            continue
        rows.append((q, [0., 0., height], quat, kind, clearance))
        if len(rows) % 512 == 0:
            print(f"Accepted {len(rows)}/{args.poses}; attempts={attempted}; rejected={rejected}", flush=True)
    np.savez_compressed(args.output, joint_names=np.array(names),
        joint_pos=np.array([r[0] for r in rows], dtype=np.float32),
        root_pos=np.array([r[1] for r in rows], dtype=np.float32),
        root_quat=np.array([r[2] for r in rows], dtype=np.float32),
        pose_kind=np.array([r[3] for r in rows], dtype=np.int64),
        floor_clearance=np.array([r[4] for r in rows], dtype=np.float32))
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = {"version": "elf3_dwaq_recovery40_init_v4", "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "poses": args.poses, "seed": args.seed, "attempts": attempted, "rejected": rejected,
        "kind_counts": {"perturbed_supine_reference": args.poses//2, "procedural_supine": args.poses//2},
        "bank_sha256": digest(args.output), "generator_sha256": digest(__file__),
        "asset_contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
        "reference_motion_sha256": digest(MOTION), "offline_FK_asset": str(ASSETS/"mjcf/elf3.xml"),
        "offline_FK_asset_sha256": digest(ASSETS/"mjcf/elf3.xml"),
        "floor_clearance_m": [.005, .02], "reject_self_penetration_below_m": -.01,
        "joint_limits": "original soft factor0.9; native names", "reference_motion_bytes_changed": False,
        "robot_assets_changed": False}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
