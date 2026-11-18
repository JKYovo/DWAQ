"""Pure Torch recovery/reset rules; Actor inputs and physical PD are unchanged."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch


TASK = "elf3_dwaq_recovery40_v2"
BODY_NAMES = ("waist_z_link", "l_hip_z_link", "l_knee_y_link", "l_ankle_x_link",
              "r_hip_z_link", "r_knee_y_link", "r_ankle_x_link", "l_shoulder_z_link",
              "l_elbow_y_link", "l_wrist_z_link", "r_shoulder_z_link", "r_elbow_y_link", "r_wrist_z_link")
MUTED_TERMS = ("stand_joint_motion", "stand_leg_pose", "stand_waist_pose",
               "joint_deviation_hip", "joint_deviation_ankle", "joint_deviation_legs", "joint_deviation_arms",
               "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
               "undesired_contacts", "fly", "body_orientation_l2", "flat_orientation_l2",
               "feet_air_time", "feet_slide", "feet_force", "feet_too_near", "feet_stumble",
               "alive", "idle_penalty", "gait_phase_contact", "feet_swing_height",
               "ankle_torque_l2", "stance_foot_speed_cost", "base_height")
SETTINGS = {
    "version": TASK, "recovery_fraction": .4, "fall_delay_steps": 250,
    "initialization": "Amp_mjlab V4: fixed random40% Recovery, remaining WalkandRun; uniform concatenated frames",
    "root": "torso_link", "amp_body_names": list(BODY_NAMES), "reference_fps": 50.,
    "enter_tilt_cos": .7071067811865476, "exit_tilt_cos": None,
    "enter_height_ratio": .65, "exit_height_ratio": .80, "stable_steps": 1,
    "exit_rule": "relative torso height >= .80 nominal; immediate, no tilt/contact/speed/hold gate",
    "fall_minimum_height": .62, "fall_tilt_cos": .3420201433256688,
    "recovery_height_weight": 3.5, "recovery_height_std": .3,
    "height_reference": "finite terrain scan mean, origin fallback; default torso link height",
    "success_definition": "initial true fall -> height>.8nominal, tilt<20deg, foot support>30N, vxvy<.5m/s, omega<1.5rad/s for1s",
    "deep_lying_definition": "reference torso<.55m and tilt>60deg; separate denominator and success counter",
    "recovery_muted_terms": list(MUTED_TERMS), "mask_upper_body_pose_auxiliary": True,
    "mirror_auxiliary_unchanged": True,
    "amp_reward_coef": .1, "amp_task_reward_lerp": .75,
    "discriminator_hidden_dims": [1024, 512, 256], "discriminator_learning_rate": 1e-4,
    "discriminator_gradient_penalty": 10., "discriminator_batch_size": 4096,
    "discriminator_trunk_decay": .001, "discriminator_head_decay": .1,
    "amp_normalization": "running policy+expert moments, epsilon1e-4, clip10; GPU float64 statistics",
    "expert_sampling": "same16 untouched AMP clips; equal per clip, uniform frames, terminal clamp",
    "replay_capacity": 50000, "discriminator_updates": 20,
    "actor_interface": "history500 -> actions29; deterministic mean context; unchanged still-v1 preprocessing",
    "physics_unchanged": True, "source_motion_bytes_unchanged": True,
    "execution": {"motion_reset": "last reset event before the single parent forward",
                  "amp_samples": "compact union of pre-action and post-action recovery phases",
                  "local_com_cache": "actual per-environment link-local COM poses after startup randomization"},
}


def recovery_fallen(gravity_z, relative_height):
    """AMP's ELF3 low-height / >70 degree fall predicate, independent of contacts."""
    return ((gravity_z > -SETTINGS["fall_tilt_cos"])
            | (relative_height < SETTINGS["fall_minimum_height"]))


def root_clearance(root_z, hits, origins_z):
    valid = hits.isfinite().all(-1)
    count = valid.sum(-1)
    ground = torch.where(valid, hits[..., 2], 0.).sum(-1)/count.clamp(min=1)
    ground = torch.where(count > 0, ground, origins_z)
    return root_z-ground


def height_objective(height, nominal_height):
    return torch.exp(-(height-nominal_height).square()/SETTINGS["recovery_height_std"]**2)


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def quat_inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def rotate_inverse(q, v):
    u, w = -q[..., 1:], q[..., :1]
    t = 2*torch.cross(u, v, dim=-1)
    return v+w*t+torch.cross(u, t, dim=-1)


def amp_features(pos, quat, lin_vel, ang_vel, body_ids, anchor_id):
    """Same feature convention as Amp_mjlab: anchor-relative pose, per-link local velocities."""
    q = quat[:, body_ids]
    anchor_q = quat[:, anchor_id, None].expand_as(q)
    position = rotate_inverse(anchor_q, pos[:, body_ids]-pos[:, anchor_id, None])
    relative = quat_mul(quat_inverse(anchor_q), q)
    # First two rotation-matrix columns, row-major, as matrix_from_quat(q)[..., :2].
    w, x, y, z = relative.unbind(-1)
    orientation = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w),
                               2*(x*y+z*w), 1-2*(x*x+z*z),
                               2*(x*z-y*w), 2*(y*z+x*w)), -1)
    return torch.cat((position.flatten(1), orientation.flatten(1),
                      rotate_inverse(q, lin_vel[:, body_ids]).flatten(1),
                      rotate_inverse(q, ang_vel[:, body_ids]).flatten(1)), -1)


def selected_amp_features(snapshot, env_ids):
    """Evaluate only selected rows; the final snapshot body is the torso anchor."""
    selected = [value.index_select(0, env_ids) for value in snapshot]
    return amp_features(*selected, range(len(BODY_NAMES)), len(BODY_NAMES))


def delayed_fall(fallen, timeout, assigned, counters, limit=250):
    """Exact AMP consecutive-fall rule. Time limits always terminate immediately."""
    done = fallen | timeout
    pending = assigned & done
    counts = torch.where(pending, counters+1, torch.zeros_like(counters))
    ready = pending & (counts >= limit)
    result = timeout | (fallen & (~assigned | ready))
    return result, torch.where(ready, torch.zeros_like(counts), counts)


def phase_update(active, stable, gravity_z, relative_height, nominal_height):
    """Hand back to DWAQ as soon as the torso is raised; keep fall re-entry.

    The old tilt/hold gate also disabled the rewards needed to straighten and
    stop after getting up. Full upright success is still measured separately
    by the environment, but no longer blocks restoring ordinary DWAQ rewards.
    """
    fallen = (gravity_z > -SETTINGS["enter_tilt_cos"]) | (relative_height < nominal_height*.65)
    raised = relative_height >= nominal_height*SETTINGS["exit_height_ratio"]
    return (active | fallen) & ~raised, raised.to(stable.dtype)


class MotionLibrary:
    def __init__(self, directory, joint_names, device="cpu"):
        self.directory, self.files = Path(directory), {}
        self.frames = {}
        self.expert_pairs = []
        self.expert_clip_lengths = []
        for category in ("Recovery", "WalkandRun"):
            collected = []
            for path in sorted((self.directory/category).glob("*.npz")):
                self.files[path.relative_to(self.directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path, allow_pickle=False) as source:
                    assert float(source["fps"].reshape(-1)[0]) == 50.
                    names = tuple(source["joint_names"].tolist())
                    bodies = tuple(source["body_names"].tolist())
                    ids = [names.index(n) for n in joint_names]
                    root = bodies.index("torso_link")
                    data = {k: torch.as_tensor(source[k], dtype=torch.float32, device=device)
                            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
                    if not all(torch.isfinite(v).all() for v in data.values()):
                        raise ValueError(f"Nonfinite motion: {path}")
                    collected.append({"root_pos": data["body_pos_w"][:, root],
                                      "root_quat": data["body_quat_w"][:, root],
                                      "root_lin_vel": data["body_lin_vel_w"][:, root],
                                      "root_ang_vel": data["body_ang_vel_w"][:, root],
                                      "joint_pos": data["joint_pos"][:, ids], "joint_vel": data["joint_vel"][:, ids]})
                    features = amp_features(data["body_pos_w"], data["body_quat_w"],
                        data["body_lin_vel_w"], data["body_ang_vel_w"], [bodies.index(n) for n in BODY_NAMES], root)
                    self.expert_pairs.append(torch.stack((features, torch.cat((features[1:], features[-1:]))), 1))
                    self.expert_clip_lengths.append(len(features))
            if not collected:
                raise ValueError(f"No motion files: {self.directory/category}")
            self.frames[category] = {k: torch.cat([item[k] for item in collected]) for k in collected[0]}
        self.expert_pairs = torch.cat(self.expert_pairs)

    def sample(self, category, count):
        source = self.frames[category]
        ids = torch.randint(len(source["root_pos"]), (count,), device=source["root_pos"].device)
        return {key: value[ids] for key, value in source.items()}

    def identity(self):
        return {"files_sha256": dict(self.files), "frames": {k: len(v["root_pos"]) for k,v in self.frames.items()},
                "expert_transitions": len(self.expert_pairs), "expert_clip_lengths": self.expert_clip_lengths}


def recovery_contract(motion_identity):
    directory = Path(__file__).parent
    return {**deepcopy(SETTINGS), "motion_data": deepcopy(motion_identity), "sources_sha256": {
        name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ("recovery_v2_core.py", "recovery_v2_amp.py", "recovery_v2_env.py", "recovery_v2_runner.py")}}
