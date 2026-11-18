"""AMP-compatible motion resets and scoped recovery for the completed DWAQ course."""
from copy import deepcopy
from functools import wraps

import torch
from isaaclab.managers import EventTermCfg, RewardTermCfg

from legged_lab.elf3_contract import CONTRACT
from legged_lab.envs.elf3.reward_curriculum import (
    Elf3DwaqRewardCurriculumEnv, make_configs as parent_configs,
)
from legged_lab.envs.elf3.recovery_v2_core import (
    BODY_NAMES, MUTED_TERMS, SETTINGS, TASK, MotionLibrary, amp_features, delayed_fall, phase_update,
    selected_amp_features,
    recovery_fallen, root_clearance, height_objective,
)


def mask_reward(env, value):
    return torch.where(env.recovery_reward_mask, torch.zeros_like(value), value)


def masked_term(source):
    # Preserve each original signature and its top-level SceneEntityCfg params
    # so Isaac resolves the same body/joint IDs, and call the original exactly.
    @wraps(source)
    def wrapped(env, *args, **kwargs):
        return mask_reward(env, source(env, *args, **kwargs))
    wrapped.__module__ = __name__
    wrapped.__name__ = wrapped.__qualname__ = "recovery_"+source.__name__
    globals()[wrapped.__name__] = wrapped
    return wrapped


def recovery_height(env):
    value = height_objective(env._relative_height(), CONTRACT["initial_position"][2])
    return torch.where(env.recovery_reward_mask, value, torch.zeros_like(value))


def recovery_reference_reset(env, env_ids):
    """Install the reference state before the existing reset performs its forward."""
    env.reset_motion_reference(env_ids)


def make_configs(data_directory, num_envs=16384, device="cuda:0", run_name="recovery40_from50000"):
    cfg, agent = parent_configs(num_envs, device, run_name, 20000, 30000)
    cfg.recovery = {**deepcopy(SETTINGS), "data_directory": str(data_directory)}
    for name in MUTED_TERMS:
        term = getattr(cfg.reward, name)
        term.func = masked_term(term.func)
    cfg.reward.recovery_height = RewardTermCfg(func=recovery_height, weight=SETTINGS["recovery_height_weight"])
    cfg.domain_rand.events.recovery_reference = EventTermCfg(func=recovery_reference_reset, mode="reset")
    agent.runner_class_name = "Elf3DwaqRecoveryRunner"
    agent.experiment_name = agent.wandb_project = TASK
    agent.max_iterations = 100001
    agent.recovery = deepcopy(SETTINGS)
    return cfg, agent


def projected_parent_cfg(cfg):
    result = deepcopy(cfg)
    reference, _ = parent_configs(cfg.scene.num_envs, cfg.device)
    for name in MUTED_TERMS:
        setattr(result.reward, name, deepcopy(getattr(reference.reward, name)))
    delattr(result.reward, "recovery_height")
    delattr(result, "recovery")
    delattr(result.domain_rand.events, "recovery_reference")
    return result


def projected_parent_recipe(recipe):
    result = deepcopy(recipe)
    result.pop("recovery")
    result.update(runner_class_name="Elf3DwaqRewardCurriculumRunner",
                  experiment_name="elf3_dwaq_still_v2_curriculum_v1",
                  wandb_project="elf3_dwaq_still_v2_curriculum_v1")
    return result


class Elf3DwaqRecoveryEnv(Elf3DwaqRewardCurriculumEnv):
    def init_buffers(self):
        super().init_buffers()
        self.motion_library = MotionLibrary(self.cfg.recovery["data_directory"], self.robot.joint_names, self.device)
        self.amp_body_ids, names = self.robot.find_bodies(list(BODY_NAMES), preserve_order=True)
        if tuple(names) != BODY_NAMES:
            raise ValueError("AMP link mapping must follow the existing named ELF3 reference")
        self.amp_anchor_id = self.robot.find_bodies("torso_link", preserve_order=True)[0][0]
        self.amp_snapshot_body_ids = torch.tensor(
            [*self.amp_body_ids, self.amp_anchor_id], device=self.device, dtype=torch.long)
        self.recovery_assigned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ids = torch.randperm(self.num_envs, device=self.device)[:int(self.num_envs*.4)]
        self.recovery_assigned[ids] = True
        self.recovery_active = self.recovery_assigned.clone()
        self.recovery_reward_mask = self.recovery_active.clone()
        self.recovery_stable = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fall_delay_counters = torch.zeros_like(self.recovery_stable)
        self.recovery_initial_fallen = torch.zeros_like(self.recovery_active)
        self.recovery_success_counted = torch.zeros_like(self.recovery_active)
        self.recovery_sampled_total = torch.zeros((), device=self.device)
        self.recovery_fallen_total = torch.zeros((), device=self.device)
        self.recovery_success_total = torch.zeros((), device=self.device)
        self.recovery_support_stable = torch.zeros_like(self.recovery_stable)
        self.recovery_initial_deep = torch.zeros_like(self.recovery_active)
        self.recovery_deep_total = torch.zeros((), device=self.device)
        self.recovery_deep_success_total = torch.zeros((), device=self.device)
        self.recovery_v2_amp = None

    def amp_snapshot(self, env_ids=None):
        """Copy named link states, postponing feature transforms until rows are selected."""
        data = self.robot.data
        values = (data.body_link_pos_w, data.body_link_quat_w,
                  data.body_link_lin_vel_w, data.body_link_ang_vel_w)
        if env_ids is None:
            return tuple(value[:, self.amp_snapshot_body_ids] for value in values)
        return tuple(value[env_ids[:, None], self.amp_snapshot_body_ids] for value in values)

    def _relative_height(self):
        return root_clearance(self.robot.data.root_pos_w[:, 2],
                              self.height_scanner.data.ray_hits_w, self.scene.env_origins[:, 2])

    def reset_motion_reference(self, env_ids):
        if len(env_ids) == 0:
            return
        if not hasattr(self, "_recovery_static_com_pose"):
            self._cache_static_com_pose()
        self.robot.data._body_com_pose_b.data = self._recovery_static_com_pose
        self.robot.data._body_com_pose_b.timestamp = float("inf")
        for category, selected in (("Recovery", self.recovery_assigned[env_ids]),
                                   ("WalkandRun", ~self.recovery_assigned[env_ids])):
            ids = env_ids[selected]
            if len(ids) == 0:
                continue
            frame = self.motion_library.sample(category, len(ids))
            position = self.scene.env_origins[ids].clone()
            position[:, 2] += frame["root_pos"][:, 2]
            pose = torch.cat((position, frame["root_quat"]), -1)
            velocity = torch.cat((frame["root_lin_vel"], frame["root_ang_vel"]), -1)
            limits = self.robot.data.soft_joint_pos_limits[ids]
            joints = frame["joint_pos"].clamp(limits[..., 0], limits[..., 1])
            self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)
            self.robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
            self.robot.write_joint_state_to_sim(joints, frame["joint_vel"], env_ids=ids)
            # This SDK invalidates local COM metadata on joint-state writes,
            # although those writes only change joint positions/velocities.
            self.robot.data._body_com_pose_b.data = self._recovery_static_com_pose
            self.robot.data._body_com_pose_b.timestamp = float("inf")

    def _cache_static_com_pose(self):
        # Local COM offsets do not change as joints move. The pinned parent
        # randomizes physical properties only at startup; reject a runtime
        # property randomizer rather than silently making this cache stale.
        for mode in self.event_manager.available_modes:
            if mode in ("startup", "prestartup"):
                continue
            for name in self.event_manager.active_terms[mode]:
                func = self.event_manager.get_term_cfg(name).func
                if any(word in func.__name__.lower() for word in ("mass", "inertia", "center_of_mass", "body_com")):
                    raise ValueError("Static COM caching requires no runtime mass/inertia/COM mutation")
        data = self.robot.data
        # Refresh from PhysX *after* all startup randomizers, retaining distinct
        # values for every environment. Only the SDK read-cache timestamp is
        # pinned; no mass, inertia, COM, collider or solver parameter is written.
        data._body_com_pose_b.timestamp = float("-inf")
        self._recovery_static_com_pose = data.body_com_pose_b
        data._body_com_pose_b.timestamp = float("inf")

    def reset(self, env_ids):
        # The final reset event writes the motion, then the parent synchronizes
        # once and resets feedback/force caches using the final sampled state.
        super().reset(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "motion_library"):
            return
        self.recovery_active[env_ids] = self.recovery_assigned[env_ids]
        self.recovery_reward_mask[env_ids] = self.recovery_active[env_ids]
        self.recovery_stable[env_ids] = 0
        self.fall_delay_counters[env_ids] = 0
        self.recovery_success_counted[env_ids] = False
        self.recovery_support_stable[env_ids] = 0
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        height = self.robot.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]
        fallen = recovery_fallen(gravity_z, height)
        self.recovery_initial_fallen[env_ids] = fallen[env_ids] & self.recovery_assigned[env_ids]
        self.recovery_initial_deep[env_ids] = ((gravity_z[env_ids] > -.5) & (height[env_ids] < .55)
                                              & self.recovery_assigned[env_ids])
        self.recovery_sampled_total += self.recovery_assigned[env_ids].sum()
        self.recovery_fallen_total += self.recovery_initial_fallen[env_ids].sum()
        self.recovery_deep_total += self.recovery_initial_deep[env_ids].sum()

    def _compute_forward_obstacle_height(self):
        super()._compute_forward_obstacle_height()
        if not hasattr(self, "recovery_active"):
            return
        active, stable = phase_update(self.recovery_active, self.recovery_stable,
            self.robot.data.projected_gravity_b[:, 2], self._relative_height(), CONTRACT["initial_position"][2])
        active &= self.recovery_assigned
        data = self.robot.data
        upright = ((data.projected_gravity_b[:, 2] < -.94)
                   & (self._relative_height() > CONTRACT["initial_position"][2]*.8))
        support = self.contact_sensor.data.net_forces_w[:, self._force_peak_sensor_ids, 2].clamp(min=0).amax(-1) > 30.
        slow = (data.root_lin_vel_w[:, :2].norm(dim=-1) < .5) & (data.root_ang_vel_w.norm(dim=-1) < 1.5)
        self.recovery_support_stable = torch.where(upright & support & slow, self.recovery_support_stable+1, 0)
        success = (self.recovery_support_stable >= 50) & self.recovery_initial_fallen & ~self.recovery_success_counted
        self.recovery_success_total += success.sum()
        self.recovery_deep_success_total += (success & self.recovery_initial_deep).sum()
        self.recovery_success_counted |= success
        self.recovery_active.copy_(active)
        self.recovery_stable.copy_(stable)
        self.recovery_reward_mask.copy_(active)

    def check_reset(self):
        done, timeout = super().check_reset()
        # Include newly fallen samples as well as those recovering at step start.
        # Capture terminal link states here, before the parent autoresets them.
        ids = (self._amp_active_before | self.recovery_active).nonzero(as_tuple=False).flatten()
        if ids.numel():
            before = selected_amp_features(self._amp_bodies_before, ids)
            after_snapshot = self.amp_snapshot(ids)
            after = amp_features(*after_snapshot, range(len(BODY_NAMES)), len(BODY_NAMES))
        else:
            before = after = torch.empty((0, len(BODY_NAMES)*15), device=self.device)
        self._amp_transition_before_reset = {"state": before, "next_state": after, "env_ids": ids}
        # AMP permits hands/knees/torso support while recovering. Its delayed
        # fall predicate is low height / large tilt, not DWAQ torso-contact-only.
        fallen = torch.where(self.recovery_assigned,
            recovery_fallen(self.robot.data.projected_gravity_b[:, 2], self._relative_height()), done & ~timeout)
        result, counts = delayed_fall(fallen, timeout, self.recovery_assigned,
                                      self.fall_delay_counters, SETTINGS["fall_delay_steps"])
        self.fall_delay_counters.copy_(counts)
        return result, timeout

    def step(self, actions):
        self._amp_bodies_before = self.amp_snapshot()
        self._amp_active_before = self.recovery_active.clone()
        obs, task_reward, done, extras = super().step(actions)
        transition = self._amp_transition_before_reset
        extras["recovery_v2_amp"] = transition
        if self.recovery_v2_amp is not None:
            reward, style = self.recovery_v2_amp.reward(
                transition["state"], transition["next_state"], task_reward, transition["env_ids"])
            self.recovery_v2_amp_style_mean = style.mean() if style.numel() else task_reward.new_zeros(())
        else:
            reward = task_reward
        return obs, reward, done, extras
