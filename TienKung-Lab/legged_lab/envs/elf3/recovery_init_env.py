"""Expand only recovery resets; walking resets and the flat-v3 task stay intact."""
from copy import deepcopy

import torch

from legged_lab.envs.elf3.flat_recovery_env import (
    Elf3DwaqFlatRecoveryEnv, make_configs as flat_configs,
)
from legged_lab.envs.elf3.recovery_init_core import TASK, SETTINGS, InitializationBank


def make_configs(data_directory, bank_path, num_envs=16384, device="cuda:0", run_name="init75zero_resume60000"):
    cfg, agent = flat_configs(data_directory, num_envs, device, run_name)
    cfg.recovery_initialization = {**deepcopy(SETTINGS), "bank_path": str(bank_path)}
    agent.recovery_initialization = deepcopy(SETTINGS)
    agent.runner_class_name = "Elf3DwaqRecoveryInitRunner"
    agent.experiment_name = agent.wandb_project = TASK
    return cfg, agent


def projected_flat_cfg(cfg):
    result = deepcopy(cfg)
    delattr(result, "recovery_initialization")
    return result


def projected_flat_recipe(recipe):
    result = deepcopy(recipe)
    del result["recovery_initialization"]
    result.update(runner_class_name="Elf3DwaqFlatRecoveryRunner",
                  experiment_name="elf3_dwaq_recovery40_flat_v3", wandb_project="elf3_dwaq_recovery40_flat_v3")
    return result


class Elf3DwaqRecoveryInitEnv(Elf3DwaqFlatRecoveryEnv):
    def init_buffers(self):
        super().init_buffers()
        self.initialization_bank = InitializationBank(self.cfg.recovery_initialization["bank_path"],
                                                     self.robot.joint_names, self.device)
        self.initialization_pose_kind = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.initialization_velocity_kind = torch.zeros_like(self.initialization_pose_kind)
        self.initialization_requested_root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        self.initialization_requested_joint_velocity = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.initialization_sampled = torch.zeros(9, device=self.device)
        self.initialization_fallen = torch.zeros_like(self.initialization_sampled)
        self.initialization_recovered = torch.zeros_like(self.initialization_sampled)

    def reset_motion_reference(self, env_ids):
        if not len(env_ids):
            return
        walking = env_ids[~self.recovery_assigned[env_ids]]
        # Preserve the existing walking sampler, reference poses and velocities.
        super().reset_motion_reference(walking)
        ids = env_ids[self.recovery_assigned[env_ids]]
        if not len(ids):
            return
        if not hasattr(self, "_recovery_static_com_pose"):
            self._cache_static_com_pose()
        reference = self.motion_library.sample("Recovery", len(ids))
        frame, kinds, velocity_kind = self.initialization_bank.sample(reference)
        position = self.scene.env_origins[ids].clone()
        position[:, 2] += frame["root_pos"][:, 2]
        pose = torch.cat((position, frame["root_quat"]), -1)
        velocity = torch.cat((frame["root_lin_vel"], frame["root_ang_vel"]), -1)
        limits = self.robot.data.soft_joint_pos_limits[ids]
        joints = frame["joint_pos"].clamp(limits[..., 0], limits[..., 1])
        self.robot.data._body_com_pose_b.data = self._recovery_static_com_pose
        self.robot.data._body_com_pose_b.timestamp = float("inf")
        self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        self.robot.write_joint_state_to_sim(joints, frame["joint_vel"], env_ids=ids)
        self.robot.data._body_com_pose_b.data = self._recovery_static_com_pose
        self.robot.data._body_com_pose_b.timestamp = float("inf")
        self.initialization_pose_kind[ids] = kinds
        self.initialization_velocity_kind[ids] = velocity_kind
        self.initialization_requested_root_velocity[ids] = velocity
        self.initialization_requested_joint_velocity[ids] = frame["joint_vel"]

    def reset(self, env_ids):
        super().reset(env_ids)
        if not len(env_ids) or not hasattr(self, "initialization_bank"):
            return
        ids = env_ids[self.recovery_assigned[env_ids]]
        groups = self.initialization_pose_kind[ids]*3+self.initialization_velocity_kind[ids]
        self.initialization_sampled += torch.bincount(groups, minlength=9)
        self.initialization_fallen += torch.bincount(groups[self.recovery_initial_fallen[ids]], minlength=9)

    def _compute_forward_obstacle_height(self):
        before = self.recovery_success_counted.clone() if hasattr(self, "initialization_bank") else None
        super()._compute_forward_obstacle_height()
        if before is not None:
            ids = (~before & self.recovery_success_counted).nonzero().flatten()
            groups = self.initialization_pose_kind[ids]*3+self.initialization_velocity_kind[ids]
            self.initialization_recovered += torch.bincount(groups, minlength=9)
