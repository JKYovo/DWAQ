"""Flat-only recovery40%; the original terrain remains intact for walking60%."""
from copy import deepcopy

import numpy as np
import torch
import trimesh

from legged_lab.terrains.staged_terrain_generator import StagedStairsTerrainGenerator
from legged_lab.envs.elf3 import recovery_v2_env as v2
from legged_lab.envs.elf3.recovery_v2_core import SETTINGS as V2_SETTINGS, TASK as V2_TASK
from legged_lab.envs.elf3.flat_recovery_core import TASK, SETTINGS, flat_layout, SplitCommands


class FlatRecoveryTerrainGenerator(StagedStairsTerrainGenerator):
    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        _, bounds = flat_layout(cfg.recovery_pad_count, cfg.size, cfg.num_rows, cfg.border_width)
        if bounds[0,0] <= self.terrain_mesh.bounds[1,0]+8.:
            raise ValueError("Recovery pad must be disjoint from the original terrain and border")
        low, high = bounds
        vertices = np.array([[low[0],low[1],0.],[high[0],low[1],0.],
                             [high[0],high[1],0.],[low[0],high[1],0.]])
        pad = trimesh.Trimesh(vertices=vertices, faces=[[0,1,2],[0,2,3]], process=False)
        # A single imported mesh gives both PhysX contacts and the existing
        # height scanner exactly the same added surface. No global plane
        # under the old stairs, and no change to any original terrain vertex.
        original_vertices,original_faces = self.terrain_mesh.vertices,self.terrain_mesh.faces
        self.terrain_mesh = trimesh.util.concatenate([self.terrain_mesh,pad])
        assert np.array_equal(self.terrain_mesh.vertices[:len(original_vertices)],original_vertices)
        assert np.array_equal(self.terrain_mesh.faces[:len(original_faces)],original_faces)
        print(f"[FLAT_RECOVERY] Original terrain unchanged: {len(original_vertices)} vertices, "
              f"{len(original_faces)} faces; added2 flat faces for {cfg.recovery_pad_count} environments",flush=True)


def push_walking_only(env, env_ids, velocity_range, asset_cfg=None):
    from isaaclab.envs.mdp.events import push_by_setting_velocity
    if env_ids is None:
        env_ids = torch.arange(env.num_envs,device=env.device)
    env_ids = env_ids[~env.recovery_assigned[env_ids]]
    if env_ids.numel():
        if asset_cfg is None:
            push_by_setting_velocity(env,env_ids,velocity_range)
        else:
            push_by_setting_velocity(env,env_ids,velocity_range,asset_cfg)


def make_configs(data_directory,num_envs=16384,device="cuda:0",run_name="flat_recovery40_from50000"):
    cfg, agent = v2.make_configs(data_directory,num_envs,device,run_name)
    count = int(num_envs*SETTINGS["recovery_fraction"])
    if count < 1:
        raise ValueError("Need enough environments for a40% recovery split")
    cfg.recovery = {**deepcopy(SETTINGS),"data_directory":str(data_directory)}
    cfg.scene.terrain_generator.class_type = FlatRecoveryTerrainGenerator
    cfg.scene.terrain_generator.recovery_pad_count = count
    cfg.domain_rand.events.push_robot.func = push_walking_only
    agent.runner_class_name = "Elf3DwaqFlatRecoveryRunner"
    agent.experiment_name = agent.wandb_project = TASK
    agent.recovery = deepcopy(SETTINGS)
    return cfg,agent


def projected_v2_cfg(cfg):
    result = deepcopy(cfg)
    reference,_ = v2.make_configs(cfg.recovery["data_directory"],cfg.scene.num_envs,cfg.device)
    result.recovery = deepcopy(reference.recovery)
    result.scene.terrain_generator.class_type = reference.scene.terrain_generator.class_type
    delattr(result.scene.terrain_generator,"recovery_pad_count")
    result.domain_rand.events.push_robot.func = reference.domain_rand.events.push_robot.func
    return result


def projected_v2_recipe(recipe):
    result = deepcopy(recipe)
    result.update(runner_class_name="Elf3DwaqRecoveryRunner",experiment_name=V2_TASK,wandb_project=V2_TASK,
                  recovery=deepcopy(V2_SETTINGS))
    return result


class Elf3DwaqFlatRecoveryEnv(v2.Elf3DwaqRecoveryEnv):
    def init_buffers(self):
        super().init_buffers()
        cfg = self.cfg.scene.terrain_generator
        self.recovery_flat_ids = self.recovery_assigned.nonzero().flatten()
        origins,bounds = flat_layout(len(self.recovery_flat_ids),cfg.size,cfg.num_rows,cfg.border_width)
        self.recovery_flat_origins = torch.as_tensor(origins,device=self.device)
        self.recovery_flat_bounds = torch.as_tensor(bounds,device=self.device)
        self.scene.env_origins[self.recovery_flat_ids] = self.recovery_flat_origins
        self.command_generator = SplitCommands(self.command_generator,self.recovery_assigned)
        self.command_generator.enforce()

    def update_terrain_levels(self,env_ids):
        normal = env_ids[~self.recovery_assigned[env_ids]]
        if normal.numel():
            super().update_terrain_levels(normal)
        # Recovery roots never enter the walking progress/distance curriculum.
        normal_levels = self.scene.terrain.terrain_levels[~self.recovery_assigned]
        return {"Curriculum/terrain_levels":normal_levels.float().mean()}
