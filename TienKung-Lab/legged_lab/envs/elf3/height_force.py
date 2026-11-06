"""Isolated height/force variant. Do not edit active motion-quality task sources."""
from copy import copy,deepcopy
import hashlib
from pathlib import Path

import torch
from isaaclab.managers import RewardTermCfg
from rsl_rl.runners import DWAQOnPolicyRunner

from legged_lab.envs.elf3.ankle_cost import Elf3DwaqAnkleCostEnv,Elf3DwaqAnkleCostRunner
from legged_lab.envs.elf3.ankle_cost_rewards import FOOT_BODY_NAMES
from legged_lab.envs.elf3.feedback_contract import canonical_sha256,validate_checkpoint_contract
from legged_lab.envs.elf3.motion_quality import (
    Elf3DwaqMotionQualityRunner,make_configs as make_parent_configs,validate_motion_quality_configuration,
)
from legged_lab.envs.elf3.height_force_contract import SETTINGS,height_force_contract
from legged_lab.envs.elf3.height_force_reference import default_height_reference
from legged_lab.envs.elf3.height_force_rewards import support_surface,moving_base_height_l2


def make_configs(num_envs=16384,device="cuda:0",run_name="height_force_v1"):
    cfg,agent=make_parent_configs(num_envs,device,run_name)
    delattr(cfg.reward,"support_knee_excess_l2")
    cfg.reward.feet_force.weight=SETTINGS["feet_force_weight"]
    cfg.reward.base_height=RewardTermCfg(func=moving_base_height_l2,weight=SETTINGS["height_weight"],params={
        "target_height":default_height_reference()["nominal_height_m"],
        "deadband":SETTINGS["height_deadband_m"],"command_epsilon":SETTINGS["command_epsilon"]})
    agent.runner_class_name="Elf3DwaqHeightForceRunner"
    return cfg,agent


def projected_motion_cfg(cfg):
    result=deepcopy(cfg)
    reference,_=make_parent_configs(cfg.scene.num_envs,cfg.device)
    delattr(result.reward,"base_height")
    result.reward.support_knee_excess_l2=reference.reward.support_knee_excess_l2
    result.reward.feet_force.weight=SETTINGS["feet_force_previous_weight"]
    return result


def validate_height_force_configuration(cfg,train_cfg):
    if train_cfg.get("runner_class_name")!="Elf3DwaqHeightForceRunner":
        raise ValueError("Use the independent height-force runner")
    if hasattr(cfg.reward,"support_knee_excess_l2"):
        raise ValueError("Remove the knee-threshold reward, do not stack it with height")
    expected,_=make_configs(cfg.scene.num_envs,cfg.device)
    for name in ("base_height","feet_force"):
        term,ref=getattr(cfg.reward,name,None),getattr(expected.reward,name)
        if term is None or term.func is not ref.func or canonical_sha256(term.to_dict())!=canonical_sha256(ref.to_dict()):
            raise ValueError(f"Wrong height-force reward/weight/parameters: {name}")
    parent=deepcopy(train_cfg);parent["runner_class_name"]="Elf3DwaqMotionQualityRunner"
    audit=validate_motion_quality_configuration(projected_motion_cfg(cfg),parent)
    if not cfg.scene.height_scanner.enable_height_scan or not cfg.scene.height_scanner.critic_only:
        raise ValueError("Require existing critic-only terrain scan; never add actor heights")
    return {"version":SETTINGS["version"],"removed_rewards":["support_knee_excess_l2"],
            "height_reference":default_height_reference(),"height_weight":SETTINGS["height_weight"],
            "height_deadband_m":SETTINGS["height_deadband_m"],"feet_force_weight":SETTINGS["feet_force_weight"],
            "zero_command_height_reward":False,"parent_audit":audit,"all_other_configuration_preserved":True}


class Elf3DwaqHeightForceEnv(Elf3DwaqAnkleCostEnv):
    def init_buffers(self):
        super().init_buffers()
        self._height_force_sensor_ids,names=self.contact_sensor.find_bodies(list(FOOT_BODY_NAMES),preserve_order=True)
        if tuple(names)!=FOOT_BODY_NAMES:
            raise ValueError("Explicit left/right contact-foot mapping required")
        self.height_force_ground=torch.zeros(self.num_envs,device=self.device)
        self.height_force_relative_height=torch.zeros_like(self.height_force_ground)
        self.height_force_terrain_gate=torch.zeros_like(self.height_force_ground)
        self.height_force_ground_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)

    def _compute_forward_obstacle_height(self):
        super()._compute_forward_obstacle_height()
        ground,gate,valid=support_surface(
            self.robot.data.body_link_pos_w[:,self._ankle_cost_feet_ids],self.height_scanner.data.ray_hits_w,
            self.contact_sensor.data.net_forces_w[:,self._height_force_sensor_ids,2],
            radius=SETTINGS["support_radius_m"],near_radius=SETTINGS["near_radius_m"],
            min_points=SETTINGS["min_points_per_foot"],flat_full=SETTINGS["flat_full_span_m"],
            flat_off=SETTINGS["flat_off_span_m"])
        self.height_force_ground.copy_(ground)
        self.height_force_relative_height.copy_(self.robot.data.root_pos_w[:,2]-ground)
        self.height_force_terrain_gate.copy_(gate)
        self.height_force_ground_valid.copy_(valid)

    def reset(self,env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self,"height_force_ground"):
            for name in ("height_force_ground","height_force_relative_height","height_force_terrain_gate","height_force_ground_valid"):
                getattr(self,name)[env_ids]=0


class Elf3DwaqHeightForceRunner(Elf3DwaqMotionQualityRunner):
    def __init__(self,env,train_cfg,log_dir=None,device="cpu",checkpoint_dir=None):
        audit=validate_height_force_configuration(env.cfg,train_cfg)
        parent_env,parent_agent=copy(env),deepcopy(train_cfg)
        parent_env.cfg=projected_motion_cfg(env.cfg)
        parent_agent["runner_class_name"]="Elf3DwaqMotionQualityRunner"
        super().__init__(parent_env,parent_agent,log_dir,device,checkpoint_dir)
        self.env,self.cfg,self.configuration_audit=env,deepcopy(train_cfg),audit

    def contract(self):
        result=Elf3DwaqAnkleCostRunner.contract(self)
        result["height_force_experiment"]=height_force_contract()
        return result

    def motion_parent_contract(self,num_envs=None,cached=False):
        parent=copy(self);parent.env=copy(self.env);parent.env.cfg=projected_motion_cfg(self.env.cfg)
        parent.cfg=deepcopy(self.cfg);parent.cfg["runner_class_name"]="Elf3DwaqMotionQualityRunner"
        if num_envs is not None:parent.env.cfg.scene.num_envs=num_envs
        result=Elf3DwaqMotionQualityRunner.contract(parent)
        if cached:
            from legged_lab.envs.elf3.motion_quality_cached import performance_contract
            result["motion_quality_performance"]=performance_contract()
        return result

    def initialize_from_motion_quality(self,path,expected_sha256):
        if self.current_learning_iteration!=0 or self.alg.optimizer.state:
            raise ValueError("Reward promotion requires a fresh runner")
        path=Path(path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError("Pinned parent SHA256 mismatch")
        saved=torch.load(path,map_location="cpu",weights_only=False)
        cached="motion_quality_performance" in saved["infos"]["elf3_contract"]
        try:validate_checkpoint_contract(saved,self.motion_parent_contract(cached=cached))
        except ValueError:validate_checkpoint_contract(saved,self.motion_parent_contract(num_envs=16384,cached=cached))
        old,current=deepcopy(saved["infos"]["training_recipe"]),deepcopy(self.cfg)
        current["runner_class_name"]="Elf3DwaqMotionQualityRunner"
        for item in (old,current):
            for key in ("run_name","device","max_iterations","resume","load_run","load_checkpoint"):item.pop(key,None)
        if canonical_sha256(old)!=canonical_sha256(current) or not saved["optimizer_state_dict"]["state"]:
            raise ValueError("Keep the complete PPO recipe and trained Adam state")
        DWAQOnPolicyRunner.load(self,str(path),load_optimizer=True)
        self.current_learning_iteration=int(saved["iter"])+1
        self._assert_tree_equal(self.alg.policy.state_dict(),saved["model_state_dict"],"model")
        self._assert_tree_equal(self.alg.optimizer.state_dict(),saved["optimizer_state_dict"],"Adam")

    def log(self,locs,width=80,pad=35):
        Elf3DwaqAnkleCostRunner.log(self,locs,width,pad)
        env=self.env
        moving=env.command_generator.command.abs().amax(-1)>SETTINGS["command_epsilon"]
        active=moving & env.height_force_ground_valid & (env.height_force_terrain_gate>0)
        count=active.sum().clamp(min=1)
        cost=moving_base_height_l2(env,**env.cfg.reward.base_height.params)
        values=torch.stack([active.float().mean(),
            torch.where(active,env.height_force_relative_height,0.).sum()/count,
            cost.mean(),env.height_force_terrain_gate.mean()]).detach().cpu().tolist()
        for key,value in zip(("active_moving_fraction","active_relative_height_m","raw_height_shortfall_m2","terrain_gate_mean"),values):
            self.writer.add_scalar("HeightForce/snapshot_"+key,value,locs["it"])
