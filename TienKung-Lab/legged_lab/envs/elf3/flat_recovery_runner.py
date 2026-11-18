"""Strictly audited flat recovery migration, preserving the original50000 policy/Adam."""
from copy import copy,deepcopy

from legged_lab.envs.elf3 import recovery_v2_runner as v2
from legged_lab.envs.elf3.flat_recovery_core import SETTINGS,recovery_contract
from legged_lab.envs.elf3.flat_recovery_env import make_configs,projected_v2_cfg,projected_v2_recipe
from legged_lab.envs.elf3.feedback_contract import canonical_sha256


def validate_configuration(cfg,recipe):
    expected,_ = make_configs(cfg.recovery["data_directory"],cfg.scene.num_envs,cfg.device)
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError("Unexpected flat recovery/physical/reward override")
    if recipe.get("runner_class_name") != "Elf3DwaqFlatRecoveryRunner" or recipe.get("recovery") != SETTINGS:
        raise ValueError("Require the explicit flat recovery40% configuration")
    return {"flat_recovery_fraction":.4,"normal_terrain_fraction":.6,
            "parent_recovery_audit":v2.validate_configuration(projected_v2_cfg(cfg),projected_v2_recipe(recipe))}


class Elf3DwaqFlatRecoveryRunner(v2.Elf3DwaqRecoveryRunner):
    def __init__(self,env,train_cfg,log_dir=None,device="cpu",checkpoint_dir=None):
        audit = validate_configuration(env.cfg,train_cfg)
        projected = copy(env)
        projected.cfg = projected_v2_cfg(env.cfg)
        super().__init__(projected,projected_v2_recipe(train_cfg),log_dir,device,checkpoint_dir)
        self.env,self.cfg = env,deepcopy(train_cfg)
        self.train_cfg,self.configuration_audit = self.cfg,audit
        env.configure_reward_curriculum(self.reward_curriculum_settings,0)
        env.recovery_v2_amp = self.amp
        self.alg.configure_recovery(env,self.amp)

    def contract(self):
        view = copy(self)
        view.env = copy(self.env)
        view.env.cfg,view.cfg = projected_v2_cfg(self.env.cfg),projected_v2_recipe(self.cfg)
        result = v2.Elf3DwaqRecoveryRunner.contract(view)
        result["flat_recovery_experiment"] = recovery_contract(self.env.motion_library.identity())
        return result

    def parent_contract(self,num_envs=16384):
        view = copy(self)
        view.env = copy(self.env)
        view.env.cfg,view.cfg = projected_v2_cfg(self.env.cfg),projected_v2_recipe(self.cfg)
        return v2.Elf3DwaqRecoveryRunner.parent_contract(view,num_envs)

    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        self.writer.add_scalar("RecoveryFlat/assigned_envs",len(self.env.recovery_flat_ids),locs["it"])
