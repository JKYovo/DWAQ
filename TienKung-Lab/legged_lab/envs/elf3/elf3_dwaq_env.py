"""Keep upstream rollout semantics while exposing Amp_mjlab's joint order.

PhysX orders articulated joints by its tree traversal. All policy-facing joint
features and actions use the explicitly named ELF3 order instead.
"""
import torch

from legged_lab.assets.elf3 import CONTRACT, JOINT_NAMES, FOOT_BODIES
from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv


class Elf3DwaqEnv(G1DwaqEnv):
    def init_buffers(self):
        self.nominal_root_height = CONTRACT["initial_position"][2]
        joint_ids, names = self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES or len(self.robot.joint_names) != len(JOINT_NAMES):
            raise ValueError(f"ELF3 joint contract mismatch: {self.robot.joint_names}")
        if self.robot.body_names[0] != CONTRACT["physical_root"]:
            raise ValueError(f"ELF3 floating root mismatch: {self.robot.body_names[0]}")
        self.policy_joint_ids = torch.tensor(joint_ids, device=self.device)
        self.sim_to_policy = torch.argsort(self.policy_joint_ids)
        # Contact sensors omit collisionless ELF3 links, so their body indices
        # cannot be used to index the full articulation state.
        self.feet_body_ids, _ = self.robot.find_bodies(list(FOOT_BODIES), preserve_order=True)
        super().init_buffers()
        self.action_scale = torch.tensor(
            [CONTRACT["action_scale"][name] for name in self.robot.joint_names], device=self.device
        )
        feet_names = tuple(self.contact_sensor.body_names[i] for i in self.feet_cfg.body_ids)
        if feet_names != FOOT_BODIES:
            raise ValueError(f"ELF3 feet must be left then right: {feet_names}")

    def init_obs_buffer(self):
        self.feet_cfg.body_ids, _ = self.contact_sensor.find_bodies(list(FOOT_BODIES), preserve_order=True)
        super().init_obs_buffer()

    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        # Upstream builds privileged observations from the same clean actor frame.
        actor = actor.clone()
        critic = critic.clone()
        for start in (9, 38, 67):
            actor[:, start:start + 29] = actor[:, start:start + 29][:, self.policy_joint_ids]
        critic[:, :actor.shape[-1]] = actor
        return actor, critic

    def step(self, actions):
        return super().step(actions[:, self.sim_to_policy])

    def close(self):
        """Unsubscribe standalone timeline callbacks before closing the stage."""
        if not getattr(self, "_closed", False):
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self._closed = True
