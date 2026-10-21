"""Isolated ELF3 sensor-feedback experiment; physical PD/actions stay unchanged."""
from __future__ import annotations

from legged_lab.envs.elf3.elf3_dwaq_env import Elf3DwaqEnv
from legged_lab.envs.elf3.feedback import JointVelocityFeedback, resolve_feedback_settings


class Elf3DwaqFeedbackEnv(Elf3DwaqEnv):
    def init_buffers(self):
        # G1.init_buffers -> init_obs_buffer dynamically dispatches back into
        # compute_current_observations. Establish feedback before that happens.
        self.velocity_feedback_settings = resolve_feedback_settings(getattr(self.cfg, "velocity_feedback", None))
        self.velocity_feedback_model = JointVelocityFeedback(
            self.robot.data.joint_vel,
            physics_dt=self.physics_dt,
            mode=self.velocity_feedback_settings["mode"],
            cutoff_hz=self.velocity_feedback_settings["cutoff_hz"],
            model_version=self.velocity_feedback_settings["model_version"],
        )
        super().init_buffers()
        if self._num_obs != 100 or self.dwaq_obs_history_length != 5:
            raise ValueError("The ELF3 feedback experiment requires the existing 5 x 100 observation contract")
        if self.cfg.scene.height_scanner.enable_height_scan and not self.cfg.scene.height_scanner.critic_only:
            raise ValueError("ELF3 feedback actor must not receive privileged height scans")

    def _post_physics_step(self):
        super()._post_physics_step()
        # Called only after scene.update(dt=physics_dt), not by observation reads.
        self.velocity_feedback_model.update(self.robot.data.joint_vel)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset(env_ids)
        # Base reset applies reset events, writes state and calls sim.forward().
        self.velocity_feedback_model.reset(self.robot.data.joint_vel, env_ids)

    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        if self.velocity_feedback_settings["mode"] != "ideal":
            # Filter the raw velocity first. Subtract defaults and scale only at
            # observation sampling, exactly as in the original clean actor frame.
            observed_velocity = self.velocity_feedback_model.value - self.robot.data.default_joint_vel
            actor[:, 38:67] = observed_velocity[:, self.policy_joint_ids] * self.obs_scales.joint_vel
            critic[:, :actor.shape[-1]] = actor

        priv_cfg = self.cfg.scene.privileged_info
        if self.velocity_feedback_settings["latest_critic_contacts"] and priv_cfg.enable_feet_contact_force:
            # Preserve every other privileged feature and the 3-step binary
            # contact detector. Only this task's explicit 6D force uses newest.
            num_feet = len(self.feet_cfg.body_ids)
            force_start = actor.shape[-1] + 3 + num_feet
            if priv_cfg.enable_feet_info:
                force_start += self.feet_pos_in_body.shape[1] * 3 + self.feet_vel_in_body.shape[1] * 3
            feet_force = self.contact_sensor.data.net_forces_w_history[:, 0, self.feet_cfg.body_ids, :]
            critic[:, force_start:force_start + 3 * num_feet] = (
                feet_force.reshape(self.num_envs, -1) * self.obs_scales.contact_force
            )
        return actor, critic

    @property
    def feedback_update_count(self) -> int:
        return self.velocity_feedback_model.update_count
