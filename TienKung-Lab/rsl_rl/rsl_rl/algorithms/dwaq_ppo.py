# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, Legged Lab,
# and DreamWaQ Projects, with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic_DWAQ
from rsl_rl.storage import RolloutStorageDWAQ


class DWAQPPO:
    """Proximal Policy Optimization with Deep Variational Autoencoder for Walking (DWAQ).
    
    This algorithm extends PPO with a β-VAE based context encoder for blind locomotion.
    The encoder infers latent states and velocity from observation history.
    
    Reference: DreamWaQ (https://github.com/Gepetto/DreamWaQ)
    """

    policy: ActorCritic_DWAQ
    """The actor critic module with DWAQ context encoder."""

    def __init__(
        self,
        policy: ActorCritic_DWAQ,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        max_learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        obs_dim: int = 45,
    ):
        """Initialize the DWAQ PPO algorithm.
        
        Args:
            policy: The actor-critic network with DWAQ context encoder.
            num_learning_epochs: Number of learning epochs per update.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda parameter.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            learning_rate: Learning rate for optimizer.
            min_learning_rate: Lower bound for the adaptive learning rate.
            max_learning_rate: Upper bound for the adaptive learning rate.
            max_grad_norm: Maximum gradient norm for clipping.
            use_clipped_value_loss: Whether to use clipped value loss.
            schedule: Learning rate schedule ("fixed" or "adaptive").
            desired_kl: Desired KL divergence for adaptive schedule.
            device: Device to run on.
            obs_dim: Observation dimension (without latent code).
        """
        # Device configuration
        self.device = device
        self.obs_dim = obs_dim

        # Learning rate schedule parameters
        self.desired_kl = desired_kl
        self.schedule = schedule
        if not 0 < min_learning_rate <= max_learning_rate:
            raise ValueError("Expected 0 < min_learning_rate <= max_learning_rate")
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.learning_rate = min(max(learning_rate, min_learning_rate), max_learning_rate)

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorageDWAQ | None = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorageDWAQ.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.upper_body_symmetry = None

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name} detected; refusing to update the policy")

    def validate_policy_parameters(self) -> None:
        """Fail before a corrupt policy can be used or saved."""
        for name, parameter in self.policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite policy parameter: {name}")

    def configure_upper_body_symmetry(
        self,
        mirror_observations,
        mirror_history,
        mirror_actions,
        action_ids,
        coefficient: float,
        pose_coefficient: float = 0.0,
    ) -> None:
        """Enable optional arm mirror and default-pose losses; standard DWAQ remains unchanged."""
        if coefficient <= 0:
            raise ValueError("Upper-body mirror-loss coefficient must be positive")
        if pose_coefficient < 0:
            raise ValueError("Upper-body pose-loss coefficient cannot be negative")
        self.upper_body_symmetry = {
            "mirror_observations": mirror_observations,
            "mirror_history": mirror_history,
            "mirror_actions": mirror_actions,
            "action_ids": tuple(action_ids),
            "coefficient": coefficient,
            "pose_coefficient": pose_coefficient,
        }

    def _upper_body_symmetry_loss(self, observations, observation_history):
        if self.upper_body_symmetry is None:
            return None
        cfg = self.upper_body_symmetry
        original_mean = self.policy.deterministic_action_mean(observations, observation_history)
        mirrored_mean = self.policy.deterministic_action_mean(
            cfg["mirror_observations"](observations),
            cfg["mirror_history"](observation_history),
        )
        mirrored_target = cfg["mirror_actions"](original_mean).detach()
        action_ids = cfg["action_ids"]
        return nn.functional.mse_loss(mirrored_mean[:, action_ids], mirrored_target[:, action_ids])

    def _upper_body_pose_loss(self, observations, observation_history):
        """Penalize deterministic arm action offsets while leaving waist and legs unconstrained."""
        if self.upper_body_symmetry is None or self.upper_body_symmetry["pose_coefficient"] == 0:
            return None
        action_ids = self.upper_body_symmetry["action_ids"]
        action_mean = self.policy.deterministic_action_mean(observations, observation_history)
        return torch.square(action_mean[:, action_ids]).mean()

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        obs_hist_shape: list[int],
        action_shape: list[int],
    ) -> None:
        """Initialize the rollout storage.
        
        Args:
            num_envs: Number of parallel environments.
            num_transitions_per_env: Number of transitions per environment per update.
            actor_obs_shape: Shape of actor observations.
            critic_obs_shape: Shape of critic observations (privileged).
            obs_hist_shape: Shape of observation history for context encoder.
            action_shape: Shape of actions.
        """
        self.storage = RolloutStorageDWAQ(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_hist_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        """Set the policy to evaluation mode."""
        self.policy.eval()

    def train_mode(self) -> None:
        """Set the policy to training mode."""
        self.policy.train()

    def broadcast_parameters(self) -> None:
        """Broadcast parameters to all processes for multi-GPU training."""
        if torch.distributed.is_initialized():
            for param in self.policy.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        prev_critic_obs: torch.Tensor,
        obs_history: torch.Tensor,
    ) -> torch.Tensor:
        """Compute actions for the given observations.
        
        Args:
            obs: Current actor observations.
            critic_obs: Current critic observations (privileged).
            prev_critic_obs: Previous critic observations for velocity target.
            obs_history: Observation history for context encoder.
            
        Returns:
            Actions to execute in the environment.
        """
        # Compute the actions and values
        # Clone tensors to avoid "Inference tensors cannot be saved for backward" error
        self.transition.actions = self.policy.act(obs, obs_history).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # Record observations before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs
        self.transition.prev_critic_obs = prev_critic_obs
        
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        """Process the environment step results.
        
        Args:
            rewards: Rewards from the environment.
            dones: Done flags from the environment.
            infos: Additional info from the environment.
        """
        # Record rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute returns and advantages using GAE.
        
        Args:
            last_critic_obs: Last critic observations for bootstrapping.
        """
        # Clone to avoid "Inference tensors cannot be saved for backward" error
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, beta: float = 1.0) -> dict[str, float]:
        """Perform a PPO update step.
        
        Args:
            beta: Weight for the VAE KL divergence loss (β-VAE).
            
        Returns:
            Dictionary containing the mean losses.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_autoenc_loss = 0.0
        mean_symmetry_loss = 0.0 if self.upper_body_symmetry is not None else None
        mean_pose_loss = (
            0.0
            if self.upper_body_symmetry is not None and self.upper_body_symmetry["pose_coefficient"] > 0
            else None
        )

        # Generator for mini batches
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            critic_obs_batch,
            prev_critic_obs_batch,
            obs_hist_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # Recompute actions log prob and values for current batch
            self.policy.act(
                obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive learning rate based on KL divergence
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    self._require_finite("adaptive KL", kl_mean)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # β-VAE loss for context encoder
            (
                code,
                code_vel,
                decode,
                mean_vel,
                logvar_vel,
                mean_latent,
                logvar_latent,
            ) = self.policy.cenet_forward(obs_hist_batch)

            # Velocity target from CURRENT privileged observations (not prev)
            # This fixes 1-step time alignment issue:
            # - obs_hist is updated AFTER physics step, so its latest frame is obs_t
            # - Therefore VAE should predict v_t, which is in critic_obs_batch (not prev)
            # In privileged obs, velocity is at [obs_dim:obs_dim+3] since
            # privileged_obs = [actor_obs, root_lin_vel(3), ...]
            vel_target = critic_obs_batch[:, self.obs_dim : self.obs_dim + 3].detach()
            # Decoder reconstructs the full observation (matching original DreamWaQ)
            decode_target = obs_batch[:, :self.obs_dim]
            vel_target.requires_grad = False
            decode_target.requires_grad = False

            # Autoencoder loss: velocity prediction + reconstruction + KL divergence
            # 
            # Normalize KL per sample so its scale does not grow with the number
            # of environments or mini-batch size.
            logvar_latent_clamped = torch.clamp(logvar_latent, min=-10.0, max=10.0)
            kl_divergence = -0.5 * (
                1 + logvar_latent_clamped - mean_latent.pow(2) - logvar_latent_clamped.exp()
            ).sum(dim=-1).mean()
            autoenc_loss = (
                nn.MSELoss()(mean_vel, vel_target)
                + nn.MSELoss()(decode, decode_target)
                + beta * kl_divergence
            )

            # Surrogate loss (PPO clipped objective)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + autoenc_loss
            )
            symmetry_loss = self._upper_body_symmetry_loss(obs_batch, obs_hist_batch)
            if symmetry_loss is not None:
                loss = loss + self.upper_body_symmetry["coefficient"] * symmetry_loss
            pose_loss = self._upper_body_pose_loss(obs_batch, obs_hist_batch)
            if pose_loss is not None:
                loss = loss + self.upper_body_symmetry["pose_coefficient"] * pose_loss

            # Gradient step
            self._require_finite("total loss", loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.validate_policy_parameters()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_pose_loss is not None:
                mean_pose_loss += pose_loss.item()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_autoenc_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_pose_loss is not None:
            mean_pose_loss /= num_updates


        # Clear storage
        self.storage.clear()

        # Construct loss dictionary (Isaac Lab style)
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "autoencoder": mean_autoenc_loss,
        }
        if mean_symmetry_loss is not None:
            loss_dict["upper_body_symmetry"] = mean_symmetry_loss
        if mean_pose_loss is not None:
            loss_dict["upper_body_pose"] = mean_pose_loss

        return loss_dict
