"""Private stand-v2 auxiliaries, isolated from PPO likelihoods and deployment.

The current-zero-command label is captured BEFORE env.step(), stored alongside obs_t,
and shuffled using exactly the same minibatch indices. Phase augmentation is a
common rotation, not the fixed-history diagnostic previously used for analysis.
"""
from copy import deepcopy
import torch

from rsl_rl.algorithms.dwaq_ppo import DWAQPPO
from rsl_rl.storage import RolloutStorageDWAQ


ARM_ACTION_IDS = tuple(range(15, 29))


def zero_command_observation_mask(observations, epsilon=1e-6):
    """Current request only: previous moving commands never delay activation."""
    commands = observations[:, 6:9]
    return torch.isfinite(commands).all(-1) & (commands.abs().amax(dim=-1) <= epsilon)


def rotate_phase_inputs(observations, history, offset):
    """Rotate both legs and all frames equally, preserving phase differences."""
    if observations.shape[-1] != 100 or history.shape[-1] != 500:
        raise ValueError("stand-v2 requires the unchanged 100 x 5 observation contract")
    current, previous = observations.clone(), history.clone()
    rotation = observations.new_tensor(offset)
    cosine, sine = torch.cos(rotation), torch.sin(rotation)
    for value in (current, previous.reshape(previous.shape[0], 5, 100)):
        old_sin, old_cos = value[..., 96:98].clone(), value[..., 98:100].clone()
        value[..., 96:98] = old_sin * cosine + old_cos * sine
        value[..., 98:100] = old_cos * cosine - old_sin * sine
    return current, previous


class StandingRolloutStorage(RolloutStorageDWAQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standing_masks = torch.zeros(self.num_transitions_per_env, self.num_envs,
                                          device=self.device, dtype=torch.bool)
        self.batch_standing_mask = None

    def clear(self):
        super().clear()
        self.standing_masks.zero_()
        self.batch_standing_mask = None

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # This is the feed-forward base generator's ordering, with one private
        # mask indexed alongside every existing tensor. Tuple layout is intact.
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        critic = (self.privileged_observations.flatten(0, 1)
                  if self.privileged_observations is not None else observations)
        fields = (observations, critic, self.prev_critic_obs.flatten(0, 1),
                  self.observation_history.flatten(0, 1), self.actions.flatten(0, 1),
                  self.values.flatten(0, 1), self.advantages.flatten(0, 1), self.returns.flatten(0, 1),
                  self.actions_log_prob.flatten(0, 1), self.mu.flatten(0, 1), self.sigma.flatten(0, 1))
        masks = self.standing_masks.flatten(0, 1)
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                batch_idx = indices[index * mini_batch_size:(index + 1) * mini_batch_size]
                self.batch_standing_mask = masks[batch_idx]
                yield (*tuple(value[batch_idx] for value in fields), (None, None), None)


class Elf3DwaqStandPPO(DWAQPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standing_settings = None
        self.standing_env = None
        self._pending_standing_mask = None
        self._standing_metric_sum = None
        self._standing_metric_count = 0
        self.last_standing_metrics = {}

    def configure_standing(self, env, settings):
        if self.obs_dim != 100 or self.policy.context_mode != "mean":
            raise ValueError("Standing auxiliaries require the original ELF3 frame and mean context")
        self.standing_env = env
        self.standing_settings = deepcopy(settings)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                     obs_hist_shape, action_shape):
        self.storage = StandingRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,
                                             critic_obs_shape, obs_hist_shape, action_shape, self.device)

    def act(self, obs, critic_obs, prev_critic_obs, obs_history):
        if self.standing_settings is None:
            raise RuntimeError("Standing objective has not been configured")
        mask = self.standing_env.get_standing_mask().detach().clone().to(self.device)
        self._pending_standing_mask = mask & zero_command_observation_mask(
            obs, self.standing_settings["command_epsilon"])
        return super().act(obs, critic_obs, prev_critic_obs, obs_history)

    def process_env_step(self, rewards, dones, infos):
        if self._pending_standing_mask is None:
            raise RuntimeError("No pre-action standing mask; refusing a misaligned auxiliary target")
        self.storage.standing_masks[self.storage.step].copy_(self._pending_standing_mask)
        super().process_env_step(rewards, dones, infos)
        self._pending_standing_mask = None

    def _additional_loss(self, observations, observation_history, critic_observations):
        settings = self.standing_settings
        if settings is None:
            raise RuntimeError("Standing objective has not been configured")
        cfg = settings["auxiliary"]
        coefficients = (cfg["phase_invariance_coeff"], cfg["arm_pose_coeff"])
        if self.storage.batch_standing_mask is None:
            raise RuntimeError("Standing auxiliary minibatch mask is missing")
        mask = self.storage.batch_standing_mask & zero_command_observation_mask(
            observations, settings["command_epsilon"])
        if self._standing_metric_sum is None:
            self._standing_metric_sum = torch.zeros(3, device=self.device)
        zero = observations.new_zeros(())
        phase_loss = pose_loss = zero
        active = observations[mask]
        if active.shape[0] and any(value != 0 for value in coefficients):
            active_history = observation_history[mask]
            mean = self.policy.deterministic_action_mean(active, active_history)[:, ARM_ACTION_IDS]
            if coefficients[0] != 0:
                for offset in cfg["phase_offsets"]:
                    changed_obs, changed_history = rotate_phase_inputs(active, active_history, offset)
                    changed = self.policy.deterministic_action_mean(changed_obs, changed_history)[:, ARM_ACTION_IDS]
                    phase_loss = phase_loss + (changed - mean).square().mean()
                phase_loss = phase_loss / len(cfg["phase_offsets"])
            if coefficients[1] != 0:
                pose_loss = mean.square().mean()
        self._standing_metric_sum += torch.stack((phase_loss.detach(), pose_loss.detach(), mask.float().mean()))
        self._standing_metric_count += 1
        if not active.shape[0] or not any(value != 0 for value in coefficients):
            return None
        return coefficients[0] * phase_loss + coefficients[1] * pose_loss

    def update(self, beta=1.0):
        self._standing_metric_sum = torch.zeros(3, device=self.device)
        self._standing_metric_count = 0
        result = super().update(beta=beta)
        values = (self._standing_metric_sum / max(self._standing_metric_count, 1)).detach().cpu().tolist()
        self.last_standing_metrics = dict(zip(("phase_invariance", "arm_pose", "active_fraction"), values))
        return result
