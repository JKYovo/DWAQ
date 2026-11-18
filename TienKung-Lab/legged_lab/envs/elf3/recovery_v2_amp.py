"""Private recovery-only AMP and minibatch pose masks; ordinary DWAQPPO stays intact."""
import torch
from torch import nn

from rsl_rl.algorithms.dwaq_ppo import DWAQPPO as ParentPPO
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.storage.rollout_storage_dwaq import RolloutStorageDWAQ
from legged_lab.envs.elf3.recovery_v2_core import SETTINGS


class RecoveryStorage(RolloutStorageDWAQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_masks = torch.zeros(self.num_transitions_per_env, self.num_envs,
                                          dtype=torch.bool, device=self.device)
        self.pending_mask = self.batch_recovery_mask = None

    def add_transitions(self, transition):
        if self.pending_mask is None or self.pending_mask.shape != (self.num_envs,):
            raise ValueError("Missing pre-action recovery mask")
        self.recovery_masks[self.step].copy_(self.pending_mask)
        super().add_transitions(transition)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Same permutation, flattening and 13-field tuple as the parent storage.
        size = self.num_envs*self.num_transitions_per_env // num_mini_batches
        indices = torch.randperm(size*num_mini_batches, device=self.device)
        names = ("observations", "privileged_observations", "prev_critic_obs", "observation_history",
                 "actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma")
        arrays = [getattr(self, n).flatten(0, 1) for n in names]
        masks = self.recovery_masks.flatten()
        try:
            for _ in range(num_epochs):
                for batch in range(num_mini_batches):
                    idx = indices[batch*size:(batch+1)*size]
                    self.batch_recovery_mask = masks[idx]
                    yield (*[a[idx] for a in arrays], (None, None), None)
        finally:
            self.batch_recovery_mask = None


def masked_pose_loss(values, recovery):
    per_sample = values.square().mean(-1)
    return (per_sample * (~recovery)).sum() / (~recovery).sum().clamp(min=1)


class RecoveryAMP:
    def __init__(self, expert_pairs, device, clip_lengths=None):
        self.device = device
        self.expert = expert_pairs.to(device)
        size = self.expert.shape[-1]
        self.mean = torch.zeros(size, dtype=torch.float64, device=device)
        self.var = torch.ones_like(self.mean)
        self.count = 1e-4
        self.clip_lengths = torch.tensor(clip_lengths or [len(self.expert)], dtype=torch.long, device=device)
        if int(self.clip_lengths.sum()) != len(self.expert) or (self.clip_lengths < 1).any():
            raise ValueError("Expert clip boundaries must cover all transitions")
        self.clip_offsets = self.clip_lengths.cumsum(0)-self.clip_lengths
        self.discriminator = Discriminator(size*2, SETTINGS["amp_reward_coef"],
            SETTINGS["discriminator_hidden_dims"], device, SETTINGS["amp_task_reward_lerp"])
        self.optimizer = torch.optim.Adam([
            {"params": self.discriminator.trunk.parameters(), "weight_decay": SETTINGS["discriminator_trunk_decay"]},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": SETTINGS["discriminator_head_decay"]},
        ], lr=SETTINGS["discriminator_learning_rate"])
        self.replay = ReplayBuffer(size, SETTINGS["replay_capacity"], device)
        self.last_metrics = {}

    def normalize(self, value):
        return ((value-self.mean.float())/torch.sqrt(self.var.float()+1e-4)).clamp(-10, 10)

    @torch.no_grad()
    def update_moments(self, value):
        value = value.double()
        var, mean = torch.var_mean(value, dim=0, unbiased=False)
        count = value.shape[0]
        delta, total = mean-self.mean, self.count+count
        self.var = (self.var*self.count + var*count + delta.square()*self.count*count/total)/total
        self.mean += delta*count/total
        self.count = total

    def sample_expert(self, batch):
        clips = torch.randint(len(self.clip_lengths), (batch,), device=self.device)
        frames = (torch.rand(batch, device=self.device)*self.clip_lengths[clips]).long()
        return self.expert[self.clip_offsets[clips]+frames]

    @torch.no_grad()
    def reward(self, state, next_state, task_reward, env_ids):
        if not env_ids.numel():
            return task_reward, task_reward.new_empty(0)
        first, second = self.normalize(state), self.normalize(next_state)
        logits = self.discriminator(torch.cat((first, second), -1)).squeeze(-1)
        style = SETTINGS["amp_reward_coef"] * (1-.25*(logits-1).square()).clamp(min=0)
        lerp = SETTINGS["amp_task_reward_lerp"]
        combined = task_reward.clone()
        combined[env_ids] = lerp*task_reward[env_ids]+(1-lerp)*style
        return combined, style

    def update(self):
        if self.replay.num_samples == 0:
            self.last_metrics = {"discriminator_loss": 0., "gradient_penalty": 0.}
            return
        batch = min(SETTINGS["discriminator_batch_size"], self.replay.num_samples)
        values = []
        for _ in range(SETTINGS["discriminator_updates"]):
            ids = torch.randint(self.replay.num_samples, (batch,), device=self.device)
            policy_state, policy_next = self.replay.states[ids], self.replay.next_states[ids]
            expert = self.sample_expert(batch)
            fake = torch.cat((self.normalize(policy_state), self.normalize(policy_next)), -1)
            real = torch.cat((self.normalize(expert[:, 0]), self.normalize(expert[:, 1])), -1).detach()
            real.requires_grad_(True)
            fake_score, real_score = self.discriminator(fake), self.discriminator(real)
            loss = .5*((fake_score+1).square().mean()+(real_score-1).square().mean())
            gradient = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0]
            penalty = SETTINGS["discriminator_gradient_penalty"]*gradient.square().sum(-1).mean()
            total = loss+penalty
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite recovery AMP loss")
            self.optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.update_moments(policy_state)
            self.update_moments(expert[:, 0])
            values.append(torch.stack((loss.detach(), penalty.detach())))
        loss, penalty = torch.stack(values).mean(0).cpu().tolist()
        self.last_metrics = {"discriminator_loss": loss, "gradient_penalty": penalty,
                             "replay_samples": self.replay.num_samples}

    def state_dict(self):
        count = self.replay.num_samples
        return {"discriminator": self.discriminator.state_dict(), "optimizer": self.optimizer.state_dict(),
                "normalizer_mean": self.mean, "normalizer_var": self.var, "normalizer_count": self.count,
                "replay_states": self.replay.states[:count], "replay_next_states": self.replay.next_states[:count],
                "replay_count": count, "replay_step": self.replay.step}

    def load_state_dict(self, saved):
        self.discriminator.load_state_dict(saved["discriminator"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.mean.copy_(saved["normalizer_mean"])
        self.var.copy_(saved["normalizer_var"])
        self.count = saved["normalizer_count"]
        count = saved["replay_count"]
        if not 0 <= count <= self.replay.buffer_size:
            raise ValueError("Invalid recovery replay count")
        self.replay.states[:count].copy_(saved["replay_states"])
        self.replay.next_states[:count].copy_(saved["replay_next_states"])
        self.replay.num_samples, self.replay.step = count, saved["replay_step"]


class DWAQPPO(ParentPPO):
    """Private factory name intentionally matches the unchanged parent recipe."""
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, obs_hist_shape, action_shape):
        self.storage = RecoveryStorage(num_envs, num_transitions_per_env, actor_obs_shape,
                                       critic_obs_shape, obs_hist_shape, action_shape, self.device)

    def configure_recovery(self, env, amp):
        self.recovery_v2_env, self.recovery_v2_amp = env, amp

    def act(self, *args, **kwargs):
        self.storage.pending_mask = self.recovery_v2_env.recovery_active.clone()
        return super().act(*args, **kwargs)

    def _upper_body_pose_loss(self, observations, observation_history):
        if self.upper_body_symmetry is None or not self.upper_body_symmetry["pose_coefficient"]:
            return None
        mask = self.storage.batch_recovery_mask
        if mask is None or mask.shape != observations.shape[:1]:
            raise ValueError("PPO pose mask must follow this minibatch's sample permutation")
        action = self.policy.deterministic_action_mean(observations, observation_history)
        return masked_pose_loss(action[:, self.upper_body_symmetry["action_ids"]], mask)

    def process_env_step(self, rewards, dones, infos):
        transition = infos["recovery_v2_amp"]
        if transition["env_ids"].numel():
            self.recovery_v2_amp.replay.insert(transition["state"], transition["next_state"])
        return super().process_env_step(rewards, dones, infos)

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        self.recovery_v2_amp.update()
        return result
