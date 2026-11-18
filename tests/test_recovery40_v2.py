"""CPU checks of motion names, recovery gates, terminal boundaries and shuffled PPO masks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"TienKung-Lab/rsl_rl"), str(ROOT/"TienKung-Lab")]
DIR = ROOT/"TienKung-Lab/legged_lab/envs/elf3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("legged_lab.envs.elf3.recovery_v2_core", DIR/"recovery_v2_core.py")
amp = load("recovery_v2_amp_cpu", DIR/"recovery_v2_amp.py")
contract = load("recovery_robot_contract_cpu", ROOT/"TienKung-Lab/legged_lab/elf3_contract.py")


class RecoveryTests(unittest.TestCase):
    def test_low_height_without_contact_and_large_tilt_are_falls(self):
        fallen = core.recovery_fallen(torch.tensor([-1., -1., -.2, -1.]), torch.tensor([.3, 1., 1., 1.]))
        self.assertEqual(fallen.tolist(), [True, False, True, False])
        done, _ = core.delayed_fall(fallen, torch.zeros(4,dtype=torch.bool),
            torch.tensor([True,True,True,False]), torch.tensor([249,0,0,0]))
        self.assertEqual(done.tolist(), [True, False, False, False])

    def test_recovery_height_uses_ground_not_world_z_and_increases_toward_default(self):
        hits = torch.tensor([[[0.,0.,2.],[0.,0.,float('inf')]], [[0.,0.,0.],[0.,0.,0.]]])
        heights = core.root_clearance(torch.tensor([2.5,.5]), hits, torch.zeros(2))
        torch.testing.assert_close(heights, torch.tensor([.5,.5]))
        objective = core.height_objective(torch.tensor([.3,.6,.9,1.2]),1.2)
        self.assertTrue((objective[1:]>objective[:-1]).all())
        self.assertEqual(objective[-1].item(),1.)

    def test_running_normalization_matches_joint_policy_expert_population(self):
        system = amp.RecoveryAMP(torch.randn(7,2,195),'cpu')
        batches = [torch.randn(9,195)*3+20,torch.randn(17,195)*.2-5]
        for batch in batches:
            system.update_moments(batch)
        data = torch.cat(batches).double()
        mean = data.sum(0)/(len(data)+1e-4)
        var = ((data-mean).square().sum(0)+1e-4*(1+mean.square()))/(len(data)+1e-4)
        torch.testing.assert_close(system.mean,mean,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(system.var,var,rtol=1e-12,atol=1e-12)
        restored = amp.RecoveryAMP(torch.randn(7,2,195),'cpu')
        restored.load_state_dict(deepcopy(system.state_dict()))
        torch.testing.assert_close(restored.normalize(data.float()),system.normalize(data.float()),rtol=0,atol=0)
        self.assertEqual(restored.count,system.count)

    def test_expert_sampling_keeps_short_clip_as_likely_as_long_clip(self):
        data = torch.zeros(100,2,195)
        data[1:] = 1.
        system = amp.RecoveryAMP(data,'cpu',[1,99])
        torch.manual_seed(123)
        frequency = system.sample_expert(6000)[:,0,0].mean().item()
        self.assertTrue(.45 < frequency < .55)

    def test_delay_exact250_consecutive_steps_and_timeout_never_suppressed(self):
        assigned = torch.tensor([True, False, True])
        counts = torch.zeros(3, dtype=torch.long)
        for step in range(250):
            done, counts = core.delayed_fall(torch.ones(3, dtype=torch.bool),
                torch.tensor([False, False, True]), assigned, counts)
            self.assertEqual(done.tolist(), [step == 249, True, True])
        counts = torch.tensor([200, 0, 0])
        _, counts = core.delayed_fall(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool), assigned, counts)
        self.assertEqual(counts.tolist(), [0, 0, 0])

    def test_raised_torso_exits_immediately_and_a_new_fall_reenters(self):
        active, stable = torch.tensor([True, True]), torch.zeros(2, dtype=torch.long)
        active, stable = core.phase_update(active, stable, torch.tensor([-.9063, 0.]), torch.tensor([1., .4]), 1.05)
        self.assertEqual(active.tolist(), [False, True])
        active, stable = core.phase_update(active, stable, torch.tensor([0., -1.]), torch.tensor([.4, 1.]), 1.)
        self.assertEqual(active.tolist(), [True, False])

    def test_height_exit_boundary_has_no_tilt_or_hold_gate_and_no_height_chatter(self):
        active = torch.ones(4, dtype=torch.bool)
        stable = torch.zeros(4, dtype=torch.long)
        active, stable = core.phase_update(active, stable,
            torch.tensor([-.9063, 0., -1., -1.]), torch.tensor([.84, 1., .839, .5]), 1.05)
        self.assertEqual(active.tolist(), [False, False, True, True])
        # An exited, upright sample stays in DWAQ in the existing .65-.80 band;
        # a recovering sample in that band still needs to reach the exit height.
        active, _ = core.phase_update(torch.tensor([False, True]), torch.zeros(2, dtype=torch.long),
            torch.tensor([-.9063, -.9063]), torch.tensor([.78, .78]), 1.05)
        self.assertEqual(active.tolist(), [False, True])

    def test_pose_loss_has_zero_recovery_gradient_and_retains_normal_gradient(self):
        values = torch.tensor([[100., 200.], [2., 4.]], requires_grad=True)
        loss = amp.masked_pose_loss(values, torch.tensor([True, False]))
        self.assertEqual(float(loss), 10.)
        loss.backward()
        torch.testing.assert_close(values.grad[0], torch.zeros(2))
        self.assertTrue((values.grad[1] != 0).all())
        all_recovery = amp.masked_pose_loss(values, torch.ones(2, dtype=torch.bool))
        self.assertEqual(float(all_recovery), 0.)

    def test_minibatch_mask_follows_exact_sample_and_history_permutation(self):
        storage = amp.RecoveryStorage(8, 3, [100], [311], [500], [29])
        sample_id = torch.arange(24).reshape(3, 8)
        storage.observations[..., 0] = sample_id
        storage.observation_history[..., 0] = sample_id
        storage.recovery_masks.copy_(sample_id % 3 == 0)
        count = 0
        for fields in storage.mini_batch_generator(4, 5):
            ids = fields[0][:, 0].long()
            torch.testing.assert_close(fields[3][:, 0], ids.float())
            self.assertTrue(torch.equal(storage.batch_recovery_mask, ids % 3 == 0))
            self.assertEqual(len(fields), 13)
            count += 1
        self.assertEqual(count, 20)
        self.assertIsNone(storage.batch_recovery_mask)

    def test_features_match_known_rigid_transform_and_local_velocities(self):
        pos = torch.tensor([[[10., 5., 1.], [11., 5., 2.]]])
        q = torch.tensor([[[1., 0, 0, 0], [1., 0, 0, 0]]])
        vel = torch.tensor([[[0., 0, 0], [1., 2., 3.]]])
        result = core.amp_features(pos, q, vel, vel*2, [1], 0)
        torch.testing.assert_close(result, torch.tensor([[1., 0., 1., 1., 0., 0., 1., 0., 0., 1., 2., 3., 2., 4., 6.]]))

    def test_existing_motion_bytes_order_and_exact_expert_transition_boundaries(self):
        library = core.MotionLibrary(ROOT/"data/elf3_amp_v4", contract.JOINT_NAMES)
        self.assertEqual(library.identity()["frames"], {"Recovery": 2575, "WalkandRun": 5089})
        self.assertEqual(len(library.expert_pairs), 7664)
        self.assertEqual(tuple(library.expert_pairs.shape[1:]), (2, 195))
        offset = 0
        self.assertEqual(len(library.expert_clip_lengths),16)
        for length in library.expert_clip_lengths:
            clip = library.expert_pairs[offset:offset+length]
            torch.testing.assert_close(clip[:-1,1],clip[1:,0],rtol=0,atol=0)
            torch.testing.assert_close(clip[-1,1],clip[-1,0],rtol=0,atol=0)
            offset += length
        reverse = core.MotionLibrary(ROOT/"data/elf3_amp_v4", list(reversed(contract.JOINT_NAMES)))
        torch.testing.assert_close(reverse.frames["Recovery"]["joint_pos"], library.frames["Recovery"]["joint_pos"].flip(-1))
        sample = library.sample("Recovery", 50)
        self.assertEqual(sample["joint_pos"].shape, (50, 29))
        self.assertTrue(torch.isfinite(sample["root_quat"]).all())

    def test_compact_features_equal_dense_features_and_preserve_snapshot(self):
        torch.manual_seed(17)
        snapshot = (torch.randn(9, 14, 3), torch.randn(9, 14, 4),
                    torch.randn(9, 14, 3), torch.randn(9, 14, 3))
        snapshot[1].div_(snapshot[1].norm(dim=-1, keepdim=True))
        rows = torch.tensor([7, 0, 4])
        dense = core.amp_features(*snapshot, range(13), 13)
        compact = core.selected_amp_features(snapshot, rows)
        torch.testing.assert_close(compact, dense[rows], rtol=0, atol=0)
        self.assertEqual(core.selected_amp_features(snapshot, torch.empty(0, dtype=torch.long)).shape, (0, 195))

    def test_amp_only_changes_recovery_reward_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        pairs = torch.randn(12, 2, 195)
        system = amp.RecoveryAMP(pairs, "cpu")
        first, second = pairs[:3, 0], pairs[:3, 1]
        task = torch.tensor([2., 3., 4.])
        mask = torch.tensor([True, False, True])
        rows = mask.nonzero().flatten()
        with torch.no_grad():
            logits = system.discriminator(torch.cat((system.normalize(first), system.normalize(second)), -1)).squeeze(-1)
            dense_style = .1*(1-.25*(logits-1).square()).clamp(min=0)
            expected = torch.where(mask, .75*task+.25*dense_style, task)
        with patch.object(system.discriminator, "forward", wraps=system.discriminator.forward) as forward:
            reward, style = system.reward(first[rows], second[rows], task, rows)
            self.assertEqual(forward.call_args.args[0].shape[0], 2)
        self.assertEqual(float(reward[1]), float(task[1]))
        torch.testing.assert_close(reward[mask], .75*task[mask]+.25*style)
        torch.testing.assert_close(reward, expected)
        with patch.object(system.discriminator, "forward", side_effect=AssertionError("Must skip inactive AMP")):
            unchanged, empty = system.reward(first[:0], second[:0], task, rows[:0])
        self.assertIs(unchanged, task)
        self.assertEqual(empty.numel(), 0)
        system.replay.insert(first, second)
        with torch.no_grad():
            saved = deepcopy(system.state_dict())
        other = amp.RecoveryAMP(pairs, "cpu")
        other.load_state_dict(saved)
        for key, value in system.discriminator.state_dict().items():
            torch.testing.assert_close(value, other.discriminator.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(other.replay.states[:3], first)
        self.assertEqual(other.replay.num_samples, 3)

    def test_policy_and_adam_resume_parameters_unchanged_by_private_algorithm(self):
        from rsl_rl.modules import ActorCritic_DWAQ
        from rsl_rl.algorithms.dwaq_ppo import DWAQPPO
        original_model = ActorCritic_DWAQ(119, 311, 29, 500, 19, 100, context_mode="mean")
        original = DWAQPPO(original_model, learning_rate=1e-4, device="cpu")
        original_model.actor[0].weight.square().mean().backward()
        original.optimizer.step()
        model = ActorCritic_DWAQ(119, 311, 29, 500, 19, 100, context_mode="mean")
        copied = amp.DWAQPPO(model, learning_rate=1e-4, device="cpu")
        model.load_state_dict(original_model.state_dict())
        copied.optimizer.load_state_dict(original.optimizer.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original_model.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(copied.optimizer.param_groups[0]['lr'], original.optimizer.param_groups[0]['lr'])


if __name__ == "__main__":
    unittest.main()
