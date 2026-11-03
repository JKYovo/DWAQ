"""CPU-only tests for the inference-only feedback checkpoint validator."""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


play = load_file("tested_elf3_play_contract",
                 "TienKung-Lab/legged_lab/utils/elf3_play_contract.py")
feedback_contract = load_file("play_test_feedback_contract",
                             "TienKung-Lab/legged_lab/envs/elf3/feedback_contract.py")


def contract(mode="lpf20"):
    experiment = feedback_contract.experiment_contract({
        "mode": mode, "cutoff_hz": 20.0,
        "model_version": "joint_velocity_tustin_v1", "latest_critic_contacts": True,
    }, .005, .02)
    experiment.update(unchanged_reward_sha256="a" * 64,
                      training_recipe_sha256="b" * 64)
    return {
        "robot": "elf3", "asset_contract_sha256": "c" * 64,
        "joint_names": [f"joint_{index}" for index in range(29)],
        "num_obs": 100, "num_privileged_obs": 311, "history_length": 5,
        "control_dt": .02, "gait_period": .8, "gait_offset": .5,
        "upper_body_symmetry": {"mirror_loss_coeff": .1, "pose_loss_coeff": .02},
        "feedback_experiment": experiment,
    }


def checkpoint(value=None):
    return {"iter": 47500, "infos": {"elf3_contract": contract() if value is None else value}}


class PlayContractTests(unittest.TestCase):
    def test_identical_contract_and_different_recipe_hash_are_allowed(self):
        saved, expected = checkpoint(), contract()
        self.assertIsNone(play.validate_play_checkpoint(saved, expected))
        expected["feedback_experiment"]["training_recipe_sha256"] = "d" * 64
        self.assertIsNone(play.validate_play_checkpoint(saved, expected))

    def test_inputs_are_unchanged_after_success_and_failure(self):
        saved, expected = checkpoint(), contract()
        expected["feedback_experiment"]["training_recipe_sha256"] = "d" * 64
        originals = copy.deepcopy((saved, expected))
        play.validate_play_checkpoint(saved, expected)
        self.assertEqual((saved, expected), originals)
        expected["asset_contract_sha256"] = "e" * 64
        originals = copy.deepcopy((saved, expected))
        with self.assertRaises(ValueError):
            play.validate_play_checkpoint(saved, expected)
        self.assertEqual((saved, expected), originals)

    def test_every_other_contract_difference_is_rejected(self):
        changes = {
            "asset": lambda value: value.update(asset_contract_sha256="d" * 64),
            "joint_order": lambda value: value["joint_names"].reverse(),
            "history": lambda value: value.update(history_length=20),
            "actor_dim": lambda value: value.update(num_obs=96),
            "critic_dim": lambda value: value.update(num_privileged_obs=105),
            "control_dt": lambda value: value.update(control_dt=.01),
            "gait": lambda value: value.update(gait_period=1.0),
            "mirror": lambda value: value["upper_body_symmetry"].update(mirror_loss_coeff=.2),
            "reward": lambda value: value["feedback_experiment"].update(unchanged_reward_sha256="d" * 64),
            "mode": lambda value: value["feedback_experiment"]["velocity_feedback"].update(mode="ideal"),
            "context": lambda value: value["feedback_experiment"].update(actor_context="sampled"),
            "feedback_dt": lambda value: value["feedback_experiment"].update(feedback_update_dt=.01),
            "extra_field": lambda value: value.update(unexpected=True),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                expected = contract()
                change(expected)
                with self.assertRaises(ValueError):
                    play.validate_play_checkpoint(checkpoint(), expected)

    def test_old_amp_missing_metadata_and_initialization_are_rejected(self):
        old_dwaq = contract()
        old_dwaq.pop("feedback_experiment")
        cases = [None, {}, {"infos": None}, {"infos": {}},
                 {"infos": {"elf3_contract": {"robot": "elf3", "algorithm": "AMP"}}},
                 checkpoint(old_dwaq),
                 {"infos": {"elf3_contract": contract(), "initialization_only": True}}]
        for saved in cases:
            with self.subTest(saved=saved):
                with self.assertRaises(ValueError):
                    play.validate_play_checkpoint(saved, contract())

    def test_equal_but_missing_or_invalid_experiment_fields_are_rejected(self):
        changes = [
            lambda experiment: experiment.pop("training_recipe_sha256"),
            lambda experiment: experiment.pop("unchanged_reward_sha256"),
            lambda experiment: experiment.pop("observation"),
            lambda experiment: experiment.update(training_recipe_sha256="not-a-sha"),
            lambda experiment: experiment.update(unchanged_reward_sha256=""),
            lambda experiment: experiment.update(version="unknown"),
            lambda experiment: experiment.update(actor_context="sampled"),
            lambda experiment: experiment["velocity_feedback"].update(mode="delay20"),
            lambda experiment: experiment["velocity_feedback"].update(cutoff_hz=float("nan")),
            lambda experiment: experiment["velocity_feedback"].update(latest_critic_contacts=False),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                expected = contract()
                change(expected["feedback_experiment"])
                with self.assertRaises(ValueError):
                    play.validate_play_checkpoint(checkpoint(copy.deepcopy(expected)), expected)

    def test_both_supported_feedback_modes_validate_but_do_not_cross_load(self):
        for mode in ("ideal", "lpf20"):
            with self.subTest(mode=mode):
                expected = contract(mode)
                self.assertIsNone(play.validate_play_checkpoint(checkpoint(copy.deepcopy(expected)), expected))
        with self.assertRaises(ValueError):
            play.validate_play_checkpoint(checkpoint(contract("ideal")), contract("lpf20"))


if __name__ == "__main__":
    unittest.main()
