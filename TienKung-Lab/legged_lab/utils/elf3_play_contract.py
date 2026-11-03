"""Simulator-independent, inference-only compatibility for feedback checkpoints.

Play may change the training recipe (for example the number of environments),
but not the experiment, robot, observation layout, or reward identity.  This
helper deliberately does not load a policy, optimizer, or training runner.
"""
from copy import deepcopy
import math
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _validate_feedback_metadata(contract):
    if not isinstance(contract, dict):
        raise ValueError("Play requires an ELF3 feedback experiment contract")
    experiment = contract.get("feedback_experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Old DWAQ and AMP checkpoints have no feedback experiment metadata")
    for key in ("training_recipe_sha256", "unchanged_reward_sha256"):
        value = experiment.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"Missing or invalid feedback experiment {key}")
    feedback = experiment.get("velocity_feedback")
    if not isinstance(feedback, dict) or set(feedback) != {
        "mode", "cutoff_hz", "model_version", "latest_critic_contacts"
    }:
        raise ValueError("Invalid velocity feedback metadata")
    cutoff = feedback["cutoff_hz"]
    if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
            or not math.isfinite(cutoff) or not 0 < cutoff < 100):
        raise ValueError("Invalid feedback cutoff frequency")
    if (feedback["mode"] not in ("ideal", "lpf20")
            or feedback["model_version"] != "joint_velocity_tustin_v1"
            or feedback["latest_critic_contacts"] is not True):
        raise ValueError("Unsupported feedback experiment configuration")
    # This is the original v1 schema.  Keeping all fields here prevents two
    # equally incomplete/malformed dictionaries from passing equality below.
    canonical = {
        "version": "elf3_feedback_v1",
        "actor_context": "encoder_mean_velocity_and_mean_latent",
        "auxiliary_vae": "original_stochastic_reparameterization",
        "velocity_feedback": feedback,
        "feedback_update_dt": .005,
        "actor_sample_dt": .02,
        "feedback_reset": "current_raw_joint_velocity_after_reset",
        "feedback_location": "proprioception_only; actuator_PD_unchanged",
        "critic_force_history_index": 0,
        "observation": {
            "frame_size": 100, "history_length": 5, "flat_size": 500,
            "joint_velocity_slice": [38, 67], "history_order": "oldest_to_newest",
        },
        "deployment": {
            "num_actions": 29, "extra_velocity_filter": False,
            "extra_action_filter": False,
        },
        "unchanged_reward_sha256": experiment["unchanged_reward_sha256"],
        "training_recipe_sha256": experiment["training_recipe_sha256"],
    }
    if experiment != canonical:
        raise ValueError("Incomplete or unsupported original feedback experiment metadata")


def validate_play_checkpoint(saved, expected_contract):
    """Validate a completed checkpoint for play, ignoring only its recipe hash.

    Both input objects remain untouched.  Training/resume code must continue to
    use its stricter validator; passing this check does not authorize resuming
    an optimizer under a changed training recipe.
    """
    if not isinstance(saved, dict) or not isinstance(saved.get("infos"), dict):
        raise ValueError("Checkpoint has no ELF3 feedback experiment metadata")
    infos = saved["infos"]
    if infos.get("initialization_only"):
        raise ValueError("An initial random snapshot is not a completed training checkpoint")
    actual = deepcopy(infos.get("elf3_contract"))
    expected = deepcopy(expected_contract)
    _validate_feedback_metadata(actual)
    _validate_feedback_metadata(expected)
    actual["feedback_experiment"].pop("training_recipe_sha256")
    expected["feedback_experiment"].pop("training_recipe_sha256")
    if actual != expected:
        raise ValueError("Checkpoint experiment/robot contract mismatch for play; only "
                         "training_recipe_sha256 may differ")
