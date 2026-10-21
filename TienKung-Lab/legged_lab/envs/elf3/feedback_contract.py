"""Simulator-independent experiment identity, separate from the robot contract."""
import hashlib
import json
import math


VERSION = "elf3_feedback_v1"
TASKS = ("elf3_dwaq_consistent_v1", "elf3_dwaq_feedback_v1")


def experiment_contract(feedback, physics_dt, control_dt, context_mode="mean"):
    if context_mode != "mean":
        raise ValueError("The feedback experiment requires mean actor context")
    required = {"mode", "cutoff_hz", "model_version", "latest_critic_contacts"}
    if set(feedback) != required:
        raise ValueError(f"Unexpected feedback configuration keys: {set(feedback) ^ required}")
    if feedback["mode"] not in ("ideal", "lpf20"):
        raise ValueError("Expected ideal or lpf20 velocity feedback")
    if feedback["model_version"] != "joint_velocity_tustin_v1":
        raise ValueError("Unknown velocity feedback model version")
    cutoff = float(feedback["cutoff_hz"])
    if not all(math.isfinite(v) and v > 0 for v in (cutoff, physics_dt, control_dt)):
        raise ValueError("Finite positive cutoff and timesteps required")
    if not math.isclose(physics_dt, .005) or not math.isclose(control_dt, .02):
        raise ValueError("This experiment retains 5 ms physics / 20 ms control")
    if cutoff >= .5 / physics_dt:
        raise ValueError("Cutoff must be below the physics sample Nyquist frequency")
    if feedback["latest_critic_contacts"] is not True:
        raise ValueError("This version requires newest privileged contact force")
    return {
        "version": VERSION,
        "actor_context": "encoder_mean_velocity_and_mean_latent",
        "auxiliary_vae": "original_stochastic_reparameterization",
        "velocity_feedback": dict(feedback),
        "feedback_update_dt": physics_dt,
        "actor_sample_dt": control_dt,
        "feedback_reset": "current_raw_joint_velocity_after_reset",
        "feedback_location": "proprioception_only; actuator_PD_unchanged",
        "critic_force_history_index": 0,
        "observation": {"frame_size": 100, "history_length": 5, "flat_size": 500,
                        "joint_velocity_slice": [38, 67], "history_order": "oldest_to_newest"},
        "deployment": {"num_actions": 29, "extra_velocity_filter": False,
                       "extra_action_filter": False},
    }


def validate_checkpoint_contract(saved, expected):
    metadata = (saved.get("infos") or {}).get("elf3_contract")
    if metadata != expected:
        raise ValueError("Checkpoint experiment/robot contract mismatch. Old DWAQ, AMP, "
                         "and the other feedback mode cannot resume this experiment.")
    if (saved.get("infos") or {}).get("initialization_only"):
        raise ValueError("An initial random snapshot is not a completed training checkpoint")


def canonical_sha256(value):
    """Stable config fingerprint, without process-specific callable addresses."""
    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(v) for v in item]
        if isinstance(item, slice):
            return {"slice": [item.start, item.stop, item.step]}
        if callable(item):
            return f"{item.__module__}.{item.__qualname__}"
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                raise ValueError("NaN is not a valid configuration constant")
            # Disabled camera/raycast configs legitimately use +/-inf sentinels.
            # Tag them explicitly rather than emitting non-standard JSON floats.
            return {"configuration_float": "+inf" if item > 0 else "-inf"}
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        return str(item)
    return hashlib.sha256(json.dumps(convert(value), sort_keys=True, allow_nan=False).encode()).hexdigest()
