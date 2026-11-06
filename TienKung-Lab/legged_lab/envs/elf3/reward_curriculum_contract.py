"""Pure, versioned reward schedule. No simulator, deployment or PPO changes."""
from copy import copy, deepcopy
import hashlib
from pathlib import Path


TASK = "elf3_dwaq_still_v2_curriculum_v1"
VERSION = "elf3_still_v2_reward_curriculum_peak500_v1"
WEIGHT_ENDPOINTS = {
    "ankle_torque_l2": (0., -.0005),
    "stance_foot_speed_cost": (0., -1.),
    "base_height": (0., -50.),
    "action_rate_l2": (-.01, -.012),
    "action_smooth": (0., -.001),
    "dof_acc_l2": (-2.5e-7, -3e-7),
    "feet_force": (-.003, -.005),
}


def schedule_settings(ramp_start=20000, ramp_end=30000):
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (ramp_start, ramp_end))
            or not 0 <= ramp_start < ramp_end):
        raise ValueError("Require integer 0 <= ramp_start < ramp_end")
    return {"version": VERSION, "ramp_start": ramp_start, "ramp_end": ramp_end,
            "clock": "absolute indexed PPO iteration; fixed for all24 rollout steps",
            "feet_force_threshold_n": 500., "feet_force_excess_cap_n": 400.,
            "force_sampling": "global all stages: simultaneous two-foot Fz norm peak over4 physics substeps",
            "baseline_difference_from_historical_still_v2": "peak sampling repair only",
            "automatic_stop_or_restart": False}


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Missing curriculum schedule")
    expected = schedule_settings(settings.get("ramp_start"), settings.get("ramp_end"))
    if settings != expected:
        raise ValueError("Unknown or altered curriculum schedule/physics/force identity")
    return deepcopy(expected)


def curriculum_alpha(iteration, settings):
    settings = validate_settings(settings)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Curriculum iteration must be a nonnegative integer")
    return min(1., max(0., (iteration-settings["ramp_start"])
                           / (settings["ramp_end"]-settings["ramp_start"])))


def effective_weights(iteration, settings):
    alpha = curriculum_alpha(iteration, settings)
    if alpha == 0.:
        return {name: values[0] for name, values in WEIGHT_ENDPOINTS.items()}
    if alpha == 1.:
        return {name: values[1] for name, values in WEIGHT_ENDPOINTS.items()}
    return {name: start + alpha*(end-start) for name, (start, end) in WEIGHT_ENDPOINTS.items()}


def apply_weights(manager, iteration, settings):
    """Update the LIVE manager, never the immutable endpoint environment cfg.

    Terms are registered from the final recipe even when initially zero. Time
    integration remains exclusively RewardManager's existing control-dt factor.
    """
    weights = effective_weights(iteration, settings)
    for name, weight in weights.items():
        term = copy(manager.get_term_cfg(name))
        term.weight = weight
        manager.set_term_cfg(name, term)
    return weights


class RolloutClock:
    """Advance only during learning, not resets, play, or arbitrary env steps."""
    def __init__(self):
        self.active = False
        self.start_iteration = 0
        self.steps = 0
        self.steps_per_iteration = 24

    def begin(self, iteration, steps_per_iteration):
        if (isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0
                or isinstance(steps_per_iteration, bool) or not isinstance(steps_per_iteration, int)
                or steps_per_iteration < 1):
            raise ValueError("Invalid rollout clock")
        self.start_iteration, self.steps_per_iteration = iteration, steps_per_iteration
        self.steps, self.active = 0, True

    @property
    def iteration(self):
        return self.start_iteration + self.steps // self.steps_per_iteration

    def advance(self):
        if self.active:
            self.steps += 1

    def end(self):
        self.active = False


def curriculum_contract(settings):
    directory = Path(__file__).resolve().parent
    return {**validate_settings(settings),
            "weight_endpoints": {k: list(v) for k, v in WEIGHT_ENDPOINTS.items()},
            "effective_rewards": "runtime linear interpolation overrides inherited final-reference weights",
            "unchanged": ["standing", "moving_phase", "height_gates", "asset", "PD", "armature",
                          "action_scale", "20Hz_feedback", "commands", "terrain", "randomization",
                          "Actor", "CENet", "PPO", "adaptive_LR", "mirror_and_pose_auxiliaries"],
            "sources_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ("reward_curriculum.py", "reward_curriculum_contract.py")}}
