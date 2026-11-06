"""Complete control-window coverage for the existing two-foot force metric."""
import math
import torch


def accumulate_peak(peak, forces):
    if forces.shape != peak.shape + (3,):
        raise ValueError("Expected peak [N,2] and foot forces [N,2,3]")
    # Cache individual-foot magnitudes for diagnostics, NOT for combining
    # unrelated left/right maxima into a fictitious simultaneous force.
    peak.copy_(torch.maximum(peak, forces[..., 2].abs()))


def force_excess(metric, threshold=400., max_reward=400.):
    if not (math.isfinite(threshold) and threshold > 0
            and math.isfinite(max_reward) and max_reward > 0):
        raise ValueError("Positive finite threshold and cap required")
    return (metric - threshold).clamp(min=0., max=max_reward)


def body_force_control_peak(env, sensor_cfg, threshold=400., max_reward=400.):
    # sensor_cfg is retained/resolved by RewardManager. The actual sampling
    # indices are verified against these named feet during environment init.
    if env._force_peak_substeps != env.cfg.sim.decimation:
        raise RuntimeError("Force reward requires exactly every physical substep")
    return force_excess(env.force_peak_metric, threshold, max_reward)
