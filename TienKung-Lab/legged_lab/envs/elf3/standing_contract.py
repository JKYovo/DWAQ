"""Identity for the opt-in walking task with zero-command gait rewards disabled."""
import hashlib
from pathlib import Path


VERSION = "elf3_stand_v1"
MODE = "walking_zero_command_fix"
COMMAND_EPSILON = 1.0e-6


def standing_experiment_contract(settings):
    """Keep this reward experiment distinct from old feedback checkpoints.

    Hash the implementations too: serializing a callable into a configuration
    only records its name and would not detect changes to the function body.
    """
    expected = {"version": VERSION, "mode": MODE, "command_epsilon": COMMAND_EPSILON}
    if settings != expected:
        raise ValueError(f"Unsupported standing experiment settings: {settings!r}; expected {expected!r}")
    directory = Path(__file__).resolve().parent
    return {
        **expected,
        "zero_command_gait_rewards_disabled": True,
        "supports_walking": True,
        "reward_source_sha256": hashlib.sha256((directory / "standing_rewards.py").read_bytes()).hexdigest(),
        "base_reward_source_sha256": hashlib.sha256(
            (directory.parents[1] / "mdp/rewards.py").read_bytes()
        ).hexdigest(),
    }
