"""Fail-closed audit against the registered task, including saved YAML and PPO."""
import yaml

from legged_lab.envs.elf3.elf3_dwaq_config import Elf3DwaqEnvCfg, Elf3DwaqUpperBodySymmetryPoseAgentCfg
from legged_lab.envs.elf3.runner import Elf3DwaqUpperBodySymmetryRunner
from legged_lab.envs.elf3.symmetry import ARM_ACTION_IDS


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                result.append({"path": path, "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], path))
        return result
    return [] if before == after else [{"path": prefix, "before": before, "after": after}]


def audit_overrides(registered_env, registered_agent, cfg, agent):
    assert type(cfg) is type(registered_env) is Elf3DwaqEnvCfg
    assert type(agent) is type(registered_agent) is Elf3DwaqUpperBodySymmetryPoseAgentCfg
    reasons = {
        "env.scene.num_envs": "requested environment count; smaller only in isolated preflight",
        "env.scene.seed": "requested seed 42",
        "agent.seed": "requested seed 42",
        "agent.resume": "fresh initialization, never load previous weights/optimizer",
        "agent.max_iterations": "requested 100001 iterations (0..100000); smaller only in preflight",
        "agent.save_interval": "requested checkpoint every 500 iterations",
        "agent.logger": "TensorBoard primary log; separate SwanLab uploader",
        "agent.run_name": "independent run identity, keep registered experiment_name",
    }
    env_changes = differences(registered_env.to_dict(), cfg.to_dict(), "env")
    agent_changes = differences(registered_agent.to_dict(), agent.to_dict(), "agent")
    for item in env_changes + agent_changes:
        if item["path"] not in reasons:
            raise AssertionError(f"Unapproved registered-task override: {item}")
        item["reason"] = reasons[item["path"]]
    assert cfg.to_dict()["reward"] == registered_env.to_dict()["reward"]
    assert agent.algorithm.to_dict() == registered_agent.algorithm.to_dict()
    assert agent.policy.to_dict() == registered_agent.policy.to_dict()
    assert agent.experiment_name == registered_agent.experiment_name == "elf3_dwaq_upper_symmetry_pose"
    before = {"env": registered_env.to_dict(), "agent": registered_agent.to_dict()}
    after = {"env": cfg.to_dict(), "agent": agent.to_dict()}
    def lookup(tree, path):
        for key in path.split("."):
            tree = tree[key]
        return tree
    assignments = [{"path": path, "before": lookup(before, path), "after": lookup(after, path),
                    "reason": reason} for path, reason in reasons.items()]
    return {"registered_task": "elf3_dwaq_upper_symmetry_pose",
            "environment_config_type": type(cfg).__name__, "agent_config_type": type(agent).__name__,
            "all_explicit_assignment_allowlist": reasons,
            "all_explicit_assignment_values": assignments,
            "actual_changes": env_changes + agent_changes,
            "algorithm_policy_reward_overrides": [],
            "asset_policy": "retained USD and MelodyAI implicit PD unchanged; explicit armature 0.01 only"}


def check_override_guard(registered_env, registered_agent):
    """Prove that undesired algorithm/task/control edits fail before simulation."""
    cases = [
        ("algorithm.learning_rate", lambda e, a: setattr(a.algorithm, "learning_rate", .001)),
        ("runner_class_name", lambda e, a: setattr(a, "runner_class_name", "Elf3DwaqOnPolicyRunner")),
        ("mirror_coefficient", lambda e, a: setattr(a, "upper_body_mirror_loss_coeff", .2)),
        ("pose_coefficient", lambda e, a: setattr(a, "upper_body_pose_loss_coeff", .04)),
        ("command_interval", lambda e, a: setattr(e.commands, "resampling_time_range", (1., 1.))),
        ("asset_armature", lambda e, a: setattr(e.scene.robot.actuators["legs"], "armature", .02)),
    ]
    for name, mutate in cases:
        cfg, agent = registered_env.copy(), registered_agent.copy()
        mutate(cfg, agent)
        try:
            audit_overrides(registered_env, registered_agent, cfg, agent)
        except AssertionError:
            continue
        raise AssertionError(f"Override guard accepted forbidden change: {name}")
    return [name for name, _ in cases]


def check_saved_configuration(directory, runner, agent):
    saved = yaml.load((directory / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    env_saved = yaml.load((directory / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    assert all(float(a["armature"]) == .01 for a in env_saved["scene"]["robot"]["actuators"].values())
    assert saved["runner_class_name"] == "Elf3DwaqUpperBodySymmetryRunner"
    assert float(saved["upper_body_mirror_loss_coeff"]) == .1
    assert float(saved["upper_body_pose_loss_coeff"]) == .02
    algorithm = saved["algorithm"]
    assert float(algorithm["learning_rate"]) == 1e-4
    assert float(algorithm["min_learning_rate"]) == 1e-5
    assert float(algorithm["max_learning_rate"]) == 3e-4
    assert algorithm["schedule"] == "adaptive"
    assert type(runner) is Elf3DwaqUpperBodySymmetryRunner
    auxiliary = runner.alg.upper_body_symmetry
    assert auxiliary is not None
    assert auxiliary["coefficient"] == .1 and auxiliary["pose_coefficient"] == .02
    assert tuple(auxiliary["action_ids"]) == tuple(ARM_ACTION_IDS)
    assert runner.alg.min_learning_rate == 1e-5 and runner.alg.max_learning_rate == 3e-4
    assert runner.alg.learning_rate == agent.algorithm.learning_rate == 1e-4
    assert runner.alg.optimizer.param_groups[0]["lr"] == 1e-4
    contract = runner.contract()["upper_body_symmetry"]
    assert contract["mirror_loss_coeff"] == .1 and contract["pose_loss_coeff"] == .02
    return {"saved_agent_yaml_checked": True, "exact_registered_runner_class": type(runner).__name__,
            "upper_body_mirror_loss_coeff": .1, "upper_body_pose_loss_coeff": .02,
            "initial_learning_rate": 1e-4, "min_learning_rate": 1e-5,
            "max_learning_rate": 3e-4, "schedule": "adaptive",
            "auxiliary_losses_configured_in_actual_ppo": True,
            "arm_action_ids": list(ARM_ACTION_IDS), "checkpoint_contract_has_both_losses": True}
