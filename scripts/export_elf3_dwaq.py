"""Export an ELF3 DWAQ checkpoint using the upstream deterministic mean path."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT / "TienKung-Lab/legged_lab/assets/elf3"
CONTRACT_PATH = ASSET_DIR.parents[1] / "configs/elf3/contract.json"
STILL_VERSION = "elf3_dwaq_still_v1"
STILL_V2_VERSION = "elf3_dwaq_still_v2"
FEEDBACK_STILL_V2_VERSION = "elf3_dwaq_feedback_still_v2"
FEEDBACK_STAND_V3_VERSION = "elf3_dwaq_feedback_stand_v3"
FEEDBACK_STAND_V3_DEFAULT_POSE_VERSION = "elf3_dwaq_feedback_stand_v3_default_pose_v1"
FEEDBACK_STAND_V3_ZERO_PHASE_VERSION = "elf3_dwaq_feedback_stand_v3_zero_phase_v1"


def load_still_preprocessor():
    """Load pure Torch preprocessing without importing Isaac/environment packages."""
    source = PROJECT / "TienKung-Lab/legged_lab/envs/elf3/stillness.py"
    spec = importlib.util.spec_from_file_location("elf3_export_stillness", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_still_inputs


class MinimalStillPolicyExporter(nn.Module):
    """Embed the task's zero-command phase treatment in the deployment graph.

    This changes inputs, never actions or PD gains: moving samples use the old
    deterministic DWAQ policy unchanged. No deployment-side mode is necessary.
    """

    def __init__(self, policy, command_epsilon=1e-6):
        super().__init__()
        self.policy = policy
        self.command_epsilon = command_epsilon
        self.prepare_inputs = load_still_preprocessor()

    def forward(self, history):
        _, prepared = self.prepare_inputs(history[:, -100:], history, self.command_epsilon)
        return self.policy(prepared)


def load_stand_v3_preprocessor():
    """Load the unchanged mirror-invariant phase helper for versioned tasks."""
    source = PROJECT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py"
    spec = importlib.util.spec_from_file_location("elf3_export_stand_v3_inputs", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_stand_inputs


class FeedbackStandV3PolicyExporter(nn.Module):
    """Embed strict-zero phase handling for explicitly versioned v3 models.

    The model keeps the raw 500-input/29-action deployment interface. Only
    strict-zero phase features change; no target locking or extra filter is
    introduced, and moving inputs retain the original deterministic mean path.
    """

    def __init__(self, policy, command_epsilon=1e-6):
        super().__init__()
        self.policy = policy
        self.command_epsilon = command_epsilon
        self.prepare_inputs = load_stand_v3_preprocessor()

    def forward(self, history):
        _, prepared = self.prepare_inputs(history[:, -100:], history, self.command_epsilon)
        return self.policy(prepared)


def legacy_stand_v3_settings():
    """Portable, immutable schema of the earlier bounded-score experiment.

    The registered task name was reused with a new checkpoint version. Old
    checkpoints must retain their original inputs even when the current v3
    contract module no longer contains the old reward/phase settings.
    """
    return {
        "version": FEEDBACK_STAND_V3_VERSION,
        "parent_task": "elf3_dwaq_feedback_v1",
        "command_epsilon": 1e-6,
        "fixed_phase": [0., 0., 1., 1.],
        "phase_scope": "current_and_all_history_if_current_zero",
        "score_weight": 1.0,
        "component_weights": {"pose": .4, "joint_motion": .3, "base_motion": .2, "upright": .1},
        "pose_std": .20,
        "joint_velocity_std": .25,
        "base_linear_std": .05,
        "base_angular_std": .10,
        "torso_gravity_std": .15,
        "pose_target": "asset.default_joint_pos",
        "replace_zero_group_pose_terms": True,
        "reset_standing_command_before_first_observation": True,
        "additional_ppo_losses": False,
        "lock_world_yaw": False,
    }


def validate_zero_phase_contract(standing):
    """Validate the new zero-phase task without reinterpreting older versions."""
    directory = PROJECT / "TienKung-Lab/legged_lab/envs/elf3"
    source = directory / "feedback_stand_v3_contract.py"
    spec = importlib.util.spec_from_file_location("export_feedback_stand_v3_zero_phase_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.stand_settings()
    selected = {key: standing.get(key) for key in expected}
    if (expected.get("version") != FEEDBACK_STAND_V3_ZERO_PHASE_VERSION
            or json.dumps(selected, sort_keys=True, allow_nan=False)
            != json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("Incompatible zero-phase standing checkpoint settings")
    if standing.get("no_phase_preprocessing") is not False:
        raise ValueError("Zero-phase checkpoint must explicitly declare input preprocessing")
    for key in ("score_weight", "component_weights",
                "replace_zero_group_pose_terms", "reset_standing_command_before_first_observation"):
        if key in standing:
            raise ValueError(f"Zero-phase standing checkpoint contains a legacy objective: {key}")
    sources = standing.get("sources_sha256")
    if not isinstance(sources, dict):
        raise ValueError("Zero-phase standing checkpoint is missing source fingerprints")
    for name in ("feedback_stand_v3_contract.py", "feedback_stand_v3_env.py",
                 "elf3_feedback_stand_v3_config.py", "feedback_stand_v3_rewards.py",
                 "feedback_stand_v3_runner.py", "feedback_stand_v3_inputs.py", "standing_rewards.py"):
        if sources.get(name) != hashlib.sha256((directory / name).read_bytes()).hexdigest():
            raise ValueError(f"Zero-phase standing checkpoint source differs from this exporter: {name}")
    return expected


def validate_still_v2_contract(metadata):
    """Validate the independent still-v2 pose objective, not feedback-still-v2.

    Its Actor uses the original still-v1 preprocessing. This metadata adds no
    network, phase convention, action processing or deployment-side filter.
    Absence retains historical still/still-pose/feedback export behavior.
    """
    if "still_v2_experiment" not in metadata:
        return
    experiment = metadata["still_v2_experiment"]
    if not isinstance(experiment, dict) or experiment.get("version") != STILL_V2_VERSION:
        raise ValueError("Unsupported or incomplete still-v2 objective metadata")
    standing = metadata.get("standing_experiment")
    if not isinstance(standing, dict) or standing.get("version") != STILL_VERSION:
        raise ValueError("Still-v2 requires its original still-v1 standing contract")
    directory = PROJECT / "TienKung-Lab/legged_lab/envs/elf3"
    spec = importlib.util.spec_from_file_location("export_still_v2_contract", directory / "still_v2_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.pose_settings()
    selected = {key: experiment.get(key) for key in expected}
    if (expected.get("version") != STILL_V2_VERSION
            or json.dumps(selected, sort_keys=True, allow_nan=False)
            != json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("Incompatible still-v2 default-pose settings")
    sources = experiment.get("sources_sha256")
    if not isinstance(sources, dict):
        raise ValueError("Still-v2 objective is missing source fingerprints")
    for name in ("still_v2_contract.py", "still_v2_rewards.py", "elf3_still_v2_config.py",
                 "still_v2_runner.py", "still_pose_rewards.py"):
        if sources.get(name) != hashlib.sha256((directory / name).read_bytes()).hexdigest():
            raise ValueError(f"Still-v2 objective source differs from this exporter: {name}")
    standing_sources = standing.get("sources_sha256")
    if (not isinstance(standing_sources, dict)
            or standing_sources.get("stillness.py") != hashlib.sha256((directory / "stillness.py").read_bytes()).hexdigest()):
        raise ValueError("Still-v2 checkpoint original still-v1 preprocessing source differs from this exporter")


def still_preprocessing_settings(metadata):
    """Fail closed for unknown/incomplete still contracts; retain legacy export."""
    validate_still_v2_contract(metadata)
    standing = metadata.get("standing_experiment") or {}
    version = standing.get("version", "")
    if version == FEEDBACK_STAND_V3_ZERO_PHASE_VERSION:
        return validate_zero_phase_contract(standing)
    if version == FEEDBACK_STAND_V3_DEFAULT_POSE_VERSION:
        raise ValueError("Intermediate default-pose checkpoint has no zero-phase preprocessing; "
                         "use its matching archived exporter, not the current zero-phase task")
    if version == FEEDBACK_STAND_V3_VERSION:
        expected = legacy_stand_v3_settings()
        for key, value in expected.items():
            if standing.get(key) != value:
                raise ValueError(f"Incompatible {FEEDBACK_STAND_V3_VERSION} setting: {key}")
        sources = standing.get("sources_sha256") or {}
        preprocessor = PROJECT / "TienKung-Lab/legged_lab/envs/elf3/feedback_stand_v3_inputs.py"
        if sources.get("feedback_stand_v3_inputs.py") != hashlib.sha256(preprocessor.read_bytes()).hexdigest():
            raise ValueError("Standing v3 checkpoint preprocessing source differs from this exporter")
        return expected
    if version == FEEDBACK_STILL_V2_VERSION:
        source = PROJECT / "TienKung-Lab/legged_lab/envs/elf3/feedback_still_v2_contract.py"
        spec = importlib.util.spec_from_file_location("export_feedback_still_v2_contract", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = module.standing_settings()
        for key, value in expected.items():
            if standing.get(key) != value:
                raise ValueError(f"Incompatible {FEEDBACK_STILL_V2_VERSION} setting: {key}")
        sources = standing.get("sources_sha256") or {}
        preprocessor = PROJECT / "TienKung-Lab/legged_lab/envs/elf3/stillness.py"
        if sources.get("stillness.py") != hashlib.sha256(preprocessor.read_bytes()).hexdigest():
            raise ValueError("Standing v2 checkpoint preprocessing source differs from this exporter")
        return expected
    if version == STILL_VERSION:
        expected = {
            "command_epsilon": 1e-6,
            "fixed_phase": [0.0, 0.0, 1.0, -1.0],
            "phase_scope": "current_and_all_history_if_current_zero",
            "joint_motion_weight": -0.5,
        }
        for key, value in expected.items():
            if standing.get(key) != value:
                raise ValueError(f"Incompatible {STILL_VERSION} setting: {key}")
        return expected
    if isinstance(version, str) and version.startswith("elf3_dwaq_still"):
        raise ValueError(f"Unsupported zero-command policy contract: {version}")
    if isinstance(version, str) and version.startswith("elf3_dwaq_feedback_still"):
        raise ValueError(f"Unsupported zero-command policy contract: {version}")
    if isinstance(version, str) and version.startswith("elf3_dwaq_feedback_stand"):
        raise ValueError(f"Unsupported zero-command policy contract: {version}")
    return None


def preprocessing_metadata(metadata):
    settings = still_preprocessing_settings(metadata)
    if settings is None:
        return {}
    return {
        "zero_command_preprocessing": {
            "embedded_in_model": True,
            "command_condition": "max(abs(current_observation[6:9])) <= 1e-6",
            "phase_features": [96, 97, 98, 99],
            "fixed_phase": settings["fixed_phase"],
            "scope": settings["phase_scope"],
            "moving_path": "unchanged deterministic DWAQ mean context",
            "extra_velocity_filter": False,
            "extra_action_filter": False,
            "joint_lock": False,
        },
        "inference": (
            "deterministic DWAQ encoder mean velocity + mean latent + current frame; "
            "zero-current-command phase preprocessing embedded for current and all history"
        ),
    }


def load_policy(checkpoint):
    source = PROJECT / "TienKung-Lab/legged_lab/scripts/export_dwaq_policy.py"
    spec = importlib.util.spec_from_file_location("upstream_dwaq_export", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = json.loads(CONTRACT_PATH.read_text())
    metadata = (saved.get("infos") or {}).get("elf3_contract", {})
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if metadata.get("asset_contract_sha256") != digest or metadata.get("joint_names") != contract["joint_names"]:
        raise ValueError("Expected a checkpoint trained with this ELF3 contract")
    if metadata.get("num_obs") != 100 or metadata.get("history_length") != 5:
        raise ValueError("This exporter requires the 100 x 5 ELF3 gait-phase observation contract")
    if "obs_norm_state_dict" in saved:
        raise ValueError("This ELF3 task uses upstream unnormalized observations")
    still_settings = still_preprocessing_settings(metadata)
    policy = module.DWAQPolicyExporter(num_obs=100, num_actions=29, dwaq_obs_history_length=5)
    state = saved["model_state_dict"]
    # Require every exported tensor; never silently retain randomly initialized layers.
    policy.load_state_dict({key: state[key] for key in policy.state_dict()}, strict=True)
    if (metadata.get("standing_experiment") or {}).get("version") in (
            FEEDBACK_STAND_V3_VERSION, FEEDBACK_STAND_V3_ZERO_PHASE_VERSION):
        policy = FeedbackStandV3PolicyExporter(policy, still_settings["command_epsilon"])
    elif still_settings is not None:
        policy = MinimalStillPolicyExporter(policy, still_settings["command_epsilon"])
    policy.eval()
    return policy, contract, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for policy.pt/onnx/json")
    args = parser.parse_args()
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy, contract, metadata = load_policy(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    sample = torch.randn(3, 500)
    with torch.inference_mode():
        expected = policy(sample)
        traced = torch.jit.trace(policy, sample)
        traced.save(str(args.output / "policy.pt"))
        torch.onnx.export(policy, sample, str(args.output / "policy.onnx"), opset_version=17,
                          input_names=["history"], output_names=["actions"],
                          dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}})
        torch.testing.assert_close(torch.jit.load(str(args.output / "policy.pt"))(sample), expected)
        session = ort.InferenceSession(str(args.output / "policy.onnx"), providers=["CPUExecutionProvider"])
        errors = []
        target_errors = []
        action_scale = np.asarray([contract["action_scale"][name] for name in contract["joint_names"]])
        validation_inputs = [torch.randn(batch, 500) for batch in (1, 3, 8)]
        if isinstance(policy, (MinimalStillPolicyExporter, FeedbackStandV3PolicyExporter)):
            # Cover both branches, including epsilon boundaries and histories
            # containing old moving commands when the current command is zero.
            mixed = torch.randn(8, 500)
            mixed[:3, -94:-91] = 0
            mixed[3, -94:-91] = 1e-6
            mixed[4, -94:-91] = -1e-6
            mixed[5, -94:-91] = 1.001e-6
            validation_inputs.append(mixed)
        for x in validation_inputs:
            actual = session.run(None, {"history": x.numpy()})[0]
            y = policy(x).numpy()
            if not np.isfinite(actual).all() or not np.isfinite(y).all():
                raise FloatingPointError("Nonfinite output during ONNX validation")
            # CPU ONNX Runtime and PyTorch use different FP32 kernel reduction
            # orders.  Large out-of-distribution test inputs can accumulate a
            # sub-milliradian joint-target difference without changing the policy.
            np.testing.assert_allclose(actual, y, atol=1e-3, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual - y))))
            target_errors.append(float(np.max(np.abs(actual - y) * action_scale)))
        if max(target_errors) > 2e-3:
            raise AssertionError(f"ONNX joint-target error exceeds 2 mrad: {max(target_errors):.6g}")
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == before
    metadata.update({
        "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": before,
        "contract": contract, "history_order": "oldest_to_newest; first frame repeated on reset",
        "features": ["root_ang_vel_b[3]", "projected_gravity_b[3]", "command[vx,vy,wz]",
                     "joint_pos_minus_default[29]", "joint_vel[29]", "previous_action[29]",
                     "sin(left),sin(right),cos(left),cos(right)"],
        "inference": "upstream export: encoder mean velocity + mean latent + current frame",
        "policy_input_dim": 500, "num_actions": 29,
        "onnx_max_abs_error": max(errors),
        "onnx_max_joint_target_error_rad": max(target_errors),
    })
    metadata.update(preprocessing_metadata(metadata))
    (args.output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Export passed: {args.output}; ONNX max absolute error {max(errors):.3g}")


if __name__ == "__main__":
    main()
