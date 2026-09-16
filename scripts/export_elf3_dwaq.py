"""Export an ELF3 DWAQ checkpoint using the upstream deterministic mean path."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT / "TienKung-Lab/legged_lab/assets/elf3"


def load_policy(checkpoint):
    source = PROJECT / "TienKung-Lab/legged_lab/scripts/export_dwaq_policy.py"
    spec = importlib.util.spec_from_file_location("upstream_dwaq_export", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = json.loads((ASSET_DIR / "contract.json").read_text())
    metadata = (saved.get("infos") or {}).get("elf3_contract", {})
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if metadata.get("asset_contract_sha256") != digest or metadata.get("joint_names") != contract["joint_names"]:
        raise ValueError("Expected a checkpoint trained with this ELF3 contract")
    if metadata.get("num_obs") != 100 or metadata.get("history_length") != 5:
        raise ValueError("This exporter requires the 100 x 5 ELF3 gait-phase observation contract")
    if "obs_norm_state_dict" in saved:
        raise ValueError("This ELF3 task uses upstream unnormalized observations")
    policy = module.DWAQPolicyExporter(num_obs=100, num_actions=29, dwaq_obs_history_length=5)
    state = saved["model_state_dict"]
    # Require every exported tensor; never silently retain randomly initialized layers.
    policy.load_state_dict({key: state[key] for key in policy.state_dict()}, strict=True)
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
        for batch in (1, 3, 8):
            x = torch.randn(batch, 500)
            actual = session.run(None, {"history": x.numpy()})[0]
            y = policy(x).numpy()
            np.testing.assert_allclose(actual, y, atol=2e-6, rtol=2e-5)
            errors.append(float(np.max(np.abs(actual - y))))
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == before
    metadata.update({
        "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": before,
        "contract": contract, "history_order": "oldest_to_newest; first frame repeated on reset",
        "features": ["root_ang_vel_b[3]", "projected_gravity_b[3]", "command[vx,vy,wz]",
                     "joint_pos_minus_default[29]", "joint_vel[29]", "previous_action[29]",
                     "sin(left),sin(right),cos(left),cos(right)"],
        "inference": "upstream export: encoder mean velocity + mean latent + current frame",
        "onnx_max_abs_error": max(errors),
    })
    (args.output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Export passed: {args.output}; ONNX max absolute error {max(errors):.3g}")


if __name__ == "__main__":
    main()
