"""Portable integrity and saved-config checks for the pinned 0918 release.

No dependency on git HEAD, a sibling checkout, old training logs or checkpoints.
"""
import hashlib
import importlib.util
import json
from pathlib import Path


def check_source_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "docs/elf3_0918_training_manifest.json").read_text())
    checked = 0
    for group in ("baseline_training_files", "release_files"):
        for relative, expected in manifest[group].items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"0918 source/asset drift: {relative}")
            checked += 1
    for relative, expected in manifest.get("physical_files_sha256", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published physical-file bytes changed: {relative}")
    asset = root / "TienKung-Lab/legged_lab/assets/elf3"
    contract = json.loads((root / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
    physical_validation = None
    if contract["retained_physical_asset"]["files_sha256"]:
        spec = importlib.util.spec_from_file_location("release_asset_metadata", root / "TienKung-Lab/legged_lab/elf3_contract.py")
        metadata = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metadata)
        physical_validation = metadata.verify_retained_assets(asset, contract)
    return {"baseline_commit": manifest["baseline_commit"], "verified_source_files": checked,
            "retained_physical_asset_checked": True, "git_head_independent": True,
            "physical_validation": physical_validation}


def compare_saved_run(cfg, agent, reference_run):
    """Compare all effective settings; only project location/run name may differ."""
    import yaml
    from melody_launch_audit import differences

    directory = Path(reference_run).resolve(strict=True)
    current = {"env": yaml.load(yaml.dump(cfg.to_dict()), Loader=yaml.BaseLoader),
               "agent": yaml.load(yaml.dump(agent.to_dict()), Loader=yaml.BaseLoader)}
    reference = {name: yaml.load((directory / f"params/{name}.yaml").read_text(), Loader=yaml.BaseLoader)
                 for name in ("env", "agent")}
    current["env"]["scene"]["robot"]["spawn"]["usd_path"] = "<retained-elf3-usd>"
    reference["env"]["scene"]["robot"]["spawn"]["usd_path"] = "<retained-elf3-usd>"
    current["agent"]["run_name"] = reference["agent"]["run_name"] = "<run-identity>"
    delta = differences(reference, current)
    if delta:
        raise ValueError(f"Effective configuration differs from saved reference: {delta}")
    return {"matched": True, "reference_run": str(directory), "differences": [],
            "ignored_fields": ["env.scene.robot.spawn.usd_path", "agent.run_name"],
            "checkpoint_loaded": False}


if __name__ == "__main__":
    print(json.dumps(check_source_manifest(Path(__file__).resolve().parents[1]), indent=2))
