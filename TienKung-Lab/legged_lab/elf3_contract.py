"""DWAQ observation/action metadata, separate from the standalone robot asset."""
import json
import hashlib
import math
import re

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets/elf3"
METADATA_DIR = Path(__file__).resolve().parent / "configs/elf3"

CONTRACT = json.loads((METADATA_DIR / "contract.json").read_text())
JOINT_NAMES = tuple(CONTRACT["joint_names"])
FOOT_BODIES = tuple(CONTRACT["foot_bodies"])


def verify_retained_assets(asset_dir=ASSET_DIR, contract=CONTRACT):
    """Verify unchanged physics while allowing only the named directory migration.

    Keep the historical checkpoint contract byte-identical. Its physical-file
    keys describe the original layout; only mesh references are canonicalized
    back for hashing. Any change to geometry, inertia, limits, or other XML
    content still fails verification. No legacy symlinks are needed at runtime.
    """
    checked = 0
    rewritten = []
    for relative, expected in contract["retained_physical_asset"]["files_sha256"].items():
        current = relative.replace("xmls/robot_meshes/", "meshes/")
        if relative == "xmls/elf3.xml":
            current = "mjcf/elf3.xml"
        data = (Path(asset_dir) / current).read_bytes()
        if relative == "xmls/elf3.xml":
            data = data.replace(b'meshdir="../meshes"', b'meshdir="robot_meshes"')
            rewritten.append(current)
        elif relative == "urdf/elf3.urdf":
            data = data.replace(b'filename="../meshes/', b'filename="../xmls/robot_meshes/')
            rewritten.append(current)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Retained physical content changed beyond path migration: {current}")
        checked += 1
    return {"verified_files": checked, "usd_and_mesh_bytes_unchanged": True,
            "xml_path_only_rewrites": rewritten,
            "checkpoint_contract_changed": False}


def validate_contract(cfg=None, contract=CONTRACT):
    """Reject edits to elf3.py that leave exported/checkpoint metadata stale."""
    if cfg is None:
        from legged_lab.assets.elf3.elf3 import ELF3_CFG
        cfg = ELF3_CFG
    fields = {
        "stiffness": "stiffness", "damping": "damping",
        "effort_limit_sim": "effort_limit", "velocity_limit_sim": "velocity_limit",
        "armature": "armature",
    }
    names = tuple(contract["joint_names"])
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("ELF3 contract must contain 29 unique native joint names")
    for name in names:
        matches = [actuator for actuator in cfg.actuators.values()
                   if any(re.fullmatch(pattern, name) for pattern in actuator.joint_names_expr)]
        if len(matches) != 1:
            raise ValueError(f"Joint {name} must be controlled by exactly one actuator group")
        for attribute, field in fields.items():
            value = getattr(matches[0], attribute)
            if isinstance(value, dict):
                values = [number for pattern, number in value.items() if re.fullmatch(pattern, name)]
                if len(values) != 1:
                    raise ValueError(f"Ambiguous/missing {attribute} for {name}")
                value = values[0]
            expected = contract[field][name]
            if value is None or not math.isclose(float(value), float(expected), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{name}.{attribute} differs from contract.json; update both configuration files")
        if not math.isclose(cfg.init_state.joint_pos[name], contract["default_joint_pos"][name], abs_tol=1e-9):
            raise ValueError(f"Home pose for {name} differs from contract.json")
    if tuple(cfg.init_state.pos) != tuple(contract["initial_position"]):
        raise ValueError("Root initial position differs from contract.json")
    if cfg.soft_joint_pos_limit_factor != contract["soft_joint_pos_limit_factor"]:
        raise ValueError("Soft joint position limit factor differs from contract.json")


__all__ = ["ASSET_DIR", "METADATA_DIR", "CONTRACT", "JOINT_NAMES", "FOOT_BODIES",
           "validate_contract", "verify_retained_assets"]
