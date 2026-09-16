"""Snapshot the Amp_mjlab ELF3 asset and convert its explicit MJCF to URDF.

Run with a Python environment containing numpy/scipy. No learned weights are read.
The generated URDF preserves the MJCF's already-fused head inertias and selected
collision meshes, rather than importing a different same-named ELF3 URDF.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(value):
    return np.fromstring(value, sep=" ")


def fmt(value):
    return " ".join(format(float(x), ".17g") for x in value)


def origin(parent, source):
    quat = numbers(source.get("quat", "1 0 0 0"))
    rpy = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
    ET.SubElement(parent, "origin", xyz=source.get("pos", "0 0 0"), rpy=fmt(rpy))


def convert(xml_path, urdf_path, efforts):
    mjcf = ET.parse(xml_path).getroot()
    mesh_files = {m.attrib["name"]: m.attrib["file"] for m in mjcf.findall("asset/mesh")}
    robot = ET.Element("robot", name="elf3")
    # Keep link-frame geometry, with no fixed-link merging or inferred inertias.
    def visit(body, parent=None):
        name = body.attrib["name"]
        link = ET.SubElement(robot, "link", name=name)
        source = body.find("inertial")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=source.get("pos", "0 0 0"), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=source.attrib["mass"])
        quat = numbers(source.get("quat", "1 0 0 0"))
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        tensor = rotation @ np.diag(numbers(source.attrib["diaginertia"])) @ rotation.T
        ET.SubElement(inertial, "inertia", **{
            k: format(tensor[i, j], ".17g")
            for k, i, j in [("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                            ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)]
        })
        for geom in body.findall("geom"):
            assert geom.get("type") == "mesh", geom.attrib
            visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            element = ET.SubElement(link, "visual" if visual else "collision", name=geom.attrib["name"])
            origin(element, geom)
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", filename="../xmls/robot_meshes/" + mesh_files[geom.attrib["mesh"]])
            if visual:
                material = ET.SubElement(element, "material", name="elf3_gray")
                ET.SubElement(material, "color", rgba=geom.get("rgba", "0.75 0.75 0.75 1"))
        if parent is not None:
            source_joint = body.find("joint")
            assert source_joint is not None and source_joint.get("pos", "0 0 0") == "0 0 0"
            joint_name = source_joint.attrib["name"]
            joint = ET.SubElement(robot, "joint", name=joint_name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            origin(joint, body)
            ET.SubElement(joint, "axis", xyz=source_joint.attrib["axis"])
            lower, upper = source_joint.attrib["range"].split()
            # MJCF has no velocity limit; this URDF parser-required field is nonbinding.
            ET.SubElement(joint, "limit", lower=lower, upper=upper,
                          effort=str(efforts[joint_name]), velocity="1000000000")
        for child in body.findall("body"):
            visit(child, name)
    visit(mjcf.find("worldbody/body"))
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp", type=Path, default=Path(__file__).resolve().parents[2] / "Amp_mjlab")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    target = project / "TienKung-Lab/legged_lab/assets/elf3"
    source = args.amp / "src/assets/robots/elf3"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "xmls", target / "xmls", dirs_exist_ok=True)
    # Extract only the literal robot contract and pure gain function; no mjlab imports.
    tree = ast.parse((source / "elf3_constants.py").read_text())
    selected = [node for node in tree.body if
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    {"ELF3_JOINT_NAMES", "ELF3_ACTION_SCALE", "ELF3_PHYSICAL_ROOT", "ELF3_POLICY_ROOT", "ELF3_FOOT_BODIES"}
                    for t in node.targets)
                or isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "ELF3_ACTION_SCALE"
                or isinstance(node, ast.FunctionDef) and node.name == "_actuator_params"]
    ns = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source / "elf3_constants.py"), "exec"), ns)
    home_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "ELF3_HOME_KEYFRAME" for t in n.targets))
    home = {k.arg: ast.literal_eval(k.value) for k in home_node.value.keywords}
    joints = ns["ELF3_JOINT_NAMES"]
    def resolve(mapping, name):
        values = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
        assert len(values) == 1, (name, values)
        return values[0]
    gains = {name: ns["_actuator_params"](name) for name in joints}
    contract = {
        "joint_names": joints,
        "physical_root": ns["ELF3_PHYSICAL_ROOT"],
        "policy_root": ns["ELF3_POLICY_ROOT"],
        "foot_bodies": ns["ELF3_FOOT_BODIES"],
        "initial_position": home["pos"],
        "default_joint_pos": {name: resolve(home["joint_pos"], name) for name in joints},
        "action_scale": {name: resolve(ns["ELF3_ACTION_SCALE"], name) for name in joints},
        "stiffness": {name: gains[name][0] for name in joints},
        "damping": {name: gains[name][1] for name in joints},
        "effort_limit": {name: gains[name][2] for name in joints},
        "armature": {name: gains[name][3] for name in joints},
        "soft_joint_pos_limit_factor": 0.9,
        "physics_dt": 0.005,
        "control_decimation": 4,
    }
    (target / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (target / "urdf").mkdir(exist_ok=True)
    convert(target / "xmls/elf3.xml", target / "urdf/elf3.urdf", contract["effort_limit"])
    provenance = {
        "amp_repository": str(args.amp.resolve()),
        "amp_commit": subprocess.check_output(["git", "-C", str(args.amp), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file() and p.suffix != ".pyc"},
        "generated_sha256": {str(p.relative_to(target)): sha(p) for p in [target / "contract.json", target / "urdf/elf3.urdf"]},
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(source / "elf3_constants.py", target / "amp_elf3_constants.py.reference")
    print(target)


if __name__ == "__main__":
    main()
