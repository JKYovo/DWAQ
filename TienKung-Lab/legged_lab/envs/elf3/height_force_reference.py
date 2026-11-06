"""ELF3 nominal torso-to-sole height from the retained URDF and named home pose."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SOLE_POINTS = ((-.075750000775, 0., -.041000001132),
               (.128939226270, -.000694592716, -.041000001132))
FOOT_NAMES = ("l_ankle_x_link", "r_ankle_x_link")


def default_height_reference():
    root = Path(__file__).resolve().parents[2]
    urdf = root / "assets/elf3/urdf/elf3.urdf"
    config = root / "configs/elf3/contract.json"
    contract = json.loads(config.read_text())
    home = contract["default_joint_pos"]
    transforms = {"torso_link": np.eye(4)}
    pending = list(ET.parse(urdf).getroot().findall("joint"))
    while pending:
        progress = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            r,p,y = rpy
            rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
            ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
            rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
            local = np.eye(4); local[:3,:3] = rz @ ry @ rx; local[:3,3] = xyz
            if joint.get("type") != "fixed":
                if joint.get("type") not in ("revolute", "continuous"):
                    raise ValueError("Unexpected ELF3 joint type")
                axis_element = joint.find("axis")
                axis = np.fromstring(axis_element.get("xyz", "1 0 0"),sep=" ") if axis_element is not None else np.array([1.,0.,0.])
                axis /= np.linalg.norm(axis)
                ax,ay,az = axis
                cross = np.array([[0,-az,ay],[az,0,-ax],[-ay,ax,0]])
                angle = home[joint.get("name")]
                rotation = np.eye(4)
                rotation[:3,:3] = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
                local = local @ rotation
            transforms[child] = transforms[parent] @ local
            pending.remove(joint); progress = True
        if not progress:
            raise ValueError("URDF is not a complete tree rooted at torso_link")
    points = np.c_[np.asarray(SOLE_POINTS),np.ones(2)]
    heights = [-float((transforms[name] @ points.T)[2].min()) for name in FOOT_NAMES]
    if abs(heights[0]-heights[1]) > 1e-5 or not .9 < np.mean(heights) < 1.2:
        raise ValueError("Unexpected nominal ELF3 leg geometry or asymmetric home height")
    return {"nominal_height_m":float(np.mean(heights)), "left_right_height_m":heights,
            "definition":"upright torso_link origin minus lowest verified sole point at named default q; NOT spawn Z",
            "urdf_sha256":hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "asset_contract_sha256":hashlib.sha256(config.read_bytes()).hexdigest()}
