"""Deterministic flat recovery area and command partition; no simulator imports."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from legged_lab.envs.elf3.recovery_v2_core import (
    SETTINGS as V2_SETTINGS, MotionLibrary, recovery_contract as v2_contract,
)


TASK = "elf3_dwaq_recovery40_flat_v3"
SETTINGS = {**deepcopy(V2_SETTINGS), "version": TASK,
    "recovery_terrain": "separate flat pad in the same collision/raycast mesh; original17x20 terrain unchanged",
    "recovery_command": "zero vx/vy/wz for assigned40%; original command sampling for normal60%",
    "recovery_push": "disabled for assigned40%; original push event for normal60%",
    "recovery_curriculum": "fixed flat origins, excluded from walking distance curriculum and its mean level",
    "flat_spacing_m": 4., "flat_margin_m": 8.}


def flat_layout(count, size, num_rows, border_width):
    if count < 1:
        raise ValueError("At least one recovery environment is required")
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count/columns)
    spacing, margin = SETTINGS["flat_spacing_m"], SETTINGS["flat_margin_m"]
    # Beyond the entire original generator including its border and tile overhang.
    start_x = size[0]*num_rows*.5 + border_width + 2*max(size) + 16.
    width, depth = (rows-1)*spacing+2*margin, (columns-1)*spacing+2*margin
    index = np.arange(count)
    origins = np.zeros((count,3), dtype=np.float32)
    origins[:,0] = start_x+margin+(index//columns)*spacing
    origins[:,1] = -depth*.5+margin+(index%columns)*spacing
    bounds = np.array([[start_x,-depth*.5,0.],[start_x+width,depth*.5,0.]],dtype=np.float64)
    return origins, bounds


class SplitCommands:
    """Run the original command generator, then zero only recovery rows."""
    def __init__(self, original, assigned):
        self.original, self.assigned = original, assigned

    def __getattr__(self, name):
        return getattr(self.original, name)

    def enforce(self):
        self.original.command[self.assigned] = 0.
        self.original.is_standing_env[self.assigned] = True
        self.original.is_heading_env[self.assigned] = False

    def compute(self, dt):
        result = self.original.compute(dt)
        self.enforce()
        return result

    def reset(self, env_ids):
        result = self.original.reset(env_ids)
        self.enforce()
        return result


def recovery_contract(motion_identity):
    result = deepcopy(SETTINGS)
    result["recovery_v2_logic"] = v2_contract(motion_identity)
    result["sources_sha256"] = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("flat_recovery_core.py","flat_recovery_env.py","flat_recovery_runner.py")}
    return result
