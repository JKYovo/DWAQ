"""ELF3 physical asset and native action contract, snapshotted from Amp_mjlab."""
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parent
CONTRACT = json.loads((ASSET_DIR / "contract.json").read_text())
JOINT_NAMES = tuple(CONTRACT["joint_names"])
FOOT_BODIES = tuple(CONTRACT["foot_bodies"])

ELF3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "usd/elf3.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(CONTRACT["initial_position"]),
        joint_pos=CONTRACT["default_joint_pos"],
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=CONTRACT["soft_joint_pos_limit_factor"],
    actuators={
        "native_pd": IdealPDActuatorCfg(
            joint_names_expr=list(JOINT_NAMES),
            stiffness=CONTRACT["stiffness"],
            damping=CONTRACT["damping"],
            effort_limit=CONTRACT["effort_limit"],
            effort_limit_sim=CONTRACT["effort_limit"],
            armature=CONTRACT["armature"],
            friction=0.0,
        ),
    },
)
