# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""ELF3 configuration in MelodyAI's named actuator-group format.

Matches the active 0918 symmetry/pose experiment's robot configuration.
The retained USD is the 43.22 kg model; all 29 armatures are explicitly 0.01.
Nominal Kp/Kd, home pose and action scales are unchanged.  If editing the
joint parameters below, update legged_lab/configs/elf3/contract.json too.
DWAQ validates that metadata when constructing its environment; the robot
asset itself remains independently importable.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parent

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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,

            "l_hip_y_joint": -0.3,
            "l_hip_x_joint": 0.0,
            "l_hip_z_joint": 0.0,
            "l_knee_y_joint": 0.6,
            "l_ankle_y_joint": -0.3,
            "l_ankle_x_joint": 0.0,

            "r_hip_y_joint": -0.3,
            "r_hip_x_joint": 0.0,
            "r_hip_z_joint": 0.0,
            "r_knee_y_joint": 0.6,
            "r_ankle_y_joint": -0.3,
            "r_ankle_x_joint": 0.0,

            "l_shoulder_y_joint": 0.2,
            "l_shoulder_x_joint": 0.2,
            "l_shoulder_z_joint": 0.0,
            "l_elbow_y_joint": 0.6,
            "l_wrist_x_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_z_joint": 0.0,

            "r_shoulder_y_joint": 0.2,
            "r_shoulder_x_joint": -0.2,
            "r_shoulder_z_joint": 0.0,
            "r_elbow_y_joint": 0.6,
            "r_wrist_x_joint": 0.0,
            "r_wrist_y_joint": 0.0,
            "r_wrist_z_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            effort_limit_sim={
                "waist_y_joint": 100,
                "waist_x_joint": 100,
                "waist_z_joint": 100,
            },
            velocity_limit_sim={
                "waist_y_joint": 20,
                "waist_x_joint": 20,
                "waist_z_joint": 20,
            },
            stiffness={
                "waist_y_joint": 108.448,
                "waist_x_joint": 162.672,
                "waist_z_joint": 176.421,
            },
            damping={
                "waist_y_joint": 6.904,
                "waist_x_joint": 10.356,
                "waist_z_joint": 11.231,
            },
        ),
        "legs": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                ".*_hip_y_joint",
                ".*_hip_x_joint",
                ".*_hip_z_joint",
                ".*_knee_y_joint",
            ],
            effort_limit_sim={
                ".*_hip_y_joint": 100,
                ".*_hip_x_joint": 100,
                ".*_hip_z_joint": 50,
                ".*_knee_y_joint": 150,
            },
            velocity_limit_sim={
                ".*_hip_y_joint": 20,
                ".*_hip_x_joint": 20,
                ".*_hip_z_joint": 20,
                ".*_knee_y_joint": 20,
            },
            stiffness={
                ".*_hip_y_joint": 176.421,
                ".*_hip_x_joint": 176.421,
                ".*_hip_z_joint": 54.224,
                ".*_knee_y_joint": 176.421,
            },
            damping={
                ".*_hip_y_joint": 11.231,
                ".*_hip_x_joint": 11.231,
                ".*_hip_z_joint": 3.452,
                ".*_knee_y_joint": 11.231,
            },
        ),
        "feet": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                ".*_ankle_y_joint",
                ".*_ankle_x_joint",
            ],
            effort_limit_sim={
                ".*_ankle_y_joint": 50,
                ".*_ankle_x_joint": 20,
            },
            velocity_limit_sim={
                ".*_ankle_y_joint": 20,
                ".*_ankle_x_joint": 20,
            },
            stiffness={
                ".*_ankle_y_joint": 33.493,
                ".*_ankle_x_joint": 21.771,
            },
            damping={
                ".*_ankle_y_joint": 2.132,
                ".*_ankle_x_joint": 1.386,
            },
        ),
        "arms": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                ".*_shoulder_y_joint",
                ".*_shoulder_x_joint",
                ".*_shoulder_z_joint",
                ".*_elbow_y_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_y_joint": 50,
                ".*_shoulder_x_joint": 50,
                ".*_shoulder_z_joint": 25,
                ".*_elbow_y_joint": 50,
            },
            velocity_limit_sim={
                ".*_shoulder_y_joint": 20,
                ".*_shoulder_x_joint": 20,
                ".*_shoulder_z_joint": 20,
                ".*_elbow_y_joint": 20,
            },
            stiffness={
                ".*_shoulder_y_joint": 54.224,
                ".*_shoulder_x_joint": 54.224,
                ".*_shoulder_z_joint": 16.747,
                ".*_elbow_y_joint": 54.224,
            },
            damping={
                ".*_shoulder_y_joint": 3.452,
                ".*_shoulder_x_joint": 3.452,
                ".*_shoulder_z_joint": 1.066,
                ".*_elbow_y_joint": 3.452,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                ".*_wrist_x_joint",
                ".*_wrist_y_joint",
                ".*_wrist_z_joint",
            ],
            effort_limit_sim={
                ".*_wrist_x_joint": 25,
                ".*_wrist_y_joint": 25,
                ".*_wrist_z_joint": 25,
            },
            velocity_limit_sim={
                ".*_wrist_x_joint": 20,
                ".*_wrist_y_joint": 20,
                ".*_wrist_z_joint": 20,
            },
            stiffness={
                ".*_wrist_x_joint": 16.747,
                ".*_wrist_y_joint": 16.747,
                ".*_wrist_z_joint": 16.747,
            },
            damping={
                ".*_wrist_x_joint": 1.066,
                ".*_wrist_y_joint": 1.066,
                ".*_wrist_z_joint": 1.066,
            },
        ),
    },
)

# Original DWAQ imports ELF3_CFG; expose MelodyAI's spelling as an alias.
ELF3LITE_CFG = ELF3_CFG

__all__ = ["ASSET_DIR", "ELF3_CFG", "ELF3LITE_CFG"]
