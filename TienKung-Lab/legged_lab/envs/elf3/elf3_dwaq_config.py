"""Robot-only adaptation of upstream G1 DWAQ; reward weights are inherited."""
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.assets.elf3 import CONTRACT, ELF3_CFG, FOOT_BODIES
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg


@configclass
class Elf3DwaqEnvCfg(G1DwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ELF3_CFG.copy()
        self.scene.terrain_generator = deepcopy(self.scene.terrain_generator)
        self.scene.height_scanner.prim_body_name = CONTRACT["physical_root"]
        self.robot.terminate_contacts_body_names = [CONTRACT["physical_root"]]
        self.robot.feet_body_names = list(FOOT_BODIES)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [CONTRACT["physical_root"]]
        # The environment expands Amp_mjlab's per-joint scales in simulator order.
        self.robot.action_scale = 1.0
        self.sim.dt = CONTRACT["physics_dt"]
        self.sim.decimation = CONTRACT["control_decimation"]
        # Local display material avoids fetching remote MDL assets at startup.
        self.scene.terrain_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
        self.scene.sky_texture_file = None
        self.reward = deepcopy(self.reward)
        feet_terms = ["fly", "feet_air_time", "feet_slide", "feet_force", "feet_too_near",
                      "feet_stumble", "gait_phase_contact", "feet_swing_height"]
        for name in feet_terms:
            for param in getattr(self.reward, name).params.values():
                if isinstance(param, SceneEntityCfg) and param.body_names is not None:
                    param.body_names = list(FOOT_BODIES)
                    param.preserve_order = True
        self.reward.undesired_contacts.params["sensor_cfg"].body_names = "(?![lr]_ankle_[xy]_link$).*"
        self.reward.body_orientation_l2.params["asset_cfg"].body_names = [CONTRACT["physical_root"]]
        joint_groups = {
            "joint_deviation_hip": ["[lr]_hip_[zx]_joint"],
            "joint_deviation_ankle": ["[lr]_ankle_[yx]_joint"],
            "joint_deviation_arms": ["waist_[yxz]_joint", "[lr]_shoulder_[yxz]_joint",
                                     "[lr]_elbow_y_joint", "[lr]_wrist_[xyz]_joint"],
            "joint_deviation_legs": ["[lr]_hip_y_joint", "[lr]_knee_y_joint"],
        }
        for name, patterns in joint_groups.items():
            getattr(self.reward, name).params["asset_cfg"].joint_names = patterns


@configclass
class Elf3DwaqFlatEnvCfg(Elf3DwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None


@configclass
class Elf3DwaqAgentCfg(G1DwaqAgentCfg):
    max_iterations: int = 100001
    experiment_name: str = "elf3_dwaq"
    wandb_project: str = "elf3_dwaq"
    runner_class_name: str = "Elf3DwaqOnPolicyRunner"


@configclass
class Elf3DwaqFlatAgentCfg(Elf3DwaqAgentCfg):
    experiment_name: str = "elf3_dwaq_flat"


@configclass
class Elf3DwaqUpperBodySymmetryAgentCfg(Elf3DwaqAgentCfg):
    """Independent 20k mirror-only test; rewards and control limits remain unchanged."""

    max_iterations: int = 20000
    experiment_name: str = "elf3_dwaq_upper_symmetry"
    wandb_project: str = "elf3_dwaq_upper_symmetry"
    runner_class_name: str = "Elf3DwaqUpperBodySymmetryRunner"
    upper_body_mirror_loss_coeff: float = 0.1


@configclass
class Elf3DwaqUpperBodySymmetryPoseAgentCfg(Elf3DwaqUpperBodySymmetryAgentCfg):
    """Independent 20k test adding a soft default-pose loss to mirrored arm actions."""

    experiment_name: str = "elf3_dwaq_upper_symmetry_pose"
    wandb_project: str = "elf3_dwaq_upper_symmetry_pose"
    upper_body_pose_loss_coeff: float = 0.02
