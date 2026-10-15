"""Validate this standalone asset in PhysX without importing tasks or training.

Run with the project's Isaac Lab Python environment and --headless.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--reference-config", type=Path, default=None)
parser.add_argument("--report", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import importlib.util
import json
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from legged_lab.assets.elf3 import ASSET_DIR, ELF3_CFG, ELF3LITE_CFG
from legged_lab.elf3_contract import CONTRACT, JOINT_NAMES, validate_contract, verify_retained_assets


@configclass
class AssetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    robot = ELF3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    assert ASSET_DIR == Path(__file__).resolve().parents[1] / "TienKung-Lab/legged_lab/assets/elf3", "Wrong project's asset imported"
    assert ELF3LITE_CFG is ELF3_CFG
    validate_contract()
    physical_validation = verify_retained_assets()
    from pxr import UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(ASSET_DIR / "usd/elf3.usd"))
    # USD's generic resolver does not use Kit's MDL module search paths.
    # Only this exact inherited shader module is non-geometric; all missing
    # layers, meshes, textures, or other references remain hard failures.
    unresolved_geometry = [path for path in unresolved if path != "OmniPBR.mdl"]
    assert not unresolved_geometry, f"Unresolved USD dependencies: {unresolved_geometry}"
    if args.reference_config is not None:
        spec = importlib.util.spec_from_file_location("reference_elf3", args.reference_config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = module.ELF3LITE_CFG.copy()
        expected.spawn.usd_path = str(ASSET_DIR / "usd/elf3.usd")
        for actuator in expected.actuators.values():
            actuator.armature = .01
        assert ELF3_CFG.to_dict() == expected.to_dict(), "Not the active MelodyAI robot profile"
    guards = []
    for field, value in (("armature", .02), ("stiffness", 1.), ("effort_limit_sim", 150.), ("velocity_limit_sim", 30.)):
        changed = ELF3_CFG.copy()
        setattr(changed.actuators["legs"], field, value)
        try:
            validate_contract(changed)
        except ValueError:
            guards.append(field)
        else:
            raise AssertionError(f"Stale contract accepted for {field}")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=CONTRACT["physics_dt"], device=args.device))
    scene = InteractiveScene(AssetSceneCfg(num_envs=args.num_envs, env_spacing=2.5))
    sim.reset()
    robot = scene["robot"]
    ids, names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    assert tuple(names) == JOINT_NAMES and robot.num_joints == 29
    assert robot.body_names[0] == CONTRACT["physical_root"]
    assert all(actuator.is_implicit_model for actuator in robot.actuators.values())
    getters = {
        "armature": robot.root_physx_view.get_dof_armatures,
        "effort_limit": robot.root_physx_view.get_dof_max_forces,
        "velocity_limit": robot.root_physx_view.get_dof_max_velocities,
        "stiffness": robot.root_physx_view.get_dof_stiffnesses,
        "damping": robot.root_physx_view.get_dof_dampings,
    }
    for field, getter in getters.items():
        actual = getter().to(robot.device)[:, ids]
        expected = torch.tensor([CONTRACT[field][name] for name in JOINT_NAMES], device=robot.device, dtype=actual.dtype)
        torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-6, atol=1e-7)
    masses = robot.data.default_mass.sum(dim=1)
    torch.testing.assert_close(masses, torch.full_like(masses, 43.2225077), atol=1e-4, rtol=0)
    expected_home = torch.tensor([CONTRACT["default_joint_pos"][name] for name in JOINT_NAMES], device=robot.device)
    torch.testing.assert_close(robot.data.default_joint_pos[:, ids], expected_home.expand(args.num_envs, -1))
    for _ in range(24):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all()
    report = {"passed": True, "asset_directory": str(ASSET_DIR), "num_envs": args.num_envs,
              "physics_steps": 24, "control_type": "ImplicitActuator", "joint_count": 29,
              "armature_kg_m2": .01, "joint_velocity_limit_rad_s": 20.,
              "total_mass_kg": float(masses[0].item()), "joint_names": list(JOINT_NAMES),
              "policy_joint_ids": ids, "retained_physical_asset_validation": physical_validation,
              "usd_dependency_layers": len(layers), "unresolved_usd_dependencies": unresolved_geometry,
              "usd_generic_resolver_builtin_mdl_modules": sorted(set(unresolved)),
              "reference_config_checked": str(args.reference_config) if args.reference_config else None,
              "stale_contract_guard_tests": guards,
              "asset_contract_sha256": hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("ELF3 ASSET CHECK PASSED:", json.dumps(report), flush=True)
    sim.clear_all_callbacks()
    sim.clear_instance()


try:
    main()
except BaseException:
    # Kit shutdown can consume the exception and return 0; retain the evidence.
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
