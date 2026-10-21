"""Read-only contact snapshot from an *already initialized* ELF3 environment.

Importing this module does not import Kit/Isaac Lab, create a simulation, reset an
environment, or cook geometry. Tensor getters report PhysX values; composed USD
mesh data and schema defaults are explicitly separate from cooked PhysX shapes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


FOOT_NAMES = ("l_ankle_x_link", "r_ankle_x_link")
SHAPE_GETTERS = ("get_contact_offsets", "get_rest_offsets", "get_material_properties")


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _runtime_getter(view, name, warnings, label):
    """Only sample instance zero; never infer shape indices from link indices."""
    getter = getattr(view, name, None)
    if not callable(getter):
        warnings.append(f"{label}.{name}: API unavailable")
        return {"status": "unavailable", "api": name}
    try:
        values = getter()
        shape = list(getattr(values, "shape", ()))
        # Slice before copying to CPU when the frontend supports slicing.
        try:
            sample = values[0]
        except (TypeError, AttributeError):
            sample = values.numpy()[0]
        return {"status": "available", "api": name, "shape": shape,
                "instance_index": 0, "values": _json_value(sample)}
    except Exception as exc:
        warnings.append(f"{label}.{name}: {type(exc).__name__}: {exc}")
        return {"status": "unavailable", "api": name, "error": str(exc)}


def _load_usd():
    # Deliberately lazy: unit tests and CLI parsing need no running Kit.
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    return Usd, UsdGeom, UsdPhysics, UsdShade


def _attribute(prim, name, schema, Usd):
    result = {"value": None, "authored": False, "source": "unavailable", "opinions": []}
    attribute = prim.GetAttribute(name)
    if attribute:
        result.update(value=_json_value(attribute.Get()),
                      authored=attribute.HasAuthoredValueOpinion(),
                      source=str(attribute.GetResolveInfo().GetSource()))
        for spec in attribute.GetPropertyStack(Usd.TimeCode.Default()):
            result["opinions"].append({"layer": spec.layer.identifier,
                                       "path": str(spec.path), "default": _json_value(spec.default)})
    # Defaults are informational, NOT a claim about a cooked/runtime value.
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema)
    if definition and name in definition.GetPropertyNames():
        result["schema_default"] = {"schema": schema,
                                    "value": _json_value(definition.GetAttributeFallbackValue(name)),
                                    "source": "loaded USD schema registry; not runtime measurement"}
    else:
        result["schema_default"] = {"schema": schema, "status": "unavailable"}
    return result


def _materials(prim, Usd, UsdPhysics, UsdShade):
    """Report physics-purpose binding and whether it really is a physics material."""
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material:
        return {"status": "unavailable", "reason": "no resolved physics-purpose material binding"}
    material_prim = material.GetPrim()
    physical = material_prim.HasAPI(UsdPhysics.MaterialAPI)
    return {"status": "available" if physical else "not_a_physics_material",
            "material_path": str(material_prim.GetPath()),
            "binding_relationship": str(relationship.GetPath()) if relationship else None,
            "attributes": {name: _attribute(material_prim, name, "PhysicsMaterialAPI", Usd)
                           for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution")}
            if physical else {},
            "note": "USD binding only; use tensor get_material_properties for runtime randomized values"}


def _capture_stage(stage, foot_paths, warnings):
    Usd, UsdGeom, UsdPhysics, UsdShade = _load_usd()
    cache = UsdGeom.XformCache()
    result = {"status": "available", "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
              "source": "composed live stage after spawn; not PhysX cooked geometry", "feet": {}}
    fields = {"physics:approximation": "PhysicsMeshCollisionAPI",
              "physics:collisionEnabled": "PhysicsCollisionAPI",
              "physxCollision:contactOffset": "PhysxCollisionAPI",
              "physxCollision:restOffset": "PhysxCollisionAPI",
              "physxConvexHullCollision:hullVertexLimit": "PhysxConvexHullCollisionAPI",
              "physxConvexHullCollision:minThickness": "PhysxConvexHullCollisionAPI"}
    for name, path in foot_paths.items():
        foot = stage.GetPrimAtPath(path)
        if not foot:
            warnings.append(f"USD foot prim missing: {path}")
            result["feet"][name] = {"status": "unavailable", "path": path}
            continue
        colliders = []
        meshes = []
        collider_paths = []
        inverse_foot = cache.GetLocalToWorldTransform(foot).GetInverse()
        for prim in Usd.PrimRange(foot, Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collider_paths.append(prim_path)
                colliders.append({"path": prim_path, "type": prim.GetTypeName(),
                                  "schemas": list(prim.GetAppliedSchemas()),
                                  "attributes": {key: _attribute(prim, key, schema, Usd)
                                                 for key, schema in fields.items()},
                                  "material_binding": _materials(prim, Usd, UsdPhysics, UsdShade)})
            if prim.IsA(UsdGeom.Mesh) and any(prim_path == c or prim_path.startswith(c + "/")
                                             for c in collider_paths):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                count = len(points) if points is not None else 0
                bounds = None
                if count:
                    bounds = {"min": [float(min(point[i] for point in points)) for i in range(3)],
                              "max": [float(max(point[i] for point in points)) for i in range(3)]}
                meshes.append({"path": prim_path, "kind": "USD source mesh (not cooked)",
                               "point_count": count, "mesh_local_bounds": bounds,
                               "mesh_to_foot_transform": _json_value(cache.GetLocalToWorldTransform(prim) * inverse_foot),
                               "attributes_on_mesh": {key: _attribute(prim, key, schema, Usd)
                                                      for key, schema in fields.items()},
                               "source_hull": "not computed; do not infer PhysX hull from source point count",
                               "material_binding": _materials(prim, Usd, UsdPhysics, UsdShade)})
        result["feet"][name] = {"status": "available", "path": path,
                                "mass": _attribute(foot, "physics:mass", "PhysicsMassAPI", Usd),
                                "colliders": colliders, "source_meshes": meshes,
                                "cooked_shape": {"status": "unavailable",
                                                 "reason": "no supported cooked-vertex/face getter used"}}
        if not colliders:
            warnings.append(f"No CollisionAPI prim beneath {path}; contact sensors are not colliders")
    return result


def capture_contact_audit(env, output_path=None):
    """Return a JSON-safe snapshot, optionally writing ONLY the caller's path.

    Call after the launcher's normal initialization/reset, before further steps.
    This function never starts, steps, resets or changes simulation/asset values.
    It may allocate read-only tensor views against the existing simulation view.
    Only articulation instance zero is sampled; it is not a population audit.
    """
    warnings = []
    env = getattr(env, "unwrapped", env)
    robot = getattr(env, "robot", None)
    if robot is None:
        try:
            robot = env.scene["robot"]
        except (AttributeError, KeyError, TypeError):
            pass
    report = {"version": "elf3_feedback_contact_audit_v1", "read_only": True,
              "sampling": "articulation instance zero only, after caller initialization",
              "warnings": warnings, "runtime": {}, "usd": {"status": "unavailable"},
              "cooked_geometry": {"status": "unavailable",
                                  "reason": "tensor offsets/materials do not expose cooked shape vertices/faces"}}
    view = getattr(robot, "root_physx_view", None)
    foot_paths = {}
    if view is None:
        warnings.append("Initialized robot.root_physx_view unavailable")
    else:
        runtime = {"shape_index_mapping": "articulation shape slots are NOT link/body indices",
                   "material_columns": ["static_friction", "dynamic_friction", "restitution"],
                   "getters": {}, "feet": {}}
        for getter in ("get_masses",) + SHAPE_GETTERS:
            runtime["getters"][getter] = _runtime_getter(view, getter, warnings, "articulation")
        try:
            paths = list(view.link_paths[0])
            runtime["link_paths_instance0"] = paths
            foot_paths = {name: path for name in FOOT_NAMES for path in paths
                          if str(path).rsplit("/", 1)[-1] == name}
            masses = runtime["getters"]["get_masses"]
            if masses["status"] == "available":
                runtime["foot_masses_kg"] = {name: masses["values"][paths.index(path)]
                                             for name, path in foot_paths.items()}
        except Exception as exc:
            warnings.append(f"Runtime link-path/mass mapping unavailable: {exc}")
        physics_view = getattr(robot, "_physics_sim_view", None)
        create_view = getattr(physics_view, "create_rigid_body_view", None)
        for name, path in foot_paths.items():
            record = {"body_path": str(path), "status": "unavailable"}
            if callable(create_view):
                try:
                    foot_view = create_view(str(path))
                    if foot_view.count != 1 or list(foot_view.prim_paths) != [str(path)]:
                        raise ValueError("exact foot body view did not resolve to the requested single path")
                    record.update(status="available", shape_mapping="slots belong to this foot; child collider mapping unavailable",
                                  getters={key: _runtime_getter(foot_view, key, warnings, name)
                                           for key in SHAPE_GETTERS})
                except Exception as exc:
                    warnings.append(f"{name} read-only body view unavailable: {exc}")
            else:
                warnings.append(f"{name}: existing physics view cannot create a read-only body view")
            runtime["feet"][name] = record
        report["runtime"] = runtime
    for missing in set(FOOT_NAMES) - set(foot_paths):
        warnings.append(f"Runtime path for {missing} unavailable; no guessed collider/sensor path used")
    try:
        stage = getattr(getattr(env, "sim", None), "stage", None)
        if stage is None:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no already-existing USD stage")
        report["usd"] = _capture_stage(stage, foot_paths, warnings)
    except Exception as exc:
        warnings.append(f"Live USD capture unavailable: {type(exc).__name__}: {exc}")
    report = _json_value(report)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
