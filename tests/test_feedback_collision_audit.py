"""CPU-only mocks; optional in-memory USD checks. Never launch Kit/PhysX."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("feedback_collision_audit", ROOT / "scripts/feedback_collision_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
LEFT = "/World/envs/env_0/Robot/l_ankle_x_link"
RIGHT = "/World/envs/env_0/Robot/r_ankle_x_link"


class Array:
    def __init__(self, data):
        self.data = data
        shape = []
        item = data
        while isinstance(item, list):
            shape.append(len(item))
            item = item[0] if item else None
        self.shape = tuple(shape)

    def __getitem__(self, item):
        return self.data[item]


class ReadOnlyView:
    def __init__(self, paths, contact=.012):
        self.count = 1
        self.prim_paths = paths
        self.link_paths = [paths]
        self.data = {"get_masses": [[.38] * len(paths)],
                     "get_contact_offsets": [[contact]], "get_rest_offsets": [[.001]],
                     "get_material_properties": [[[.9, .7, .02]]]}
        self.calls = []

    def __getattr__(self, name):
        if name in self.data:
            def getter():
                self.calls.append(name)
                return Array(self.data[name])
            return getter
        if name.startswith(("set_", "write_")):
            raise AssertionError("Mutation API must not be accessed")
        raise AttributeError(name)


class ExistingPhysicsView:
    def __init__(self):
        self.paths = []
        self.views = {}

    def create_rigid_body_view(self, path):
        self.paths.append(path)
        self.views[path] = ReadOnlyView([path], .014 if path == LEFT else .016)
        return self.views[path]


class ContactAuditTests(unittest.TestCase):
    def make_env(self):
        root_view = ReadOnlyView(["/World/envs/env_0/Robot/torso_link", LEFT, RIGHT])
        physics = ExistingPhysicsView()
        env = SimpleNamespace(robot=SimpleNamespace(root_physx_view=root_view, _physics_sim_view=physics),
                              sim=SimpleNamespace(stage=object()))
        return env, root_view, physics

    def test_no_top_level_kit_or_simulation_import(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [node.module if isinstance(node, ast.ImportFrom) else node.names[0].name for node in imports]
        self.assertFalse(any(name.startswith(("pxr", "omni", "isaac", "torch")) for name in names))

    def test_api_values_and_exact_foot_mapping_without_mutation(self):
        env, view, physics = self.make_env()
        before = deepcopy(view.data)
        with patch.object(audit, "_capture_stage", return_value={"status": "mocked"}):
            result = audit.capture_contact_audit(env)
        self.assertEqual(view.data, before)
        self.assertEqual(physics.paths, [LEFT, RIGHT])
        self.assertEqual(result["runtime"]["foot_masses_kg"], dict.fromkeys(audit.FOOT_NAMES, .38))
        left = result["runtime"]["feet"]["l_ankle_x_link"]["getters"]
        right = result["runtime"]["feet"]["r_ankle_x_link"]["getters"]
        self.assertEqual(left["get_contact_offsets"]["values"], [.014])
        self.assertEqual(right["get_contact_offsets"]["values"], [.016])
        self.assertEqual(left["get_material_properties"]["values"], [[.9, .7, .02]])
        self.assertEqual(left["get_material_properties"]["shape"], [1, 1, 3])
        self.assertEqual(result["cooked_geometry"]["status"], "unavailable")
        self.assertEqual(view.calls, ["get_masses", *audit.SHAPE_GETTERS])

    def test_absent_tensor_apis_and_no_guess_from_contact_sensor(self):
        view = SimpleNamespace(link_paths=[[LEFT, RIGHT]])
        env = SimpleNamespace(robot=SimpleNamespace(root_physx_view=view),
                              contact_sensor=SimpleNamespace(prim_path="/not/a/collider"),
                              sim=SimpleNamespace(stage=object()))
        with patch.object(audit, "_capture_stage", return_value={"status": "mocked"}):
            result = audit.capture_contact_audit(env)
        self.assertEqual(result["runtime"]["getters"]["get_contact_offsets"]["status"], "unavailable")
        self.assertTrue(any("API unavailable" in warning for warning in result["warnings"]))
        self.assertNotIn("/not/a/collider", json.dumps(result))
        self.assertEqual(result["runtime"]["feet"]["r_ankle_x_link"]["status"], "unavailable")

    def test_backend_error_does_not_fabricate_default(self):
        def fail():
            raise RuntimeError("backend is not initialized")
        view = SimpleNamespace(link_paths=[[LEFT, RIGHT]], get_contact_offsets=fail)
        env = SimpleNamespace(robot=SimpleNamespace(root_physx_view=view), sim=SimpleNamespace(stage=object()))
        with patch.object(audit, "_capture_stage", side_effect=ImportError("pxr absent")):
            result = audit.capture_contact_audit(env)
        offsets = result["runtime"]["getters"]["get_contact_offsets"]
        self.assertEqual(offsets["status"], "unavailable")
        self.assertNotIn("values", offsets)
        self.assertEqual(result["usd"]["status"], "unavailable")

    def test_explicit_output_only_and_strict_json(self):
        env, view, _ = self.make_env()
        view.data["get_contact_offsets"] = [[float("inf")]]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "caller_chosen" / "contact.json"
            with patch.object(audit, "_capture_stage", return_value={"status": "mocked"}):
                result = audit.capture_contact_audit(SimpleNamespace(unwrapped=env), target)
            self.assertEqual(json.loads(target.read_text()), result)
            self.assertEqual(result["runtime"]["getters"]["get_contact_offsets"]["values"], ["inf"])
            self.assertEqual([p for p in Path(directory).rglob("*") if p.is_file()], [target])

    def test_uninitialized_env_is_diagnostic_not_startup(self):
        with patch.object(audit, "_capture_stage", return_value={"status": "mocked"}):
            result = audit.capture_contact_audit(SimpleNamespace(sim=SimpleNamespace(stage=object())))
        self.assertTrue(any("root_physx_view unavailable" in warning for warning in result["warnings"]))

    def test_ambiguous_foot_view_never_used_as_exact_mapping(self):
        env, _, physics = self.make_env()
        def wrong_view(path):
            return ReadOnlyView(["/some/other/body"])
        physics.create_rigid_body_view = wrong_view
        with patch.object(audit, "_capture_stage", return_value={"status": "mocked"}):
            result = audit.capture_contact_audit(env)
        self.assertEqual(result["runtime"]["feet"]["l_ankle_x_link"]["status"], "unavailable")
        self.assertNotIn("getters", result["runtime"]["feet"]["l_ankle_x_link"])
        self.assertTrue(any("did not resolve" in warning for warning in result["warnings"]))

    def test_in_memory_usd_provenance_and_readonly_when_pxr_available(self):
        try:
            Usd, UsdGeom, UsdPhysics, _ = audit._load_usd()
            from pxr import Sdf
        except ImportError:
            self.skipTest("Optional pxr package unavailable; other tests require only stdlib")
        stage = Usd.Stage.CreateInMemory()
        for name, path in zip(audit.FOOT_NAMES, (LEFT, RIGHT)):
            foot = UsdGeom.Xform.Define(stage, path).GetPrim()
            UsdPhysics.MassAPI.Apply(foot).CreateMassAttr(.38)
            collider = UsdGeom.Xform.Define(stage, path + "/collisions").GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
            collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.015)
            mesh = UsdGeom.Mesh.Define(stage, path + "/collisions/mesh")
            mesh.CreatePointsAttr([(0, 0, -.04), (.1, 0, -.04), (0, .04, -.04)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            # A sensor-looking prim without CollisionAPI must not become a collider.
            UsdGeom.Xform.Define(stage, path + "/contact_sensor")
        before = stage.GetRootLayer().ExportToString()
        warnings = []
        result = audit._capture_stage(stage, dict(zip(audit.FOOT_NAMES, (LEFT, RIGHT))), warnings)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        self.assertEqual(warnings, [])
        left = result["feet"]["l_ankle_x_link"]
        self.assertEqual(len(left["colliders"]), 1)
        attrs = left["colliders"][0]["attributes"]
        self.assertEqual(attrs["physics:approximation"]["value"], "convexHull")
        self.assertTrue(attrs["physxCollision:contactOffset"]["authored"])
        self.assertAlmostEqual(attrs["physxCollision:contactOffset"]["value"], .015)
        self.assertTrue(attrs["physxCollision:contactOffset"]["opinions"])
        self.assertFalse(attrs["physxCollision:restOffset"]["authored"])
        self.assertIsNone(attrs["physxCollision:restOffset"]["value"])
        self.assertEqual(left["source_meshes"][0]["point_count"], 3)
        self.assertEqual(left["cooked_shape"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
