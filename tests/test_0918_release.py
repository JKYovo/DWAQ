"""CPU-only checks; importing this test never starts Isaac Sim or training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_checks", ROOT / "scripts/release_0918_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ReleaseTests(unittest.TestCase):
    def test_actual_source_and_physical_asset_manifest(self):
        result = checks.check_source_manifest(ROOT)
        self.assertTrue(result["retained_physical_asset_checked"])
        self.assertGreater(result["verified_source_files"], 90)

    def test_drift_rejected_without_git_or_old_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            source = root / "source.py"
            source.write_text("pinned source\n")
            asset = root / "TienKung-Lab/legged_lab/configs/elf3"
            asset.mkdir(parents=True)
            (asset / "contract.json").write_text(json.dumps({"retained_physical_asset": {"files_sha256": {}}}))
            (root / "docs/elf3_0918_training_manifest.json").write_text(json.dumps({
                "baseline_commit": "test", "release_files": {},
                "baseline_training_files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
            checks.check_source_manifest(root)
            source.write_text("modified source\n")
            with self.assertRaisesRegex(ValueError, "source/asset drift"):
                checks.check_source_manifest(root)

    def test_only_0918_elf3_tasks_registered(self):
        source = ROOT / "TienKung-Lab/legged_lab/envs/__init__.py"
        tasks = [node.args[0].value for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "register" and node.args and isinstance(node.args[0], ast.Constant)
                 and isinstance(node.args[0].value, str) and node.args[0].value.startswith("elf3_")]
        self.assertEqual(set(tasks), {"elf3_dwaq", "elf3_dwaq_flat", "elf3_dwaq_upper_symmetry", "elf3_dwaq_upper_symmetry_pose"})

    def test_asset_limits_and_unchanged_deployment_scale(self):
        contract = json.loads((ROOT / "TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
        names = contract["joint_names"]
        self.assertEqual(len(names), 29)
        self.assertEqual(set(contract["armature"].values()), {.01})
        self.assertEqual(set(contract["velocity_limit"].values()), {20})
        for side in ("l", "r"):
            for axis in ("x", "y"):
                self.assertEqual(contract["effort_limit"][f"{side}_hip_{axis}_joint"], 100)
            self.assertEqual(contract["effort_limit"][f"{side}_knee_y_joint"], 150)
            self.assertEqual(contract["action_scale"][f"{side}_hip_y_joint"], .213)
        self.assertEqual(contract["effort_limit"]["waist_z_joint"], 100)

    def test_launcher_portable_and_correct_task(self):
        source = (ROOT / "scripts/train_0918_melody.py").read_text()
        self.assertIn('TASK = "elf3_dwaq_upper_symmetry_pose"', source)
        self.assertNotIn("/home/cheng/", source)
        self.assertNotIn('"rev-parse", "HEAD"', source)
        self.assertIn("check_source_manifest(ROOT)", source)
        self.assertIn("agent.save_interval = 500", source)
        self.assertIn("agent.resume = False", source)

    def test_asset_package_has_no_training_metadata_dependency(self):
        asset = ROOT / "TienKung-Lab/legged_lab/assets/elf3"
        self.assertEqual({path.name for path in asset.iterdir() if path.is_file()}, {"__init__.py", "elf3.py"})
        init = (asset / "__init__.py").read_text()
        self.assertNotIn("contract", init)
        self.assertNotIn("json", init)
        for name in ("usd/elf3.usd", "urdf/elf3.urdf", "mjcf/elf3.xml"):
            self.assertTrue((asset / name).is_file())

    def test_relocated_contract_keeps_existing_checkpoint_identity(self):
        source = ROOT / "TienKung-Lab/legged_lab/elf3_contract.py"
        spec = importlib.util.spec_from_file_location("isolated_elf3_contract", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        digest = hashlib.sha256(json.dumps(module.CONTRACT, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "d3ae8408b6789523724c52d090bd3a0d2824df897c2ba5b174f6655e24994093")
        self.assertEqual(module.ASSET_DIR, ROOT / "TienKung-Lab/legged_lab/assets/elf3")

    def test_root_meshes_and_mjcf_are_self_contained(self):
        asset = ROOT / "TienKung-Lab/legged_lab/assets/elf3"
        self.assertFalse((asset / "xmls").exists())
        self.assertEqual(len(list((asset / "meshes").glob("*.STL"))), 34)
        urdf = asset / "urdf/elf3.urdf"
        for mesh in ET.parse(urdf).findall(".//mesh"):
            path = (urdf.parent / mesh.attrib["filename"]).resolve()
            self.assertEqual(path.parent, asset / "meshes")
            self.assertTrue(path.is_file())
        mjcf = asset / "mjcf/elf3.xml"
        xml = ET.parse(mjcf)
        self.assertEqual(xml.find("compiler").attrib["meshdir"], "../meshes")
        for mesh in xml.findall("asset/mesh"):
            self.assertTrue((mjcf.parent / "../meshes" / mesh.attrib["file"]).resolve().is_file())

    def test_path_migration_does_not_allow_physics_changes(self):
        spec = importlib.util.spec_from_file_location("migration_checks", ROOT / "TienKung-Lab/legged_lab/elf3_contract.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            asset = Path(folder)
            (asset / "mjcf").mkdir()
            old = b'<mujoco><compiler meshdir="robot_meshes"/><mass value="1"/></mujoco>'
            current = old.replace(b'"robot_meshes"', b'"../meshes"')
            path = asset / "mjcf/elf3.xml"
            path.write_bytes(current)
            contract = {"retained_physical_asset": {"files_sha256": {"xmls/elf3.xml": hashlib.sha256(old).hexdigest()}}}
            result = module.verify_retained_assets(asset, contract)
            self.assertEqual(result["verified_files"], 1)
            path.write_bytes(current.replace(b'value="1"', b'value="2"'))
            with self.assertRaisesRegex(ValueError, "physical content changed"):
                module.verify_retained_assets(asset, contract)


if __name__ == "__main__":
    unittest.main()
