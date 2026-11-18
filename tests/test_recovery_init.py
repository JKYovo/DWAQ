"""Recovery pose-bank geometry, name mapping and zero/nonzero reset sampling."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_init_cpu", ROOT/"TienKung-Lab/legged_lab/envs/elf3/recovery_init_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CONTRACT = json.loads((ROOT/"TienKung-Lab/legged_lab/configs/elf3/contract.json").read_text())
BANK = ROOT/"data/elf3_recovery_init_v4/supine_4096_seed20261117.npz"


def reference(n):
    return {"root_pos": torch.full((n, 3), .13), "root_quat": torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            "joint_pos": torch.full((n, 29), .27), "joint_vel": torch.full((n, 29), 11.),
            "root_lin_vel": torch.full((n, 3), 11.), "root_ang_vel": torch.full((n, 3), 11.)}


class RecoveryInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = core.InitializationBank(BANK, CONTRACT["joint_names"])

    def test_mixture_and_velocity_probabilities_with_exact_zero_and_bounds(self):
        torch.manual_seed(71)
        source = reference(40000)
        sampled, pose_kind, velocity_kind = self.bank.sample(source)
        probabilities = torch.bincount(velocity_kind, minlength=3)/len(velocity_kind)
        torch.testing.assert_close(probabilities, torch.tensor([.75, .20, .05]), rtol=0, atol=.01)
        self.assertTrue(.48 < (pose_kind > 0).float().mean() < .52)
        self.assertTrue((sampled["root_lin_vel"][velocity_kind == 0] == 0).all())
        self.assertTrue((sampled["root_ang_vel"][velocity_kind == 0] == 0).all())
        self.assertTrue((sampled["joint_vel"][velocity_kind == 0] == 0).all())
        for mode in range(3):
            ids = velocity_kind == mode
            limit = torch.tensor(core.SETTINGS["root_linear_bounds_mps"][mode])
            self.assertTrue((sampled["root_lin_vel"][ids].abs() <= limit).all())
            self.assertTrue((sampled["root_ang_vel"][ids].abs() <= core.SETTINGS["root_angular_bounds_radps"][mode]).all())
            self.assertTrue((sampled["joint_vel"][ids].abs() <= core.SETTINGS["joint_velocity_bounds_radps"][mode]).all())
            if mode:
                self.assertTrue((sampled["joint_vel"][ids].abs().sum(-1) > 0).all())

    def test_kept_reference_poses_and_all_input_tensors_are_unchanged(self):
        source = reference(2000)
        before = {k: v.clone() for k, v in source.items()}
        sampled, kinds, _ = self.bank.sample(source)
        for key in source:
            torch.testing.assert_close(source[key], before[key], rtol=0, atol=0)
        for key in ("joint_pos", "root_pos", "root_quat"):
            torch.testing.assert_close(sampled[key][kinds == 0], source[key][kinds == 0], rtol=0, atol=0)

    def test_bank_mapping_by_name_preserves_root_and_reorders_joints(self):
        reverse = core.InitializationBank(BANK, list(reversed(CONTRACT["joint_names"])))
        torch.testing.assert_close(reverse.frames["joint_pos"], self.bank.frames["joint_pos"].flip(-1), rtol=0, atol=0)
        torch.testing.assert_close(reverse.frames["root_quat"], self.bank.frames["root_quat"], rtol=0, atol=0)

    def test_empty_batch_is_valid(self):
        frame, kinds, speed = self.bank.sample(reference(0))
        self.assertEqual(frame["joint_pos"].shape, (0, 29))
        self.assertEqual(kinds.numel()+speed.numel(), 0)

    def test_bank_rejects_corrupted_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"bad.npz"
            target.write_bytes(BANK.read_bytes())
            meta = json.loads(BANK.with_suffix(".json").read_text())
            meta["bank_sha256"] = "0"*64
            target.with_suffix(".json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "identity"):
                core.InitializationBank(target, CONTRACT["joint_names"])

    def test_coverage_has_both_sources_and_independent_left_right_leg_poses(self):
        kinds = self.bank.frames["pose_kind"]
        self.assertEqual(torch.bincount(kinds, minlength=3).tolist(), [0, 2048, 2048])
        names = CONTRACT["joint_names"]
        knees = self.bank.frames["joint_pos"][:, [names.index("l_knee_y_joint"), names.index("r_knee_y_joint")]]
        self.assertGreater(int((knees[:, 0]-knees[:, 1]).abs().gt(.3).sum()), 1000)

    def test_representative_poses_respect_original_soft_limits_floor_and_self_contacts(self):
        assets = ROOT/"TienKung-Lab/legged_lab/assets/elf3"
        xml = ET.parse(assets/"mjcf/elf3.xml")
        xml.find("compiler").set("meshdir", str(assets/"meshes"))
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
        data = mujoco.MjData(model)
        joints = [model.joint(n).id for n in CONTRACT["joint_names"]]
        qids = model.jnt_qposadr[joints]
        limits = model.jnt_range[joints]
        mid, half = limits.mean(-1), np.diff(limits, axis=-1).ravel()*.9*.5
        with np.load(BANK, allow_pickle=False) as bank:
            self.assertTrue((bank["joint_pos"] >= mid-half-1e-6).all())
            self.assertTrue((bank["joint_pos"] <= mid+half+1e-6).all())
            for i in np.linspace(0, len(bank["joint_pos"])-1, 96).astype(int):
                mujoco.mj_resetData(model, data)
                data.qpos[:3], data.qpos[3:7], data.qpos[qids] = bank["root_pos"][i], bank["root_quat"][i], bank["joint_pos"][i]
                mujoco.mj_forward(model, data)
                lowest = np.inf
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                        continue
                    mesh = model.geom_dataid[g]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    z = model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3, 3)[2]+data.geom_xpos[g, 2]
                    lowest = min(lowest, float(z.min()))
                self.assertAlmostEqual(lowest, float(bank["floor_clearance"][i]), delta=2e-6)
                for contact in data.contact:
                    if model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]:
                        self.assertGreaterEqual(contact.dist, -.01001)


if __name__ == "__main__":
    unittest.main()
