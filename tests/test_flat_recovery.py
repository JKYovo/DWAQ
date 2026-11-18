"""Flat area isolation and original command generator delegation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/"TienKung-Lab/legged_lab/envs/elf3"
for name in ("recovery_v2_core","flat_recovery_core"):
    spec = importlib.util.spec_from_file_location("legged_lab.envs.elf3."+name,DIR/(name+".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
flat = sys.modules["legged_lab.envs.elf3.flat_recovery_core"]


class FlatRecoveryTests(unittest.TestCase):
    def test_fullsize_flat_origins_have_margin_spacing_and_no_rough_overlap(self):
        origins,bounds = flat.flat_layout(6553,(8.,8.),17,20.)
        self.assertEqual(origins.shape,(6553,3))
        self.assertEqual(len(np.unique(origins,axis=0)),6553)
        self.assertTrue((origins[:,2] == 0).all())
        self.assertTrue((origins[:,:2] >= bounds[0,:2]+8).all())
        self.assertTrue((origins[:,:2] <= bounds[1,:2]-8).all())
        self.assertGreater(bounds[0,0],8*17/2+20+8)
        self.assertTrue((np.diff(np.unique(origins[:,0])) == 4).all())
        self.assertTrue((np.diff(np.unique(origins[:,1])) == 4).all())

    def test_reset_and_resample_zero_recovery_only_and_keep_original_normal_commands(self):
        class Original:
            def __init__(self):
                self.command = torch.zeros(5,3)
                self.is_standing_env = torch.zeros(5,dtype=torch.bool)
                self.is_heading_env = torch.ones(5,dtype=torch.bool)
            def compute(self,dt):
                self.command.copy_(torch.arange(15).reshape(5,3))
                return dt
            def reset(self,ids):
                self.command[ids] = -1
                return {"reset_count":len(ids)}
        base = Original()
        assigned = torch.tensor([True,False,True,False,False])
        wrapper = flat.SplitCommands(base,assigned)
        self.assertEqual(wrapper.compute(.02),.02)
        self.assertTrue(torch.equal(base.command[~assigned],torch.arange(15).reshape(5,3).float()[~assigned]))
        self.assertTrue((wrapper.command[assigned] == 0).all())
        result = wrapper.reset(torch.arange(5))
        self.assertEqual(result,{"reset_count":5})
        self.assertTrue((wrapper.command[~assigned] == -1).all())
        self.assertTrue((wrapper.command[assigned] == 0).all())
        self.assertTrue(wrapper.is_standing_env[assigned].all())
        self.assertFalse(wrapper.is_heading_env[assigned].any())


if __name__ == "__main__":
    unittest.main()
