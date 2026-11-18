"""Public 50k recovery entry and portable resume of the frozen init-v4 recipe."""
from copy import copy, deepcopy
from pathlib import Path

import torch

from legged_lab.envs.elf3.feedback_contract import canonical_sha256
from legged_lab.envs.elf3.flat_recovery_runner import Elf3DwaqFlatRecoveryRunner
from legged_lab.envs.elf3.recovery_init_env import projected_flat_cfg, projected_flat_recipe
from legged_lab.envs.elf3.recovery_init_runner import Elf3DwaqRecoveryInitRunner
from legged_lab.envs.elf3.recovery_course_contract import validate_portable_contract


class Elf3DwaqRecoveryCourseRunner(Elf3DwaqRecoveryInitRunner):
    """Keep init-v4 math and checkpoint identity; add the curriculum-50000 bridge."""

    def initialize_from_curriculum(self, path, expected_sha256, parent_num_envs=16384):
        if isinstance(parent_num_envs, bool) or not isinstance(parent_num_envs, int) or parent_num_envs < 4:
            raise ValueError("Require the actual parent environment count")
        view = copy(self)
        view.env = copy(self.env)
        view.cfg = projected_flat_recipe(self.cfg)
        view.env.cfg = projected_flat_cfg(self.env.cfg)
        view.parent_contract = lambda num_envs=parent_num_envs: Elf3DwaqFlatRecoveryRunner.parent_contract(view, num_envs)
        result = Elf3DwaqFlatRecoveryRunner.initialize_from_curriculum(view, path, expected_sha256)
        self.current_learning_iteration = view.current_learning_iteration
        self.env.set_training_iteration(self.current_learning_iteration)
        return {**result, "recovery_initialization": "init-v4 enabled from the first recovery update"}

    def load(self, path, load_optimizer=True):
        if not load_optimizer:
            raise ValueError("Recovery continuation requires policy Adam and complete AMP state")
        path = Path(path).resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_portable_contract(saved, self.contract())
        old, current = deepcopy(saved["infos"]["training_recipe"]), deepcopy(self.cfg)
        for recipe in (old, current):
            for key in ("run_name", "device", "max_iterations", "resume", "load_run", "load_checkpoint"):
                recipe.pop(key, None)
        if canonical_sha256(old) != canonical_sha256(current):
            raise ValueError("Resume must preserve the full PPO/Actor/auxiliary/initialization recipe")
        return self._restore_training_state(path, saved)
