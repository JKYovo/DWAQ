"""Terrain curriculum that extends stair levels without remapping existing rows."""

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg


class StagedStairsTerrainGenerator(TerrainGenerator):
    """Keep levels 0-9 unchanged, then add one centimeter per stair level."""

    base_num_levels = 10
    total_num_levels = 17
    base_max_step_height = 0.23
    extended_max_step_height = 0.30

    @classmethod
    def map_difficulty(cls, difficulty: float, is_stairs: bool) -> float:
        """Map a 17-row curriculum onto the original 10 rows plus seven stair rows."""
        original_progress = difficulty * cls.total_num_levels / cls.base_num_levels
        if not is_stairs:
            return min(original_progress, 1.0)
        if original_progress <= 1.0:
            return original_progress

        added_progress = original_progress - 1.0
        added_span = cls.total_num_levels / cls.base_num_levels - 1.0
        effective_difficulty_span = (
            cls.extended_max_step_height / cls.base_max_step_height - 1.0
        )
        return 1.0 + added_progress / added_span * effective_difficulty_span

    def _get_terrain_mesh(self, difficulty, cfg):
        mapped_difficulty = self.map_difficulty(
            float(difficulty), isinstance(cfg, MeshPyramidStairsTerrainCfg)
        )
        return super()._get_terrain_mesh(mapped_difficulty, cfg)
