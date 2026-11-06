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
"""Public ELF3 recipes, including original still-v2 and current force500.

Keep existing config factories and runner implementations byte-for-byte: their
source fingerprints are part of completed checkpoint contracts. Registration
does not rename experiment metadata, change rewards, or migrate old weights.
"""
from legged_lab.utils.task_registry import task_registry
from legged_lab.envs.elf3.elf3_dwaq_env import Elf3DwaqEnv
from legged_lab.envs.elf3.elf3_feedback_config import Elf3DwaqFeedbackEnvCfg, Elf3DwaqFeedbackAgentCfg
from legged_lab.envs.elf3.elf3_feedback_env import Elf3DwaqFeedbackEnv
from legged_lab.envs.elf3.feedback_runner import Elf3DwaqFeedbackRunner
from legged_lab.envs.elf3.still_env import Elf3DwaqStillEnv
from legged_lab.envs.elf3.elf3_still_v2_config import Elf3DwaqStillV2EnvCfg, Elf3DwaqStillV2AgentCfg
from legged_lab.envs.elf3.still_v2_runner import Elf3DwaqStillV2Runner
from legged_lab.envs.elf3.ankle_cost import (
    Elf3DwaqAnkleCostEnv, Elf3DwaqAnkleCostRunner, make_configs as _ankle_configs,
)
from legged_lab.envs.elf3.foot_force_500 import (
    Elf3DwaqFootForce500Env, Elf3DwaqFootForce500Runner, make_configs as _force500_configs,
)

task_registry.register("elf3_dwaq_feedback_v1", Elf3DwaqFeedbackEnv,
                       Elf3DwaqFeedbackEnvCfg(), Elf3DwaqFeedbackAgentCfg())
task_registry.register("elf3_dwaq_still_v2", Elf3DwaqStillEnv,
                       Elf3DwaqStillV2EnvCfg(), Elf3DwaqStillV2AgentCfg())
_ankle_env, _ankle_agent = _ankle_configs()
task_registry.register("elf3_dwaq_still_v2_ankle_cost_v1", Elf3DwaqAnkleCostEnv,
                       _ankle_env, _ankle_agent)
_force500_env, _force500_agent = _force500_configs()
task_registry.register("elf3_dwaq_still_v2_force500_v1", Elf3DwaqFootForce500Env,
                       _force500_env, _force500_agent)

__all__ = [
    "Elf3DwaqEnv", "Elf3DwaqFeedbackEnv", "Elf3DwaqFeedbackRunner",
    "Elf3DwaqStillEnv", "Elf3DwaqStillV2Runner",
    "Elf3DwaqAnkleCostEnv", "Elf3DwaqAnkleCostRunner",
    "Elf3DwaqFootForce500Env", "Elf3DwaqFootForce500Runner",
]
