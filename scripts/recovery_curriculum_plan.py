"""Simulator-free absolute-iteration planning for the 0..80000 course."""
BOUNDARY = 50000
FINAL_ITERATION = 80000
TASK = "elf3_dwaq_recovery_curriculum_v1"
RECOVERY_TASK = "elf3_dwaq_recovery40_init_v4"


def checkpoint_stage(saved):
    iteration = saved.get("iter")
    infos = saved.get("infos") or {}
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint must have a completed, nonnegative iteration")
    if infos.get("initialization_only") or not (saved.get("optimizer_state_dict") or {}).get("state"):
        raise ValueError("Require a completed checkpoint with trained policy Adam")
    runner = (infos.get("training_recipe") or {}).get("runner_class_name")
    contract = infos.get("elf3_contract") or {}
    if runner == "Elf3DwaqRewardCurriculumRunner" and iteration <= BOUNDARY:
        settings = (infos.get("training_recipe") or {}).get("reward_curriculum") or {}
        if settings.get("ramp_start") != 20000 or settings.get("ramp_end") != 30000:
            raise ValueError("The recovery course requires the original 20000..30000 reward ramp")
        return "curriculum_boundary" if iteration == BOUNDARY else "curriculum"
    if (runner == "Elf3DwaqRecoveryInitRunner" and iteration > BOUNDARY
            and "recovery_initialization_experiment" in contract and "recovery_training_state" in infos):
        return "recovery"
    if (runner == "Elf3DwaqFlatRecoveryRunner" and iteration == 60000
            and "flat_recovery_experiment" in contract and "recovery_training_state" in infos):
        return "historical_flat60000"
    raise ValueError("Use this course's curriculum checkpoint, init-v4 recovery, or historical flat60000")


def stage_plan(iterations=80001, completed_iteration=None, source_stage=None):
    if (isinstance(iterations, bool) or not isinstance(iterations, int)
            or not 1 <= iterations <= FINAL_ITERATION+1):
        raise ValueError("Course --iterations is a total count in1..80001; final index is count-1")
    if completed_iteration is None:
        if source_stage is not None:
            raise ValueError("Source stage requires a checkpoint")
        start = 0
    else:
        if (isinstance(completed_iteration, bool) or not isinstance(completed_iteration, int)
                or completed_iteration < 0):
            raise ValueError("Invalid completed iteration")
        expected = ("curriculum" if completed_iteration < BOUNDARY else
                    "curriculum_boundary" if completed_iteration == BOUNDARY else "recovery")
        if source_stage != expected and not (completed_iteration == 60000 and source_stage == "historical_flat60000"):
            raise ValueError("Source stage disagrees with the completed iteration")
        start = completed_iteration+1
    if iterations <= start:
        raise ValueError("Target must include at least one new update; --iterations is total, not additional")
    result = []
    if start <= BOUNDARY:
        end = min(BOUNDARY, iterations-1)
        result.append({"stage": "walking", "first_iteration": start, "last_iteration": end,
                       "updates": end-start+1, "mode": "resume" if completed_iteration is not None else "fresh"})
        start = end+1
    if start < iterations:
        promote = source_stage in (None, "curriculum", "curriculum_boundary", "historical_flat60000")
        result.append({"stage": "recovery", "first_iteration": start, "last_iteration": iterations-1,
                       "updates": iterations-start, "mode": "parent" if promote else "resume"})
    return result
