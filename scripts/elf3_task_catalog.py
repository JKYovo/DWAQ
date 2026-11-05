"""List supported tasks and reject retired public commands before Isaac starts.

Standard library only. Historical scripts remain importable as compatibility
helpers; elf3.sh does not expose them as training entrypoints.
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "elf3_dwaq_feedback_v1": "scripts/train_elf3_feedback.py",
    "elf3_dwaq_still_v2_ankle_cost_v1": "scripts/train_elf3_ankle_cost.py",
}
RETIRED_ENTRYPOINTS = frozenset({
    "scripts/train_0918_melody.py", "scripts/train_0918_melody.sh",
    "scripts/train_elf3_cached_visuals.py", "scripts/train_elf3_stand.py",
    "scripts/train_elf3_still_pose.py", "scripts/train_elf3_still_v2.py",
    "scripts/train_elf3_feedback_still_v2.py", "scripts/train_elf3_feedback_stand_v3.py",
    "scripts/resume_elf3_still_v2.py",
    # Use train_elf3_ankle_cost.py --resume instead of private old-log migration.
    "scripts/resume_elf3_ankle_cost.py",
})
TASK_ENTRYPOINTS = frozenset({
    "TienKung-Lab/legged_lab/scripts/train.py",
    "TienKung-Lab/legged_lab/scripts/play.py",
    "scripts/train_elf3_feedback.py",
})


def check_entrypoint(arguments):
    """Validate an elf3.sh invocation without importing the simulator or torch."""
    arguments = list(arguments)
    while arguments and arguments[0] in ("-u", "-B", "--"):
        arguments.pop(0)
    if not arguments or arguments[0].startswith("-"):
        return
    path = Path(arguments[0]).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return
    supported = ", ".join(TASKS)
    if relative in RETIRED_ENTRYPOINTS:
        raise ValueError(f"Retired training entrypoint: {relative}. Supported tasks: {supported}. "
                         "Run ./elf3.sh scripts/elf3_task_catalog.py to list entrypoints. "
                         "Existing weights/logs and internal compatibility helpers are retained.")
    if relative not in TASK_ENTRYPOINTS:
        return
    selected = None
    for index, value in enumerate(arguments[1:], 1):
        if value == "--task" and index + 1 < len(arguments):
            selected = arguments[index + 1]
        elif value.startswith("--task="):
            selected = value.split("=", 1)[1]
    if selected is not None and selected not in TASKS:
        raise ValueError(f"Retired/unknown task: {selected}. Supported tasks: {supported}")
    if relative == "scripts/train_elf3_feedback.py" and selected not in (None, "elf3_dwaq_feedback_v1"):
        raise ValueError("The feedback launcher only accepts elf3_dwaq_feedback_v1")


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments and arguments[0] == "--check-entrypoint":
        try:
            check_entrypoint(arguments[1:])
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0
    if arguments and arguments not in (["--help"], ["-h"]):
        print("Usage: ./elf3.sh scripts/elf3_task_catalog.py", file=sys.stderr)
        return 2
    for task, entrypoint in TASKS.items():
        print(f"{task}\n  ./elf3.sh {entrypoint}")
    print("Generic train.py accepts these two --task IDs; resume requires identical configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
