"""Run the existing0..50000 reward course, then current recovery through80000.

The handoff happens once after a successful model50000 save. Each stage runs
in a separate process/log directory; failures propagate without retrying.
--resume accepts an earlier course checkpoint, completed50000, or init-v4.
--dry-run prints the absolute update plan and commands without starting Isaac.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

from recovery_curriculum_plan import BOUNDARY, RECOVERY_TASK, TASK, checkpoint_stage, stage_plan

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--iterations", type=int, default=80001, help="Total indexed updates;80001 ends at model80000")
    p.add_argument("--seed", type=int, choices=(42,), default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--run-name", default="curriculum_recovery80k")
    p.add_argument("--resume", type=Path, help="Completed original course or init-v4 checkpoint;50000 promotes to recovery")
    p.add_argument("--parent-num-envs", type=int,
                   help="Actual curriculum parent batch when intentionally changing batch; default matches --num-envs")
    p.add_argument("--motion-dir", type=Path, default=ROOT/"data/elf3_amp_v4")
    p.add_argument("--bank", type=Path, default=ROOT/"data/elf3_recovery_init_v4/supine_4096_seed20261117.npz")
    p.add_argument("--swanlab-project", default=None)
    p.add_argument("--use-cached-visual", action="store_true")
    p.add_argument("--preflight-steps", type=int, default=0, help="Optional recovery-stage check before learning")
    p.add_argument("--dry-run", action="store_true")
    return p


def prepare(args):
    if (args.num_envs < 4 or args.preflight_steps < 0 or (args.parent_num_envs is not None and args.parent_num_envs < 4)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_name)
            or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", args.device)):
        raise ValueError("Invalid environment count, name, device or preflight steps")
    source, digest, completed, kind = None, None, None, None
    if args.resume is not None:
        import torch
        source = args.resume.resolve(strict=True)
        digest = sha256(source)
        saved = torch.load(source, map_location="cpu", weights_only=False)
        kind, completed = checkpoint_stage(saved), saved["iter"]
        del saved
    plan = stage_plan(args.iterations, completed, kind)
    if any(item["stage"] == "recovery" for item in plan):
        args.bank.resolve(strict=True)
        for name in ("Recovery", "WalkandRun"):
            if not list((args.motion_dir/name).glob("*.npz")):
                raise ValueError(f"No reference motion in {args.motion_dir/name}")
    return {"source": source, "sha256": digest, "plan": plan, "source_stage": kind}


def command(args, item, name, source=None, digest=None):
    script = "train_elf3_reward_curriculum.py" if item["stage"] == "walking" else "train_elf3_recovery_stage.py"
    result = [sys.executable, str(ROOT/"scripts"/script), "--num-envs", str(args.num_envs),
              "--iterations", str(item["last_iteration"]+1), "--seed", str(args.seed),
              "--device", args.device, "--run-name", name]
    for enabled, flag in ((args.headless, "--headless"), (args.use_cached_visual, "--use-cached-visual")):
        if enabled:
            result.append(flag)
    if args.swanlab_project:
        result += ["--swanlab-project", args.swanlab_project]
    if item["stage"] == "walking":
        result += ["--ramp-start", "20000", "--ramp-end", "30000"]
    else:
        result += ["--motion-dir", str(args.motion_dir.resolve()), "--bank", str(args.bank.resolve()),
                   "--preflight-steps", str(args.preflight_steps)]
        if item["mode"] == "parent":
            result += ["--parent-num-envs", str(args.parent_num_envs or args.num_envs)]
    if source is not None:
        if item["mode"] == "parent":
            result += ["--parent-checkpoint", str(source), "--parent-sha256", digest]
        else:
            result += ["--resume", str(source)]
    return result


def write_report(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def execute(args, prepared):
    preview = []
    source, digest = prepared["source"], prepared["sha256"]
    for item in prepared["plan"]:
        preview.append({**item, "argv": command(args, item, args.run_name+"_"+item["stage"], source, digest)})
        source, digest = "<successful_walking_run>/model_50000.pt", "<computed_SHA256>"
    if args.dry_run:
        print(json.dumps({"course": TASK, "stages": preview, "creates_environment": False,
                          "learning_updates": 0, "dry_run": True}, indent=2))
        return
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    directory = ROOT/"logs"/TASK/f"{stamp}_{args.run_name}_{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"course": TASK, "source": str(prepared["source"]) if prepared["source"] else None,
              "source_sha256": prepared["sha256"], "plan": prepared["plan"], "num_envs": args.num_envs,
              "seed": 42, "stages": [], "success": False, "automatic_restart": False}
    source, digest = prepared["source"], prepared["sha256"]
    child = None
    try:
        for item in prepared["plan"]:
            name = f"{args.run_name}_{item['stage']}_{os.getpid()}"
            argv = command(args, item, name, source, digest)
            entry = {**item, "argv": argv, "source_sha256": digest}
            report["stages"].append(entry)
            write_report(directory/"course.json", report)
            print("[COURSE_STAGE] "+json.dumps(entry), flush=True)
            child = subprocess.Popen(argv, cwd=ROOT)
            entry["pid"] = child.pid
            write_report(directory/"course.json", report)
            code = child.wait()
            entry["exit_code"] = code
            if code != 0:
                raise RuntimeError(f"{item['stage']} exited with{code}; next stage was not started")
            namespace = "elf3_dwaq_still_v2_curriculum_v1" if item["stage"] == "walking" else RECOVERY_TASK
            checkpoints = list((ROOT/"logs"/namespace).glob(f"*_{name}_{child.pid}/model_{item['last_iteration']}.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError("Stage exited without its unique requested final checkpoint")
            source = checkpoints[0].resolve()
            digest = sha256(source)
            entry.update(final_checkpoint=str(source), final_checkpoint_sha256=digest)
            if item["stage"] == "walking":
                # The newly completed parent was produced by this invocation.
                args.parent_num_envs = args.num_envs
            write_report(directory/"course.json", report)
            child = None
        report.update(success=True, final_checkpoint=str(source), final_checkpoint_sha256=digest)
        print("[COURSE_COMPLETE] "+json.dumps({"checkpoint": str(source), "sha256": digest}), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write_report(directory/"course.json", report)


def main():
    p = parser()
    args = p.parse_args()
    try:
        prepared = prepare(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    execute(args, prepared)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Explicit signal{signum}")
    signal.signal(signal.SIGTERM, interrupted)
    main()
