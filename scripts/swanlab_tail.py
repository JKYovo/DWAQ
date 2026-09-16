#!/usr/bin/env python3
"""Continuously mirror an existing training run's TensorBoard scalars to SwanLab."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import psutil
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.util import tensor_util


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ScalarTail:
    def __init__(self, directory, cursors=None):
        self.directory = Path(directory)
        self.cursors = dict(cursors or {})
        self.loaders = {}

    def poll(self, emit):
        count = 0
        for path in sorted(self.directory.glob("events.out.tfevents.*")):
            if path not in self.loaders:
                self.loaders[path] = LegacyEventFileLoader(str(path))
            for event in self.loaders[path].Load():
                values = {}
                for value in event.summary.value:
                    if event.step <= self.cursors.get(value.tag, -1):
                        continue
                    if value.HasField("simple_value"):
                        scalar = value.simple_value
                    elif value.HasField("tensor"):
                        array = tensor_util.make_ndarray(value.tensor)
                        if array.size != 1 or array.dtype.kind not in "biuf":
                            continue
                        scalar = float(array.reshape(-1)[0])
                    else:
                        continue
                    values[value.tag] = scalar
                if values:
                    emit(values, step=int(event.step))
                    self.cursors.update({tag: int(event.step) for tag in values})
                    count += len(values)
        return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--project", default="locomotion")
    parser.add_argument("--name", default=None)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    directory = args.log_dir.resolve(strict=True)
    lock = (directory / "swanlab_bridge.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = psutil.Process(args.training_pid)
    started = process.create_time()
    metadata_path = directory / "swanlab_run.json"
    state_path = directory / "swanlab_bridge_state.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if metadata and metadata.get("mode") != "tensorboard-tail":
        raise RuntimeError("This run already has a native SwanLab logger.")
    import swanlab
    import yaml

    agent = yaml.safe_load((directory / "params/agent.yaml").read_text())
    # IsaacLab dumps Python tuple tags; BaseLoader reads data without constructing objects.
    env = yaml.load((directory / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    run = swanlab.init(
        project=metadata.get("project", args.project),
        name=metadata.get("name", args.name or directory.name),
        id=metadata.get("id"), resume="must" if metadata else "never",
        mode="online", log_dir=str(directory / "swanlab"),
        config={"task": agent["experiment_name"], "robot": "ELF3", "algorithm": "DWAQ",
                "num_envs": int(env["scene"]["num_envs"]), "max_iterations": agent["max_iterations"],
                "seed": agent["seed"], "tensorboard_logdir": str(directory),
                "logging_method": "tensorboard-tail", "training_pid": args.training_pid},
    )
    write_json(metadata_path, {"id": run.id, "url": run.url,
                              "project": metadata.get("project", args.project),
                              "name": metadata.get("name", args.name or directory.name),
                              "mode": "tensorboard-tail"})
    print(f"SwanLab experiment: {run.url}", flush=True)
    tail = ScalarTail(directory, state.get("cursors"))
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    total = state.get("scalars_queued", 0)
    dead_polls = 0
    try:
        while True:
            count = tail.poll(run.log)
            total += count
            alive = (process.is_running() and process.create_time() == started
                     and process.status() != psutil.STATUS_ZOMBIE)
            dead_polls = 0 if alive else dead_polls + 1
            write_json(state_path, {"bridge_pid": os.getpid(), "training_pid": args.training_pid,
                                   "training_alive": alive, "updated_at": time.time(),
                                   "scalars_queued": total, "cursors": tail.cursors,
                                   "last_iteration": tail.cursors.get("Train/mean_reward"),
                                   "url": run.url})
            if count:
                print(f"Queued {count} scalars; total={total}; iteration="
                      f"{tail.cursors.get('Train/mean_reward')}", flush=True)
            if stopping or dead_polls >= 2:
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
