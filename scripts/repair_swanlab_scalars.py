"""Backfill and tail TensorBoard into an existing SwanLab run without changing its lifecycle."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time
import threading

import psutil
from swanlab import Api
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.util import tensor_util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--run-path", required=True, help="Existing owner/project/run slug")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--restore-running", action="store_true", help="Resume cloud state and maintain its heartbeat while training is alive")
    args = parser.parse_args()
    directory = args.log_dir.resolve(strict=True)
    lock = (directory / "swanlab_scalar_repair.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = psutil.Process(args.training_pid)
    created = process.create_time()
    state_path = directory / "swanlab_scalar_repair_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous and previous["run_path"] != args.run_path:
        raise ValueError("Repair state belongs to a different run")
    cursors = previous.get("cursors", {})
    sent = previous.get("scalars_uploaded", 0)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    def request_deadline(*_):
        raise TimeoutError("SwanLab request exceeded its deadline")

    signal.signal(signal.SIGALRM, request_deadline)

    def bounded_call(callback, *call_args, **kwargs):
        # The SDK adapter uses setdefault(timeout), but requests passes None;
        # without explicit timeouts a broken connection can block for hours.
        signal.setitimer(signal.ITIMER_REAL, 45)
        try:
            return callback(*call_args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    run = bounded_call(lambda: Api().run(args.run_path))
    # Public API resolves the internal IDs required by /house/metrics. Never
    # initialize or finish a second SDK run while the native logger is active.
    project_id, experiment_id = bounded_call(lambda: (run.project_id, run.run_id))
    if not project_id or not experiment_id:
        raise RuntimeError("Could not resolve the existing cloud run")
    known_columns = bounded_call(lambda: {column.key for column in run.columns(all=True)})
    loaders, pending = {}, []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def http(method, path, body):
        return bounded_call(run._ctx.client.request, method, path, json=body,
                            retries=0, timeout=(5, 25))

    def training_alive():
        return (process.is_running() and process.create_time() == created
                and process.status() != psutil.STATUS_ZOMBIE)

    last_heartbeat = 0.0

    def heartbeat():
        nonlocal last_heartbeat
        if args.restore_running and training_alive() and time.monotonic() - last_heartbeat >= 30:
            http("POST", f"/house/experiments/{experiment_id}/heartbeat", {"flagId": "123456"})
            last_heartbeat = time.monotonic()

    if args.restore_running and training_alive() and bounded_call(lambda: run.state) != "RUNNING":
        owner, project, slug = args.run_path.split("/")
        # Use the SDK's existing-run resume endpoint, with the same slug and
        # original name/time. Do not initialize another SDK logger or new run.
        existing = bounded_call(lambda: run._ensure_data())
        body = {"cuid": slug, "name": run.name, "createdAt": run.created_at}
        for key in ("colors", "description", "labels", "job", "cluster"):
            if existing.get(key) is not None:
                body[key] = existing[key]
        response = http("POST", f"/project/{owner}/{project}/experiment", body)
        if response.raw.status_code != 200:
            raise RuntimeError("Expected to resume the existing cloud run")
        print("Resumed the existing cloud run; training process unchanged", flush=True)
    heartbeat()
    worker_local = threading.local()
    executor = ThreadPoolExecutor(max_workers=args.workers)

    def upload_batch(batch):
        # Each worker owns its authenticated session; requests.Session is not
        # shared between threads. Explicit timeouts also cover worker requests.
        if not hasattr(worker_local, "client"):
            worker_local.client = Api()._ctx.client
        return worker_local.client.request(
            "POST", "/house/metrics", json={"type": "scalar", "projectId": project_id,
                                             "experimentId": experiment_id, "metrics": batch},
            retries=0, timeout=(5, 25))

    def save_state(alive):
        data = {"run_path": args.run_path, "url": run.url, "repair_pid": os.getpid(),
                "training_pid": args.training_pid, "training_alive": alive,
                "updated_at": time.time(), "scalars_uploaded": sent,
                "last_iteration": cursors.get("Train/mean_reward"), "cursors": cursors,
                "pending_scalars": len(pending)}
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(state_path)

    dead_polls = 0
    print(f"Repairing existing run {run.url}; training PID {args.training_pid}", flush=True)
    while True:
        try:
            if not pending:
                for path in sorted(directory.glob("events.out.tfevents.*")):
                    if path not in loaders:
                        loaders[path] = LegacyEventFileLoader(str(path))
                    for event in loaders[path].Load():
                        for item in event.summary.value:
                            if event.step <= cursors.get(item.tag, -1):
                                continue
                            if item.HasField("simple_value"):
                                value = item.simple_value
                            elif item.HasField("tensor"):
                                array = tensor_util.make_ndarray(item.tensor)
                                if array.size != 1 or array.dtype.kind not in "biuf":
                                    continue
                                value = float(array.reshape(-1)[0])
                            else:
                                continue
                            pending.append({"key": item.tag, "index": int(event.step),
                                            "data": float(value) if math.isfinite(value) else "nan",
                                            "create_time": datetime.fromtimestamp(event.wall_time, timezone.utc).isoformat()})
            missing = {metric["key"] for metric in pending} - known_columns
            if missing:
                http("POST", f"/experiment/{experiment_id}/columns",
                     [{"key": key, "type": "FLOAT", "name": key} for key in sorted(missing)])
                known_columns.update(missing)
            while pending:
                if stopping:
                    break
                heartbeat()
                batches = [pending[offset:offset + args.batch_size]
                           for offset in range(0, min(len(pending), args.batch_size * args.workers), args.batch_size)]
                futures = [executor.submit(upload_batch, batch) for batch in batches]
                succeeded = []
                failure = None
                for future in futures:
                    try:
                        future.result(timeout=45)
                        succeeded.append(True)
                    except Exception as error:
                        succeeded.append(False)
                        failure = error
                # Persist only acknowledged batches; failures remain queued and
                # commit only a contiguous prefix so a failed earlier batch
                # cannot be skipped after restart. Later successful batches
                # may be replayed with the same tag/step/value.
                accepted = 0
                for batch, success in zip(batches, succeeded):
                    if not success:
                        break
                    accepted += len(batch)
                batch = pending[:accepted]
                for metric in batch:
                    cursors[metric["key"]] = max(cursors.get(metric["key"], -1), metric["index"])
                sent += len(batch)
                del pending[:len(batch)]
                save_state(True)
                print(f"Uploaded {len(batch)} scalars; total={sent}; iteration={cursors.get('Train/mean_reward')}; pending={len(pending)}", flush=True)
                if failure is not None:
                    raise failure
            heartbeat()
            alive = training_alive()
            save_state(alive)
            dead_polls = 0 if alive else dead_polls + 1
            if stopping or dead_polls >= 2:
                break
        except Exception as error:
            # Do not print credentials, request headers, or authentication data.
            print(f"Upload retry after {type(error).__name__}; pending={len(pending)}", flush=True)
            if stopping:
                break
        time.sleep(5)
    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
