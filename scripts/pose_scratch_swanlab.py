"""SwanLab lifecycle and ACK-based uploads outside the training process.

Only uploader failures restart. Training is NEVER restarted by this process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.log_dir.resolve()
    process = psutil.Process(args.training_pid)
    created = process.create_time()
    def alive():
        try:
            return process.is_running() and process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    request = json.loads((directory / "swanlab_request.json").read_text())
    import swanlab
    import importlib.util
    spec = importlib.util.spec_from_file_location("isolated_transport", root / "TienKung-Lab/legged_lab/envs/elf3/swanlab_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.bound_native_requests()
    run = None
    while alive() and run is None:
        try:
            run = swanlab.init(project=request["project"], name=request["name"], config=request["config"],
                               mode="online", log_dir=str(directory / "swanlab"), resume="never")
        except Exception as error:
            print(f"SwanLab initialization failed (training unchanged): {error}", flush=True)
            time.sleep(5)
    if run is None:
        return
    (directory / "swanlab_run.json").write_text(json.dumps({
        "id": run.id, "url": run.url, "project": request["project"], "mode": "isolated_sidecar",
    }, indent=2) + "\n")
    print(f"SwanLab run: {run.url}", flush=True)
    slug = run.id
    # Use the owner in the SDK URL; no hardcoded user namespace.
    owner = run.url.split("/@", 1)[1].split("/", 1)[0]
    command = [sys.executable, str(root / "scripts/repair_swanlab_scalars.py"), "--log-dir", str(directory),
               "--training-pid", str(args.training_pid), "--run-path", f"{owner}/{request['project']}/{slug}",
               "--batch-size", "5000", "--workers", "4", "--restore-running"]
    try:
        with (directory / "swanlab_repair.log").open("a") as output:
            while True:
                result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
                if result.returncode == 0 or not alive():
                    break
                time.sleep(5)
    finally:
        exit_path = directory / "training_exit.json"
        success = exit_path.exists() and json.loads(exit_path.read_text()).get("success", False)
        run.finish(state="success" if success else "crashed",
                   error=None if success else "Training stopped/crashed; no automatic restart",
                   async_log_timeout=10)


if __name__ == "__main__":
    main()
