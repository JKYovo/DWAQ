"""Use one existing display-only arrow cache without changing training config.

The standard Omni Client alias is process-local. No robot/terrain asset,
reward, observation, checkpoint contract, or global Omni setting is changed.
All CLI arguments are forwarded to the original strict training launcher.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
ARROW = Path(os.environ.get("ELF3_ARROW_CACHE", str(
    Path.home() / ".cache/ov/client/http/4a8ab93074b320c51f69135679cb742b1c8d69fd1f1fb4b70816d14f2babe6ea.usd"
)))
ARROW_SHA256 = "d321116a8fecfef657ac95a5392115b2bfbe7d5eeae2a74556a7037f15259132"
URL_PATH = "omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd"


def main():
    import omni.client

    if not ARROW.is_file():
        raise FileNotFoundError(f"Existing display-only arrow cache unavailable: {ARROW}")
    if hashlib.sha256(ARROW.read_bytes()).hexdigest() != ARROW_SHA256:
        raise ValueError("Display-only arrow cache differs from the inspected file")
    if not omni.client.initialize():
        raise RuntimeError("Omni Client initialization failed")
    aliases = [f"{scheme}://{URL_PATH}" for scheme in ("http", "https")]
    try:
        for url in aliases:
            omni.client.set_alias(url, str(ARROW))
            result, local_path = omni.client.get_local_file(url)
            if result != omni.client.Result.OK or Path(local_path).resolve() != ARROW.resolve():
                raise RuntimeError(f"Display-only cache alias did not resolve: {url}")
        print("[INFO] Process-local visual cache: " + json.dumps({
            "aliases": aliases, "local_file": str(ARROW),
            "sha256": hashlib.sha256(ARROW.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "global_config_changed": False, "training_config_changed": False,
        }), flush=True)
        runpy.run_path(str(ROOT / "scripts/train_elf3_feedback.py"), run_name="__main__")
    finally:
        for url in aliases:
            omni.client.set_alias(url, None)
        omni.client.shutdown()


if __name__ == "__main__":
    main()
