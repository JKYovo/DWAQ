#!/usr/bin/env bash
# Reproduce the installation on this host without editing its existing envs.
set -euo pipefail
ELF3_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ELF3_CONDA="${ELF3_CONDA:-/home/cheng/anaconda3/bin/conda}"
ELF3_SOURCE_ENV="${ELF3_SOURCE_ENV:-/home/cheng/anaconda3/envs/isaacsim45}"
ELF3_ISAACLAB_DIR="${ELF3_ISAACLAB_DIR:-/home/cheng/Desktop/IsaacLab}"
if [[ ! -x "$ELF3_PROJECT_DIR/.venv/bin/python" ]]; then
    "$ELF3_CONDA" create --offline --prefix "$ELF3_PROJECT_DIR/.venv" --clone "$ELF3_SOURCE_ENV" -y
fi
ELF3_PYTHON="$ELF3_PROJECT_DIR/.venv/bin/python"
"$ELF3_PYTHON" -m pip install -r "$ELF3_PROJECT_DIR/requirements-elf3.txt"
"$ELF3_PYTHON" -m pip install --no-deps --no-build-isolation \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_assets" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_rl" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_tasks" \
    -e "$ELF3_PROJECT_DIR/TienKung-Lab" \
    -e "$ELF3_PROJECT_DIR/TienKung-Lab/rsl_rl"
"$ELF3_PYTHON" -m pip check
