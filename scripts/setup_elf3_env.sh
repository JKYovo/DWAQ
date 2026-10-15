#!/usr/bin/env bash
# Reproduce the verified Python 3.10 / Isaac Sim 4.5 / Isaac Lab 2.1.1 stack
# without editing any existing environment on the host.
set -euo pipefail

ELF3_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ELF3_ISAACLAB_DIR="${ELF3_ISAACLAB_DIR:-$ELF3_PROJECT_DIR/IsaacLab}"

if [[ ! -x "$ELF3_PROJECT_DIR/.venv/bin/python" ]]; then
    if [[ -n "${ELF3_CONDA:-}" ]]; then
        ELF3_CONDA_BIN="$ELF3_CONDA"
    elif command -v conda >/dev/null 2>&1; then
        ELF3_CONDA_BIN="$(command -v conda)"
    elif [[ -x /home/cheng/test/miniconda3-local/bin/conda ]]; then
        ELF3_CONDA_BIN=/home/cheng/test/miniconda3-local/bin/conda
    else
        echo "Conda was not found; set ELF3_CONDA to its executable path." >&2
        exit 1
    fi
    "$ELF3_CONDA_BIN" create --prefix "$ELF3_PROJECT_DIR/.venv" python=3.10.20 pip -y
fi

ELF3_PYTHON="$ELF3_PROJECT_DIR/.venv/bin/python"
export PYTHONNOUSERSITE=1

"$ELF3_PYTHON" -m pip install 'setuptools<81' toml
"$ELF3_PYTHON" -m pip install -r "$ELF3_PROJECT_DIR/requirements-elf3.txt"
"$ELF3_PYTHON" -m pip install 'isaacsim[all,extscache]==4.5.0' \
    --extra-index-url https://pypi.nvidia.com

if [[ ! -d "$ELF3_ISAACLAB_DIR/.git" ]]; then
    git clone --branch v2.1.1 --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "$ELF3_ISAACLAB_DIR"
fi

"$ELF3_PYTHON" -m pip install --no-build-isolation \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_assets" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_rl" \
    -e "$ELF3_ISAACLAB_DIR/source/isaaclab_tasks" \
    -e "$ELF3_PROJECT_DIR/TienKung-Lab/rsl_rl" \
    -e "$ELF3_PROJECT_DIR/TienKung-Lab" \
    -e "$ELF3_PROJECT_DIR/unitree_sdk2_python"

"$ELF3_PYTHON" -m pip check
