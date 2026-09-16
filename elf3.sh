#!/usr/bin/env bash
# Use this project's Python and the host's standalone Isaac Sim installation.
set -e
ELF3_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ELF3_ISAAC_SIM_DIR="${ELF3_ISAAC_SIM_DIR:-/home/cheng/isaacsim}"
source "$ELF3_ISAAC_SIM_DIR/setup_conda_env.sh"
export PATH="$ELF3_PROJECT_DIR/.venv/bin:$PATH"
export PYTHONPATH="$ELF3_PROJECT_DIR/TienKung-Lab:$ELF3_PROJECT_DIR/TienKung-Lab/rsl_rl:$PYTHONPATH"
export PYTHONUNBUFFERED=1
exec "$ELF3_PROJECT_DIR/.venv/bin/python" "$@"
