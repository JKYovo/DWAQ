#!/usr/bin/env bash
set -euo pipefail
ELF3_RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ELF3_RELEASE_ROOT"
if (( $# != 0 )); then
    echo "This fixed experiment accepts no overrides; use the Python entry for isolated checks." >&2
    exit 2
fi
mkdir -p logs
exec 9>logs/melody_training.lock
flock -n 9 || { echo "This experiment is already running in this checkout." >&2; exit 3; }
exec bash elf3.sh scripts/train_0918_melody.py \
    --headless --num-envs 4096 --iterations 100001 --swanlab-project DWAQ
