#!/usr/bin/env bash
# Run DWAQ with its isolated environment. Supports either the pip Isaac Sim
# installed by scripts/setup_elf3_env.sh or an optional standalone install.
set -euo pipefail

ELF3_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ELF3_PYTHON="$ELF3_PROJECT_DIR/.venv/bin/python"
ELF3_ISAAC_SIM_DIR="${ELF3_ISAAC_SIM_DIR:-}"

if [[ ! -x "$ELF3_PYTHON" ]]; then
    echo "DWAQ environment is missing. Run: bash scripts/setup_elf3_env.sh" >&2
    exit 1
fi

# Only public command dispatch changes; imported checkpoint helpers are untouched.
"$ELF3_PYTHON" "$ELF3_PROJECT_DIR/scripts/elf3_task_catalog.py" --check-entrypoint "$@"

if [[ -n "$ELF3_ISAAC_SIM_DIR" ]]; then
    if [[ ! -f "$ELF3_ISAAC_SIM_DIR/setup_conda_env.sh" ]]; then
        echo "Invalid ELF3_ISAAC_SIM_DIR: $ELF3_ISAAC_SIM_DIR" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$ELF3_ISAAC_SIM_DIR/setup_conda_env.sh"
fi

ELF3_SITE_PACKAGES="$($ELF3_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ELF3_NVIDIA_LIBS=""
for ELF3_LIB_DIR in "$ELF3_SITE_PACKAGES"/nvidia/*/lib; do
    if [[ -d "$ELF3_LIB_DIR" ]]; then
        ELF3_NVIDIA_LIBS="${ELF3_NVIDIA_LIBS:+$ELF3_NVIDIA_LIBS:}$ELF3_LIB_DIR"
    fi
done

export PATH="$ELF3_PROJECT_DIR/.venv/bin:$PATH"
export PYTHONPATH="$ELF3_PROJECT_DIR/TienKung-Lab:$ELF3_PROJECT_DIR/TienKung-Lab/rsl_rl:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ELF3_NVIDIA_LIBS}${ELF3_NVIDIA_LIBS:+:}${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# This host exposes the same NVIDIA Vulkan ICD from two directories. Point Kit
# at one copy so the RTX 4090 is not enumerated twice.
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json \
    && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

exec "$ELF3_PYTHON" "$@"
