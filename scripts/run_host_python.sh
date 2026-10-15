#!/usr/bin/env bash
# Compatibility entry point; dependencies and imports belong to this checkout.
set -euo pipefail
ELF3_RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ELF3_RELEASE_ROOT/elf3.sh" "$@"
