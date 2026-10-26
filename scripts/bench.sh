#!/bin/sh
# Local benchmark runner — mirrors .github/workflows/bench.yml so developers
# produce the same evidence format the CI archives (docs/benchmarks.md).
# Usage: scripts/bench.sh [scale]   (scale: kernel_bench --scale value,
#                                    default 1.0 = full reference protocol;
#                                    CI smoke uses 0.1)
set -e
cd "$(dirname "$0")/.."
STAMP="$(date +%Y-%m-%d)"
SCALE_ARGS=""
if [ "$#" -ge 1 ]; then
    SCALE_ARGS="--scale $1"
fi
mkdir -p .cep/evidence
echo "==> anvil_bench -> .cep/evidence/bench-${STAMP}-local.txt"
cargo run --release --bin anvil_bench | tee ".cep/evidence/bench-${STAMP}-local.txt"
echo "==> kernel_bench ${SCALE_ARGS:-full scale} -> .cep/evidence/kernel-bench-${STAMP}-local.txt"
# shellcheck disable=SC2086  # SCALE_ARGS is intentionally word-split
cargo run --release --bin kernel_bench -- ${SCALE_ARGS} | tee ".cep/evidence/kernel-bench-${STAMP}-local.txt"
