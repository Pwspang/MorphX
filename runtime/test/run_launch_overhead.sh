#!/bin/bash
# Build launch_overhead_test.cu with clang + the MorphX pass and time patched
# launches without the runtime and under each sync backend / memcpy init mode.
# Needs a GPU node:
#   module load cuda/13.0 gnu/gcc-12.3; source scripts/ae/env-h100.sh
#   bash runtime/test/run_launch_overhead.sh
# Env: OUT (build dir), BEFORE_REPO (optional second checkout to compare against,
#      e.g. an older commit's runtime/ + include/), CONFIGS (override the runs).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LLVM="${MORPHX_LLVM_ROOT:?set MORPHX_LLVM_ROOT}"
PASS="$REPO/smsched-pass/build/libsmsched.so"
OUT="${OUT:-$REPO/runtime/test/build}"
mkdir -p "$OUT"

REAL_LIBCUDA=""
for lib in /usr/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; do
    if [[ -e "$lib" ]]; then REAL_LIBCUDA="$lib"; break; fi
done
[[ -n "$REAL_LIBCUDA" ]] || { echo "libcuda.so not found" >&2; exit 1; }

# build_hook <repo> <backend> <name>
build_hook() {
    local def gdr=
    case "$2" in
        hostmapped) def=-DMORPHX_SYNC_BACKEND_HOSTMAPPED ;;
        memcpy) def=-DMORPHX_SYNC_BACKEND_MEMCPY ;;
        gdr) def="${MORPHX_GDRCOPY_ROOT:+-I$MORPHX_GDRCOPY_ROOT/include}"; gdr="${MORPHX_GDRCOPY_ROOT:+-L$MORPHX_GDRCOPY_ROOT/lib} -lgdrapi" ;;
        *) echo "unknown backend $2" >&2; exit 1 ;;
    esac
    mkdir -p "$OUT/$3"
    g++ -std=c++20 -O3 -g -fPIC -shared $def \
        -I"$CUDA_HOME/include" -I"$1/include" "$1/runtime/src/cuda.cpp" -o "$OUT/$3/libcuda.so" \
        -L"$CUDA_HOME/lib64" -lcudart -ldl ${gdr:-} "$REAL_LIBCUDA"
    gcc -O3 -g -fPIC -shared "$1/runtime/src/preload.c" -o "$OUT/$3/libpreload.so" -ldl
}
build_hook "$REPO" memcpy memcpy
build_hook "$REPO" hostmapped hostmapped
[[ -e /dev/gdrdrv ]] && build_hook "$REPO" gdr gdr
if [[ -n "${BEFORE_REPO:-}" ]]; then
    build_hook "$BEFORE_REPO" memcpy before-memcpy
fi
"$LLVM/bin/clang++" --cuda-path="$CUDA_HOME" --cuda-gpu-arch=sm_90 -O3 \
    -fpass-plugin="$PASS" "$REPO/runtime/test/launch_overhead_test.cu" -o "$OUT/launch_overhead_test" \
    -L"$CUDA_HOME/lib64" -lcudart -Wl,-rpath,"$CUDA_HOME/lib64"

# config = <hook dir or none>:<extra env, comma-separated>
DEFAULT_CONFIGS="none:"
[[ -n "${BEFORE_REPO:-}" ]] && DEFAULT_CONFIGS+=" before-memcpy:"
DEFAULT_CONFIGS+=" memcpy: memcpy:MORPHX_MEMCPY_INIT=stream memcpy:MORPHX_AGENTS=grid memcpy:MORPHX_LAUNCH_CACHE=0 memcpy:MORPHX_AGENT_BATCH=0 memcpy:MORPHX_AGENT_CPU=-1 hostmapped:"
[[ -e /dev/gdrdrv ]] && DEFAULT_CONFIGS+=" gdr: gdr:MORPHX_AGENTS=grid"
for cfg in ${CONFIGS:-$DEFAULT_CONFIGS}; do
    hook=${cfg%%:*}
    extra=${cfg#*:}
    echo "== $hook ${extra//,/ }"
    envs=(SMSCHED_TARGET_PATTERNS=spin_work MORPHX_LAUNCH_STATS=1)
    [[ -n "$extra" ]] && IFS=, read -r -a more <<< "$extra" && envs+=("${more[@]}")
    if [[ "$hook" == none ]]; then
        env "${envs[@]}" "$OUT/launch_overhead_test" 2>&1 | grep -v "^$" || true
    else
        env "${envs[@]}" LD_PRELOAD="$OUT/$hook/libcuda.so:$OUT/$hook/libpreload.so" \
            "$OUT/launch_overhead_test" 2>&1 | grep -E "^\[(tiny|medium|MorphX-stats)\]|rror" || true
    fi
done
