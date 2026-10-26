#!/bin/bash
# Build the MorphX runtime hook directly (no CMake) and run hook_launch_test.cu
# under it. Needs a GPU node, e.g. inside a PBS job on hpcc:
#   module load cuda/13.0 gnu/gcc-12.3
#   bash runtime/test/run_hook_test.sh
# Env: CUDA_HOME (toolkit), MORPHX_SYNC_BACKEND (hostmapped|memcpy|gdr, default hostmapped),
#      NEUTRINO_REAL_DRIVER (optional; auto-detected), OUT (build dir).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OUT="${OUT:-$REPO/runtime/test/build}"
BACKEND="${MORPHX_SYNC_BACKEND:-hostmapped}"
mkdir -p "$OUT"

DEFS=()
GDR_LIB=()
if [[ "$BACKEND" == hostmapped ]]; then
    DEFS+=(-DMORPHX_SYNC_BACKEND_HOSTMAPPED)
elif [[ "$BACKEND" == memcpy ]]; then
    DEFS+=(-DMORPHX_SYNC_BACKEND_MEMCPY)
else
    if [[ -n "${MORPHX_GDRCOPY_ROOT:-}" ]]; then
        DEFS+=(-I"$MORPHX_GDRCOPY_ROOT/include")
        GDR_LIB+=(-L"$MORPHX_GDRCOPY_ROOT/lib")
    fi
    GDR_LIB+=(-lgdrapi)
fi
REAL_LIBCUDA=""
for lib in /usr/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; do
    if [[ -e "$lib" ]]; then REAL_LIBCUDA="$lib"; break; fi
done
[[ -n "$REAL_LIBCUDA" ]] || { echo "libcuda.so not found" >&2; exit 1; }

for variant in cuda cuda-llm; do
    g++ -std=c++20 -O3 -g -fPIC -shared "${DEFS[@]}" \
        -I"$CUDA_HOME/include" -I"$REPO/include" \
        "$REPO/runtime/src/$variant.cpp" -o "$OUT/lib$variant-hook.so" \
        -L"$CUDA_HOME/lib64" -lcudart -ldl "${GDR_LIB[@]}" "$REAL_LIBCUDA"
done
gcc -O3 -g -fPIC -shared "$REPO/runtime/src/preload.c" -o "$OUT/libpreload.so" -ldl
nvcc -O2 -arch=sm_90 -o "$OUT/hook_launch_test" "$REPO/runtime/test/hook_launch_test.cu"

echo "== without hook"
"$OUT/hook_launch_test"

status=0
for variant in cuda cuda-llm; do
    for loading in LAZY EAGER; do
        # libpreload redirects dlopen("libcuda.so*") to libcuda.so next to itself
        cp "$OUT/lib$variant-hook.so" "$OUT/libcuda.so"
        echo "== hook=$variant CUDA_MODULE_LOADING=$loading"
        log="$OUT/$variant-$loading.log"
        if CUDA_MODULE_LOADING=$loading SMSCHED_DEBUG=1 SMSCHED_DEBUG_LIMIT=100000 \
            SMSCHED_LLM_TRACE_LIMIT=100000 \
            LD_PRELOAD="$OUT/libcuda.so:$OUT/libpreload.so" \
            "$OUT/hook_launch_test" > "$log" 2>&1; then
            :
        else
            status=1
        fi
        grep -E "^\[[0-9]|RESULT" "$log" || true
        # evidence that launches went through the hook and which path they took
        echo "   cuGetProcAddress lookups: $(grep -c 'cuGetProcAddress' "$log" || true)"
        grep -oE "\[MorphX-debug\] (launch func=[A-Za-z_]+|bypass-[a-z-]+|profile func=[A-Za-z_]+)|phase=[a-z-]+ .* name=[A-Za-z_]+" "$log" \
            | sed -E 's/phase=([a-z-]+) .* name=/llm \1 /' \
            | sort | uniq -c | sed 's/^/   /' || true
    done
done
exit $status
