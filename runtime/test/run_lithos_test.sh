#!/bin/bash
# Build the LithOS-style hook (runtime/src/cuda-lithos.cpp) directly and run the
# hook, pass and two-stream tests under it. Needs a GPU node:
#   module load cuda/13.0 gnu/gcc-12.3; source scripts/ae/env-h100.sh
#   bash runtime/test/run_lithos_test.sh
# Env: MORPHX_SYNC_BACKEND (memcpy|hostmapped|gdr, default memcpy), OUT (build dir).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LLVM="${MORPHX_LLVM_ROOT:?set MORPHX_LLVM_ROOT}"
PASS="$REPO/smsched-pass/build/libsmsched.so"
OUT="${OUT:-$REPO/runtime/test/build-lithos}"
NUM_SMS="${MORPHX_NUM_SMS:-132}"
BACKEND="${MORPHX_SYNC_BACKEND:-memcpy}"
mkdir -p "$OUT"

REAL_LIBCUDA=""
for lib in /usr/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; do
    if [[ -e "$lib" ]]; then REAL_LIBCUDA="$lib"; break; fi
done
[[ -n "$REAL_LIBCUDA" ]] || { echo "libcuda.so not found" >&2; exit 1; }
case "$BACKEND" in
    hostmapped) SYNC_DEF=-DMORPHX_SYNC_BACKEND_HOSTMAPPED; GDR_LIB= ;;
    memcpy) SYNC_DEF=-DMORPHX_SYNC_BACKEND_MEMCPY; GDR_LIB= ;;
    gdr) SYNC_DEF=; GDR_LIB=-lgdrapi ;;
    *) echo "unknown MORPHX_SYNC_BACKEND" >&2; exit 1 ;;
esac
g++ -std=c++20 -O3 -g -fPIC -shared $SYNC_DEF \
    -I"$CUDA_HOME/include" -I"$REPO/include" "$REPO/runtime/src/cuda-lithos.cpp" -o "$OUT/libcuda.so" \
    -L"$CUDA_HOME/lib64" -lcudart -ldl $GDR_LIB "$REAL_LIBCUDA"
gcc -O3 -g -fPIC -shared "$REPO/runtime/src/preload.c" -o "$OUT/libpreload.so" -ldl
nvcc -O2 -arch=sm_90 -o "$OUT/hook_launch_test" "$REPO/runtime/test/hook_launch_test.cu"
for t in pass_smid_test lithos_two_stream_test; do
    "$LLVM/bin/clang++" --cuda-path="$CUDA_HOME" --cuda-gpu-arch=sm_90 -O3 \
        -fpass-plugin="$PASS" "$REPO/runtime/test/$t.cu" -o "$OUT/$t" \
        -L"$CUDA_HOME/lib64" -lcudart -ldl -lpthread -Wl,-rpath,"$CUDA_HOME/lib64" -rdynamic
done
echo "backend=$BACKEND hook=$OUT/libcuda.so"

status=0
run() {
    local name=$1; shift
    echo "== $name: $*"
    if ! env "$@" LD_PRELOAD="$OUT/libcuda.so:$OUT/libpreload.so" "$OUT/$name" 2>"$OUT/err.log"; then
        status=1
    fi
    grep -E "^\[smsched\]\[lithos\] (profile|split)" "$OUT/err.log" | head -1 || true
    echo "   slices launched: $(grep -c '\[smsched\]\[lithos\] launch' "$OUT/err.log" || true)" \
         " multi-slice cuts: $(grep -c '\[smsched\]\[lithos\] done' "$OUT/err.log" || true)"
    grep -E "\[smsched\]\[lithos\] (launch|done)" "$OUT/err.log" | sed -n '1,4p' | sed 's/^/   /' || true
    grep -iE "error|abort|assert" "$OUT/err.log" | head -3 | sed 's/^/   ERR /' || true
}

half=$((NUM_SMS / 2))
for loading in LAZY EAGER; do
    run hook_launch_test CUDA_MODULE_LOADING=$loading
done
run pass_smid_test SMSCHED_LITHOS_VERBOSE=1
run pass_smid_test SMSCHED_LITHOS_VERBOSE=1 PASS_TEST_MIN=0 PASS_TEST_MAX=$half PASS_TEST_STRICT=1
run pass_smid_test SMSCHED_LITHOS_SLICE_WAVES=4 PASS_TEST_MIN=0 PASS_TEST_MAX=$half PASS_TEST_STRICT=1
run pass_smid_test SMSCHED_LITHOS_SYNC=0
run lithos_two_stream_test SMSCHED_SPLIT_HINT=$half SMSCHED_LITHOS_VERBOSE=1
run lithos_two_stream_test SMSCHED_SPLIT_HINT=$half SMSCHED_LITHOS_RANK=priority LITHOS_TEST_B_HIGH=1
run lithos_two_stream_test SMSCHED_SPLIT_HINT=$half SMSCHED_LITHOS_RANK=priority LITHOS_TEST_B_HIGH=1 SMSCHED_LITHOS_STEAL=1
run lithos_two_stream_test SMSCHED_SPLIT_HINT=$half SMSCHED_LITHOS_SYNC=0
exit $status
