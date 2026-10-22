#!/bin/bash
# Build pass_smid_test.cu with clang + the MorphX pass and run it under the
# runtime hook, without and with fix_SM ranges. Needs a GPU node:
#   module load cuda/13.0 gnu/gcc-12.3; source scripts/ae/env-h100.sh
#   bash runtime/test/run_pass_test.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LLVM="${MORPHX_LLVM_ROOT:?set MORPHX_LLVM_ROOT}"
PASS="$REPO/smsched-pass/build/libsmsched.so"
OUT="${OUT:-$REPO/runtime/test/build}"
NUM_SMS="${MORPHX_NUM_SMS:-132}"
mkdir -p "$OUT"

# Reuse run_hook_test.sh's direct build of the hook libraries.
REAL_LIBCUDA=""
for lib in /usr/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; do
    if [[ -e "$lib" ]]; then REAL_LIBCUDA="$lib"; break; fi
done
[[ -n "$REAL_LIBCUDA" ]] || { echo "libcuda.so not found" >&2; exit 1; }
g++ -std=c++20 -O3 -g -fPIC -shared -DMORPHX_SYNC_BACKEND_HOSTMAPPED \
    -I"$CUDA_HOME/include" -I"$REPO/include" "$REPO/runtime/src/cuda.cpp" -o "$OUT/libcuda.so" \
    -L"$CUDA_HOME/lib64" -lcudart -ldl "$REAL_LIBCUDA"
gcc -O3 -g -fPIC -shared "$REPO/runtime/src/preload.c" -o "$OUT/libpreload.so" -ldl
"$LLVM/bin/clang++" --cuda-path="$CUDA_HOME" --cuda-gpu-arch=sm_90 -O3 \
    -fpass-plugin="$PASS" "$REPO/runtime/test/pass_smid_test.cu" -o "$OUT/pass_smid_test" \
    -L"$CUDA_HOME/lib64" -lcudart -ldl -Wl,-rpath,"$CUDA_HOME/lib64" -rdynamic
cuobjdump -symbols "$OUT/pass_smid_test" | grep -c "_pk" | sed 's/^/patched kernels in binary: /'

run() {
    echo "== $*"
    env "$@" SMSCHED_DEBUG=1 SMSCHED_DEBUG_LIMIT=50 \
        LD_PRELOAD="$OUT/libcuda.so:$OUT/libpreload.so" "$OUT/pass_smid_test" 2>"$OUT/err.log" || status=1
    grep -oE "\[MorphX-debug\] [a-z-]+( func=[A-Za-z_0-9]+)?( range=\[[0-9,]+\))?" "$OUT/err.log" | sort | uniq -c | head -8 | sed 's/^/   /' || true
}
status=0
echo "== without runtime"
"$OUT/pass_smid_test" || status=1
half=$((NUM_SMS / 2))
run PASS_TEST_MIN=-1
run PASS_TEST_MIN=0 PASS_TEST_MAX=$half
run PASS_TEST_MIN=$half PASS_TEST_MAX=$NUM_SMS
exit $status
