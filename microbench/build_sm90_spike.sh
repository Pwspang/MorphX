#!/bin/bash
# Build microbench/src/sm90_spike.cu with clang for sm_90a, without and with the
# SMSched pass, into microbench/build-spike/sm90_spike-{nopass,pass}. No GPU needed.
#   module load gnu/gcc-12.3 cuda/13.0; source scripts/ae/env-h100.sh
#   bash microbench/build_sm90_spike.sh
# CUTLASS takes CUDA 12.4 for the CUDA version (clang reports none), hence the
# PFN_cuTensorMapEncode* aliases to the _v12000 typedefs that CUDA 13 still has.
# -DNDEBUG drops the device asserts: a call to __assertfail inside the mainloop
# makes ptxas serialize the wgmma pipeline (C7510): 1.8x slower at 4096^3.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA="${MORPHX_CUDA_ROOT:?source scripts/ae/env-h100.sh}"
CUTLASS="${MORPHX_CUTLASS_ROOT:?source scripts/ae/env-h100.sh}"
PASS="$REPO/smsched-pass/build/libsmsched.so"
OUT="$REPO/microbench/build-spike"
mkdir -p "$OUT"

FLAGS=(-x cuda "$REPO/microbench/src/sm90_spike.cu"
    -I"$CUDA/include" -I"$CUDA/include/cccl" -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include"
    -std=c++17 -O3 -DNDEBUG --cuda-path="$CUDA" --cuda-gpu-arch=sm_90a -Wno-unknown-cuda-version
    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4 -include cuda_fp16.h
    -DPFN_cuTensorMapEncodeTiled=PFN_cuTensorMapEncodeTiled_v12000
    -DPFN_cuTensorMapEncodeIm2col=PFN_cuTensorMapEncodeIm2col_v12000
    -L"$CUDA/lib64" -lcudart -lcublas -lcuda -ldl -Wl,-rpath,"$CUDA/lib64")
"$MORPHX_LLVM_ROOT/bin/clang++" "${FLAGS[@]}" -o "$OUT/sm90_spike-nopass" & p1=$!
"$MORPHX_LLVM_ROOT/bin/clang++" "${FLAGS[@]}" -fplugin="$PASS" -fpass-plugin="$PASS" -mllvm -smsched-smem-handoff-barriers -o "$OUT/sm90_spike-pass" & p2=$!
wait $p1; wait $p2
cuobjdump -symbols "$OUT/sm90_spike-pass" | grep -c "_pk$" | sed 's/^/patched kernels: /'
