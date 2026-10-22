# shellcheck shell=bash
# MorphX AE settings for Hopper GPUs (H100 SXM / H200: sm_90, 132 SMs) on CUDA 13.0.
# Defaults target the hpcc cluster; any variable already set in the environment wins.
#
#   module load gnu/gcc-12.3 cuda/13.0 anaconda/2025 && conda activate smsched
#   source scripts/ae/env-h100.sh
#   ./run.sh smoke
#
# Unset variables fall back to the original A100 machine (scripts/ae/morphx_env.py).

# Toolchain
# hpcc's /usr/bin/cc is gcc 11; use the gnu/gcc-12.3 module's compilers for CMake.
if command -v gcc >/dev/null && gcc -dumpversion | grep -q '^12'; then
  export CC="${CC:-gcc}" CXX="${CXX:-g++}"
fi
export MORPHX_CUDA_ROOT="${MORPHX_CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda-13.0}}"
export MORPHX_LLVM_ROOT="${MORPHX_LLVM_ROOT:-${HOME}/morphx/llvm}"
# CUDA 13 ships CCCL (Thrust/CUB/libcudacxx) under include/cccl.
export MORPHX_CCCL_ROOT="${MORPHX_CCCL_ROOT:-${MORPHX_CUDA_ROOT}/include/cccl}"
export MORPHX_CUTLASS_ROOT="${MORPHX_CUTLASS_ROOT:-${HOME}/morphx/src/cutlass}"
if [[ -z "${MORPHX_PYTHON:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  export MORPHX_PYTHON="${CONDA_PREFIX}/bin/python"
fi

# GPU
export MORPHX_CUDA_ARCH="${MORPHX_CUDA_ARCH:-90}"
export MORPHX_NUM_SMS="${MORPHX_NUM_SMS:-132}"  # H100 PCIe: 114

# No GDRCopy on hpcc: keep scheduler control words in pinned host memory
# (functionally correct, slower per block; see docs/h100-port-plan.md section 5).
export MORPHX_SYNC_BACKEND="${MORPHX_SYNC_BACKEND:-hostmapped}"

# Data and baselines
export MORPHX_HF_CACHE="${MORPHX_HF_CACHE:-${HOME}/morphx/hf/hub}"
export MORPHX_NANOFLOW_LIB="${MORPHX_NANOFLOW_LIB:-${HOME}/morphx/envs/nanoflow/lib}"
# Unix socket paths are limited to 108 bytes; keep this directory short.
export MORPHX_TGS_SOCKET_DIR="${MORPHX_TGS_SOCKET_DIR:-${HOME}/morphx/tgs}"
mkdir -p "${MORPHX_TGS_SOCKET_DIR}"

export LLVM_DIR="${LLVM_DIR:-${MORPHX_LLVM_ROOT}/lib/cmake/llvm}"
