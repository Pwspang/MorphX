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

# No GDRCopy on hpcc: keep scheduler control words in device memory and move
# them with async copies (see docs/h100-port-plan.md section 5). hostmapped also
# works but adds ~5 us per thread block.
export MORPHX_SYNC_BACKEND="${MORPHX_SYNC_BACKEND:-memcpy}"

# LLM engine (docs/h100-port-plan.md section 9): SGLang in-process instead of the
# authors' unpublished vLLM fork, with dense GEMMs (LLM linear layers and the
# GEMM/GEMV workloads) through the pass-built CUTLASS library so MorphX can
# schedule them. libmycutlass.so comes from the pass-enabled microbench build.
_morphx_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MORPHX_LLM_BACKEND="${MORPHX_LLM_BACKEND:-sglang}"
export MORPHX_LLM_GEMM="${MORPHX_LLM_GEMM:-mycutlass}"
# Default (2026-10-28): the sm90 library (CUTLASS 3 WGMMA/TMA, microbench/build-model-sm90),
# with pass-built FA3 when MORPHX_FLASHINFER_PASS=1 (scripts/sglang/flashinfer_pass.py).
# MORPHX_LEGACY_SM80_FA2=1: the earlier setup, Sm80 library (microbench/build-model) + FA2.
if [[ "${MORPHX_LEGACY_SM80_FA2:-0}" == 1 ]]; then
  export MYCUTLASS_LIB_DIR="${MYCUTLASS_LIB_DIR:-${_morphx_repo}/microbench/build-model}"
else
  export MYCUTLASS_LIB_DIR="${MYCUTLASS_LIB_DIR:-${_morphx_repo}/microbench/build-model-sm90}"
fi
# Gated Llama weights aren't downloaded yet: every LLM experiment runs on Qwen3-8B.
export MORPHX_MODEL_ALIASES="${MORPHX_MODEL_ALIASES-llama-3-8b=qwen3-8b,llama-2-7b=qwen3-8b}"
# Fig 13-14: the A100 request rates leave the H200 lightly loaded; 3x puts the
# baseline's saturation knee on the same rows (docs/h100-port-plan.md section 9, J4).
export MORPHX_PD_RPS_SCALE="${MORPHX_PD_RPS_SCALE:-3}"
# Fig 10-12: same for the matmul load; 3x takes llm-1-mm from light (mm 3/s) to
# saturated (mm 30/s), as the A100 grid did (section 9, Track L). GGNN unmeasured.
export MORPHX_MM_RPS_SCALE="${MORPHX_MM_RPS_SCALE:-3}"
unset _morphx_repo

# Data and baselines
export MORPHX_HF_CACHE="${MORPHX_HF_CACHE:-${HOME}/morphx/hf/hub}"
export MORPHX_NANOFLOW_LIB="${MORPHX_NANOFLOW_LIB:-${HOME}/morphx/envs/nanoflow/lib}"
# Unix socket paths are limited to 108 bytes; keep this directory short.
export MORPHX_TGS_SOCKET_DIR="${MORPHX_TGS_SOCKET_DIR:-${HOME}/morphx/tgs}"
mkdir -p "${MORPHX_TGS_SOCKET_DIR}"
# Fig 15 profiler baselines and build tools (docs/h100-port-plan.md section 8.1):
# ncu ships in the toolkit; CMake >= 3.31 and patchelf (TGS build.sh) are pip
# installs in ~/.local/bin; `neutrino` comes from scripts/baselines/neutrino_setup/.
case ":${PATH}:" in
  *":${MORPHX_CUDA_ROOT}/bin:"*) ;;
  *) export PATH="${MORPHX_CUDA_ROOT}/bin:${PATH}" ;;
esac
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

export LLVM_DIR="${LLVM_DIR:-${MORPHX_LLVM_ROOT}/lib/cmake/llvm}"
