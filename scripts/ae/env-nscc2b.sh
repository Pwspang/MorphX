# shellcheck shell=bash
# MorphX AE settings for NSCC ASPIRE 2B (nscc2b): H200 nodes (sm_90, 132 SMs),
# CUDA 13.0, and GDRCopy (gdrdrv + libgdrapi) on the GPU nodes, so the runtime
# uses the gdr sync backend like the original A100 artifact. Everything lives
# under ~/scratch/morphx (home is 50 GB). Variables already set win; the rest
# comes from env-h100.sh.
#
#   module load cuda/13.0
#   source ~/scratch/morphx/envs/sglang-stock/bin/activate
#   source scripts/ae/env-nscc2b.sh
#   ./run.sh smoke
#
# See docs/h100-port-plan.md, "nscc2b (ASPIRE 2B) setup".

export MORPHX_ROOT="${MORPHX_ROOT:-${HOME}/scratch/morphx}"
# The default PrgEnv makes `cc`/`CC` the Cray compiler wrappers, which link
# MPI/xpmem into everything; build with the plain gcc-toolset-14 compilers.
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
export MORPHX_LLVM_ROOT="${MORPHX_LLVM_ROOT:-${MORPHX_ROOT}/llvm}"
export MORPHX_CUTLASS_ROOT="${MORPHX_CUTLASS_ROOT:-${MORPHX_ROOT}/src/cutlass}"
export MORPHX_HF_CACHE="${MORPHX_HF_CACHE:-${MORPHX_ROOT}/hf/hub}"
export MORPHX_NANOFLOW_LIB="${MORPHX_NANOFLOW_LIB:-${MORPHX_ROOT}/envs/nanoflow/lib}"
# Unix socket paths are limited to 108 bytes; ~/scratch/... is a long path.
export MORPHX_TGS_SOCKET_DIR="${MORPHX_TGS_SOCKET_DIR:-${HOME}/.mxtgs}"

# GDRCopy: the GPU nodes have /dev/gdrdrv and /usr/lib64/libgdrapi.so.2, the
# login node has neither, so builds use headers and a link stub from a local
# user-space build (MORPHX_GDRCOPY_ROOT). It is deliberately not on
# LD_LIBRARY_PATH: at run time the node's libgdrapi must match its gdrdrv.
export MORPHX_GDRCOPY_ROOT="${MORPHX_GDRCOPY_ROOT:-${MORPHX_ROOT}/gdrcopy}"
export MORPHX_SYNC_BACKEND="${MORPHX_SYNC_BACKEND:-gdr}"

# The clang install was built on hpcc with gcc 12; RHEL 9.6's libstdc++ is
# older (no GLIBCXX_3.4.30), so ship gcc 12's libstdc++ next to it. Newer
# libstdc++ is backward compatible with everything else in the process.
case ":${LD_LIBRARY_PATH:-}:" in
  *":${MORPHX_LLVM_ROOT}/compat:"*) ;;
  *) export LD_LIBRARY_PATH="${MORPHX_LLVM_ROOT}/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

# Boost headers (the runtime's lock-free pool queue) aren't installed on
# nscc2b; a copy of hpcc's /usr/include/boost (1.75) lives here. CPATH reaches
# gcc and clang alike.
export MORPHX_BOOST_INCLUDE="${MORPHX_BOOST_INCLUDE:-${MORPHX_ROOT}/boost/include}"
case ":${CPATH:-}:" in
  *":${MORPHX_BOOST_INCLUDE}:"*) ;;
  *) export CPATH="${MORPHX_BOOST_INCLUDE}${CPATH:+:${CPATH}}" ;;
esac

if [[ -z "${MORPHX_PYTHON:-}" && -n "${VIRTUAL_ENV:-}" ]]; then
  export MORPHX_PYTHON="${VIRTUAL_ENV}/bin/python"
fi

# The pass-built ggnn module links libpython3.12.so, which the portable CPython
# keeps in its own lib/ (no system Python 3.12 here); it is also imported at
# build time to generate stubs. sysconfig's LIBDIR is the path the portable
# build was made at, so use sys.base_prefix.
if [[ -n "${MORPHX_PYTHON:-}" ]]; then
  _morphx_pylib="$("${MORPHX_PYTHON}" -c 'import sys; print(sys.base_prefix + "/lib")' 2>/dev/null)"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_morphx_pylib}:"*) ;;
    *) [[ -n "${_morphx_pylib}" ]] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_morphx_pylib}" ;;
  esac
  unset _morphx_pylib
fi

# shellcheck source=scripts/ae/env-h100.sh
source "$(dirname "${BASH_SOURCE[0]}")/env-h100.sh"
