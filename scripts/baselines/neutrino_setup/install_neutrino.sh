#!/bin/bash
# Build and install Neutrino (OSDI'25 GPU kernel probing, github.com/open-neutrino/neutrino),
# the `neutrino` CLI behind the Fig 15 "neutrino" baseline (`neutrino -p dmat <cmd>`).
# The authors' env had neutrino 0.1.0; PyPI's "neutrino" is an unrelated package.
#
#   module load cuda/13.0 gnu/gcc-12.3
#   PYTHON=~/morphx/envs/sglang-stock/bin/python bash scripts/baselines/neutrino_setup/install_neutrino.sh
#
# Changes applied on top of upstream 0.1.0 for CUDA 13 / driver 580 / RHEL:
#  - src/parse.py replaced by ./parse.py (CUDA 13 _v2 macro handling, same fix as
#    MorphX runtime/src/parse.py)
#  - build.py: detect CUDA by the nvidia-smi binary (login nodes without GPUs return
#    non-zero) and search $CUDA_HOME for cuda.h
#  - cli.py: resolve the real driver to this node's versioned libcuda file at run time
#    (see patch_neutrino.py)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${NEUTRINO_REPO:-https://github.com/open-neutrino/neutrino.git}"
COMMIT="${NEUTRINO_COMMIT:-4a82cd22f474c31ac2fecfa174d381a19bb3f469}"
SRC="${NEUTRINO_SRC:-$HOME/morphx/src/neutrino}"
PY="${PYTHON:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [[ ! -d "$SRC/.git" ]]; then
    git clone -q "$REPO_URL" "$SRC"
fi
git -C "$SRC" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$SRC" checkout -q -f "$COMMIT"

cp "$HERE/parse.py" "$SRC/neutrino/src/parse.py"
"$PY" "$HERE/patch_neutrino.py" "$SRC" "$CUDA_HOME"

"$PY" -m pip install -q toml
mkdir -p "$SRC/neutrino/build"
(cd "$SRC" && "$PY" neutrino/build.py)
# build.py records the versioned driver file (e.g. /lib/libcuda.so.580.65.06), which
# differs between the login node and the GPU nodes; point at the SONAME link instead.
REAL_DRIVER="${NEUTRINO_REAL_DRIVER:-}"
if [[ -z "$REAL_DRIVER" ]]; then
    for lib in /usr/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
        if [[ -e "$lib" ]]; then REAL_DRIVER="$lib"; break; fi
    done
fi
"$PY" - "$SRC/neutrino/build/config.toml" "$REAL_DRIVER" <<'PY'
import os, sys, toml
path, driver = sys.argv[1], sys.argv[2]
config = toml.load(path)
config["system"]["NEUTRINO_REAL_DRIVER_LIB_DIR"] = os.path.dirname(driver)
config["system"]["NEUTRINO_REAL_DRIVER_LIB_NAME"] = os.path.basename(driver)
toml.dump(config, open(path, "w"))
PY
"$PY" -m pip install -q --no-deps --no-build-isolation -e "$SRC"
NEUTRINO_BIN="$(dirname "$PY")/neutrino"
"$NEUTRINO_BIN" --help > /dev/null
echo "neutrino installed: $NEUTRINO_BIN ($(git -C "$SRC" rev-parse --short HEAD))"
