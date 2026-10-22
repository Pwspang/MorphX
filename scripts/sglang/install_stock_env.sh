#!/bin/bash
# Stock (no MorphX pass) SGLang env for the in-process runner, on hpcc.
# Mirrors ~/sglang-main/python (torch 2.9.1+cu130, flashinfer 0.6.8.post1,
# sglang-kernel 0.4.1.post1+cu130). ~/sglang-main is only read: it is copied to
# $SRC, and the copy's pyproject drops the rust gRPC extension (../rust is not
# part of ~/sglang-main, and there is no Rust toolchain on hpcc).
#   bash scripts/sglang/install_stock_env.sh
set -euo pipefail
module load python/3.12.13
SGLANG_SRC=${SGLANG_SRC:-$HOME/sglang-main/python}
SRC=$HOME/morphx/src/sglang-python
ENV=${ENV:-$HOME/morphx/envs/sglang-stock}

rsync -a --exclude __pycache__ --exclude build --exclude '*.egg-info' "$SGLANG_SRC/" "$SRC/"
python3 - "$SRC/pyproject.toml" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('''[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
''', "# MorphX copy: rust gRPC extension removed (../rust is not part of ~/sglang-main)\n")
open(p, "w").write(s)
PY

[ -d "$ENV" ] || uv venv --python /usr/local/python/3.12.13/bin/python3.12 "$ENV"
source "$ENV/bin/activate"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+sglang-main.20260429
export UV_HTTP_TIMEOUT=600
uv pip install "$SRC"
# kernels 0.17.2 breaks the transformers 5.6 import ("Either a revision or a
# version must be specified"); 0.13.0 matches ~/sglang-codec/.venv.
uv pip install "kernels==0.13.0"
python - <<'PY'
import importlib.metadata as m
for p in ["sglang", "torch", "flashinfer-python", "flashinfer-cubin", "sglang-kernel", "transformers", "triton", "kernels"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
PY
uv pip freeze > "$ENV.freeze.txt"
echo INSTALL_DONE
