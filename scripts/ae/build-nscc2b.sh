#!/bin/bash
# Build MorphX's repo-local pieces on nscc2b (login node is fine), all with the
# gdr sync backend: the pass, every runtime variant, the pass-built microbench
# (libmycutlass), TGS, NVBit mem_trace and the pass-built ggnn module.
#   cd ~/scratch/morphx/MorphX && STEPS="pass runtime microbench tgs nvbit ggnn" bash scripts/ae/build-nscc2b.sh
# ggnn needs the Python env: VENV=~/scratch/morphx/envs/sglang-stock (default).
set -uo pipefail
module load cuda/13.0 cmake/4.3.4 >/dev/null 2>&1
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VENV=${VENV:-$HOME/scratch/morphx/envs/sglang-stock}
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
source scripts/ae/env-nscc2b.sh
STEPS=${STEPS:-"pass runtime microbench tgs nvbit ggnn"}
J=${J:-8}
L=${TMPDIR:-/tmp}/morphx-build-$USER
mkdir -p "$L"
log() { echo "[$(date +%T)] $*"; }
fail=0
echo "sync=$MORPHX_SYNC_BACKEND gdrcopy=$MORPHX_GDRCOPY_ROOT llvm=$MORPHX_LLVM_ROOT cuda=$MORPHX_CUDA_ROOT gcc=$(gcc -dumpversion) python=$(command -v python)"

if [[ " $STEPS " == *" pass "* ]]; then
  log pass
  rm -rf smsched-pass/build
  if cmake -S smsched-pass -B smsched-pass/build -DLLVM_DIR="$MORPHX_LLVM_ROOT/lib/cmake/llvm" > "$L/pass.log" 2>&1 \
     && cmake --build smsched-pass/build -j "$J" >> "$L/pass.log" 2>&1; then log "pass ok"; else log "pass FAILED"; tail -20 "$L/pass.log"; fail=1; fi
fi

if [[ " $STEPS " == *" runtime "* ]]; then
  # build: cuda.cpp (Fig 10-12, 15, 16); build-llm: cuda-llm.cpp (Fig 13-14);
  # build-orion / build-lithos: the extra Fig 10-12 schedulers.
  for v in "build:-DUSE_CUDA_LLM=OFF" "build-llm:-DUSE_CUDA_LLM=ON" "build-orion:-DUSE_CUDA_ORION=ON" "build-lithos:-DUSE_CUDA_LITHOS=ON"; do
    d=${v%%:*}; opt=${v#*:}
    log "runtime/$d $opt"
    rm -rf "runtime/$d"
    if cmake -S runtime -B "runtime/$d" "$opt" -DMORPHX_SYNC_BACKEND="$MORPHX_SYNC_BACKEND" -DMORPHX_GDRCOPY_ROOT="$MORPHX_GDRCOPY_ROOT" > "$L/rt-$d.log" 2>&1 \
       && cmake --build "runtime/$d" -j "$J" >> "$L/rt-$d.log" 2>&1; then log "runtime/$d ok"; else log "runtime/$d FAILED"; grep -E "error" "$L/rt-$d.log" | head -20; fail=1; fi
    grep -i "sync backend" "$L/rt-$d.log" | head -1
    gdr_imports=$(nm -D --undefined-only "runtime/$d/libcuda.so" 2>/dev/null | grep -cw gdr_open)
    if [[ "$MORPHX_SYNC_BACKEND" == gdr && "$gdr_imports" == 0 ]]; then
      [[ "$d" == build-orion ]] || { log "runtime/$d: libcuda.so doesn't import gdr_open (not a gdr build?)"; fail=1; }
    fi
  done
fi

if [[ " $STEPS " == *" microbench "* ]]; then
  log microbench
  rm -rf microbench/build-model
  if cmake -S microbench -B microbench/build-model -DSMSCHED_ENABLE_PASS=ON -DSMSCHED_LLVM_ROOT="$MORPHX_LLVM_ROOT" \
       -DMORPHX_SYNC_BACKEND="$MORPHX_SYNC_BACKEND" -DMORPHX_GDRCOPY_ROOT="$MORPHX_GDRCOPY_ROOT" > "$L/mb.log" 2>&1 \
     && cmake --build microbench/build-model --target bench -j "$J" >> "$L/mb.log" 2>&1; then log "microbench ok"; else log "microbench FAILED"; grep -E "error" "$L/mb.log" | head -20; fail=1; fi
  log "patched kernels in libmycutlass: $(cuobjdump -symbols microbench/build-model/libmycutlass.so 2>/dev/null | grep -c _pk)"
fi

if [[ " $STEPS " == *" tgs "* ]]; then
  log tgs
  if (cd TGS/hijack && bash build.sh > "$L/tgs.log" 2>&1); then log "tgs ok"; else log "tgs FAILED"; tail -15 "$L/tgs.log"; fail=1; fi
fi

if [[ " $STEPS " == *" nvbit "* ]]; then
  log nvbit
  if make -C nvbit-tutorial/tools/mem_trace -j "$J" > "$L/nvbit.log" 2>&1; then log "nvbit ok"; else log "nvbit FAILED"; tail -10 "$L/nvbit.log"; fail=1; fi
fi

if [[ " $STEPS " == *" ggnn "* ]]; then
  log ggnn
  # The build imports the module to generate stubs; the login node has no
  # libgdrapi, so borrow the local build's for that import only.
  if LD_LIBRARY_PATH="$MORPHX_GDRCOPY_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
       python -m pip install -e ggnn --no-build-isolation > "$L/ggnn.log" 2>&1; then log "ggnn ok"; else log "ggnn FAILED"; tail -20 "$L/ggnn.log"; fail=1; fi
fi
log "done fail=$fail"
exit $fail
