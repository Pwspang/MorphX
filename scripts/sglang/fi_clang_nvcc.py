#!/usr/bin/env python3
"""nvcc stand-in for FlashInfer's JIT: compile with clang++ and the SMSched pass.

FlashInfer builds its attention kernels at run time with ninja, calling
`$FLASHINFER_NVCC <nvcc flags> -c x.cu -o x.cuda.o`. Pointing FLASHINFER_NVCC at
this script (scripts/sglang/flashinfer_pass.py does that when
MORPHX_FLASHINFER_PASS=1) builds the same sources with clang's CUDA mode plus
the MorphX pass, so every __global__ kernel gets its morphable `_pk` clone, like
microbench/ (SMSCHED_ENABLE_PASS=ON) builds libmycutlass.

Only the flags FlashInfer's JIT emits are translated (flashinfer/jit/cpp_ext.py,
core.py); anything else is an error so a silent mis-translation can't happen.
Env:
  MORPHX_LLVM_ROOT     clang install (bin/clang++), default ~/morphx/llvm
  SMSCHED_PASS_SO      pass plugin, default <repo>/smsched-pass/build/libsmsched.so
  CUDA_HOME            CUDA toolkit (FlashInfer sets $cuda_home in build.ninja)
  MORPHX_FI_CLANG_EXTRA  extra clang flags (shell-split)
  MORPHX_FI_CLANG_NOPASS=1  clang without the pass (to separate compiler from pass effects)
  MORPHX_FI_CLANG_LOG  append each translated command line to this file
  MORPHX_FI_FA3_NONPERSISTENT=1  build the FA3 (*sm90*) sources with one CTA per
                       work tile (set by flashinfer_pass.py; see FA3_NP_PATCHES)
  MORPHX_FI_NVCC_DIV=1 approximate f32 div/sqrt as nvcc -use_fast_math does (set by
                       flashinfer_pass.py in FA3 mode; the FA2-only build keeps IEEE div/sqrt)
"""

import hashlib
import os
import re
import shlex
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def die(msg):
    sys.stderr.write(f"fi_clang_nvcc: {msg}\n")
    sys.exit(2)


def gencode_arch(value):
    # arch=compute_90a,code=sm_90a  (code may be a list: code=[sm_90a,compute_90a])
    archs = []
    for part in value.split(","):
        part = part.strip("[]")
        if part.startswith("code="):
            part = part[len("code="):]
        part = part.strip("[]")
        if part.startswith("sm_"):
            archs.append(part)
    return archs


# FlashInfer calls member templates of dependent objects without the `template`
# disambiguator (smem.load_128b_async<fill_mode>(...), vec_cast<T, U>::cast<8>(...)).
# nvcc's EDG front end accepts that; clang (correctly) doesn't. The wrapper
# compiles against a copy of FlashInfer's include/ with the keyword added, put
# ahead of the original on the include path. Semantics are unchanged.
TEMPLATE_MEMBER_NAMES = ("load_128b_async", "get_permuted_offset", "cast")
PATCH_VERSION = "1"


def _template_names():
    extra = tuple(n for n in os.environ.get("MORPHX_FI_TEMPLATE_NAMES", "").split(",") if n)
    return TEMPLATE_MEMBER_NAMES + extra


def patch_source(text):
    names = "|".join(_template_names())
    # obj.name<  obj->name<  Type<..>::name<   (unless already `template name<`)
    return re.sub(r"(\.|->|>::)(" + names + r")<", r"\1template \2<", text)


# Non-persistent FA3 (MORPHX_FA3_NONPERSISTENT, opt-in; see flashinfer_pass.py).
# FlashInfer's FA3 batch prefill launches one CTA per SM, and each CTA walks the
# tiles the planner (PrefillSM90Plan) gave it, found by blockIdx.x in work_indptr.
# Under MorphX that is 132 long logical blocks: confined to fewer SMs the tail
# serialises and there is nothing to preempt between. With the macro the planner
# plans one CTA per work tile (the heap gives every CTA exactly one tile, since a
# tile's cost is > 0), records that CTA count as a 10th plan_info element, and the
# launch uses it as the grid. The kernel is unchanged: its scheduler loop runs
# once. The run function reads the count back in PrefillPlanSM90Info::FromVector
# and hands it to the launcher (same thread, same call) through a thread_local.
# Without the macro every patched line compiles to the original code, so FA2
# modules and FA3 builds without the flag are unaffected.
FA3_PATCH_VERSION = "4"
FA3_NP_HEADER = r"""// Added by scripts/sglang/fi_clang_nvcc.py (MorphX non-persistent FA3).
#ifndef MORPHX_FA3_NP_CUH_
#define MORPHX_FA3_NP_CUH_
namespace flashinfer {
#ifdef MORPHX_FA3_NONPERSISTENT
// grid of the next FA3 batch-prefill launch on this thread (set by FromVector)
inline thread_local int morphx_fa3_num_ctas = 0;
#define MORPHX_FA3_PLAN_INFO_SIZE 10
#else
#define MORPHX_FA3_PLAN_INFO_SIZE 9
#endif
}  // namespace flashinfer
#endif
"""

_FA3_GRID_OLD = "  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);\n"
_FA3_GRID_NEW = """#ifdef MORPHX_FA3_NONPERSISTENT
  if constexpr (!std::is_same_v<Scheduler, SingleTileScheduler>) {
    if (morphx_fa3_num_ctas <= 0) return cudaErrorInvalidConfiguration;  // plan without the count
    multiprocessor_count = morphx_fa3_num_ctas;  // one CTA per work tile
  }
#endif
""" + _FA3_GRID_OLD

# (file under include/flashinfer, old, new, expected count)
FA3_NP_PATCHES = [
    ("attention/scheduler.cuh", '#include "../allocator.h"\n',
     '#include "../allocator.h"\n#include <flashinfer/attention/morphx_fa3_np.cuh>\n', 1),
    ("attention/scheduler.cuh",
     "  int64_t batch_indices_offset;\n  bool same_schedule_for_all_heads;\n\n  PrefillPlanSM90Info()",
     "  int64_t batch_indices_offset;\n  bool same_schedule_for_all_heads;\n"
     "#ifdef MORPHX_FA3_NONPERSISTENT\n  int64_t morphx_num_ctas = 0;\n#endif\n\n"
     "  PrefillPlanSM90Info()", 1),
    ("attention/scheduler.cuh",
     "work_indptr_offset,     batch_indices_offset, same_schedule_for_all_heads};",
     "work_indptr_offset,     batch_indices_offset, same_schedule_for_all_heads\n"
     "#ifdef MORPHX_FA3_NONPERSISTENT\n            , morphx_num_ctas\n#endif\n    };", 1),
    ("attention/scheduler.cuh", "if (vec.size() != 9) {", "if (vec.size() != MORPHX_FA3_PLAN_INFO_SIZE) {", 1),
    ("attention/scheduler.cuh", "    same_schedule_for_all_heads = vec[8];\n",
     "    same_schedule_for_all_heads = vec[8];\n#ifdef MORPHX_FA3_NONPERSISTENT\n"
     "    morphx_num_ctas = vec[9];\n    morphx_fa3_num_ctas = int(vec[9]);\n#endif\n", 1),
    ("attention/scheduler.cuh",
     "      cudaDeviceGetAttribute(&num_sm90_ctas, cudaDevAttrMultiProcessorCount, device));\n",
     "      cudaDeviceGetAttribute(&num_sm90_ctas, cudaDevAttrMultiProcessorCount, device));\n"
     "#ifdef MORPHX_FA3_NONPERSISTENT\n"
     "  if (enable_cuda_graph) {\n"
     "    FLASHINFER_ERROR(\"MorphX non-persistent FA3 (MORPHX_FA3_NONPERSISTENT) does not support CUDA graphs\");\n"
     "  }\n"
     "  {\n"
     "    // one CTA per work tile: the same works the loop below creates\n"
     "    bool same = (int(ceil_div(total_num_rows, cta_tile_q)) + int(batch_size) - 1) > 4096;\n"
     "    int64_t n = 0;\n"
     "    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) n += ceil_div(qo_len, cta_tile_q);\n"
     "    if (!same) n *= num_qo_heads;\n"
     "    num_sm90_ctas = int(n > 0 ? n : 1);\n"
     "  }\n"
     "  plan_info.morphx_num_ctas = num_sm90_ctas;\n"
     "#endif\n", 1),
    ("attention/hopper/prefill_sm90.cuh", '#include "tile_scheduler.cuh"\n',
     '#include "tile_scheduler.cuh"\n#include <flashinfer/attention/morphx_fa3_np.cuh>\n', 1),
    ("attention/hopper/prefill_sm90.cuh", _FA3_GRID_OLD, _FA3_GRID_NEW, 3),
    # clang (not EDG) needs these two disambiguators in the FA3 kernel; no semantic change
    ("attention/hopper/prefill_sm90.cuh", "collective_mainloop.load<LEFT_SLIDING_WINDOW>(",
     "collective_mainloop.template load<LEFT_SLIDING_WINDOW>(", 1),
    ("attention/hopper/prefill_sm90.cuh", "              CollectiveMainloop::WarpScheduler>(",
     "              typename CollectiveMainloop::WarpScheduler>(", -1),
    ("attention/hopper/mainloop_mma.cuh", "attention_updater.update</*init=*/",
     "attention_updater.template update</*init=*/", -1),
    # Re-entrant kernel body (MORPHX_FA3_REENTRANT, every pass build): the pass runs the
    # whole kernel body again for each logical block an agent fetches, but FA3 leaves
    # hardware named barriers half-arrived at exit (fine when the CTA exits, fatal when
    # the body restarts: the next mma_init()'s arrivals land in the old phase and the
    # warpgroups deadlock). Per CTA lifetime the only unmatched arrivals are mma_init()'s:
    # kQueryEmpty (consumers -> producer) and the warp-scheduler token(s). Complete them
    # at the end of the body, the way the next tile would.
    ("attention/hopper/named_barrier.cuh", "  static CUTLASS_DEVICE void mma_init() {",
     """#ifdef MORPHX_FA3_REENTRANT
  // Called by every consumer thread after its last tile: complete the scheduler
  // barrier phases left open by mma_init() so the body can run again.
  static CUTLASS_DEVICE void morphx_drain() {
    if constexpr (UseSchedulerBarrier) {
      int wg = cutlass::canonical_warp_group_idx();
      if constexpr (NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup) {
        if (wg == 1) barrier_sync();  // WG2's init arrival is pending on WG1's barrier
      } else {
        // pending: WG1's barrier WG2+WG3 (init), WG2's barrier WG3 (init)
        if (wg == 1) {
          cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS, get_warp_group_barrier_idx(2));
          barrier_sync();
        } else if (wg == 2) {
          barrier_sync();
        }
      }
    }
  }
#endif

  static CUTLASS_DEVICE void mma_init() {""", 1),
    ("attention/hopper/prefill_sm90.cuh",
     "      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);\n",
     "      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);\n"
     "#ifdef MORPHX_FA3_REENTRANT\n"
     "      // consumers' last kQueryEmpty arrival (mma_init's or the last tile's) has no producer sync\n"
     "      cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,\n"
     "                                        static_cast<int>(NamedBarriers::kQueryEmpty));\n"
     "#endif\n", 1),
    ("attention/hopper/prefill_sm90.cuh", "    collective_epilogue.store_tail();\n",
     "    collective_epilogue.store_tail();\n#ifdef MORPHX_FA3_REENTRANT\n"
     "    CollectiveMainloop::WarpScheduler::morphx_drain();\n#endif\n", 1),
    ("attention/hopper/quantization/prefill_sm90.cuh", _FA3_GRID_OLD, _FA3_GRID_NEW, 3),
]


def apply_fa3_patches(root):
    """Apply FA3_NP_PATCHES to the copied include tree <root>/flashinfer; die on any anchor mismatch."""
    fi = os.path.join(root, "flashinfer")
    with open(os.path.join(fi, "attention", "morphx_fa3_np.cuh"), "w") as fh:
        fh.write(FA3_NP_HEADER)
    for rel, old, new, count in FA3_NP_PATCHES:
        path = os.path.join(fi, rel)
        with open(path) as fh:
            text = fh.read()
        if (text.count(old) != count) if count >= 0 else (text.count(old) == 0):
            die(f"FA3 patch anchor found {text.count(old)}x (expected {count if count >= 0 else '>= 1'}) in {rel}: "
                f"{old[:60]!r}; "
                "FlashInfer version changed? (patches written for 0.6.8.post1)")
        with open(path, "w") as fh:
            fh.write(text.replace(old, new))
    # quantization/prefill_sm90.cuh includes ../prefill_sm90.cuh-level headers; make sure it sees the header
    q = os.path.join(fi, "attention", "hopper", "quantization", "prefill_sm90.cuh")
    with open(q) as fh:
        text = fh.read()
    if "morphx_fa3_np.cuh" not in text:
        text = text.replace("namespace flashinfer {", "#include <flashinfer/attention/morphx_fa3_np.cuh>\n"
                            "namespace flashinfer {", 1)
        with open(q, "w") as fh:
            fh.write(text)


def patched_include(orig, fa3=False):
    """Copy <orig>/flashinfer with patch_source applied, once per (orig, rules); return the copy's root.

    fa3=True: a separate copy that also carries the FA3 non-persistent patches
    (used for the *sm90* sources only), so FA2 builds see exactly the old copy."""
    key = hashlib.sha1(f"{os.path.realpath(orig)}|{PATCH_VERSION}|{','.join(_template_names())}".encode()).hexdigest()[:12]
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE", os.path.join(REPO, "smsched-pass", "build"))
    dst = os.path.join(base, f"include-patched-{key}" + (f"-fa3np{FA3_PATCH_VERSION}" if fa3 else ""))
    if os.path.isdir(dst):
        return dst
    tmp = f"{dst}.tmp{os.getpid()}"
    shutil.copytree(os.path.join(orig, "flashinfer"), os.path.join(tmp, "flashinfer"))
    changed = 0
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.endswith((".cuh", ".h", ".hpp", ".inc")):
                path = os.path.join(root, f)
                with open(path) as fh:
                    text = fh.read()
                new = patch_source(text)
                if new != text:
                    changed += 1
                    with open(path, "w") as fh:
                        fh.write(new)
    if fa3:
        apply_fa3_patches(tmp)
    with open(os.path.join(tmp, "PATCHED_FROM"), "w") as fh:
        fh.write(f"{orig}\n{changed} files patched by scripts/sglang/fi_clang_nvcc.py (+ FA3 non-persistent, "
                 "macro-guarded)\n")
    try:
        os.rename(tmp, dst)  # parallel compiles: the first rename wins
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


def translate(argv):
    out = []
    archs = []
    depfile = None
    gen_deps = False
    shared = False
    i = 0

    def take():
        nonlocal i
        i += 1
        if i >= len(argv):
            die(f"missing value after {argv[i - 1]}")
        return argv[i]

    while i < len(argv):
        a = argv[i]
        if a in ("--generate-dependencies-with-compile", "-MD"):
            gen_deps = True
        elif a in ("--dependency-output", "-MF"):
            depfile = take()
        elif a.startswith("--dependency-output="):
            depfile = a.split("=", 1)[1]
        elif a in ("-ccbin", "--compiler-bindir"):
            take()
        elif a.startswith("-ccbin=") or a.startswith("--compiler-bindir="):
            pass
        elif a.startswith("--compiler-options=") or a.startswith("-Xcompiler="):
            out += [x for x in a.split("=", 1)[1].split(",") if x]
        elif a in ("-Xcompiler", "--compiler-options"):
            out += [x for x in take().split(",") if x]
        elif a in ("--expt-relaxed-constexpr", "-expt-relaxed-constexpr",
                   "--expt-extended-lambda", "-expt-extended-lambda", "--extended-lambda",
                   "-lineinfo", "--keep", "-static-global-template-stub=false",
                   "-static-global-template-stub=true"):
            pass  # clang: constexpr/lambdas allowed in device code; no stub/lineinfo knobs
        elif a.startswith("--threads") or a.startswith("-t=") or a.startswith("-diag-suppress") \
                or a.startswith("--diag-suppress") or a.startswith("-Xfatbin"):
            if a in ("--threads", "-Xfatbin"):
                take()
        elif a in ("-use_fast_math", "--use_fast_math"):
            # nvcc: ftz + approximate div/sqrt/transcendentals. Not -ffast-math:
            # that implies finite-math-only, and FlashInfer masks with -inf.
            out += ["-fgpu-flush-denormals-to-zero", "-fgpu-approx-transcendentals"]
        elif a.startswith("-gencode=") or a.startswith("--generate-code="):
            archs += gencode_arch(a.split("=", 1)[1])
        elif a in ("-gencode", "--generate-code"):
            archs += gencode_arch(take())
        elif a.startswith("-arch=sm_") or a.startswith("--gpu-architecture=sm_"):
            archs.append(a.split("=", 1)[1])
        elif a.startswith("--ptxas-options="):
            for x in a.split("=", 1)[1].split(","):
                out += ["-Xcuda-ptxas", x]
        elif a in ("-Xptxas", "--ptxas-options"):
            for x in take().split(","):
                out += ["-Xcuda-ptxas", x]
        elif a == "-shared":
            shared = True
            out.append(a)
        elif a == "-isystem":
            out += [a, take()]
        elif a in ("-c", "-o", "-x") or a.startswith(("-D", "-U", "-I", "-O", "-std=", "-f", "-W",
                                                      "-L", "-l", "-g")):
            out.append(a)
            if a in ("-o", "-x"):
                out.append(take())
        elif a.startswith("-"):
            die(f"unsupported nvcc flag {a!r} (extend scripts/sglang/fi_clang_nvcc.py)")
        else:
            out.append(a)  # source / object file
        i += 1

    if gen_deps:
        if depfile is None:
            die("--generate-dependencies-with-compile without --dependency-output")
        out += ["-MD", "-MF", depfile]
    return out, sorted(set(archs)), shared


def main():
    args, archs, shared = translate(sys.argv[1:])
    llvm = os.environ.get("MORPHX_LLVM_ROOT", os.path.expanduser("~/morphx/llvm"))
    clang = os.path.join(llvm, "bin", "clang++")
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("MORPHX_CUDA_ROOT") or "/usr/local/cuda"
    if shared:
        # Device linking isn't needed by the attention ops; keep it explicit.
        die("device-link (-shared) builds are not supported")
    if not archs:
        die("no -gencode sm_XX target")
    cmd = [clang, "-x", "cuda", f"--cuda-path={cuda}", "-Wno-unknown-cuda-version"]
    cmd += [f"--cuda-gpu-arch={a}" for a in archs]
    # As cmake/morphx.cmake (CUDA >= 13, plain clang CUDA mode): clang doesn't
    # define __CUDACC_VER_*; claim 12.4 so CUTLASS/cute take their tensor-core
    # paths without nvcc-only __nv_atomic_* builtins.
    cmd += ["-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=4", "-Wno-unused-command-line-argument"]
    # host-side ::min/::max that nvcc's math headers provide (see the header)
    cmd += ["-include", os.path.join(REPO, "scripts", "sglang", "fi_clang_compat.h")]
    if os.environ.get("MORPHX_FI_CLANG_NOPASS", "0") != "1":
        so = os.environ.get("SMSCHED_PASS_SO", os.path.join(REPO, "smsched-pass", "build", "libsmsched.so"))
        if not os.path.exists(so):
            die(f"pass plugin {so} not found (build smsched-pass or set SMSCHED_PASS_SO)")
        cmd += [f"-fplugin={so}", f"-fpass-plugin={so}", "-mllvm", "-smsched-smem-handoff-barriers=1"]
    # FA3 (Hopper) sources: FlashInfer's batch_prefill_*_sm90 JIT module (MORPHX_FLASHINFER_FA3=1).
    srcs = [a for a in args if a.endswith(".cu")]
    sm90 = any("sm90" in os.path.basename(s) for s in srcs)
    if sm90:
        # TMA descriptors live in the kernel params and must stay in param space:
        # cute enables __grid_constant__ on the device side only (it keys on
        # __CUDA_ARCH__); CUTLASS's own macro is off under clang. Same for both sides.
        cmd += ["-DCUTE_GRID_CONSTANT=__grid_constant__", "-DCUTLASS_GRID_CONSTANT=__grid_constant__"]
        # CUTLASS takes CUDA 12.4 (see above); CUDA 13 only keeps the versioned typedefs
        cmd += ["-DPFN_cuTensorMapEncodeTiled=PFN_cuTensorMapEncodeTiled_v12000",
                "-DPFN_cuTensorMapEncodeIm2col=PFN_cuTensorMapEncodeIm2col_v12000"]
        if os.environ.get("MORPHX_FI_FA3_NONPERSISTENT", "0") == "1":
            cmd += ["-DMORPHX_FA3_NONPERSISTENT=1"]
        if os.environ.get("MORPHX_FI_CLANG_NOPASS", "0") != "1":
            cmd += ["-DMORPHX_FA3_REENTRANT=1"]  # the pass re-runs the body per logical block
    if os.environ.get("MORPHX_FI_NVCC_DIV", "0") == "1":
        # nvcc -use_fast_math also means -prec-div=false -prec-sqrt=false; with these the
        # clang build of FA3 + FA2 + cascade is bitwise identical to stock nvcc (fi_kernel_test)
        cmd += ["-mllvm", "-nvptx-prec-divf32=1", "-mllvm", "-nvptx-prec-sqrtf32=0"]
    for i, a in enumerate(args[:-1]):
        inc = args[i + 1]
        if a == "-isystem" and os.path.basename(os.path.normpath(inc)) == "include" \
                and os.path.isdir(os.path.join(inc, "flashinfer")):
            cmd += ["-I", patched_include(inc, fa3=sm90)]
            break
    cmd += args
    cmd += shlex.split(os.environ.get("MORPHX_FI_CLANG_EXTRA", ""))
    log = os.environ.get("MORPHX_FI_CLANG_LOG")
    if log:
        with open(log, "a") as f:
            f.write(shlex.join(cmd) + "\n")
    os.execv(clang, cmd)


if __name__ == "__main__":
    main()
