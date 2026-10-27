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
"""

import os
import shlex
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
    if os.environ.get("MORPHX_FI_CLANG_NOPASS", "0") != "1":
        so = os.environ.get("SMSCHED_PASS_SO", os.path.join(REPO, "smsched-pass", "build", "libsmsched.so"))
        if not os.path.exists(so):
            die(f"pass plugin {so} not found (build smsched-pass or set SMSCHED_PASS_SO)")
        cmd += [f"-fplugin={so}", f"-fpass-plugin={so}", "-mllvm", "-smsched-smem-handoff-barriers=1"]
    cmd += args
    cmd += shlex.split(os.environ.get("MORPHX_FI_CLANG_EXTRA", ""))
    log = os.environ.get("MORPHX_FI_CLANG_LOG")
    if log:
        with open(log, "a") as f:
            f.write(shlex.join(cmd) + "\n")
    os.execv(clang, cmd)


if __name__ == "__main__":
    main()
