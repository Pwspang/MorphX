"""Build SGLang's FlashInfer attention kernels with clang + the MorphX pass.

MORPHX_FLASHINFER_PASS=1 (default off) makes FlashInfer's run-time JIT compile
its CUDA sources with scripts/sglang/fi_clang_nvcc.py (clang++ CUDA mode plus
smsched-pass/build/libsmsched.so) instead of nvcc, into a separate JIT
workspace, so the attention kernels get `_pk` clones that the MorphX runtime
can confine to SM ranges. The stock nvcc cache (~/.cache/flashinfer) is never
touched.

It also pins every FlashInfer prefill/decode wrapper to the FA2 backend: on
sm_90 FlashInfer otherwise picks the Hopper FA3 kernels for SGLang's ragged
prefill (warp-specialized, TMA, setmaxnreg), which the pass's persistent-agent
transform doesn't support. FA2 is what the paper's A100 ran.
MORPHX_FLASHINFER_FA2=1 applies only the FA2 pin (stock nvcc build), to
separate the FA3 -> FA2 change from the compiler/pass change.

MORPHX_FLASHINFER_FA3=1 (opt-in, with MORPHX_FLASHINFER_PASS=1) drops the FA2
pin, so SGLang gets what it picks on Hopper: FA3 for the ragged prefill
(`BatchPrefillWithRaggedKVCacheWrapper(backend="auto")`); its paged prefill and
decode wrappers are hard-coded to FA2 in SGLang and stay FA2. The FA3 module is
then built with clang + the pass too, as a non-persistent variant (one CTA per
work tile instead of one per SM; fi_clang_nvcc.py FA3_NP_PATCHES), in its own
workspace. MORPHX_FA3_PERSISTENT=1 keeps FlashInfer's persistent scheduler
(grid = number of SMs; for comparison only).

Env:
  MORPHX_FLASHINFER_PASS=1        enable (needs MORPHX_LLVM_ROOT and the built pass)
  MORPHX_FLASHINFER_PASS_CACHE    JIT workspace base (default <repo>/smsched-pass/build/flashinfer-ws[-fa3|-fa3p])
  MORPHX_FI_CLANG_NOPASS=1        clang without the pass (own workspace)
  MORPHX_FLASHINFER_FA2=1         FA2 pin only, stock nvcc kernels
  MORPHX_FLASHINFER_FA3=1         no FA2 pin; pass-built non-persistent FA3 for the ragged prefill
  MORPHX_FA3_PERSISTENT=1         with FA3: keep the persistent tile scheduler

Call setup_env() before anything imports flashinfer (it fixes the JIT paths at
import time), then patch_backends() once flashinfer is importable.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _flag("MORPHX_FLASHINFER_PASS")


def fa3() -> bool:
    """Pass-built FA3 for the ragged prefill (no FA2 pin)."""
    return enabled() and _flag("MORPHX_FLASHINFER_FA3")


def fa3_persistent() -> bool:
    return fa3() and _flag("MORPHX_FA3_PERSISTENT")


def fa2_only() -> bool:
    return (enabled() and not fa3()) or _flag("MORPHX_FLASHINFER_FA2")


def workspace_base() -> str:
    name = "flashinfer-ws-nopass" if _flag("MORPHX_FI_CLANG_NOPASS") else "flashinfer-ws"
    if fa3():
        # FA3 objects differ between the variants; FA2 modules are rebuilt per workspace (~10 s).
        # Flags the wrapper adds are invisible to ninja, so key the workspace on the wrapper.
        import hashlib
        with open(os.path.join(_HERE, "fi_clang_nvcc.py"), "rb") as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:8]
        name += ("-fa3p-" if fa3_persistent() else "-fa3-") + tag
    default = os.path.join(_REPO, "smsched-pass", "build", name)
    return os.path.abspath(os.environ.get("MORPHX_FLASHINFER_PASS_CACHE", default))


def config_name() -> str:
    """Short tag for logs and dumps: stock|pass, fa2|auto|fa3|fa3p."""
    if fa3():
        att = "fa3p" if fa3_persistent() else "fa3"
    else:
        att = "fa2" if fa2_only() else "auto"
    return ("pass" if enabled() else "stock") + "-" + att


def setup_env() -> None:
    """Point FlashInfer's JIT at clang + the pass. Idempotent; no-op unless enabled."""
    if not enabled():
        return
    base = workspace_base()
    if "flashinfer.jit.env" in sys.modules:
        env = sys.modules["flashinfer.jit.env"]
        if not str(env.FLASHINFER_CACHE_DIR).startswith(base):
            raise RuntimeError(
                "MORPHX_FLASHINFER_PASS=1 but flashinfer was imported before flashinfer_pass.setup_env(); "
                f"its JIT cache is {env.FLASHINFER_CACHE_DIR}")
    os.makedirs(base, exist_ok=True)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = base
    os.environ["FLASHINFER_NVCC"] = os.path.join(_HERE, "fi_clang_nvcc.py")
    os.environ["MORPHX_FI_FA3_NONPERSISTENT"] = "1" if fa3() and not fa3_persistent() else "0"
    # FA3 mode: nvcc's fast-math div/sqrt too, so the build matches stock nvcc bit for bit
    os.environ["MORPHX_FI_NVCC_DIV"] = "1" if fa3() else "0"
    # Host-only sources and the .so link stay with the host compiler; every
    # .cu (kernels and the tvm-ffi bindings) goes through the wrapper.
    if os.environ.get("FLASHINFER_JIT_VERBOSE") is None:
        os.environ["FLASHINFER_JIT_VERBOSE"] = "0"
    so = os.environ.get("SMSCHED_PASS_SO", os.path.join(_REPO, "smsched-pass", "build", "libsmsched.so"))
    if not _flag("MORPHX_FI_CLANG_NOPASS") and not os.path.exists(so):
        raise RuntimeError(f"MORPHX_FLASHINFER_PASS=1 but the pass plugin {so} is missing")
    logger.info("FlashInfer JIT: clang + SMSched pass (%s), workspace %s", config_name(), base)


def runtime_report() -> None:
    """Print the MorphX runtime's [MorphX-profile] summary (os._exit skips its atexit hook)."""
    if "libcuda.so" not in os.environ.get("LD_PRELOAD", ""):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        import ctypes
        ctypes.CDLL(None).morphx_report()
    except (OSError, AttributeError):
        pass  # runtimes without the report (cuda.cpp)


def patch_backends() -> None:
    """Pin FlashInfer's prefill/decode wrappers to FA2 (see module docstring)."""
    if not (fa2_only() or fa3()):
        return
    import flashinfer  # noqa: F401
    from flashinfer import decode, prefill
    from flashinfer.jit import env as jit_env

    if enabled():
        # Prebuilt kernels would bypass the pass build: refuse an AOT/jit-cache package.
        if jit_env.has_flashinfer_jit_cache():
            raise RuntimeError("flashinfer-jit-cache is installed; its prebuilt .so files would bypass the pass")
        jit_env.FLASHINFER_AOT_DIR = jit_env.FLASHINFER_WORKSPACE_DIR / "no-aot"
        assert str(jit_env.FLASHINFER_JIT_DIR).startswith(workspace_base()), jit_env.FLASHINFER_JIT_DIR

    if fa3():
        logger.info("FlashInfer: no FA2 pin (MORPHX_FLASHINFER_FA3=1): ragged prefill auto -> FA3 on sm90")
        return
    for cls in (prefill.BatchPrefillWithRaggedKVCacheWrapper, prefill.BatchPrefillWithPagedKVCacheWrapper,
                decode.BatchDecodeWithPagedKVCacheWrapper):
        init = cls.__init__
        if getattr(init, "_morphx_fa2", False):
            continue

        def make(init, cls):
            def __init__(self, *args, **kwargs):
                backend = kwargs.get("backend", "auto")
                if backend not in ("auto", "fa2"):
                    raise RuntimeError(f"{cls.__name__}(backend={backend!r}) under MORPHX_FLASHINFER_PASS/FA2")
                kwargs["backend"] = "fa2"
                init(self, *args, **kwargs)
            __init__._morphx_fa2 = True
            return __init__

        cls.__init__ = make(init, cls)
    logger.info("FlashInfer: prefill/decode wrappers pinned to the FA2 backend")
