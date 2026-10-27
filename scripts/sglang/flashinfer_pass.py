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

Env:
  MORPHX_FLASHINFER_PASS=1        enable (needs MORPHX_LLVM_ROOT and the built pass)
  MORPHX_FLASHINFER_PASS_CACHE    JIT workspace base (default <repo>/smsched-pass/build/flashinfer-ws)
  MORPHX_FI_CLANG_NOPASS=1        clang without the pass (own workspace)
  MORPHX_FLASHINFER_FA2=1         FA2 pin only, stock nvcc kernels

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


def fa2_only() -> bool:
    return enabled() or _flag("MORPHX_FLASHINFER_FA2")


def workspace_base() -> str:
    default = os.path.join(_REPO, "smsched-pass", "build",
                           "flashinfer-ws-nopass" if _flag("MORPHX_FI_CLANG_NOPASS") else "flashinfer-ws")
    return os.path.abspath(os.environ.get("MORPHX_FLASHINFER_PASS_CACHE", default))


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
    # Host-only sources and the .so link stay with the host compiler; every
    # .cu (kernels and the tvm-ffi bindings) goes through the wrapper.
    if os.environ.get("FLASHINFER_JIT_VERBOSE") is None:
        os.environ["FLASHINFER_JIT_VERBOSE"] = "0"
    so = os.environ.get("SMSCHED_PASS_SO", os.path.join(_REPO, "smsched-pass", "build", "libsmsched.so"))
    if not _flag("MORPHX_FI_CLANG_NOPASS") and not os.path.exists(so):
        raise RuntimeError(f"MORPHX_FLASHINFER_PASS=1 but the pass plugin {so} is missing")
    logger.info("FlashInfer JIT: clang + SMSched pass, workspace %s", base)


def patch_backends() -> None:
    """Pin FlashInfer's prefill/decode wrappers to FA2 (see module docstring)."""
    if not fa2_only():
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
