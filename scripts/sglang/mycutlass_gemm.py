"""Route SGLang's dense linear layers through mycutlass (MORPHX_LLM_GEMM=mycutlass).

Stock SGLang runs every linear layer as cuBLASLt (`nvjet_*`), which the MorphX
LLVM pass cannot instrument. microbench/src/my_cutlass_gemm.cu
(libmycutlass.so, built by the AE with clang + the pass) is a CUTLASS fp16
GEMM with a shape lookup table; the authors' env ships its Python wrapper as
`mycutlass==0.1` (microbench/python/mycutlass). This module patches
UnquantizedLinearMethod.apply (all QKV/O/gate-up/down projections) and
LogitsProcessor._compute_lm_head (lm_head, incl. tied embeddings) to use it.

Layout: every operand of myCutlassHgemm is CUTLASS ColumnMajor. For
y[T, N] = x[T, K] @ W[N, K]^T (all row-major) we compute the column-major
y^T[N, T] = W[N, K] @ x^T[K, T]:
  m = N, n = T, k = K
  A = W as column-major N x K  -> W^T stored row-major (K, N), lda = N
  B = x as column-major K x T  -> x itself, ldb = x.stride(0)
  C = y as column-major N x T  -> y itself, ldc = N
so W^T is materialized once per weight (cached on the layer; costs one extra
copy of the dense weights). Unsupported cases (dtype != fp16, CUTLASS
alignment: N, K and leading dimensions multiple of 8, 16-byte aligned
pointers) fall back to torch and are counted in stats().
"""

from __future__ import annotations

import collections
import os
import sys
import threading
from typing import Optional

import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ALIGN = 8  # 128-bit accesses of fp16: kAlignmentA/B/C of the CUTLASS kernels

_stats = collections.Counter()
_lock = threading.Lock()
_installed = False
_hgemm = None
_enabled = True  # A/B switch for correctness checks in one process
_capture = None  # list collecting lm_head logits when set


def _load():
    global _hgemm
    if _hgemm is not None:
        return
    pkg = os.path.join(_REPO, "microbench", "python")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from mycutlass.binding import _lib, myCutlassHgemm  # noqa: F401

    if _lib is None:
        raise RuntimeError("libmycutlass.so not found; set MYCUTLASS_LIB_DIR to the pass-built microbench build dir")
    _hgemm = myCutlassHgemm


def set_enabled(flag: bool) -> None:
    global _enabled
    _enabled = bool(flag)


def capture_logits(sink) -> None:
    """Append every lm_head output (float32) to sink; None stops capturing."""
    global _capture
    _capture = sink


def stats() -> dict:
    with _lock:
        return dict(_stats)


def _count(key: str) -> None:
    with _lock:
        _stats[key] += 1


def _transposed_weight(owner: torch.nn.Module, weight: torch.Tensor) -> torch.Tensor:
    wt = getattr(owner, "_morphx_wt", None)
    if wt is None or getattr(owner, "_morphx_wt_src", None) != (weight.data_ptr(), tuple(weight.shape)):
        wt = weight.detach().t().contiguous()  # (K, N) row-major == W column-major
        owner._morphx_wt = wt
        owner._morphx_wt_src = (weight.data_ptr(), tuple(weight.shape))
    return wt


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def linear(owner: torch.nn.Module, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           tag: str = "linear") -> torch.Tensor:
    """y = x @ weight^T (+ bias) through mycutlass, or torch if the shape/dtype is unsupported."""
    n_out, k = weight.shape
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        _count(f"{tag}:fallback_dtype")
        return F.linear(x, weight, bias)
    if x.numel() == 0 or n_out % _ALIGN or k % _ALIGN:
        _count(f"{tag}:fallback_shape")
        return F.linear(x, weight, bias)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.stride(1) != 1 or x2.stride(0) % _ALIGN or not _aligned(x2):
        x2 = x2.contiguous()
        _count(f"{tag}:copied_input")
    t = x2.shape[0]
    wt = _transposed_weight(owner, weight)
    y = torch.empty((t, n_out), dtype=torch.float16, device=x.device)
    if not (_aligned(wt) and _aligned(y)):
        _count(f"{tag}:fallback_align")
        return F.linear(x, weight, bias)
    _hgemm(n_out, t, k, wt.data_ptr(), n_out, x2.data_ptr(), x2.stride(0), y.data_ptr(), n_out,
           1.0, 0.0, torch.cuda.current_stream(x.device).cuda_stream)
    if bias is not None:
        y.add_(bias)
    _count(f"{tag}:mycutlass")
    return y.view(*lead, n_out)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c = a @ b for row-major 2-D fp16 tensors, through mycutlass without copies.

    Row-major c[M, N] is column-major c^T[N, M] = b^T[N, K] @ a^T[K, M], and the
    row-major a and b already are column-major a^T and b^T, so
    myCutlassHgemm(m=N, n=M, k=K, A=b, lda=N, B=a, ldb=K, C=c, ldc=N).
    Used by the GEMM/GEMV workloads (Fig 11, Fig 15); falls back to torch.matmul.
    """
    _load()
    if a.dim() != 2 or b.dim() != 2 or a.dtype != torch.float16 or b.dtype != torch.float16:
        _count("matmul:fallback_dtype")
        return torch.matmul(a, b)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}")
    if n % _ALIGN or k % _ALIGN or m == 0:
        _count("matmul:fallback_shape")
        return torch.matmul(a, b)
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    if not (_aligned(a) and _aligned(b) and _aligned(c)):
        _count("matmul:fallback_align")
        return torch.matmul(a, b)
    _hgemm(n, m, k, b.data_ptr(), n, a.data_ptr(), k, c.data_ptr(), n,
           1.0, 0.0, torch.cuda.current_stream(a.device).cuda_stream)
    _count("matmul:mycutlass")
    return c


def prepare(model: torch.nn.Module) -> int:
    """Materialize W^T for every routed fp16 weight now, instead of on the first forward."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    n = 0
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.dtype != torch.float16:
            continue
        routed = isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod) or name.endswith("lm_head")
        if routed and weight.shape[0] % _ALIGN == 0 and weight.shape[1] % _ALIGN == 0:
            _transposed_weight(module, weight)
            n += 1
    return n


def install() -> None:
    """Patch SGLang (idempotent). Call before the model runs its first forward."""
    global _installed
    if _installed:
        return
    _load()
    from sglang.srt.layers import logits_processor
    from sglang.srt.layers.quantization import unquant

    orig_apply = unquant.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        if not _enabled:
            return orig_apply(self, layer, x, bias)
        if type(layer.weight.data) is not torch.Tensor or getattr(layer, "use_intel_amx_backend", False):
            _count("linear:fallback_other")
            return orig_apply(self, layer, x, bias)
        return linear(layer, x, layer.weight, bias, tag="linear")

    unquant.UnquantizedLinearMethod.apply = apply

    orig_lm_head = logits_processor.LogitsProcessor._compute_lm_head

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        plain = (
            hasattr(lm_head, "weight")
            and not (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
            and not getattr(self, "use_fp32_lm_head", False)
        )
        if not _enabled or not plain:
            if _enabled:
                _count("lm_head:fallback_other")
            logits = orig_lm_head(self, hidden_states, lm_head, embedding_bias)
        else:
            logits = linear(lm_head, hidden_states.to(lm_head.weight.dtype), lm_head.weight, None, tag="lm_head")
        if _capture is not None:
            _capture.append(logits.float())
        return logits

    logits_processor.LogitsProcessor._compute_lm_head = _compute_lm_head
    _installed = True
