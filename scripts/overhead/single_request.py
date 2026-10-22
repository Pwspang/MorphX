#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run single-request latency measurements for GPU workloads.

Each workload executes a dedicated warmup followed by multiple single-request
runs to compute an average latency. All work happens on the main thread/process
for easier debugging when colocated workloads are not required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import torch

try:
    import ggnn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ggnn = None

# LLM engine and dense GEMMs are selectable for the H100/H200 port (scripts/sglang):
#   MORPHX_LLM_BACKEND=vllm|sglang  (default vllm, the original stack)
#   MORPHX_LLM_GEMM=torch|mycutlass  (mycutlass: pass-built CUTLASS, so MorphX can schedule it)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sglang"))
import backend as llm_backend  # noqa: E402

_LLM: Any = None

# Machine-specific settings (see scripts/ae/morphx_env.py); defaults match the A100 AE machine.
HF_CACHE = os.environ.get("MORPHX_HF_CACHE") or "/huggingface-cache/hub"
_CUDA_ARCH = int(os.environ.get("MORPHX_CUDA_ARCH") or "80")

# Match environment hints from run.py for consistency across scripts.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_CUDA_ARCH // 10}.{_CUDA_ARCH % 10}")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_USE_V1", "0")


@contextmanager
def without_child_ld_preload():
    old_ld_preload = os.environ.pop("LD_PRELOAD", None)
    try:
        yield
    finally:
        if old_ld_preload is not None:
            os.environ["LD_PRELOAD"] = old_ld_preload


def exit_after_flush_if_morphx_preloaded() -> None:
    preload = os.environ.get("LD_PRELOAD", "")
    if "libcuda.so" not in preload or "libpreload.so" not in preload:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

MODEL_PATHS = {
    "llama-2-7b": f"{HF_CACHE}/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "llama-3-8b": f"{HF_CACHE}/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen3-4b": f"{HF_CACHE}/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": f"{HF_CACHE}/models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2",
    "gpt2-l": f"{HF_CACHE}/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519",
}


def set_smsched_bypass(enabled: bool) -> None:
    os.environ["SMSCHED_BYPASS"] = "1" if enabled else "0"


def get_llm_backend() -> Any:
    global _LLM
    if _LLM is None:
        try:
            _LLM = llm_backend.get_backend()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"LLM backend '{llm_backend.backend_name()}' is not available in this Python environment") from exc
    return _LLM


def gemm_backend() -> str:
    return os.environ.get("MORPHX_LLM_GEMM", "torch").strip().lower() or "torch"


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if gemm_backend() == "mycutlass":
        import mycutlass_gemm

        return mycutlass_gemm.matmul(a, b)
    return torch.matmul(a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-request latency measurements for selected workload"
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=("llm", "gemm", "ggnn", "dnn"),
        help="Which workload to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured runs (single request each) after warmup",
    )

    llm_group = parser.add_argument_group("LLM options")
    llm_group.add_argument(
        "--model",
        default="llama-2-7b",
        help="Model alias or path (defaults to llama-2-7b alias)",
    )
    llm_group.add_argument(
        "--input-length",
        type=int,
        default=128,
        help="LLM input (prompt) length in tokens",
    )
    llm_group.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="LLM output length in tokens",
    )
    llm_group.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable vLLM chunked prefill",
    )

    gemm_group = parser.add_argument_group("GEMM options")
    gemm_group.add_argument(
        "--mm-size",
        type=int,
        default=8192,
        help="Square matrix dimension N for GEMM (N x N)",
    )
    gemm_group.add_argument(
        "--mm-m",
        type=int,
        default=None,
        help="Matrix dimension M for GEMM (M x K) * (K x N) = (M x N)",
    )
    gemm_group.add_argument(
        "--mm-n",
        type=int,
        default=None,
        help="Matrix dimension N for GEMM",
    )
    gemm_group.add_argument(
        "--mm-k",
        type=int,
        default=None,
        help="Matrix dimension K for GEMM",
    )
    gemm_group.add_argument(
        "--mm-dtype",
        default="float16",
        help="Torch dtype for GEMM tensors (e.g., float16, float32)",
    )

    ggnn_group = parser.add_argument_group("GGNN options")
    ggnn_group.add_argument(
        "--ggnn-base-size",
        type=int,
        default=100_000,
        help="Number of base vectors in GGNN index",
    )
    ggnn_group.add_argument(
        "--ggnn-query-size",
        type=int,
        default=100_000,
        help="Number of query vectors per GGNN request",
    )
    ggnn_group.add_argument(
        "--ggnn-dim",
        type=int,
        default=128,
        help="Feature dimension for GGNN tensors",
    )
    ggnn_group.add_argument(
        "--ggnn-k-build",
        type=int,
        default=24,
        help="k parameter for GGNN build",
    )
    ggnn_group.add_argument(
        "--ggnn-tau-build",
        type=float,
        default=0.5,
        help="tau parameter for GGNN build",
    )
    ggnn_group.add_argument(
        "--ggnn-k-query",
        type=int,
        default=10,
        help="k parameter for GGNN query",
    )
    ggnn_group.add_argument(
        "--ggnn-tau-query",
        type=float,
        default=0.64,
        help="tau parameter for GGNN query",
    )
    ggnn_group.add_argument(
        "--ggnn-max-iter",
        type=int,
        default=400,
        help="Maximum GGNN query iterations",
    )
    ggnn_group.add_argument(
        "--ggnn-measure",
        default="euclidean",
        help="Distance measure for GGNN (euclidean, cosine, ...)",
    )
    ggnn_group.add_argument(
        "--ggnn-dtype",
        default="float32",
        help="Torch dtype for GGNN tensors",
    )
    ggnn_group.add_argument(
        "--ggnn-no-randomize",
        action="store_true",
        help="Disable per-run randomization of GGNN query tensor",
    )

    return parser.parse_args()


def ensure_cuda_device() -> None:
    if not torch.cuda.is_available():
        print("This benchmark requires a CUDA-capable GPU.", file=sys.stderr)
        sys.exit(1)


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {}
    summary: Dict[str, float] = {
        "runs": float(len(latencies)),
        "avg_ms": mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    if len(latencies) > 1:
        summary["stdev_ms"] = stdev(latencies)
    return summary


def pretty_print_summary(label: str, summary: Dict[str, float]) -> None:
    if not summary:
        print(f"{label}: no measurements recorded")
        return
    runs = int(summary.pop("runs", 0.0))
    stats_str = ", ".join(f"{key}={value:.3f} ms" for key, value in summary.items())
    print(f"{label}: {runs} runs -> {stats_str}")


def resolve_model_path(model: Optional[str]) -> str:
    if not model:
        model = "llama-2-7b"
    model = llm_backend.resolve_model_alias(model)
    return MODEL_PATHS.get(model, model)


def initialize_engine(model: str, input_length: int, output_length: int, enable_chunked_prefill: bool, stream: Any = None) -> Any:
    context_length = int(input_length) + int(output_length)
    with without_child_ld_preload():
        return get_llm_backend().create_engine(
            model,
            max_model_len=context_length,
            max_num_batched_tokens=context_length,
            max_num_seqs=1,
            enable_chunked_prefill=enable_chunked_prefill,
            gpu_memory_utilization=0.9,
            stream=stream,
        )


def build_prompt_ids(tokenizer, input_length: int) -> List[int]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)  # type: ignore[arg-type]
        except Exception:
            vocab_size = 32_000
    vocab_size = max(int(vocab_size), 2)
    return [((i % (vocab_size - 1)) + 1) for i in range(input_length)]


def wait_for_request(engine: Any, request_id: str) -> Any:
    while True:
        outputs = engine.step()
        torch.cuda.synchronize()
        if outputs is None or len(outputs) == 0:
            outputs = engine.get_prefill_result(isBlock=True)
            if not outputs:
                continue
        for ro in outputs:
            if ro.request_id == request_id and ro.finished:
                return ro

using_ncu = False

def maybe_start_cuda_profiler() -> None:
    if using_ncu:
        torch.cuda.profiler.cudart().cudaProfilerStart()


def maybe_stop_cuda_profiler() -> None:
    if using_ncu:
        torch.cuda.profiler.cudart().cudaProfilerStop()


def benchmark_llm(args: argparse.Namespace) -> Dict[str, float]:
    llm = get_llm_backend()
    runs = max(args.runs, 1)
    model_path = resolve_model_path(args.model)
    input_len = int(args.input_length)
    output_len = int(args.output_length)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        set_smsched_bypass(True)
        engine = initialize_engine(model_path, input_len, output_len, args.enable_chunked_prefill, stream=stream)
        tokenizer = engine.get_tokenizer()
        prompt_ids = build_prompt_ids(tokenizer, input_len)
        sampling_params = llm.make_sampling_params(max_tokens=output_len, min_tokens=output_len)

        prompt = llm.make_prompt(prompt_ids)
        engine.add_request("warmup", prompt, sampling_params)
        while engine.has_unfinished_requests():
            wait_for_request(engine, "warmup")
        print("[LLM] Warmup finished.")
        set_smsched_bypass(False)

        maybe_start_cuda_profiler()
        latencies = []
        ttfts = []
        for run_idx in range(runs):
            request_id = f"req-{run_idx}"
            engine.add_request(request_id, prompt, sampling_params)
            result = wait_for_request(engine, request_id)
            arrival = result.metrics.arrival_time
            finished = result.metrics.finished_time
            first_token = result.metrics.first_token_time
            latency_ms = (finished - arrival) * 1000.0
            latencies.append(latency_ms)
            if first_token is not None:
                ttfts.append((first_token - arrival) * 1000.0)
            print(f"[LLM] Run {run_idx}: latency={latency_ms:.3f} ms, tokens={len(result.outputs[0].token_ids) if result.outputs else 0}")

        set_smsched_bypass(True)
        engine.wait_prefill_shutdown()
        maybe_stop_cuda_profiler()

    summary = summarize_latencies(latencies)
    if ttfts:
        summary["ttft_avg_ms"] = mean(ttfts)
    return summary


def benchmark_gemm(args: argparse.Namespace) -> Dict[str, float]:
    runs = max(args.runs, 1)
    default_size = int(args.mm_size)
    m = int(args.mm_m) if args.mm_m is not None else default_size
    n = int(args.mm_n) if args.mm_n is not None else default_size
    k = int(args.mm_k) if args.mm_k is not None else default_size

    dtype_name = args.mm_dtype
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype '{dtype_name}' for GEMM benchmark")

    device = torch.device("cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        A = torch.randn((m, k), dtype=dtype, device=device)
        B = torch.randn((k, n), dtype=dtype, device=device)
        matmul(A, B)
    torch.cuda.synchronize()
    print(f"[GEMM] Warmup finished (gemm={gemm_backend()}).")

    latencies = []
    flops_per_matmul = 2 * m * n * k
    gflops = []
    maybe_start_cuda_profiler()
    for run_idx in range(runs):
        with torch.cuda.stream(stream):
            start = time.perf_counter()
            for _ in range(1):
                C = matmul(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0 / 1
        latencies.append(latency_ms)
        gflops.append(flops_per_matmul / (end - start) / 1e9)
        print(f"[GEMM] Run {run_idx}: latency={latency_ms:.3f} ms, approx GFLOPS={gflops[-1]:.2f}")
        _ = C  # suppress lint about unused variable
    maybe_stop_cuda_profiler()

    summary = summarize_latencies(latencies)
    summary["gflops_avg"] = mean(gflops)
    return summary


def resolve_distance_measure(measure_name: str):
    if ggnn is None or not hasattr(ggnn, "DistanceMeasure"):
        return None
    target = measure_name.upper()
    for attr in dir(ggnn.DistanceMeasure):
        if attr.upper() == target:
            return getattr(ggnn.DistanceMeasure, attr)
    return ggnn.DistanceMeasure.Euclidean


def benchmark_ggnn(args: argparse.Namespace) -> Dict[str, float]:
    if ggnn is None:
        raise RuntimeError("ggnn module is not available")

    os.environ["MEMTRACE_ENABLE"] = "0"

    runs = max(args.runs, 1)
    base_size = int(args.ggnn_base_size)
    query_size = int(args.ggnn_query_size)
    feature_dim = int(args.ggnn_dim)
    dtype_name = args.ggnn_dtype
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype '{dtype_name}' for GGNN benchmark")

    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    distance_measure = resolve_distance_measure(args.ggnn_measure)

    randomize = not args.ggnn_no_randomize

    with torch.cuda.stream(stream):
        base = torch.rand((base_size, feature_dim), dtype=dtype, device=device)
        query = torch.rand((query_size, feature_dim), dtype=dtype, device=device)

        index = ggnn.GGNN()
        index.set_base(base)
        index.set_return_results_on_gpu(True)
        index.build(
            k_build=int(args.ggnn_k_build),
            tau_build=float(args.ggnn_tau_build),
            measure=distance_measure,
        )
        torch.cuda.synchronize()

        index.query(
            query,
            int(args.ggnn_k_query),
            float(args.ggnn_tau_query),
            int(args.ggnn_max_iter),
            distance_measure,
        )
    torch.cuda.synchronize()
    print("[GGNN] Warmup finished.")

    os.environ["MEMTRACE_ENABLE"] = "1"
    latencies = []
    maybe_start_cuda_profiler()
    for run_idx in range(runs):
        if randomize:
            with torch.cuda.stream(stream):
                query.uniform_(0.0, 1.0)
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            index.query(
                query,
                int(args.ggnn_k_query),
                float(args.ggnn_tau_query),
                int(args.ggnn_max_iter),
                distance_measure,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        latencies.append(latency_ms)
        print(f"[GGNN] Run {run_idx}: latency={latency_ms:.3f} ms")
    maybe_stop_cuda_profiler()

    os.environ["MEMTRACE_ENABLE"] = "0"
    return summarize_latencies(latencies)

def benchmark_dnn(args: argparse.Namespace) -> Dict[str, float]:
    import torchvision
    from torch.amp import autocast

    # 检查硬件是否支持 bf16
    if not torch.cuda.is_bf16_supported():
        print("Warning: Your GPU does not support BF16 natively.")
    stream = torch.cuda.Stream()
    latencies = []
    with torch.cuda.stream(stream):
        model = torchvision.models.resnet50().cuda()
        model = model.to(memory_format=torch.channels_last)

        input_data = torch.randn(256, 3, 256, 256).cuda()
        input_data = input_data.to(memory_format=torch.channels_last)
        print("[DNN] Warming up...", flush=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(input_data)
        print("[DNN] Warmup finished.")
        maybe_start_cuda_profiler()
        for run_idx in range(args.runs):
            start_time = time.time()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)
        maybe_stop_cuda_profiler()
    return summarize_latencies(latencies)


def main() -> None:
    global using_ncu
    args = parse_args()
    ensure_cuda_device()
    if os.environ.get("NCU_PROFILING", "0") == "1":
        using_ncu = True
    else:
        using_ncu = False

    workload = args.workload.lower()
    if workload == "llm":
        summary = benchmark_llm(args)
        pretty_print_summary("LLM latency", summary)
    elif workload == "gemm":
        summary = benchmark_gemm(args)
        pretty_print_summary("GEMM latency", summary)
    elif workload == "ggnn":
        summary = benchmark_ggnn(args)
        pretty_print_summary("GGNN latency", summary)
    elif workload == "dnn":
        summary = benchmark_dnn(args)
        pretty_print_summary("DNN latency", summary)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported workload '{workload}'")


if __name__ == "__main__":
    main()
    exit_after_flush_if_morphx_preloaded()
