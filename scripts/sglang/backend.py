"""LLM engine selection for the MorphX workload scripts.

The scripts (scripts/colocate/run.py, scripts/test-pd-online.py,
scripts/overhead/single_request.py) drive a vLLM V0 `LLMEngine` directly:

    engine.add_request(request_id, TokensPrompt(prompt_token_ids=...), SamplingParams(...))
    outputs = engine.step()                     # -> [RequestOutput]
    engine.has_unfinished_requests()
    engine.get_tokenizer()                      # .vocab_size / len() / .encode()
    engine.get_prefill_result(isBlock=True)     # vLLM-fork only (VLLM_PREFILL_THREAD)
    ro.request_id, ro.finished, ro.outputs[0].token_ids,
    ro.metrics.{arrival_time, first_scheduled_time, first_token_time, finished_time, time_in_queue}

Both backends return an engine with exactly that surface, so the scripts only
have to swap `EngineArgs`/`LLMEngine`/`TokensPrompt`/`SamplingParams` for
`backend.create_engine(...)`, `backend.make_prompt(...)` and
`backend.make_sampling_params(...)`.

Select with MORPHX_LLM_BACKEND=vllm|sglang (default: vllm, the original stack).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclasses.dataclass
class GenerationParams:
    """Backend-neutral sampling parameters (vLLM names)."""

    max_tokens: int
    min_tokens: int = 0
    temperature: float = 0.0


class LLMBackend:
    name = "base"

    def create_engine(
        self,
        model_path: str,
        max_model_len: int,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        enable_chunked_prefill: bool = False,
        gpu_memory_utilization: float = 0.9,
        stream: Any = None,
    ) -> Any:
        raise NotImplementedError

    def make_prompt(self, prompt_token_ids: Sequence[int]) -> Any:
        raise NotImplementedError

    def make_sampling_params(self, max_tokens: int, min_tokens: int = 0, temperature: float = 0.0) -> Any:
        raise NotImplementedError


class VllmBackend(LLMBackend):
    """The original path: vLLM V0 LLMEngine in this process, eager mode."""

    name = "vllm"

    def __init__(self):
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        os.environ.setdefault("VLLM_USE_V1", "0")
        from vllm import EngineArgs, LLMEngine, SamplingParams  # type: ignore
        from vllm.inputs import TokensPrompt  # type: ignore

        self._EngineArgs = EngineArgs
        self._LLMEngine = LLMEngine
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt

    def create_engine(
        self,
        model_path,
        max_model_len,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.9,
        stream=None,
    ):
        kwargs = dict(
            model=model_path,
            enforce_eager=True,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens or max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs
        # vLLM V0 runs on the current stream; callers wrap this in torch.cuda.stream(...)
        return self._LLMEngine.from_engine_args(self._EngineArgs(**kwargs))

    def make_prompt(self, prompt_token_ids):
        return self._TokensPrompt(prompt_token_ids=list(prompt_token_ids))

    def make_sampling_params(self, max_tokens, min_tokens=0, temperature=0.0):
        return self._SamplingParams(temperature=temperature, max_tokens=max_tokens, min_tokens=min_tokens)


class SglangEngine:
    """vLLM-LLMEngine-shaped facade over runner.SglangRunner."""

    def __init__(self, runner):
        self.runner = runner

    def add_request(self, request_id, prompt, sampling_params, arrival_time=None):
        token_ids = prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt.prompt_token_ids
        self.runner.add_request(
            request_id,
            token_ids,
            max_tokens=sampling_params.max_tokens,
            min_tokens=sampling_params.min_tokens or 0,
            temperature=sampling_params.temperature,
            arrival_time=arrival_time,
        )

    def step(self):
        return self.runner.step()

    def has_unfinished_requests(self):
        return self.runner.has_unfinished_requests()

    def get_tokenizer(self):
        return self.runner.get_tokenizer()

    def get_prefill_result(self, isBlock=True):  # noqa: N803 (vLLM-fork name)
        # Prefill results come back from step(); the separate prefill thread is a later track.
        return []


class SglangBackend(LLMBackend):
    """SGLang ModelRunner in this process and thread (scripts/sglang/runner.py)."""

    name = "sglang"

    def __init__(self):
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from runner import SglangRunner

        self._SglangRunner = SglangRunner

    def create_engine(
        self,
        model_path,
        max_model_len,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.9,
        stream=None,
        **runner_kwargs,
    ):
        if enable_chunked_prefill:
            raise NotImplementedError("chunked prefill is not implemented in the in-process SGLang runner yet")
        runner = self._SglangRunner(
            model_path=model_path,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs or 256,
            max_num_batched_tokens=max_num_batched_tokens,
            mem_fraction_static=gpu_memory_utilization,
            stream=stream,
            **runner_kwargs,  # e.g. dtype, gemm (see runner.SglangRunner)
        )
        return SglangEngine(runner)

    def make_prompt(self, prompt_token_ids):
        return {"prompt_token_ids": list(prompt_token_ids)}

    def make_sampling_params(self, max_tokens, min_tokens=0, temperature=0.0):
        return GenerationParams(max_tokens=max_tokens, min_tokens=min_tokens, temperature=temperature)


_BACKENDS = {"vllm": VllmBackend, "sglang": SglangBackend}


def backend_name() -> str:
    return os.environ.get("MORPHX_LLM_BACKEND", "vllm").strip().lower() or "vllm"


def get_backend(name: Optional[str] = None) -> LLMBackend:
    name = (name or backend_name()).lower()
    if name not in _BACKENDS:
        raise ValueError(f"MORPHX_LLM_BACKEND={name!r}; expected one of {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def backend_names() -> List[str]:
    return sorted(_BACKENDS)
