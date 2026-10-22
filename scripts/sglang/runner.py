"""In-process SGLang engine for MorphX experiments.

SGLang's Engine runs the model in a scheduler subprocess, but the MorphX
runtime can only split SMs between CUDA streams of one process. This wrapper
drives SGLang's ModelRunner directly (the same low-level path as
sglang.bench_one_batch) in the calling process and thread, on a caller-provided
CUDA stream, with a minimal continuous-batching loop:

- requests wait in FIFO order; a step admits as many as fit (KV tokens are
  reserved for prompt + max_tokens up front, so no preemption is needed) and
  runs them as one prefill (extend) batch;
- otherwise the step decodes the running batch by one token.

CUDA graphs, the radix cache, chunked prefill and the overlap scheduler are
off, so every kernel goes through cuLaunchKernel(Ex) where the MorphX hook can
see it. prefill() and decode() are separate so a later prefill thread
(Fig 13-14) can call them on different streams.

Not a package on purpose: scripts/sglang has no __init__.py, so it never
shadows the real `sglang` package (a regular package always wins over a
namespace directory on sys.path).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RequestMetrics:
    """Wall-clock timestamps (time.time()), named like vLLM's RequestMetrics."""

    arrival_time: float
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    finished_time: Optional[float] = None
    token_times: List[float] = dataclasses.field(default_factory=list)

    @property
    def time_in_queue(self) -> Optional[float]:
        if self.first_scheduled_time is None:
            return None
        return self.first_scheduled_time - self.arrival_time


@dataclasses.dataclass
class CompletionOutput:
    token_ids: List[int]


@dataclasses.dataclass
class RequestOutput:
    """Duck-types the parts of vLLM's RequestOutput the MorphX scripts read."""

    request_id: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool
    metrics: RequestMetrics


@dataclasses.dataclass
class _ReqState:
    req: Req
    max_tokens: int
    metrics: RequestMetrics
    reserved_tokens: int


class SglangRunner:
    """Single-GPU, single-process SGLang model runner with continuous batching."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        max_num_seqs: int = 256,
        max_num_batched_tokens: Optional[int] = None,
        mem_fraction_static: float = 0.85,
        stream: Optional[torch.cuda.Stream] = None,
        gpu_id: int = 0,
        attention_backend: str = "flashinfer",
        dtype: str = "auto",
        log_level: str = "error",
        **server_kwargs,
    ):
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = int(max_num_batched_tokens or max_model_len)
        torch.cuda.set_device(gpu_id)
        self.stream = stream if stream is not None else torch.cuda.current_stream()

        self.server_args = ServerArgs(
            model_path=model_path,
            dtype=dtype,
            context_length=self.max_model_len,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=self.max_num_seqs,
            attention_backend=attention_backend,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            log_level=log_level,
            **server_kwargs,
        )
        _set_envs_and_config(self.server_args)
        initialize_moe_config(self.server_args)
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)
        port_args = PortArgs.init_new(self.server_args)

        with torch.cuda.stream(self.stream):
            self.model_config = ModelConfig.from_server_args(self.server_args)
            self.model_runner = ModelRunner(
                model_config=self.model_config,
                mem_fraction_static=self.server_args.mem_fraction_static,
                gpu_id=gpu_id,
                tp_rank=0,
                tp_size=1,
                moe_ep_rank=0,
                moe_ep_size=1,
                pp_rank=0,
                pp_size=1,
                nccl_port=port_args.nccl_port,
                server_args=self.server_args,
            )
        self.stream.synchronize()

        self.tokenizer = get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        self.tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                page_size=self.server_args.page_size,
            )
        )
        self.kv_capacity = int(self.model_runner.max_total_num_tokens)
        self.reserved_tokens = 0
        self.waiting: Deque[_ReqState] = deque()
        self.running: Optional[ScheduleBatch] = None
        self.states: Dict[str, _ReqState] = {}
        self.num_prefill_steps = 0
        self.num_decode_steps = 0

    # ------------------------------------------------------------------ API

    def get_tokenizer(self):
        return self.tokenizer

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        max_tokens: int,
        min_tokens: int = 0,
        temperature: float = 0.0,
        arrival_time: Optional[float] = None,
    ) -> None:
        request_id = str(request_id)
        prompt_token_ids = [int(t) for t in prompt_token_ids]
        need = len(prompt_token_ids) + int(max_tokens)
        if need > self.max_model_len:
            raise ValueError(f"request {request_id}: prompt + max_tokens = {need} > max_model_len {self.max_model_len}")
        if need > self.kv_capacity:
            raise ValueError(f"request {request_id}: needs {need} KV tokens, pool holds {self.kv_capacity}")
        sampling_params = SamplingParams(
            max_new_tokens=int(max_tokens),
            min_new_tokens=int(min_tokens),
            temperature=temperature,
            # vLLM's min_tokens == max_tokens means "always generate exactly max_tokens"
            ignore_eos=int(min_tokens) >= int(max_tokens),
        )
        sampling_params.normalize(None)
        req = Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=prompt_token_ids,
            sampling_params=sampling_params,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        state = _ReqState(
            req=req,
            max_tokens=int(max_tokens),
            metrics=RequestMetrics(arrival_time=arrival_time if arrival_time is not None else time.time()),
            reserved_tokens=need,
        )
        self.states[request_id] = state
        self.waiting.append(state)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or (self.running is not None and not self.running.is_empty())

    def step(self) -> List[RequestOutput]:
        """One scheduling step: a prefill batch if any waiting request fits, else one decode step."""
        admitted = self._admit()
        if admitted:
            return self.prefill(admitted)
        if self.running is not None and not self.running.is_empty():
            return self.decode()
        return []

    # ------------------------------------------------------------ internals

    def _admit(self) -> List[_ReqState]:
        admitted: List[_ReqState] = []
        num_running = 0 if self.running is None else len(self.running.reqs)
        batch_tokens = 0
        free_req_slots = self.model_runner.req_to_token_pool.available_size()
        while self.waiting:
            state = self.waiting[0]
            prompt_len = len(state.req.origin_input_ids)
            if num_running + len(admitted) >= self.max_num_seqs or len(admitted) >= free_req_slots:
                break
            if self.reserved_tokens + state.reserved_tokens > self.kv_capacity:
                break
            if admitted and batch_tokens + prompt_len > self.max_num_batched_tokens:
                break
            self.waiting.popleft()
            self.reserved_tokens += state.reserved_tokens
            batch_tokens += prompt_len
            admitted.append(state)
        return admitted

    @torch.no_grad()
    def prefill(self, states: List[_ReqState]) -> List[RequestOutput]:
        now = time.time()
        for state in states:
            state.metrics.first_scheduled_time = now
        with torch.cuda.stream(self.stream):
            batch = ScheduleBatch.init_new(
                reqs=[s.req for s in states],
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            batch.prepare_for_extend()
            next_token_ids = self._forward(batch)
            batch.output_ids = next_token_ids
            next_ids = next_token_ids.tolist()  # waits for this stream only
        self.num_prefill_steps += 1
        outputs = self._process_tokens(batch, next_ids, first=True)
        batch.filter_batch()
        if not batch.is_empty():
            if self.running is None or self.running.is_empty():
                self.running = batch
            else:
                self.running.merge_batch(batch)
        return outputs

    @torch.no_grad()
    def decode(self) -> List[RequestOutput]:
        batch = self.running
        with torch.cuda.stream(self.stream):
            batch.prepare_for_decode()
            next_token_ids = self._forward(batch)
            batch.output_ids = next_token_ids
            next_ids = next_token_ids.tolist()
        self.num_decode_steps += 1
        outputs = self._process_tokens(batch, next_ids, first=False)
        batch.filter_batch()
        return outputs

    def _forward(self, batch: ScheduleBatch) -> torch.Tensor:
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        return self.model_runner.sample(logits_output, forward_batch)

    def _process_tokens(self, batch: ScheduleBatch, next_ids: List[int], first: bool) -> List[RequestOutput]:
        now = time.time()
        outputs: List[RequestOutput] = []
        for req, token_id in zip(batch.reqs, next_ids):
            state = self.states[req.rid]
            req.output_ids.append(int(token_id))
            m = state.metrics
            if first:
                m.first_token_time = now
            m.last_token_time = now
            m.token_times.append(now)
            req.check_finished()
            if req.finished():
                release_kv_cache(req, self.tree_cache)
                self.reserved_tokens -= state.reserved_tokens
                m.finished_time = now
                del self.states[req.rid]
            elif first:
                self.tree_cache.cache_unfinished_req(req)
            outputs.append(
                RequestOutput(
                    request_id=req.rid,
                    prompt_token_ids=req.origin_input_ids,
                    outputs=[CompletionOutput(token_ids=req.output_ids)],
                    finished=req.finished(),
                    metrics=m,
                )
            )
        return outputs
