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

CUDA graphs, the radix cache and the overlap scheduler are off, so every
kernel goes through cuLaunchKernel(Ex) where the MorphX hook can see it.

Two optional scheduling modes stand in for the authors' vLLM fork in the PD
co-location experiment (Fig 13-14, scripts/test-pd-online.py):

- prefill thread (MORPHX_PREFILL_THREAD=1, or the fork's VLLM_PREFILL_THREAD=1):
  a worker thread runs prefill batches on its own CUDA stream and attention
  backend while the caller keeps decoding on its stream. Finished prefills join
  the decode batch at the next step(); get_prefill_result() blocks for one when
  there is nothing to decode. Under MorphX the two streams let the runtime give
  prefill and decode kernels different SM ranges.
- chunked prefill (chunked_prefill=True): vLLM-style mixed batches. Each step
  decodes every running request and fills the rest of the max_num_batched_tokens
  budget with prompt chunks, in one forward pass.

Not a package on purpose: scripts/sglang has no __init__.py, so it never
shadows the real `sglang` package (a regular package always wins over a
namespace directory on sys.path).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import sys
import threading
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # mycutlass_gemm


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
    prefill_done: int = 0  # prompt tokens already prefilled (chunked prefill)


def _env_flag(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on") for name in names)


def _reserve_for_weight_copy(model_path: str, mem_fraction_static: float, gpu_id: int) -> float:
    """Lower mem_fraction_static by the checkpoint size (fp16/bf16 safetensors ~ dense weights)."""
    try:
        weight_bytes = sum(
            os.path.getsize(os.path.join(model_path, name))
            for name in os.listdir(model_path)
            if name.endswith(".safetensors")
        )
    except OSError:
        return mem_fraction_static
    total = torch.cuda.get_device_properties(gpu_id).total_memory
    reduced = max(0.3, mem_fraction_static - weight_bytes / total)
    logger.info("mycutlass: mem_fraction_static %.3f -> %.3f for the transposed weights", mem_fraction_static, reduced)
    return reduced


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
        gemm: Optional[str] = None,
        prefill_thread: Optional[bool] = None,
        chunked_prefill: bool = False,
        **server_kwargs,
    ):
        # Dense linear layers: "torch" (cuBLASLt) or "mycutlass" (pass-buildable CUTLASS, fp16 only)
        self.gemm = (gemm or os.environ.get("MORPHX_LLM_GEMM", "torch")).strip().lower()
        if self.gemm not in ("torch", "mycutlass"):
            raise ValueError(f"MORPHX_LLM_GEMM={self.gemm!r}; expected torch or mycutlass")
        if self.gemm == "mycutlass":
            import mycutlass_gemm

            mycutlass_gemm.install()
            if dtype == "auto":
                dtype = "float16"
            if dtype != "float16":
                raise ValueError("MORPHX_LLM_GEMM=mycutlass needs dtype float16")
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = int(max_num_batched_tokens or max_model_len)
        if prefill_thread is None:
            prefill_thread = _env_flag("MORPHX_PREFILL_THREAD", "VLLM_PREFILL_THREAD")
        self.prefill_thread = bool(prefill_thread)
        self.chunked_prefill = bool(chunked_prefill)
        if self.prefill_thread and self.chunked_prefill:
            raise ValueError("the prefill thread and chunked prefill are separate modes")
        torch.cuda.set_device(gpu_id)
        self.stream = stream if stream is not None else torch.cuda.current_stream()
        if self.gemm == "mycutlass":
            # mycutlass_gemm.prepare() adds a transposed copy of the dense weights after
            # ModelRunner has sized the KV pool, so leave room for it.
            mem_fraction_static = _reserve_for_weight_copy(model_path, mem_fraction_static, gpu_id)

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
        # SGLang turns on programmatic dependent launch (TRTLLM_ENABLE_PDL) process-wide
        # unless it is "0". The MorphX hook passes PDL launches through unscheduled and
        # co-located workloads share this process, so keep it off unless asked for
        # (MORPHX_SGLANG_PDL=1). FlashInfer's norm kernels still pick PDL from the device.
        os.environ["TRTLLM_ENABLE_PDL"] = os.environ.get("MORPHX_SGLANG_PDL", "0")
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
        if self.gemm == "mycutlass":
            with torch.cuda.stream(self.stream):
                self.num_prepared_weights = mycutlass_gemm.prepare(self.model_runner.model)
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

        # KV pool, req slots, reservations and the waiting queue are shared with the
        # prefill thread. _kv_section() serializes their bookkeeping and orders the GPU
        # side of the allocator ops (slices/cats of the free list) across streams.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._kv_event: Optional[torch.cuda.Event] = None
        self._num_running = 0  # len(self.running.reqs), readable from the prefill thread
        self._chunk_state: Optional[_ReqState] = None
        self._prefill_worker: Optional[threading.Thread] = None
        self._inflight = 0
        if self.prefill_thread:
            self._start_prefill_thread()

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
        # vLLM's min_tokens == max_tokens means "always generate exactly max_tokens":
        # ignore EOS instead, which avoids the min-new-tokens penalizer's extra kernels.
        exact = int(min_tokens) >= int(max_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=int(max_tokens),
            min_new_tokens=0 if exact else int(min_tokens),
            temperature=temperature,
            ignore_eos=exact,
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
        req.tokenizer = self.tokenizer  # read by the min-new-tokens penalizer and stop checks
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        state = _ReqState(
            req=req,
            max_tokens=int(max_tokens),
            metrics=RequestMetrics(arrival_time=arrival_time if arrival_time is not None else time.time()),
            reserved_tokens=need,
        )
        with self._cond:
            self.states[request_id] = state
            self.waiting.append(state)
            self._cond.notify_all()

    def has_unfinished_requests(self) -> bool:
        with self._lock:
            pending = bool(self.waiting) or self._chunk_state is not None
            if self.prefill_thread:
                pending = pending or self._inflight > 0 or bool(self._done)
        return pending or self._has_running()

    def step(self) -> List[RequestOutput]:
        """One scheduling step.

        Default: a prefill batch if any waiting request fits, else one decode step.
        Prefill thread: merge finished prefills, then one decode step (prefills run
        concurrently in the worker). Chunked prefill: one mixed decode + chunk step.
        """
        if self.chunked_prefill:
            return self._step_chunked()
        if self.prefill_thread:
            outputs = self._collect_prefilled()
            if self._has_running():
                outputs += self.decode()
            return outputs
        with self._kv_section():
            admitted = self._admit()
        if admitted:
            return self.prefill(admitted)
        if self._has_running():
            return self.decode()
        return []

    def get_prefill_result(self, isBlock: bool = True) -> List[RequestOutput]:  # noqa: N803 (vLLM-fork name)
        """Outputs of finished prefills (prefill thread mode); waits for one if isBlock."""
        if not self.prefill_thread:
            return []
        with self._cond:
            while isBlock and not self._done and (self.waiting or self._inflight) and self._prefill_error is None:
                self._cond.wait()
        return self._collect_prefilled()

    def shutdown(self) -> None:
        """Stop the prefill thread (vLLM fork: wait_prefill_shutdown)."""
        worker = self._prefill_worker
        if worker is None:
            return
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        worker.join()
        self._prefill_worker = None

    def warmup(self, input_len: int, output_len: int = 4) -> None:
        """Run one request end to end so JIT compiles and first-launch work (e.g. the
        MorphX runtime's per-kernel profiling) happen before the timed workload."""
        vocab = self.model_config.vocab_size
        ids = [(7 + j) % vocab for j in range(int(input_len))]
        self.add_request("__warmup__", ids, max_tokens=int(output_len), min_tokens=int(output_len))
        while self.has_unfinished_requests():
            if not self.step():
                self.get_prefill_result(isBlock=True)
        self.num_prefill_steps = 0
        self.num_decode_steps = 0

    # ------------------------------------------------------------ internals

    def _has_running(self) -> bool:
        return self.running is not None and not self.running.is_empty()

    def _set_running(self, batch: Optional[ScheduleBatch]) -> None:
        self.running = batch
        with self._lock:
            self._num_running = 0 if batch is None or batch.is_empty() else len(batch.reqs)

    @contextlib.contextmanager
    def _kv_section(self):
        """Lock the shared KV/req-slot bookkeeping. With the prefill thread, also make
        the current stream wait for the previous section's GPU work (the allocator's
        free list is a GPU tensor sliced and concatenated by both threads) and
        record where this section's work ends."""
        with self._lock:
            if not self.prefill_thread:
                yield
                return
            stream = torch.cuda.current_stream()
            if self._kv_event is not None:
                stream.wait_event(self._kv_event)
            try:
                yield
            finally:
                event = torch.cuda.Event()
                event.record(stream)
                self._kv_event = event

    def _admit(self) -> List[_ReqState]:
        """Pop waiting requests that fit (call with the lock held)."""
        admitted: List[_ReqState] = []
        num_running = self._num_running + self._inflight
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

    def _new_batch(self, reqs: List[Req], chunked_req: Optional[Req] = None) -> ScheduleBatch:
        return ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            chunked_req=chunked_req,
        )

    @torch.no_grad()
    def _run_prefill(self, states: List[_ReqState], stream: torch.cuda.Stream, attn_backend=None):
        """Prefill `states` on `stream`; returns the filtered batch (not merged) and outputs."""
        now = time.time()
        for state in states:
            state.metrics.first_scheduled_time = now
        # Everything touching the GPU stays on `stream`, including the
        # post-processing (KV index copies in cache_unfinished_req, filter).
        with torch.cuda.stream(stream):
            batch = self._new_batch([s.req for s in states])
            with self._kv_section():
                batch.prepare_for_extend()
            next_token_ids = self._forward(batch, attn_backend)
            batch.output_ids = next_token_ids
            next_ids = next_token_ids.tolist()  # waits for this stream only
            self.num_prefill_steps += 1
            outputs = self._process_tokens(batch, next_ids, ["first"] * len(next_ids))
            batch.filter_batch()
        return batch, outputs

    @torch.no_grad()
    def prefill(self, states: List[_ReqState]) -> List[RequestOutput]:
        batch, outputs = self._run_prefill(states, self.stream)
        with torch.cuda.stream(self.stream):
            self._merge_into_running(batch)
        return outputs

    def _merge_into_running(self, batch: ScheduleBatch) -> None:
        if batch.is_empty():
            return
        if not self._has_running():
            self._set_running(batch)
        else:
            self.running.merge_batch(batch)
            self._set_running(self.running)

    @torch.no_grad()
    def decode(self) -> List[RequestOutput]:
        batch = self.running
        with torch.cuda.stream(self.stream):
            with self._kv_section():
                batch.prepare_for_decode()
            next_token_ids = self._forward(batch)
            batch.output_ids = next_token_ids
            next_ids = next_token_ids.tolist()
            self.num_decode_steps += 1
            outputs = self._process_tokens(batch, next_ids, ["decode"] * len(next_ids))
            batch.filter_batch()
            self._set_running(batch)
        return outputs

    def _forward(self, batch: ScheduleBatch, attn_backend=None) -> torch.Tensor:
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        if attn_backend is None:
            logits_output = self.model_runner.forward(forward_batch).logits_output
        else:
            # The prefill thread has its own attention backend (FlashInfer wrappers and
            # workspace) so its planning never races the decode thread's.
            forward_batch.attn_backend = attn_backend
            attn_backend.init_forward_metadata(forward_batch)
            logits_output = self.model_runner.forward(forward_batch, skip_attn_backend_init=True).logits_output
        return self.model_runner.sample(logits_output, forward_batch)

    def _process_tokens(self, batch: ScheduleBatch, next_ids: List[int], kinds: List[str]) -> List[RequestOutput]:
        """kinds[i]: "first" (prompt finished, first token), "decode", or "chunk" (a
        non-final prompt chunk: its sampled token is discarded)."""
        now = time.time()
        outputs: List[RequestOutput] = []
        with self._cond, self._kv_section():
            for req, token_id, kind in zip(batch.reqs, next_ids, kinds):
                if kind == "chunk":
                    self.tree_cache.cache_unfinished_req(req, chunked=True)
                    continue
                state = self.states[req.rid]
                req.output_ids.append(int(token_id))
                m = state.metrics
                if kind == "first":
                    m.first_token_time = now
                m.last_token_time = now
                m.token_times.append(now)
                req.check_finished()
                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    self.reserved_tokens -= state.reserved_tokens
                    m.finished_time = now
                    del self.states[req.rid]
                    self._cond.notify_all()  # freed KV may admit a waiting prefill
                elif kind == "first":
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

    # ------------------------------------------------------ prefill thread

    def _start_prefill_thread(self) -> None:
        self.prefill_stream = torch.cuda.Stream()
        # A second attention backend with its own workspace: FlashInfer's default
        # workspace buffer is shared by every wrapper of one backend.
        self.prefill_attn_backend = self.model_runner._get_attention_backend(init_new_workspace=True)
        self._done: Deque = deque()
        self._stop = False
        self._prefill_error: Optional[BaseException] = None
        self._prefill_worker = threading.Thread(target=self._prefill_loop, name="morphx-prefill", daemon=True)
        self._prefill_worker.start()

    def _prefill_loop(self) -> None:
        torch.cuda.set_device(self.model_runner.gpu_id)
        try:
            while True:
                with self._cond:
                    admitted: List[_ReqState] = []
                    while not self._stop:
                        admitted = self._admit()
                        if admitted:
                            break
                        self._cond.wait()
                    if self._stop:
                        return
                    self._inflight += len(admitted)
                batch, outputs = self._run_prefill(admitted, self.prefill_stream, self.prefill_attn_backend)
                with self._cond:
                    self._done.append((batch, outputs))
                    self._inflight -= len(admitted)
                    self._cond.notify_all()
        except BaseException as exc:  # surfaced by the caller's next step()
            logger.exception("prefill thread failed")
            with self._cond:
                self._prefill_error = exc
                self._inflight = 0
                self._cond.notify_all()

    def _collect_prefilled(self) -> List[RequestOutput]:
        """Merge finished prefill batches into the decode batch (caller's thread)."""
        if self._prefill_error is not None:
            raise RuntimeError("prefill thread failed") from self._prefill_error
        with self._lock:
            done = list(self._done)
            self._done.clear()
        outputs: List[RequestOutput] = []
        if not done:
            return outputs
        with torch.cuda.stream(self.stream):
            for batch, batch_outputs in done:
                self._merge_into_running(batch)
                outputs += batch_outputs
            # The merged-away prefill tensors were allocated on the prefill stream;
            # don't let the caching allocator hand them back out before the merge
            # kernels on this stream have read them.
            self.stream.synchronize()
        return outputs

    # ------------------------------------------------------- chunked prefill

    @torch.no_grad()
    def _step_chunked(self) -> List[RequestOutput]:
        """Decode all running requests and fill the remaining token budget with
        prompt chunks (the continuing chunked request first), in one forward."""
        running = self.running if self._has_running() else None
        budget = self.max_num_batched_tokens - (len(running.reqs) if running is not None else 0)
        plan = []  # (state, prompt tokens prefilled after this step)
        now = time.time()
        with self._kv_section():
            if self._chunk_state is not None and budget > 0:
                state = self._chunk_state
                n = min(len(state.req.origin_input_ids) - state.prefill_done, budget)
                plan.append((state, state.prefill_done + n))
                budget -= n
            num_running = len(running.reqs) if running is not None else 0
            free_req_slots = self.model_runner.req_to_token_pool.available_size()
            new = 0
            while budget > 0 and self.waiting:
                state = self.waiting[0]
                if num_running + len(plan) >= self.max_num_seqs or new >= free_req_slots:
                    break
                if self.reserved_tokens + state.reserved_tokens > self.kv_capacity:
                    break
                self.waiting.popleft()
                self.reserved_tokens += state.reserved_tokens
                state.metrics.first_scheduled_time = now
                n = min(len(state.req.origin_input_ids), budget)
                plan.append((state, n))
                budget -= n
                new += 1
        if not plan:
            return self.decode() if running is not None else []

        partial = None
        for state, end in plan:
            req = state.req
            req.fill_ids = req.origin_input_ids[:end]
            req.set_extend_input_len(end - len(req.prefix_indices))
            state.prefill_done = end
            if end < len(req.origin_input_ids):
                partial = state
        self._chunk_state = partial

        with torch.cuda.stream(self.stream):
            batch = self._new_batch([s.req for s, _ in plan], chunked_req=partial.req if partial else None)
            kinds = ["chunk" if s is partial else "first" for s, _ in plan]
            with self._kv_section():
                batch.prepare_for_extend()
                if running is not None:
                    running.prepare_for_decode()
                    batch.mix_with_running(running)
                    kinds += ["decode"] * len(running.reqs)
            next_token_ids = self._forward(batch)
            batch.output_ids = next_token_ids
            next_ids = next_token_ids.tolist()
            self.num_prefill_steps += 1
            outputs = self._process_tokens(batch, next_ids, kinds)
            batch.filter_batch(chunked_req_to_exclude=partial.req if partial else None)
            self._set_running(batch if not batch.is_empty() else None)
        return outputs
