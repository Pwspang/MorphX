#!/usr/bin/env python3
"""Machine-specific AE settings read from MORPHX_* environment variables.

Every default matches the original A100 evaluation machine (108 SMs, sm_80,
toolchain under /home/rtx), so nothing changes there when the variables are
unset. scripts/ae/env-h100.sh sets the H100/H200 values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence


REFERENCE_NUM_SMS = 108
LLVM_TRIPLE_LIB = Path("lib") / "x86_64-unknown-linux-gnu"

LEGACY_PYTHONS = (
    "/home/rtx/miniconda3/envs/smsched/bin/python",
    "/mnt/data/rtx/miniconda3/envs/smsched/bin/python",
)
# Search order of the original scripts; MORPHX_* values, when set, go first.
LEGACY_LLVM_ROOTS = (
    Path("/mnt/data/rtx/llvm-project/install"),
    Path("/home/rtx/llvm-project/install"),
)
LEGACY_NANOFLOW_LIBS = (
    Path("/mnt/data/rtx/miniconda3/envs/nanoflow/lib"),
    Path("/home/rtx/miniconda3/envs/nanoflow/lib"),
)

DEFAULTS = {
    "MORPHX_LLVM_ROOT": "/home/rtx/llvm-project/install",
    "MORPHX_CUDA_ROOT": "/usr/local/cuda-12.4",
    "MORPHX_CCCL_ROOT": "/home/rtx/gpu/cccl",
    "MORPHX_CUTLASS_ROOT": "/home/rtx/gpu/cutlass",
    "MORPHX_CUDA_ARCH": "80",
    "MORPHX_NUM_SMS": str(REFERENCE_NUM_SMS),
    "MORPHX_HF_CACHE": "/huggingface-cache/hub",
    "MORPHX_NANOFLOW_LIB": "/home/rtx/miniconda3/envs/nanoflow/lib",
    "MORPHX_SYNC_BACKEND": "gdr",
    "MORPHX_TGS_SOCKET_DIR": "/home/rtx/gpu/sm-sched/TGS/gsharing",
}


def env_str(name: str) -> str:
    return os.environ.get(name) or DEFAULTS[name]


def env_path(name: str) -> Path:
    return Path(env_str(name)).expanduser()


def env_int(name: str) -> int:
    raw = env_str(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def unique_paths(paths: Sequence[Optional[Path]]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def python_candidates() -> list[str]:
    candidates = [os.environ.get("MORPHX_PYTHON"), os.environ.get("PYTHON"), *LEGACY_PYTHONS]
    return [candidate for candidate in candidates if candidate]


def default_python() -> str:
    for candidate in python_candidates():
        if Path(candidate).exists():
            return candidate
    return sys.executable


def explicit_path(name: str) -> list[Path]:
    value = os.environ.get(name)
    return [Path(value).expanduser()] if value else []


def llvm_cmake_dirs() -> list[Path]:
    """LLVM_DIR, MORPHX_LLVM_ROOT, then the original machine's install locations."""
    dirs: list[Path] = explicit_path("LLVM_DIR")
    roots = [*explicit_path("MORPHX_LLVM_ROOT"), Path("/home/rtx/llvm-project/install"), Path("/mnt/data/rtx/llvm-project/install")]
    dirs.extend(root / "lib" / "cmake" / "llvm" for root in roots)
    return unique_paths(dirs)


def llvm_runtime_lib_dirs() -> list[Path]:
    """libc++ runtime directories of the pass toolchain (install and sibling build trees)."""
    dirs: list[Path] = []
    for root in [*explicit_path("MORPHX_LLVM_ROOT"), *LEGACY_LLVM_ROOTS]:
        dirs.append(root / LLVM_TRIPLE_LIB)
        dirs.append(root.parent / "build" / LLVM_TRIPLE_LIB)
    return unique_paths(dirs)


def nanoflow_lib_dirs() -> list[Path]:
    return unique_paths([*explicit_path("MORPHX_NANOFLOW_LIB"), *LEGACY_NANOFLOW_LIBS])


def hf_cache() -> Path:
    return env_path("MORPHX_HF_CACHE")


# HuggingFace snapshots the workload scripts load (same table as scripts/colocate/run.py).
MODEL_SNAPSHOTS = {
    "llama-2-7b": "models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "llama-3-8b": "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen3-4b": "models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": "models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2",
    "gpt2-l": "models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519",
}


def model_alias(model: str) -> str:
    """Apply MORPHX_MODEL_ALIASES ("from=to,..."), as scripts/sglang/backend.py does."""
    for item in os.environ.get("MORPHX_MODEL_ALIASES", "").split(","):
        if "=" in item:
            src, dst = (part.strip() for part in item.split("=", 1))
            if src == model and dst:
                return dst
    return model


def model_path(model: str) -> Optional[Path]:
    """Snapshot directory of a model alias after MORPHX_MODEL_ALIASES, or None if unknown."""
    snapshot = MODEL_SNAPSHOTS.get(model_alias(model))
    return hf_cache() / snapshot if snapshot else None


def num_sms() -> int:
    return env_int("MORPHX_NUM_SMS")


def cuda_arch() -> int:
    return env_int("MORPHX_CUDA_ARCH")


def scale_sm_count(value: int) -> int:
    """Scale an SM count tuned on the 108-SM A100 to MORPHX_NUM_SMS."""
    total = num_sms()
    if total == REFERENCE_NUM_SMS:
        return value
    return max(1, min(total, round(value * total / REFERENCE_NUM_SMS)))


def scale_split_hint(split_hint: str) -> str:
    """Scale a numeric split hint; non-numeric or empty hints pass through."""
    try:
        value = int(split_hint)
    except (TypeError, ValueError):
        return split_hint
    return str(scale_sm_count(value))


def sm_sweep(reference: Sequence[int]) -> tuple[int, ...]:
    """Re-anchor an SM sweep measured on the 108-SM A100 at MORPHX_NUM_SMS.

    The runtime profiles a kernel at nrSM, nrSM - dec, nrSM - 2*dec, ... with
    dec = nrSM / baseWaves (generateProfileWaves in runtime/src/cuda.cpp), so
    baseWaves is recovered from the reference step and re-applied. Small
    trailing points that are off that grid (the last 4 or 8) depend on the
    grid size, not the SM count, and are kept as-is.
    """
    total = num_sms()
    if total == REFERENCE_NUM_SMS or len(reference) < 2:
        return tuple(reference)
    step = reference[0] - reference[1]
    base_waves = REFERENCE_NUM_SMS // step
    tail = tuple(value for value in reference if (reference[0] - value) % step != 0)
    grid = [value for value in reference if (reference[0] - value) % step == 0]
    new_step = total // base_waves
    points = [total - i * new_step for i in range(len(grid))]
    return (*(value for value in points if value > 0), *tail)


def morphx_env_vars() -> dict[str, str]:
    """All MORPHX_* variables currently set, for passing to child processes."""
    return {key: value for key, value in os.environ.items() if key.startswith("MORPHX_")}


def sync_backend_errors() -> list[str]:
    """Problems that would make the gdr sync backend fail on this node.

    The runtime is built for one backend (MORPHX_SYNC_BACKEND); with gdr it needs
    the gdrdrv device and the GDRCopy user library at run time. Nothing falls
    back to memcpy, so report it up front instead of failing in every child.
    """
    backend = env_str("MORPHX_SYNC_BACKEND")
    if backend != "gdr":
        return []
    errors: list[str] = []
    dev = Path("/dev/gdrdrv")
    if not dev.exists():
        errors.append("MORPHX_SYNC_BACKEND=gdr but /dev/gdrdrv is missing (gdrdrv kernel module not loaded); "
                      "use a node with GDRCopy or set MORPHX_SYNC_BACKEND=memcpy and rebuild the runtime")
    elif not os.access(dev, os.R_OK | os.W_OK):
        errors.append("MORPHX_SYNC_BACKEND=gdr but /dev/gdrdrv is not readable and writable by this user")
    import ctypes.util
    lib_dirs = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    lib_dirs += ["/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/usr/local/lib64"]
    found = any(Path(d, name).exists() for d in lib_dirs for name in ("libgdrapi.so.2", "libgdrapi.so"))
    if not found and ctypes.util.find_library("gdrapi") is None:
        errors.append("MORPHX_SYNC_BACKEND=gdr but libgdrapi (GDRCopy user library) was not found")
    return errors
