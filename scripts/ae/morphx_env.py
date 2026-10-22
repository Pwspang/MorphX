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
