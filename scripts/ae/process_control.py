"""Process helpers shared by the AE experiment runners."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import IO, Mapping, Optional, Sequence


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate a command and every descendant in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        pass

    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _close_process_pipes(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    stdin_text: Optional[str] = None,
    stdout: Optional[IO] = None,
    stderr: Optional[IO] = None,
    text: bool = True,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run a command and clean up its process tree on timeout or interruption."""
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=None if env is None else dict(env),
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )
    try:
        stdout_data, stderr_data = process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        _close_process_pipes(process)
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=exc.output,
            stderr=exc.stderr,
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        _close_process_pipes(process)
        raise

    return subprocess.CompletedProcess(command, process.returncode, stdout_data, stderr_data)
