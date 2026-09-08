"""Subprocess helpers for isolated GPU stages."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

Command = Sequence[str | os.PathLike[str]]


def normalize_command(command: Command) -> list[str]:
    return [os.fspath(value) for value in command]


def run_command(
    command: Command,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    normalized = normalize_command(command)
    print("[RUN] " + " ".join(normalized))
    subprocess.run(normalized, cwd=cwd, env=env, check=True)


def start_preloaded_worker(
    command: Command,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.Popen:
    preload_command = [*normalize_command(command), "--wait-for-trigger"]
    print("[PRELOAD] Starting SAM3D worker", flush=True)
    print("[RUN] " + " ".join(preload_command), flush=True)
    return subprocess.Popen(
        preload_command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
    )


def finish_preloaded_worker(process: subprocess.Popen) -> None:
    """Trigger a preloaded one-shot worker and propagate its exit status."""
    if process.poll() is not None:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    if process.stdin is None:
        raise RuntimeError("SAM3D preload worker has no trigger pipe")
    print("[PRELOAD] Mask accepted; triggering preloaded SAM3D", flush=True)
    process.stdin.write(b"run\n")
    process.stdin.flush()
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def cancel_preloaded_worker(process: subprocess.Popen | None) -> None:
    """Stop a background loader when capture or annotation is cancelled."""
    if process is None or process.poll() is not None:
        return
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
