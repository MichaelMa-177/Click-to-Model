#!/usr/bin/env python3
"""Compatibility entry point for the packaged Click-to-Model pipeline."""

from click_to_model.pipeline import build_reconstruction_command, main
from click_to_model.processes import (
    cancel_preloaded_worker,
    finish_preloaded_worker,
    start_preloaded_worker,
)

# Compatibility aliases for existing integrations and older tests.
build_sam3d_command = build_reconstruction_command
finish_sam3d_preload = finish_preloaded_worker
cancel_sam3d_preload = cancel_preloaded_worker


def start_sam3d_preload(command, env):
    """Start the worker from the repository root (legacy API)."""
    from click_to_model.config import REPOSITORY_ROOT

    return start_preloaded_worker(command, REPOSITORY_ROOT, env)


if __name__ == "__main__":
    main()
