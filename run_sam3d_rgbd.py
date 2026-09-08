#!/usr/bin/env python3
"""Compatibility entry point for metric SAM3D reconstruction."""

from click_to_model.reconstruction.cli import main
from click_to_model.reconstruction.geometry import (
    first_file,
    first_rgbd_file,
    load_depth_scale,
    load_intrinsics,
    make_pointmap,
)

__all__ = [
    "first_file",
    "first_rgbd_file",
    "load_depth_scale",
    "load_intrinsics",
    "main",
    "make_pointmap",
]


if __name__ == "__main__":
    main()
