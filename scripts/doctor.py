#!/usr/bin/env python3
"""Validate a Click-to-Model checkout without starting GPU inference."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from click_to_model.config import RuntimePaths  # noqa: E402


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status when a required item is missing",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="also require pyrealsense2 (no device is opened)",
    )
    args = parser.parse_args()
    paths = RuntimePaths.from_environment()

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, condition: bool, value: object) -> None:
        checks.append((label, bool(condition), str(value)))

    check("Python >= 3.10", sys.version_info >= (3, 10), sys.executable)
    for module in (
        "torch",
        "cv2",
        "numpy",
        "open3d",
        "scipy",
        "trimesh",
        "hydra",
        "omegaconf",
        "pytorch3d",
        "nvdiffrast",
    ):
        check(f"Python module: {module}", _module_exists(module), module)
    if args.camera:
        check(
            "Python module: pyrealsense2",
            _module_exists("pyrealsense2"),
            "pyrealsense2",
        )

    check(
        "FoundationPose submodule",
        (paths.repository_root / "FoundationPose" / ".git").exists(),
        paths.repository_root / "FoundationPose",
    )
    check(
        "SAM3D submodule",
        (paths.sam3d_root / ".git").exists(),
        paths.sam3d_root,
    )
    check(
        "SAM3D entry point",
        (paths.sam3d_root / "notebook" / "inference.py").is_file(),
        paths.sam3d_root / "notebook" / "inference.py",
    )
    check(
        "SAM3D checkpoint manifest",
        (paths.sam3d_checkpoint_dir / "pipeline.yaml").is_file(),
        paths.sam3d_checkpoint_dir / "pipeline.yaml",
    )
    check(
        "SPARK-6D offline tracker",
        (paths.spark_root / "run_demo.py").is_file(),
        paths.spark_root / "run_demo.py",
    )
    check(
        "SPARK-6D live tracker",
        (paths.spark_root / "tools" / "run_realsense_foundationpose.py").is_file(),
        paths.spark_root / "tools" / "run_realsense_foundationpose.py",
    )
    check(
        "SAM2 source",
        (paths.sam2_root / "sam2" / "build_sam.py").is_file(),
        paths.sam2_root,
    )
    check(
        "SAM2 checkpoint",
        paths.sam2_checkpoint.is_file(),
        paths.sam2_checkpoint,
    )

    for label, passed, value in checks:
        print(f"[{'OK' if passed else 'MISSING'}] {label}: {value}")
    missing = [label for label, passed, _ in checks if not passed]
    if missing:
        print(f"\n{len(missing)} check(s) missing. See docs/DEPLOYMENT.md.")
        return 1 if args.strict else 0
    print("\nEnvironment layout is ready for an explicit hardware smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
