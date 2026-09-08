"""Runtime path configuration resolved from environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    """Resolve environment paths relative to the repository, not the caller."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class RuntimePaths:
    """External repositories, checkpoints, and interpreter paths."""

    repository_root: Path
    spark_root: Path
    tracker_python: str
    sam2_root: Path
    sam2_checkpoint: Path
    sam3d_root: Path
    sam3d_checkpoint_dir: Path

    @classmethod
    def from_environment(cls) -> RuntimePaths:
        repository_root = REPOSITORY_ROOT
        workspace_root = repository_root.parent
        spark_root = _resolve_path(
            os.environ.get(
                "SPARK6D_REPO",
                os.environ.get("FOUNDATIONPOSE_REPO", workspace_root / "SPARK-6D"),
            ),
            repository_root,
        )
        tracker_python = os.environ.get(
            "SPARK6D_PY",
            os.environ.get("FP_PY", sys.executable),
        )
        sam2_root = _resolve_path(
            os.environ.get("SAM2_REPO", spark_root / "third_party" / "sam2"),
            repository_root,
        )
        sam2_checkpoint = _resolve_path(
            os.environ.get(
                "SAM2_CHECKPOINT",
                sam2_root / "checkpoints" / "sam2.1_hiera_small.pt",
            ),
            repository_root,
        )
        sam3d_root = _resolve_path(
            os.environ.get("SAM3D_REPO", repository_root / "sam-3d-objects"),
            repository_root,
        )
        sam3d_checkpoint_dir = _resolve_path(
            os.environ.get(
                "SAM3D_CHECKPOINT_DIR",
                sam3d_root / "checkpoints" / "hf",
            ),
            repository_root,
        )
        return cls(
            repository_root=repository_root,
            spark_root=spark_root,
            tracker_python=tracker_python,
            sam2_root=sam2_root,
            sam2_checkpoint=sam2_checkpoint,
            sam3d_root=sam3d_root,
            sam3d_checkpoint_dir=sam3d_checkpoint_dir,
        )
