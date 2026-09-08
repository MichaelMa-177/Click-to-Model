"""FoundationPose-compatible sequence directory operations."""

from __future__ import annotations

import shutil
from pathlib import Path

SEQUENCE_DIRECTORIES = ("rgb", "depth", "masks", "mesh", "debug")


def allocate_data_dir(root: Path) -> Path:
    """Create the next numeric sequence directory under *root*."""
    root.mkdir(parents=True, exist_ok=True)
    identifiers = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    data_dir = root / str(max(identifiers, default=0) + 1)
    for name in SEQUENCE_DIRECTORIES:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    return data_dir


def archive_mask_rgbd_inputs(
    data_dir: Path,
    rgb_path: Path,
    depth_path: Path,
    move_from_sequence: bool,
) -> tuple[Path, Path]:
    """Keep the exact RGB-D pair used by SAM2 beside its generated mask."""
    rgb_target = data_dir / "masks" / "rgb" / rgb_path.name
    depth_target = data_dir / "masks" / "depth" / rgb_path.name
    rgb_target.parent.mkdir(parents=True, exist_ok=True)
    depth_target.parent.mkdir(parents=True, exist_ok=True)
    if rgb_target.exists() and depth_target.exists():
        print(
            f"[MASK] Reusing archived mask RGB-D inputs in {rgb_target.parent.parent}"
        )
        return rgb_target, depth_target
    if rgb_target.exists() or depth_target.exists():
        raise FileExistsError(
            f"Mask RGB-D archive already exists: {rgb_target}, {depth_target}"
        )

    if move_from_sequence:
        moved: list[tuple[Path, Path]] = []
        try:
            shutil.move(str(rgb_path), str(rgb_target))
            moved.append((rgb_target, rgb_path))
            shutil.move(str(depth_path), str(depth_target))
            moved.append((depth_target, depth_path))
        except BaseException:
            for archived, original in reversed(moved):
                if archived.exists() and not original.exists():
                    shutil.move(str(archived), str(original))
            raise
        action = "Moved"
    else:
        shutil.copy2(rgb_path, rgb_target)
        shutil.copy2(depth_path, depth_target)
        action = "Copied"
    print(
        f"[MASK] {action} mask RGB-D inputs to "
        f"{rgb_target.parent} and {depth_target.parent}"
    )
    return rgb_target, depth_target
