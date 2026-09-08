"""Command-line interface for the SAM3D reconstruction stage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from click_to_model.config import RuntimePaths
from click_to_model.reconstruction.pipeline import (
    ReconstructionOptions,
    run_reconstruction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a metric, object-local mesh from one RGB-D frame and mask."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--depth-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int)
    parser.add_argument("--stage2-steps", type=int)
    parser.add_argument("--distilled", action="store_true")
    parser.add_argument("--no-low-vram", action="store_true")
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="preload SAM3D and wait for one line on stdin before inference",
    )
    parser.add_argument("--max-faces", type=int, default=200_000)
    parser.add_argument(
        "--scale-mode",
        choices=("icp", "sam3d", "extent"),
        default="icp",
        help="metric scale source; icp uses the masked RGB-D observation",
    )
    parser.add_argument("--icp-voxel-size", type=float, default=0.003)
    parser.add_argument("--icp-samples", type=int, default=20_000)
    parser.add_argument("--icp-iterations", type=int, default=60)
    parser.add_argument("--icp-mask-erode", type=int, default=3)
    parser.add_argument("--min-depth-coverage", type=float, default=0.25)
    parser.add_argument(
        "--allow-border-mask",
        action="store_true",
        help="allow scale export when the object mask touches an image border",
    )
    parser.add_argument(
        "--require-icp-confidence",
        action="store_true",
        help="return failure when the RGB-D registration is rejected",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_depth_coverage <= 1.0:
        parser.error("--min-depth-coverage must be between 0 and 1")
    if args.icp_voxel_size <= 0 or args.icp_samples < 100:
        parser.error("--icp-voxel-size must be positive and --icp-samples at least 100")
    if args.icp_iterations <= 0 or args.icp_mask_erode < 0:
        parser.error(
            "--icp-iterations must be positive and --icp-mask-erode nonnegative"
        )
    if args.max_faces < 0:
        parser.error("--max-faces must be nonnegative")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    options = ReconstructionOptions(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        config_path=(Path(args.config).expanduser().resolve() if args.config else None),
        depth_scale=args.depth_scale,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        distilled=args.distilled,
        low_vram=not args.no_low_vram,
        wait_for_trigger=args.wait_for_trigger,
        max_faces=args.max_faces,
        scale_mode=args.scale_mode,
        icp_voxel_size=args.icp_voxel_size,
        icp_samples=args.icp_samples,
        icp_iterations=args.icp_iterations,
        icp_mask_erode=args.icp_mask_erode,
        min_depth_coverage=args.min_depth_coverage,
        allow_border_mask=args.allow_border_mask,
        require_icp_confidence=args.require_icp_confidence,
    )
    run_reconstruction(options, RuntimePaths.from_environment())


if __name__ == "__main__":
    main()
