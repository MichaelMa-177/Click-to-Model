"""End-to-end RealSense, SAM2, SAM3D, and 6D tracking orchestration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from click_to_model.camera import RealSenseConfig, capture_realsense
from click_to_model.config import RuntimePaths
from click_to_model.dataset import allocate_data_dir, archive_mask_rgbd_inputs
from click_to_model.processes import (
    cancel_preloaded_worker,
    finish_preloaded_worker,
    normalize_command,
    run_command,
    start_preloaded_worker,
)
from click_to_model.segmentation import Sam2Session
from click_to_model.tracking import TrackingOptions, build_tracking_command


def build_parser(paths: RuntimePaths | None = None) -> argparse.ArgumentParser:
    paths = paths or RuntimePaths.from_environment()
    parser = argparse.ArgumentParser(
        description=(
            "RealSense -> interactive SAM2 mask -> metric SAM3D mesh -> "
            "SPARK-6D tracking."
        )
    )
    data = parser.add_argument_group("data and capture")
    data.add_argument(
        "--data-root",
        default=str(paths.repository_root / "data_online"),
    )
    data.add_argument("--data-dir", help="reuse an existing RGB-D sequence")
    data.add_argument("--no-capture", action="store_true")
    data.add_argument("--n-frames", type=int, default=300)
    data.add_argument(
        "--online",
        action="store_true",
        help="capture one frame, reconstruct, then track the live camera",
    )
    data.add_argument("--mask-file", help="reuse a mask instead of the click UI")
    data.add_argument("--no-gui", action="store_true")

    camera = parser.add_argument_group("RealSense")
    camera.add_argument("--camera-serial")
    camera.add_argument("--camera-width", type=int, default=640)
    camera.add_argument("--camera-height", type=int, default=480)
    camera.add_argument("--camera-fps", type=int, default=60)
    camera.add_argument("--frame-timeout-ms", type=int, default=10_000)
    camera.add_argument("--startup-retries", type=int, default=2)

    models = parser.add_argument_group("models and scheduling")
    models.add_argument("--cuda-device", type=int, default=0)
    models.add_argument("--skip-sam3d", action="store_true")
    models.add_argument(
        "--no-model-preload",
        action="store_true",
        help="disable concurrent SAM2/SAM3D loading",
    )
    models.add_argument("--distilled", action="store_true")

    scale = parser.add_argument_group("metric scale recovery")
    scale.add_argument(
        "--scale-mode",
        choices=("icp", "sam3d", "extent"),
        default="icp",
    )
    scale.add_argument("--icp-voxel-size", type=float, default=0.003)
    scale.add_argument("--icp-samples", type=int, default=20_000)
    scale.add_argument("--icp-iterations", type=int, default=60)
    scale.add_argument("--icp-mask-erode", type=int, default=3)
    scale.add_argument("--min-depth-coverage", type=float, default=0.25)
    scale.add_argument("--allow-border-mask", action="store_true")
    scale.add_argument("--require-icp-confidence", action="store_true")
    scale.add_argument(
        "--max-tracking-faces",
        type=int,
        default=50_000,
        help="simplify the SAM3D mesh before tracking; 0 disables it",
    )

    tracker = parser.add_argument_group("6D tracking")
    tracker.add_argument(
        "--tracker",
        choices=("spark6d", "foundationpose"),
        default="spark6d",
    )
    tracker.add_argument(
        "--skip-tracker",
        "--skip-foundationpose",
        dest="skip_tracker",
        action="store_true",
    )
    tracker.add_argument(
        "--live-tracker",
        "--live-foundationpose",
        dest="live_tracker",
        action="store_true",
    )
    tracker.add_argument(
        "--use-kalman",
        action="store_true",
        help="deprecated compatibility alias for --tracker spark6d",
    )
    tracker.add_argument(
        "--max-tracker-frames",
        "--max-fp-frames",
        dest="max_tracker_frames",
        type=int,
        default=0,
    )
    tracker.add_argument(
        "--spark-refine-interval",
        "--kalman-refine-interval",
        dest="spark_refine_interval",
        type=int,
        default=2,
    )
    tracker.add_argument("--save-tracker-render", action="store_true")
    tracker.add_argument(
        "--save-tracking-video",
        "--save-video",
        dest="save_tracking_video",
        action="store_true",
    )
    tracker.add_argument("--tracking-video-path")
    tracker.add_argument("--tracking-video-fps", type=float)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_camera_values = (
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )
    if args.n_frames <= 0:
        parser.error("--n-frames must be positive")
    if any(value <= 0 for value in positive_camera_values):
        parser.error("camera width, height, and FPS must be positive")
    if args.frame_timeout_ms <= 0 or args.startup_retries < 0:
        parser.error(
            "--frame-timeout-ms must be positive and --startup-retries nonnegative"
        )
    if args.max_tracker_frames < 0 or args.max_tracking_faces < 0:
        parser.error("frame and mesh limits must be nonnegative")
    if args.spark_refine_interval <= 0:
        parser.error("--spark-refine-interval must be positive")
    if args.tracking_video_fps is not None and args.tracking_video_fps <= 0:
        parser.error("--tracking-video-fps must be positive")
    if not 0.0 <= args.min_depth_coverage <= 1.0:
        parser.error("--min-depth-coverage must be between 0 and 1")
    if args.icp_voxel_size <= 0 or args.icp_samples < 100:
        parser.error("--icp-voxel-size must be positive and --icp-samples at least 100")
    if args.icp_iterations <= 0 or args.icp_mask_erode < 0:
        parser.error(
            "--icp-iterations must be positive and --icp-mask-erode nonnegative"
        )


def build_reconstruction_command(
    args: argparse.Namespace,
    data_dir: Path | str,
    paths: RuntimePaths | None = None,
) -> list[str]:
    """Build the isolated SAM3D worker command."""
    paths = paths or RuntimePaths.from_environment()
    command: list[str | os.PathLike[str]] = [
        paths.tracker_python,
        "-m",
        "click_to_model.reconstruction.cli",
        "--data-dir",
        data_dir,
        "--scale-mode",
        args.scale_mode,
        "--icp-voxel-size",
        str(args.icp_voxel_size),
        "--icp-samples",
        str(args.icp_samples),
        "--icp-iterations",
        str(args.icp_iterations),
        "--icp-mask-erode",
        str(args.icp_mask_erode),
        "--min-depth-coverage",
        str(args.min_depth_coverage),
        "--max-faces",
        str(args.max_tracking_faces),
    ]
    if args.distilled:
        command.append("--distilled")
    if args.require_icp_confidence:
        command.append("--require-icp-confidence")
    if args.allow_border_mask:
        command.append("--allow-border-mask")
    return normalize_command(command)


def _resolve_data_dir(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Path:
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            parser.error(f"--data-dir does not exist: {data_dir}")
        return data_dir
    return allocate_data_dir(Path(args.data_root).expanduser().resolve())


def _first_rgbd_paths(data_dir: Path) -> tuple[Path, Path]:
    """Resolve a same-frame pair, including an interrupted online capture.

    Online initialization archives its input before starting reconstruction.
    A retry must read that archive directly so the live recording directories
    remain reserved for frames actually consumed by the tracker.
    """
    for root in (data_dir, data_dir / "masks"):
        rgb_files = sorted((root / "rgb").glob("*.png"))
        if not rgb_files:
            continue
        first_rgb = rgb_files[0]
        first_depth = root / "depth" / first_rgb.name
        if not first_depth.is_file():
            raise FileNotFoundError(
                f"No matching depth frame for {first_rgb}: expected {first_depth}"
            )
        return first_rgb, first_depth
    raise FileNotFoundError(
        f"No RGB frames in {data_dir / 'rgb'} or {data_dir / 'masks' / 'rgb'}"
    )


def _prepare_mask(
    args: argparse.Namespace,
    paths: RuntimePaths,
    data_dir: Path,
    first_rgb: Path,
    first_depth: Path,
    sam2_session: Sam2Session | None,
) -> Path:
    mask_path = data_dir / "masks" / first_rgb.name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mask_file:
        source_mask = Path(args.mask_file).expanduser().resolve()
        if source_mask != mask_path.resolve():
            shutil.copy2(source_mask, mask_path)
        print(f"[MASK] Reused {source_mask} -> {mask_path}")
        return mask_path
    if args.no_gui and mask_path.is_file():
        print(f"[MASK] Reusing existing {mask_path}")
        return mask_path
    if args.no_gui:
        raise RuntimeError(
            "--no-gui requires --mask-file or an existing first-frame mask"
        )

    image_bgr = cv2.imread(str(first_rgb), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(first_depth), cv2.IMREAD_UNCHANGED)
    if image_bgr is None or depth_raw is None:
        raise RuntimeError(
            f"Failed to read first RGB-D pair: {first_rgb}, {first_depth}"
        )
    owns_session = sam2_session is None
    session = sam2_session or Sam2Session.load(
        paths,
        f"cuda:{args.cuda_device}",
    )
    try:
        prediction = session.interactive_mask(
            image_bgr,
            depth_raw,
            args.min_depth_coverage,
        )
    finally:
        if owns_session:
            session.close()
    if not cv2.imwrite(
        str(mask_path),
        prediction.mask.astype(np.uint8) * 255,
    ):
        raise OSError(f"Failed to write mask: {mask_path}")

    archive_root = data_dir / "masks"
    prompt_metadata = {
        "image": str(archive_root / "rgb" / first_rgb.name),
        "depth": str(archive_root / "depth" / first_rgb.name),
        "sam2_score": prediction.score,
        "valid_depth_fraction": prediction.valid_depth_fraction,
        "point_coords": prediction.points.tolist(),
        "point_labels": prediction.labels.tolist(),
    }
    mask_path.with_suffix(".prompts.json").write_text(
        json.dumps(prompt_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"[MASK] SAM2 score={prediction.score:.4f}, saved {mask_path}")
    return mask_path


def _tracking_options(args: argparse.Namespace, data_dir: Path) -> TrackingOptions:
    save_video = bool(args.save_tracking_video or args.tracking_video_path)
    video_path = (
        Path(args.tracking_video_path).expanduser().resolve()
        if args.tracking_video_path
        else data_dir / "debug" / "tracking.mp4"
    )
    video_fps = (
        args.tracking_video_fps
        if args.tracking_video_fps is not None
        else (float(args.camera_fps) if args.live_tracker else 30.0)
    )
    return TrackingOptions(
        backend=args.tracker,
        live=args.live_tracker,
        camera_serial=args.camera_serial,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        no_gui=args.no_gui,
        max_frames=args.max_tracker_frames,
        spark_refine_interval=args.spark_refine_interval,
        save_render=args.save_tracker_render,
        save_video=save_video,
        video_path=video_path,
        video_fps=video_fps,
    )


def run_pipeline(
    args: argparse.Namespace,
    paths: RuntimePaths | None = None,
) -> Path:
    paths = paths or RuntimePaths.from_environment()
    parser = build_parser(paths)
    validate_args(parser, args)
    if args.online:
        args.n_frames = 1
        args.live_tracker = True
    if args.use_kalman:
        args.tracker = "spark6d"
    data_dir = _resolve_data_dir(parser, args)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    reconstruction_command = (
        None if args.skip_sam3d else build_reconstruction_command(args, data_dir, paths)
    )
    needs_sam2_ui = not args.no_gui and not args.mask_file
    sam2_session = None
    sam3d_process = None
    try:
        if needs_sam2_ui and not args.no_model_preload:
            sam2_session = Sam2Session.load(
                paths,
                f"cuda:{args.cuda_device}",
            )
        if reconstruction_command is not None and not args.no_model_preload:
            sam3d_process = start_preloaded_worker(
                reconstruction_command,
                paths.repository_root,
                environment,
            )
        if not args.no_capture:
            capture_realsense(
                data_dir,
                args.n_frames,
                preview=not args.no_gui,
                settings=RealSenseConfig(
                    serial=args.camera_serial,
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                    timeout_ms=args.frame_timeout_ms,
                    startup_retries=args.startup_retries,
                ),
            )
        first_rgb, first_depth = _first_rgbd_paths(data_dir)
        mask_path = _prepare_mask(
            args,
            paths,
            data_dir,
            first_rgb,
            first_depth,
            sam2_session,
        )
        archive_mask_rgbd_inputs(
            data_dir,
            first_rgb,
            first_depth,
            move_from_sequence=bool(args.online and not args.no_capture),
        )
    except BaseException:
        cancel_preloaded_worker(sam3d_process)
        raise
    finally:
        if sam2_session is not None:
            sam2_session.close()

    if reconstruction_command is not None:
        if sam3d_process is not None and sam3d_process.poll() is None:
            try:
                finish_preloaded_worker(sam3d_process)
            finally:
                cancel_preloaded_worker(sam3d_process)
        else:
            if sam3d_process is not None:
                print(
                    "[PRELOAD] SAM3D worker exited before trigger; "
                    "retrying sequentially after releasing SAM2"
                )
            run_command(
                reconstruction_command,
                paths.repository_root,
                environment,
            )

    mesh_path = data_dir / "mesh" / "model.obj"
    if not args.skip_tracker:
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        options = _tracking_options(args, data_dir)
        print(f"[TRACKER] backend={options.backend}, live={options.live}")
        command = build_tracking_command(
            paths,
            data_dir,
            mask_path,
            mesh_path,
            options,
        )
        run_command(command, paths.spark_root, environment)

    print(f"[DONE] Click-to-Model output: {data_dir}")
    return data_dir


def main(argv: Sequence[str] | None = None) -> None:
    paths = RuntimePaths.from_environment()
    parser = build_parser(paths)
    args = parser.parse_args(argv)
    run_pipeline(args, paths)
