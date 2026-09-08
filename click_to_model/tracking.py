"""SPARK-6D and FoundationPose command construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from click_to_model.config import RuntimePaths


@dataclass(frozen=True)
class TrackingOptions:
    """Options shared by live and offline tracking backends."""

    backend: str
    live: bool
    camera_serial: str | None
    camera_width: int
    camera_height: int
    camera_fps: int
    no_gui: bool
    max_frames: int
    spark_refine_interval: int
    save_render: bool
    save_video: bool
    video_path: Path
    video_fps: float


def build_tracking_command(
    paths: RuntimePaths,
    data_dir: Path,
    mask_path: Path,
    mesh_path: Path,
    options: TrackingOptions,
) -> list[str | Path]:
    """Build the external tracker command without executing it."""
    debug_dir = data_dir / "debug"
    if options.live:
        command: list[str | Path] = [
            paths.tracker_python,
            paths.spark_root / "tools" / "run_realsense_foundationpose.py",
            "--mesh-file",
            mesh_path,
            "--mask-file",
            mask_path,
            "--debug-dir",
            debug_dir,
            "--pose-output-dir",
            debug_dir / "ob_in_cam",
            "--rgbd-output-dir",
            data_dir,
            "--width",
            str(options.camera_width),
            "--height",
            str(options.camera_height),
            "--fps",
            str(options.camera_fps),
            "--benchmark-report",
            debug_dir / f"{options.backend}_live_benchmark.json",
        ]
        if options.no_gui:
            command.append("--no-display")
        if options.camera_serial:
            command.extend(["--serial", options.camera_serial])
        if options.backend == "spark6d":
            command.extend(
                [
                    "--use-kalman",
                    "--kalman-refine-interval",
                    str(options.spark_refine_interval),
                    "--quiet",
                ]
            )
        if options.max_frames:
            command.extend(["--max-frames", str(options.max_frames)])
        if options.save_video:
            command.extend(
                [
                    "--save-video",
                    "--video-path",
                    options.video_path,
                    "--video-fps",
                    str(options.video_fps),
                ]
            )
        return command

    command = [
        paths.tracker_python,
        paths.spark_root / "run_demo.py",
        "--mesh_file",
        mesh_path,
        "--test_scene_dir",
        data_dir,
        "--debug_dir",
        debug_dir,
        "--debug",
        "1" if options.save_render else "0",
        "--no_display",
    ]
    if options.save_render:
        command.append("--save_render")
    command.extend(
        [
            "--benchmark_report",
            debug_dir / f"{options.backend}_benchmark.json",
        ]
    )
    if options.backend == "spark6d":
        command.extend(
            [
                "--use_kalman",
                "--kalman_refine_interval",
                str(options.spark_refine_interval),
                "--fast_io",
                "--quiet",
            ]
        )
    if options.max_frames:
        command.extend(["--max_frames", str(options.max_frames)])
    if options.save_video:
        command.extend(
            [
                "--save_video",
                "--video_path",
                options.video_path,
                "--video_fps",
                str(options.video_fps),
            ]
        )
    return command
