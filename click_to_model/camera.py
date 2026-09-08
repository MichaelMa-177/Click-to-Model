"""Intel RealSense RGB-D capture for FoundationPose-style datasets."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RealSenseConfig:
    """RealSense stream and retry settings."""

    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 60
    timeout_ms: int = 10_000
    startup_retries: int = 2


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def _start_pipeline(rs, config, settings: RealSenseConfig):
    pipeline = None
    for attempt in range(settings.startup_retries + 1):
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
            pipeline.wait_for_frames(settings.timeout_ms)
            return pipeline, profile
        except RuntimeError as error:
            with suppress(RuntimeError):
                pipeline.stop()
            if (
                "Frame didn't arrive" not in str(error)
                or attempt >= settings.startup_retries
            ):
                raise
            print(
                "[CAM] Startup frame timeout; restarting stream "
                f"({attempt + 1}/{settings.startup_retries})"
            )
            time.sleep(1.0)
    raise RuntimeError("Failed to start the RealSense pipeline")


def capture_realsense(
    data_dir: Path,
    frame_count: int,
    preview: bool,
    settings: RealSenseConfig,
) -> None:
    """Capture aligned color/depth frames and persist camera calibration."""
    import pyrealsense2 as rs

    stream_config = rs.config()
    if settings.serial:
        stream_config.enable_device(settings.serial)
    stream_config.enable_stream(
        rs.stream.color,
        settings.width,
        settings.height,
        rs.format.bgr8,
        settings.fps,
    )
    stream_config.enable_stream(
        rs.stream.depth,
        settings.width,
        settings.height,
        rs.format.z16,
        settings.fps,
    )
    pipeline, profile = _start_pipeline(rs, stream_config, settings)
    align = rs.align(rs.stream.color)
    device = profile.get_device()

    def device_info(field):
        return device.get_info(field) if device.supports(field) else "unknown"

    camera = {
        "name": device_info(rs.camera_info.name),
        "serial": device_info(rs.camera_info.serial_number),
        "firmware": device_info(rs.camera_info.firmware_version),
        "usb": device_info(rs.camera_info.usb_type_descriptor),
    }
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    depth_scale = device.first_depth_sensor().get_depth_scale()
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ]
    )
    np.savetxt(data_dir / "cam_K.txt", camera_matrix)
    camera_parameters = {
        "camera": camera,
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fps": settings.fps,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "depth_scale": depth_scale,
    }
    (data_dir / "camera_params.json").write_text(
        json.dumps(camera_parameters, indent=2),
        encoding="utf-8",
    )

    recording = not preview
    index = 0
    consecutive_timeouts = 0
    try:
        print(
            f"[CAM] {camera['name']} | serial={camera['serial']} | "
            f"firmware={camera['firmware']} | USB={camera['usb']}"
        )
        print(f"[CAM] Stream {intrinsics.width}x{intrinsics.height}@{settings.fps} FPS")
        print("[CAM] Press s to capture; q cancels" if preview else "[CAM] Capturing")
        while index < frame_count:
            try:
                frames = align.process(pipeline.wait_for_frames(settings.timeout_ms))
                consecutive_timeouts = 0
            except RuntimeError as error:
                if "Frame didn't arrive" not in str(error):
                    raise
                consecutive_timeouts += 1
                if consecutive_timeouts > 3:
                    raise
                print(f"[CAM] Transient frame timeout ({consecutive_timeouts}/3)")
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if recording:
                _write_png(data_dir / "rgb" / f"{index:06d}.png", color)
                _write_png(data_dir / "depth" / f"{index:06d}.png", depth)
                index += 1
            if preview:
                view = color.copy()
                cv2.putText(
                    view,
                    f"{'REC' if recording else 'READY'} {index}/{frame_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Click-to-Model RealSense", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    recording = True
                elif key in (ord("q"), 27):
                    raise RuntimeError("Capture cancelled")
    finally:
        pipeline.stop()
        if preview:
            cv2.destroyAllWindows()
    print(f"[CAM] Saved {index} aligned RGB-D frames to {data_dir}")
