import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from click_to_model.config import RuntimePaths

LIVE_TRACKER = (
    RuntimePaths.from_environment().spark_root
    / "tools"
    / "run_realsense_foundationpose.py"
)


def load_live_tracker_module():
    spec = importlib.util.spec_from_file_location("spark6d_live_tracker", LIVE_TRACKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiveVideoWriterTest(unittest.TestCase):
    def test_writes_readable_mp4(self):
        module = load_live_tracker_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tracking.mp4"
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = 180
            writer = module._open_video_writer(path, frame, 20.0)
            for _ in range(3):
                writer.write(frame)
            writer.release()

            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            ok, decoded = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(decoded.shape[:2], (48, 64))

    def test_bounded_executor_flushes_pose_and_video(self):
        module = load_live_tracker_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "async.mp4"
            frame = np.full((48, 64, 3), 90, dtype=np.uint8)
            writer = module._open_video_writer(video_path, frame, 24.0)
            executor = module.BoundedOutputExecutor(max_pending=2)
            for index in range(4):
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = index
                executor.submit(
                    module._write_frame_outputs,
                    root / f"{index:06d}.txt",
                    pose,
                    writer,
                    frame.copy(),
                )
            executor.close()
            writer.release()

            self.assertEqual(len(list(root.glob("*.txt"))), 4)
            capture = cv2.VideoCapture(str(video_path))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            capture.release()

    def test_online_performance_arguments(self):
        module = load_live_tracker_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mesh = root / "mesh.obj"
            mask = root / "mask.png"
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            cv2.imwrite(str(mask), np.ones((4, 4), dtype=np.uint8))
            report = root / "benchmark.json"
            rgbd = root / "sequence"

            args = module.parse_args(
                [
                    "--mesh-file",
                    str(mesh),
                    "--mask-file",
                    str(mask),
                    "--fps",
                    "60",
                    "--output-queue-size",
                    "8",
                    "--rgbd-output-dir",
                    str(rgbd),
                    "--rgbd-output-workers",
                    "2",
                    "--rgbd-png-compression",
                    "1",
                    "--benchmark-report",
                    str(report),
                ]
            )

            self.assertEqual(args.fps, 60)
            self.assertEqual(args.output_queue_size, 8)
            self.assertEqual(args.rgbd_output_dir, rgbd)
            self.assertEqual(args.rgbd_output_workers, 2)
            self.assertEqual(args.rgbd_png_compression, 1)
            self.assertEqual(args.benchmark_report, report)

    def test_async_rgbd_writer_preserves_aligned_frames(self):
        module = load_live_tracker_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            color = np.zeros((24, 32, 3), dtype=np.uint8)
            color[..., 0] = 17
            color[..., 1] = 93
            depth = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
            executor = module.BoundedOutputExecutor(
                max_pending=2,
                max_workers=2,
                thread_name_prefix="test-rgbd",
            )
            for index in range(3):
                executor.submit(
                    module._write_rgbd_outputs,
                    root / "rgb" / f"{index:06d}.png",
                    color.copy(),
                    root / "depth" / f"{index:06d}.png",
                    depth.copy(),
                    1,
                )
            executor.close()

            self.assertEqual(module._next_rgbd_index(root), 3)
            saved_color = cv2.imread(str(root / "rgb" / "000002.png"))
            saved_depth = cv2.imread(
                str(root / "depth" / "000002.png"), cv2.IMREAD_UNCHANGED
            )
            np.testing.assert_array_equal(saved_color, color)
            np.testing.assert_array_equal(saved_depth, depth)


if __name__ == "__main__":
    unittest.main()
