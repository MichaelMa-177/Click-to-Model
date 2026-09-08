import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from click_to_model import pipeline
from click_to_model.dataset import archive_mask_rgbd_inputs
from click_to_model.processes import (
    cancel_preloaded_worker,
    finish_preloaded_worker,
)
from click_to_model.reconstruction.geometry import first_rgbd_file
from click_to_model.tracking import TrackingOptions, build_tracking_command


class PipelinePreloadTest(unittest.TestCase):
    def test_live_tracker_receives_selected_camera_serial(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = SimpleNamespace(
                tracker_python=sys.executable,
                spark_root=root / "SPARK-6D",
            )
            options = TrackingOptions(
                backend="spark6d",
                live=True,
                camera_serial="camera-under-test",
                camera_width=640,
                camera_height=480,
                camera_fps=30,
                no_gui=True,
                max_frames=1,
                spark_refine_interval=2,
                save_render=False,
                save_video=False,
                video_path=root / "tracking.mp4",
                video_fps=30.0,
            )

            command = build_tracking_command(
                paths,
                root / "sequence",
                root / "mask.png",
                root / "mesh.obj",
                options,
            )

            self.assertEqual(
                command[command.index("--serial") + 1],
                "camera-under-test",
            )

    def test_sam3d_command_preserves_inference_options(self):
        args = SimpleNamespace(
            scale_mode="icp",
            icp_voxel_size=0.002,
            icp_samples=1234,
            icp_iterations=17,
            icp_mask_erode=2,
            min_depth_coverage=0.4,
            max_tracking_faces=50_000,
            distilled=True,
            require_icp_confidence=True,
            allow_border_mask=True,
        )

        command = pipeline.build_reconstruction_command(args, "/tmp/sequence")

        self.assertIn("--distilled", command)
        self.assertIn("--require-icp-confidence", command)
        self.assertIn("--allow-border-mask", command)
        self.assertEqual(command[command.index("--data-dir") + 1], "/tmp/sequence")
        self.assertEqual(command[command.index("--icp-samples") + 1], "1234")
        self.assertEqual(command[command.index("--max-faces") + 1], "50000")

    def test_preloaded_worker_is_triggered_through_stdin(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "raise SystemExit(0 if "
                    "sys.stdin.readline().strip() == 'run' else 2)"
                ),
            ],
            stdin=subprocess.PIPE,
        )

        finish_preloaded_worker(process)

        self.assertEqual(process.returncode, 0)

    def test_preloaded_worker_can_be_cancelled(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.readline()"],
            stdin=subprocess.PIPE,
        )

        cancel_preloaded_worker(process)

        self.assertEqual(process.returncode, 0)

    def test_online_mask_inputs_are_moved_and_sam3d_can_find_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            (root / "masks").mkdir()
            rgb = root / "rgb" / "000000.png"
            depth = root / "depth" / "000000.png"
            cv2.imwrite(str(rgb), np.zeros((8, 10, 3), dtype=np.uint8))
            cv2.imwrite(str(depth), np.ones((8, 10), dtype=np.uint16))

            archived_rgb, archived_depth = archive_mask_rgbd_inputs(
                root, rgb, depth, move_from_sequence=True
            )

            self.assertFalse(rgb.exists())
            self.assertFalse(depth.exists())
            self.assertEqual(first_rgbd_file(root, "rgb"), archived_rgb)
            self.assertEqual(first_rgbd_file(root, "depth"), archived_depth)


if __name__ == "__main__":
    unittest.main()
