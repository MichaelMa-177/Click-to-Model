"""Regression coverage for resuming an archived online initialization."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from click_to_model import pipeline
from click_to_model.config import RuntimePaths


class CaptureResumeTest(unittest.TestCase):
    def test_archived_capture_resumes_without_camera_or_reannotation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("rgb", "depth", "masks/rgb", "masks/depth", "mesh"):
                (root / name).mkdir(parents=True, exist_ok=True)
            rgb_path = root / "masks/rgb/000000.png"
            depth_path = root / "masks/depth/000000.png"
            mask_path = root / "masks/000000.png"
            cv2.imwrite(str(rgb_path), np.zeros((8, 10, 3), dtype=np.uint8))
            cv2.imwrite(str(depth_path), np.full((8, 10), 600, dtype=np.uint16))
            cv2.imwrite(str(mask_path), np.full((8, 10), 255, dtype=np.uint8))
            original_bytes = {
                path: path.read_bytes() for path in (rgb_path, depth_path, mask_path)
            }
            paths = RuntimePaths.from_environment()
            args = pipeline.build_parser(paths).parse_args(
                [
                    "--data-dir",
                    str(root),
                    "--no-capture",
                    "--live-tracker",
                    "--mask-file",
                    str(mask_path),
                    "--no-model-preload",
                    "--camera-fps",
                    "30",
                    "--require-icp-confidence",
                    "--save-tracking-video",
                ]
            )
            commands = []

            def fake_run(command, cwd, environment):
                commands.append(command)
                if "click_to_model.reconstruction.cli" in command:
                    self.assertEqual(
                        pipeline._first_rgbd_paths(root),
                        (rgb_path, depth_path),
                    )
                    (root / "mesh/model.obj").touch()

            with (
                patch.object(pipeline, "run_command", side_effect=fake_run),
                patch.object(pipeline, "capture_realsense") as capture,
                patch.object(pipeline.Sam2Session, "load") as load_sam2,
            ):
                self.assertEqual(pipeline.run_pipeline(args, paths), root)

            capture.assert_not_called()
            load_sam2.assert_not_called()
            self.assertEqual(len(commands), 2)
            self.assertIn("--require-icp-confidence", commands[0])
            self.assertIn("--rgbd-output-dir", commands[1])
            self.assertIn("--pose-output-dir", commands[1])
            self.assertIn("--save-video", commands[1])
            self.assertEqual(list((root / "rgb").iterdir()), [])
            self.assertEqual(list((root / "depth").iterdir()), [])
            for path, contents in original_bytes.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_mismatched_depth_is_not_substituted_for_first_rgb(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            (root / "rgb/000000.png").touch()
            (root / "depth/000001.png").touch()
            with self.assertRaisesRegex(FileNotFoundError, "matching depth"):
                pipeline._first_rgbd_paths(root)

    def test_sequence_pair_remains_preferred_over_archive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for base in (root, root / "masks"):
                for name in ("rgb", "depth"):
                    (base / name).mkdir(parents=True)
                    (base / name / "000000.png").touch()
            self.assertEqual(
                pipeline._first_rgbd_paths(root),
                (root / "rgb/000000.png", root / "depth/000000.png"),
            )


if __name__ == "__main__":
    unittest.main()
