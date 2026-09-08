import os
import sys
import unittest
from unittest.mock import patch

from click_to_model.config import REPOSITORY_ROOT, RuntimePaths


class RuntimePathsTest(unittest.TestCase):
    def test_defaults_are_checkout_relative(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = RuntimePaths.from_environment()

        self.assertEqual(paths.repository_root, REPOSITORY_ROOT)
        self.assertEqual(paths.spark_root, REPOSITORY_ROOT.parent / "SPARK-6D")
        self.assertEqual(paths.tracker_python, sys.executable)
        self.assertEqual(paths.sam3d_root, REPOSITORY_ROOT / "sam-3d-objects")
        self.assertEqual(
            paths.sam2_root,
            REPOSITORY_ROOT.parent / "SPARK-6D" / "third_party" / "sam2",
        )

    def test_environment_overrides_are_resolved(self):
        with patch.dict(
            os.environ,
            {
                "SPARK6D_REPO": "./tracker",
                "SPARK6D_PY": "custom-python",
                "SAM3D_CHECKPOINT_DIR": "./weights",
            },
            clear=True,
        ):
            paths = RuntimePaths.from_environment()

        self.assertEqual(paths.spark_root, (REPOSITORY_ROOT / "tracker").resolve())
        self.assertEqual(paths.tracker_python, "custom-python")
        self.assertEqual(
            paths.sam3d_checkpoint_dir,
            (REPOSITORY_ROOT / "weights").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
