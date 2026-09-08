"""Ensure the documented legacy script imports remain valid."""

import unittest

import metric_scale_icp
import run_click_to_model
import run_sam3d_rgbd


class CompatibilityEntryPointTest(unittest.TestCase):
    def test_pipeline_wrapper_exports_previous_helper_names(self):
        self.assertTrue(callable(run_click_to_model.build_sam3d_command))
        self.assertTrue(callable(run_click_to_model.finish_sam3d_preload))
        self.assertTrue(callable(run_click_to_model.cancel_sam3d_preload))

    def test_reconstruction_wrapper_exports_data_helpers(self):
        self.assertTrue(callable(run_sam3d_rgbd.first_rgbd_file))
        self.assertTrue(callable(run_sam3d_rgbd.make_pointmap))

    def test_metric_scale_wrapper_exports_public_api(self):
        self.assertTrue(callable(metric_scale_icp.masked_depth_points))
        self.assertTrue(callable(metric_scale_icp.recover_metric_scale_icp))


if __name__ == "__main__":
    unittest.main()
