import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from click_to_model.reconstruction.metric_scale import (
    masked_depth_points,
    recover_metric_scale_icp,
)


class MetricScaleIcpTest(unittest.TestCase):
    def test_reports_valid_depth_fraction_inside_mask(self):
        height, width = 30, 30
        row, column = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        pointmap = np.stack(
            (column * 0.001, row * 0.001, np.ones_like(row)),
            axis=-1,
        ).astype(float)
        pointmap[:, width // 2 :] = np.nan
        mask = np.ones((height, width), dtype=bool)

        _, metadata = masked_depth_points(
            pointmap,
            mask,
            mask_erode_px=0,
            voxel_size_m=0.0,
        )

        self.assertEqual(metadata["mask_pixels_after_erosion"], 900)
        self.assertAlmostEqual(metadata["valid_depth_fraction"], 0.5)

    def test_recovers_scale_from_partial_depth_surface(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.vertices[:] *= np.array([1.0, 0.72, 0.48])
        points, _ = trimesh.sample.sample_surface(mesh, 12_000, seed=17)

        true_scale = 0.115
        rotation = Rotation.from_euler(
            "xyz", [18.0, -27.0, 36.0], degrees=True
        ).as_matrix()
        translation = np.array([0.035, -0.022, 0.72])
        target = points * true_scale @ rotation.T + translation
        target_xy_extent = (
            np.percentile(target, 95.0, axis=0) - np.percentile(target, 5.0, axis=0)
        )[:2]
        target = target[target[:, 2] <= np.quantile(target[:, 2], 0.56)]
        target += np.random.default_rng(9).normal(0.0, 0.0006, target.shape)

        predicted = np.eye(4)
        predicted[:3, :3] = rotation * (true_scale * 1.08)
        predicted[:3, 3] = translation
        result = recover_metric_scale_icp(
            mesh,
            target,
            predicted_similarity=predicted,
            target_xy_extent_m=target_xy_extent,
            sample_points=5_000,
            voxel_size_m=0.0025,
            max_iterations=25,
            seed=5,
        )

        self.assertTrue(result.accepted)
        self.assertGreater(result.metrics.target_coverage, 0.9)
        self.assertLess(abs(result.scale / true_scale - 1.0), 0.12)


if __name__ == "__main__":
    unittest.main()
