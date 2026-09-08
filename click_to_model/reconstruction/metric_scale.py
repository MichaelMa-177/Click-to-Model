"""Metric scale recovery for a generated mesh from one RGB-D observation.

The registration is deliberately bounded around a depth-derived silhouette
scale. A free similarity ICP tends to shrink a complete mesh onto the visible
surface from a single depth frame. We therefore search scale explicitly and
run rigid coarse-to-fine ICP at every scale candidate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


@dataclass
class RegistrationMetrics:
    score: float
    target_coverage: float
    target_rmse_m: float
    visible_source_rmse_m: float
    icp_fitness: float
    icp_inlier_rmse_m: float


@dataclass
class MetricScaleResult:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    similarity_transform: np.ndarray
    aligned_points: np.ndarray
    source_center: np.ndarray
    target_points: np.ndarray
    initial_extent_scale: float
    predicted_scale: float | None
    rotation_source: str
    coarse_threshold_m: float
    fine_threshold_m: float
    metrics: RegistrationMetrics
    accepted: bool
    candidate_summary: list[dict]

    def metadata(self) -> dict:
        result = asdict(self)
        result.pop("aligned_points")
        result.pop("target_points")
        return _jsonable(result)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def robust_extent(points: np.ndarray, lower: float = 5.0, upper: float = 95.0):
    bounds = np.percentile(points, [lower, upper], axis=0)
    return bounds[1] - bounds[0]


def masked_depth_points(
    pointmap: np.ndarray,
    mask: np.ndarray,
    *,
    mask_erode_px: int = 3,
    voxel_size_m: float = 0.003,
    max_points: int = 30_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Extract a robust metric object cloud from an aligned pointmap and mask."""
    import cv2

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_erode_px > 0:
        size = 2 * mask_erode_px + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1) > 0
        mask_bool = eroded if np.count_nonzero(eroded) >= 200 else mask_u8 > 0
    else:
        mask_bool = mask_u8 > 0

    mask_pixels = int(np.count_nonzero(mask_bool))
    points = np.asarray(pointmap, dtype=np.float64)[mask_bool]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    raw_count = len(points)
    if raw_count < 100:
        raise ValueError(f"Only {raw_count} valid depth points inside the mask")

    # Mask edges and small mask holes often contain the background plane. Keep
    # the robust object depth distribution without imposing a fixed thickness.
    q1, q3 = np.percentile(points[:, 2], [25.0, 75.0])
    iqr = max(float(q3 - q1), 0.002)
    z_low = max(0.001, float(q1 - 2.5 * iqr))
    z_high = float(q3 + 2.5 * iqr)
    depth_filtered = points[(points[:, 2] >= z_low) & (points[:, 2] <= z_high)]
    if len(depth_filtered) >= 100:
        points = depth_filtered

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel_size_m > 0:
        cloud = cloud.voxel_down_sample(voxel_size_m)
    if len(cloud.points) >= 300:
        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=min(30, max(10, len(cloud.points) // 100)),
            std_ratio=2.5,
        )
        if len(filtered.points) >= 100:
            cloud = filtered
    points = np.asarray(cloud.points)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    if len(points) < 100:
        raise ValueError(f"Only {len(points)} depth points remain after filtering")

    metadata = {
        "raw_masked_points": int(raw_count),
        "mask_pixels_after_erosion": mask_pixels,
        "valid_depth_fraction": float(raw_count / max(mask_pixels, 1)),
        "filtered_points": int(len(points)),
        "z_filter_m": [z_low, z_high],
        "extent_m": robust_extent(points).tolist(),
    }
    return np.ascontiguousarray(points), metadata


def estimate_similarity(source: np.ndarray, target: np.ndarray):
    """Umeyama similarity mapping source points to target points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    scale = float(np.sum(singular_values * sign) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _proper_pca_rotations(
    source: np.ndarray,
    target: np.ndarray,
) -> Iterable[np.ndarray]:
    def axes(points):
        covariance = np.cov((points - points.mean(axis=0)).T)
        _, vectors = np.linalg.eigh(covariance)
        return vectors[:, ::-1]

    source_axes = axes(source)
    target_axes = axes(target)
    for signs in product((-1.0, 1.0), repeat=3):
        mapping = np.diag(signs)
        rotation = target_axes @ mapping @ source_axes.T
        if np.linalg.det(rotation) > 0.0:
            yield rotation


def _unique_rotations(candidates: list[tuple[str, np.ndarray]]):
    unique: list[tuple[str, np.ndarray]] = []
    for name, rotation in candidates:
        rotation = np.asarray(rotation, dtype=np.float64)
        if not np.isfinite(rotation).all() or np.linalg.det(rotation) < 0.5:
            continue
        if any(np.trace(existing.T @ rotation) > 2.999 for _, existing in unique):
            continue
        unique.append((name, rotation))
    return unique


def _extent_scale(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    target_xy_extent_m: np.ndarray | None,
) -> float:
    rotated = source @ rotation.T
    source_extent = robust_extent(rotated)
    target_extent = robust_extent(target)
    if target_xy_extent_m is not None:
        target_extent[:2] = np.asarray(target_xy_extent_m, dtype=np.float64)[:2]
    ratios = target_extent[:2] / np.maximum(source_extent[:2], 1e-8)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if not len(ratios):
        return float(np.max(target_extent) / max(np.max(source_extent), 1e-8))
    return float(np.median(ratios))


def _make_cloud(points: np.ndarray):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def _score_registration(
    source_aligned: np.ndarray,
    target: np.ndarray,
    threshold: float,
    registration,
) -> RegistrationMetrics:
    source_tree = cKDTree(source_aligned)
    target_tree = cKDTree(target)
    target_distances = source_tree.query(target, workers=-1)[0]
    source_distances = target_tree.query(source_aligned, workers=-1)[0]

    target_inliers = target_distances[target_distances <= threshold]
    target_coverage = float(len(target_inliers) / len(target_distances))
    if len(target_inliers):
        target_rmse = float(np.sqrt(np.mean(target_inliers**2)))
    else:
        target_rmse = threshold * 2.0

    # Only a fraction of the complete mesh is visible in one RGB-D frame.
    # Score the best-matching 35% instead of forcing hidden surfaces onto the
    # observed front surface.
    visible_count = min(
        len(source_distances), max(100, int(round(0.35 * len(source_distances))))
    )
    visible = np.partition(source_distances, visible_count - 1)[:visible_count]
    visible_rmse = float(np.sqrt(np.mean(np.minimum(visible, threshold * 2.0) ** 2)))
    score = target_rmse + 0.35 * visible_rmse + (1.0 - target_coverage) * threshold
    return RegistrationMetrics(
        score=float(score),
        target_coverage=target_coverage,
        target_rmse_m=target_rmse,
        visible_source_rmse_m=visible_rmse,
        icp_fitness=float(registration.fitness),
        icp_inlier_rmse_m=float(registration.inlier_rmse),
    )


def _run_rigid_icp(
    source_centered: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    coarse_threshold: float,
    fine_threshold: float,
    max_iterations: int,
):
    scaled_source = source_centered * scale
    source_median = np.median(scaled_source, axis=0)
    target_median = np.median(target, axis=0)
    initial = np.eye(4)
    initial[:3, :3] = rotation
    initial[:3, 3] = target_median - rotation @ source_median

    source_cloud = _make_cloud(scaled_source)
    target_cloud = _make_cloud(target)
    coarse = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        coarse_threshold,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iterations,
        ),
    )
    fine = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        fine_threshold,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=max_iterations,
        ),
    )
    rigid = fine.transformation
    aligned = scaled_source @ rigid[:3, :3].T + rigid[:3, 3]
    metrics = _score_registration(aligned, target, fine_threshold, fine)
    return rigid, aligned, metrics


def recover_metric_scale_icp(
    mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    *,
    predicted_similarity: np.ndarray | None = None,
    target_xy_extent_m: np.ndarray | None = None,
    sample_points: int = 20_000,
    voxel_size_m: float = 0.003,
    max_iterations: int = 60,
    seed: int = 0,
) -> MetricScaleResult:
    """Recover uniform metric scale with bounded scale search and rigid ICP."""
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("Mesh is too small for metric registration")
    if len(target_points) < 100:
        raise ValueError("At least 100 target depth points are required")

    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_points, seed=seed)
    source_center = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    source_centered = np.asarray(surface_points, dtype=np.float64) - source_center
    target = np.asarray(target_points, dtype=np.float64)

    predicted_scale = None
    predicted_rotation = None
    if predicted_similarity is not None:
        predicted_similarity = np.asarray(predicted_similarity, dtype=np.float64)
        if (
            predicted_similarity.shape == (4, 4)
            and np.isfinite(predicted_similarity).all()
        ):
            linear = predicted_similarity[:3, :3]
            predicted_scale = float(np.cbrt(abs(np.linalg.det(linear))))
            if predicted_scale > 0:
                raw_rotation = linear / predicted_scale
                u, _, vt = np.linalg.svd(raw_rotation)
                predicted_rotation = u @ vt
                if np.linalg.det(predicted_rotation) < 0:
                    u[:, -1] *= -1
                    predicted_rotation = u @ vt

    rotations: list[tuple[str, np.ndarray]] = []
    if predicted_rotation is not None:
        rotations.append(("sam3d_pose", predicted_rotation))
    rotations.extend(
        (f"pca_{index}", rotation)
        for index, rotation in enumerate(_proper_pca_rotations(source_centered, target))
    )
    rotations.append(("identity", np.eye(3)))
    rotations = _unique_rotations(rotations)

    target_extent = robust_extent(target)
    if target_xy_extent_m is not None:
        target_extent[:2] = np.asarray(target_xy_extent_m, dtype=np.float64)[:2]
    object_size = float(max(target_extent[0], target_extent[1], voxel_size_m * 5))
    coarse_threshold = max(voxel_size_m * 5.0, object_size * 0.18, 0.012)
    fine_threshold = max(voxel_size_m * 2.0, object_size * 0.055, 0.004)

    candidates = []
    for rotation_name, rotation in rotations:
        extent_scale = _extent_scale(
            source_centered,
            target,
            rotation,
            target_xy_extent_m,
        )
        factors = (
            (0.84, 1.0, 1.16)
            if rotation_name != "sam3d_pose"
            else (0.8, 0.9, 1.0, 1.1, 1.2)
        )
        scales = [extent_scale * factor for factor in factors]
        if (
            rotation_name == "sam3d_pose"
            and predicted_scale is not None
            and 0.5 * extent_scale <= predicted_scale <= 2.0 * extent_scale
        ):
            scales.append(float(predicted_scale))
        for scale in sorted(set(round(value, 10) for value in scales)):
            rigid, aligned, metrics = _run_rigid_icp(
                source_centered,
                target,
                scale,
                rotation,
                coarse_threshold,
                fine_threshold,
                max_iterations,
            )
            # Prefer the network rotation only as a weak tie breaker; metric
            # depth fit remains the dominant criterion.
            adjusted_score = metrics.score
            if rotation_name != "sam3d_pose" and predicted_rotation is not None:
                adjusted_score += fine_threshold * 0.03
            candidates.append(
                {
                    "scale": float(scale),
                    "rotation_source": rotation_name,
                    "rigid": rigid,
                    "aligned": aligned,
                    "metrics": metrics,
                    "adjusted_score": float(adjusted_score),
                    "extent_scale": float(extent_scale),
                }
            )

    if not candidates:
        raise RuntimeError("No valid scale/rotation candidates were generated")
    best = min(candidates, key=lambda item: item["adjusted_score"])

    # A narrow second pass turns the discrete coarse scale into a locally
    # refined estimate without allowing the full-to-partial shrink failure.
    coarse_best = best
    for factor in (0.94, 0.97, 1.03, 1.06):
        scale = coarse_best["scale"] * factor
        rigid, aligned, metrics = _run_rigid_icp(
            source_centered,
            target,
            scale,
            coarse_best["rigid"][:3, :3],
            coarse_threshold,
            fine_threshold,
            max_iterations,
        )
        candidate = {
            "scale": float(scale),
            "rotation_source": f"{coarse_best['rotation_source']}_fine",
            "rigid": rigid,
            "aligned": aligned,
            "metrics": metrics,
            "adjusted_score": float(metrics.score),
            "extent_scale": float(coarse_best["extent_scale"]),
        }
        candidates.append(candidate)
        if candidate["adjusted_score"] < best["adjusted_score"]:
            best = candidate

    scale = float(best["scale"])
    rigid = best["rigid"]
    similarity = np.eye(4)
    similarity[:3, :3] = rigid[:3, :3] * scale
    similarity[:3, 3] = rigid[:3, 3]
    accepted = bool(
        best["metrics"].target_coverage >= 0.45
        and best["metrics"].target_rmse_m <= fine_threshold
        and np.isfinite(scale)
        and scale > 0
    )

    ranked = sorted(candidates, key=lambda item: item["adjusted_score"])[:10]
    summary = [
        {
            "scale": item["scale"],
            "rotation_source": item["rotation_source"],
            **asdict(item["metrics"]),
        }
        for item in ranked
    ]
    return MetricScaleResult(
        scale=scale,
        rotation=rigid[:3, :3],
        translation=rigid[:3, 3],
        similarity_transform=similarity,
        aligned_points=best["aligned"],
        source_center=np.asarray(source_center),
        target_points=target,
        initial_extent_scale=float(best["extent_scale"]),
        predicted_scale=(
            float(predicted_scale) if predicted_scale is not None else None
        ),
        rotation_source=str(best["rotation_source"]),
        coarse_threshold_m=coarse_threshold,
        fine_threshold_m=fine_threshold,
        metrics=best["metrics"],
        accepted=accepted,
        candidate_summary=summary,
    )
