"""RGB-D projection, mesh transforms, simplification, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from click_to_model.reconstruction.metric_scale import robust_extent


def first_file(folder: Path, suffix: str = "*.png") -> Path:
    files = sorted(folder.glob(suffix))
    if not files:
        raise FileNotFoundError(f"No {suffix} files in {folder}")
    return files[0]


def first_rgbd_file(data_dir: Path, modality: str) -> Path:
    """Prefer sequence RGB-D, then use the pair archived with the mask."""
    for folder in (data_dir / modality, data_dir / "masks" / modality):
        files = sorted(folder.glob("*.png"))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No RGB-D {modality} PNG in {data_dir / modality} or "
        f"{data_dir / 'masks' / modality}"
    )


def load_intrinsics(data_dir: Path) -> np.ndarray:
    matrix_path = data_dir / "cam_K.txt"
    if matrix_path.is_file():
        return np.loadtxt(matrix_path, dtype=np.float32).reshape(3, 3)
    for name in ("camera_params.json", "camera_info.json", "intrinsics.json"):
        path = data_dir / name
        if path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            values = values.get("intrinsics", values)
            cx = values.get("ppx", values.get("cx"))
            cy = values.get("ppy", values.get("cy"))
            return np.array(
                [[values["fx"], 0, cx], [0, values["fy"], cy], [0, 0, 1]],
                dtype=np.float32,
            )
    raise FileNotFoundError(f"No cam_K.txt/camera_params.json in {data_dir}")


def load_depth_scale(data_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    for name in ("camera_params.json", "camera_info.json", "intrinsics.json"):
        path = data_dir / name
        if path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            value = values.get("depth_scale")
            if value is None:
                value = values.get("stream", {}).get("sensor_depth_scale_m_per_unit")
            if value is not None:
                return float(value)
    return 0.001


def make_pointmap(depth_m: np.ndarray, intrinsics: np.ndarray) -> torch.Tensor:
    height, width = depth_m.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z = depth_m
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    # RealSense/R3 (+x right, +y down) -> PyTorch3D (+x left, +y up).
    points = np.stack((-x, -y, z), axis=-1)
    points[~np.isfinite(z) | (z <= 0)] = np.nan
    return torch.from_numpy(points.astype(np.float32, copy=False))


def tensor_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def sam3d_similarity_transform(output: dict[str, Any]) -> np.ndarray:
    """Return the SAM3D local-to-PyTorch3D-camera Sim(3) matrix."""
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    quaternion = torch.as_tensor(output["rotation"]).detach().float().reshape(-1, 4)[:1]
    translation = (
        torch.as_tensor(output["translation"]).detach().float().reshape(-1, 3)[:1]
    )
    scale = torch.as_tensor(output["scale"]).detach().float().reshape(-1, 3)[:1]
    transform = compose_transform(
        scale=scale,
        rotation=quaternion_to_matrix(quaternion),
        translation=translation,
    )
    # Transform3d stores row-vector matrices. Transpose to column convention.
    return transform.get_matrix()[0].detach().cpu().numpy().T


def fallback_metric_local_vertices(
    vertices: np.ndarray,
    predicted_scale: np.ndarray,
    depth_extent: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    vertices = vertices.astype(np.float64, copy=True)
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    mesh_extent = np.percentile(vertices, 95, axis=0) - np.percentile(
        vertices, 5, axis=0
    )
    predicted = np.asarray(predicted_scale, dtype=np.float64).reshape(-1)
    predicted = predicted[np.isfinite(predicted) & (predicted > 0)]
    if predicted.size:
        scale_factor = float(np.median(predicted))
        method = "sam3d_pose_decoder_fallback"
    else:
        scale_factor = float(np.max(depth_extent) / max(np.max(mesh_extent), 1e-8))
        method = "rgbd_extent_only"

    scaled_extent = np.max(mesh_extent) * scale_factor
    depth_size = float(np.max(depth_extent))
    if not (0.25 * depth_size <= scaled_extent <= 4.0 * depth_size):
        scale_factor = depth_size / max(float(np.max(mesh_extent)), 1e-8)
        method = "rgbd_extent_sanity_fallback"
    return vertices * scale_factor, scale_factor, method


def p3d_to_realsense(points: np.ndarray) -> np.ndarray:
    converted = np.asarray(points, dtype=np.float64).copy()
    converted[:, :2] *= -1.0
    return converted


def metric_mask_silhouette(
    mask: np.ndarray,
    intrinsics: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate the mask's metric x/y span at its median depth."""
    rows, columns = np.where(mask)
    if len(columns) < 100:
        raise ValueError("The object mask has fewer than 100 pixels")
    u_low, u_high = np.percentile(columns, [1.0, 99.0])
    v_low, v_high = np.percentile(rows, [1.0, 99.0])
    median_depth = float(np.median(target_points[:, 2]))
    extent = np.array(
        [
            (u_high - u_low) * median_depth / intrinsics[0, 0],
            (v_high - v_low) * median_depth / intrinsics[1, 1],
            robust_extent(target_points)[2],
        ],
        dtype=np.float64,
    )
    border_margin = 2
    touches_border = bool(
        np.any(columns <= border_margin)
        or np.any(columns >= mask.shape[1] - 1 - border_margin)
        or np.any(rows <= border_margin)
        or np.any(rows >= mask.shape[0] - 1 - border_margin)
    )
    return extent, {
        "extent_m": extent.tolist(),
        "median_depth_m": median_depth,
        "mask_pixels": int(len(columns)),
        "touches_image_border": touches_border,
    }


def save_registration_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    aligned_p3d: np.ndarray,
    intrinsics: np.ndarray,
    path: Path,
) -> None:
    """Project the camera-aligned mesh for a registration sanity check."""
    points = p3d_to_realsense(aligned_p3d)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    canvas = image_bgr.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    if len(points):
        u = np.rint(
            intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]
        ).astype(int)
        v = np.rint(
            intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]
        ).astype(int)
        valid = (u >= 0) & (u < canvas.shape[1]) & (v >= 0) & (v < canvas.shape[0])
        projected = np.zeros(canvas.shape[:2], dtype=np.uint8)
        projected[v[valid], u[valid]] = 255
        projected = cv2.dilate(projected, np.ones((3, 3), np.uint8))
        overlay = np.zeros_like(canvas)
        overlay[projected > 0] = (255, 80, 30)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.6, 0.0)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Failed to write registration overlay: {path}")


def simplify_for_tracking(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_faces <= 0 or len(faces) <= max_faces:
        return vertices, faces, vertex_colors
    import open3d as o3d

    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False)),
    )
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        source.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    reduced = source.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    reduced_vertices = np.asarray(reduced.vertices)
    reduced_faces = np.asarray(reduced.triangles)
    reduced_colors = np.asarray(reduced.vertex_colors)
    if not len(reduced_colors):
        reduced_colors = None
    return reduced_vertices, reduced_faces, reduced_colors
