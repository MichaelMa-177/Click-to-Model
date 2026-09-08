"""SAM3D inference followed by metric RGB-D scale recovery."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh

from click_to_model.config import RuntimePaths
from click_to_model.reconstruction.geometry import (
    fallback_metric_local_vertices,
    first_file,
    first_rgbd_file,
    load_depth_scale,
    load_intrinsics,
    make_pointmap,
    metric_mask_silhouette,
    p3d_to_realsense,
    sam3d_similarity_transform,
    save_registration_overlay,
    simplify_for_tracking,
    tensor_numpy,
)
from click_to_model.reconstruction.metric_scale import (
    MetricScaleResult,
    masked_depth_points,
    recover_metric_scale_icp,
    robust_extent,
)


@dataclass(frozen=True)
class ReconstructionOptions:
    data_dir: Path
    config_path: Path | None = None
    depth_scale: float | None = None
    seed: int = 42
    stage1_steps: int | None = None
    stage2_steps: int | None = None
    distilled: bool = False
    low_vram: bool = True
    wait_for_trigger: bool = False
    max_faces: int = 200_000
    scale_mode: str = "icp"
    icp_voxel_size: float = 0.003
    icp_samples: int = 20_000
    icp_iterations: int = 60
    icp_mask_erode: int = 3
    min_depth_coverage: float = 0.25
    allow_border_mask: bool = False
    require_icp_confidence: bool = False


@dataclass(frozen=True)
class RgbdObservation:
    image_path: Path
    depth_path: Path
    mask_path: Path
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    depth_scale: float
    pointmap: torch.Tensor
    target_points: np.ndarray
    depth_extent: np.ndarray
    depth_filter_metadata: dict[str, Any]
    silhouette_extent: np.ndarray
    silhouette_metadata: dict[str, Any]


@dataclass(frozen=True)
class GeneratedMesh:
    mesh: trimesh.Trimesh
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    predicted_scale: np.ndarray
    predicted_similarity: np.ndarray | None


@dataclass(frozen=True)
class ScaleSolution:
    vertices: np.ndarray
    aligned_p3d: np.ndarray
    scale_factor: float
    scale_method: str
    metric_to_camera: np.ndarray
    raw_to_camera: np.ndarray | None
    icp_result: MetricScaleResult | None
    quality_issues: list[str]
    confidence_failure: bool


def _configure_sam3d_imports(paths: RuntimePaths) -> None:
    for directory in (paths.sam3d_root, paths.sam3d_root / "notebook"):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def _resolve_config_path(
    options: ReconstructionOptions,
    paths: RuntimePaths,
) -> Path:
    config_path = options.config_path or paths.sam3d_checkpoint_dir / "pipeline.yaml"
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM3D pipeline config not found: {config_path}")
    return config_path


def _load_inference(config_path: Path, options: ReconstructionOptions):
    from notebook.inference import Inference

    print(f"[SAM3D] config={config_path}", flush=True)
    print("[SAM3D] preloading inference model", flush=True)
    started = time.perf_counter()
    inference = Inference(
        str(config_path),
        compile=False,
        low_vram=options.low_vram,
        mesh_only=True,
        use_depth_model=False,
    )
    print(
        f"[SAM3D] model preload complete in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return inference


def _wait_for_trigger() -> bool:
    print("[SAM3D] waiting for accepted mask", flush=True)
    if not sys.stdin.readline():
        print("[SAM3D] preload cancelled before inference", flush=True)
        return False
    print("[SAM3D] trigger received; starting inference", flush=True)
    return True


def _load_observation(options: ReconstructionOptions) -> RgbdObservation:
    data_dir = options.data_dir
    image_path = first_rgbd_file(data_dir, "rgb")
    depth_path = first_rgbd_file(data_dir, "depth")
    mask_path = first_file(data_dir / "masks")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or depth_raw is None or mask_raw is None:
        raise RuntimeError(
            "Failed to read reconstruction inputs: "
            f"rgb={image_path}, depth={depth_path}, mask={mask_path}"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = mask_raw > 127
    if image_rgb.shape[:2] != depth_raw.shape[:2] or mask.shape != depth_raw.shape[:2]:
        raise ValueError("RGB, depth, and mask dimensions must match")

    intrinsics = load_intrinsics(data_dir)
    depth_scale = load_depth_scale(data_dir, options.depth_scale)
    depth_m = depth_raw.astype(np.float32) * depth_scale
    pointmap = make_pointmap(depth_m, intrinsics)
    target_points, depth_filter_metadata = masked_depth_points(
        pointmap.numpy(),
        mask,
        mask_erode_px=options.icp_mask_erode,
        voxel_size_m=options.icp_voxel_size,
        seed=options.seed,
    )
    depth_extent = robust_extent(target_points)
    silhouette_extent, silhouette_metadata = metric_mask_silhouette(
        mask,
        intrinsics,
        target_points,
    )
    print(
        f"[SAM3D] image={image_path.name}, depth={depth_path.name}, "
        f"mask={mask_path.name}"
    )
    print(
        f"[SAM3D] valid masked depth points={len(target_points)}, "
        f"extent(m)={depth_extent}"
    )
    return RgbdObservation(
        image_path=image_path,
        depth_path=depth_path,
        mask_path=mask_path,
        image_bgr=image_bgr,
        image_rgb=image_rgb,
        mask=mask,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        pointmap=pointmap,
        target_points=target_points,
        depth_extent=depth_extent,
        depth_filter_metadata=depth_filter_metadata,
        silhouette_extent=silhouette_extent,
        silhouette_metadata=silhouette_metadata,
    )


def _run_sam3d(
    inference,
    observation: RgbdObservation,
    options: ReconstructionOptions,
) -> GeneratedMesh:
    stage1_steps = options.stage1_steps
    stage2_steps = options.stage2_steps
    if options.distilled:
        stage1_steps = 2 if stage1_steps is None else stage1_steps
        stage2_steps = 12 if stage2_steps is None else stage2_steps
    output = inference(
        observation.image_rgb,
        observation.mask,
        seed=options.seed,
        pointmap=observation.pointmap,
        stage1_inference_steps=stage1_steps,
        stage2_inference_steps=stage2_steps,
        use_stage1_distillation=options.distilled,
        use_stage2_distillation=options.distilled,
    )
    if not output.get("mesh"):
        raise RuntimeError("SAM3D returned no mesh")

    mesh_result = output["mesh"][0]
    vertices = tensor_numpy(mesh_result.vertices).astype(np.float64)
    faces = tensor_numpy(mesh_result.faces).astype(np.int64)
    predicted_scale = tensor_numpy(output.get("scale", np.array([])))
    vertex_colors = None
    attributes = getattr(mesh_result, "vertex_attrs", None)
    if attributes is not None:
        colors = tensor_numpy(attributes)[..., :3]
        if colors.size:
            if np.nanmin(colors) < 0:
                colors = (colors + 1.0) / 2.0
            vertex_colors = np.clip(colors, 0, 1)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    try:
        predicted_similarity = sam3d_similarity_transform(output)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"[SAM3D] pose transform unavailable: {error}")
        predicted_similarity = None
    return GeneratedMesh(
        mesh=mesh,
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        predicted_scale=predicted_scale,
        predicted_similarity=predicted_similarity,
    )


def _recover_scale(
    generated: GeneratedMesh,
    observation: RgbdObservation,
    options: ReconstructionOptions,
) -> ScaleSolution:
    quality_issues: list[str] = []
    if options.scale_mode == "icp":
        print("[ICP] Recovering metric scale from the masked depth cloud")
        icp_result = recover_metric_scale_icp(
            generated.mesh,
            observation.target_points,
            predicted_similarity=generated.predicted_similarity,
            target_xy_extent_m=observation.silhouette_extent[:2],
            sample_points=options.icp_samples,
            voxel_size_m=options.icp_voxel_size,
            max_iterations=options.icp_iterations,
            seed=options.seed,
        )
        if not icp_result.accepted:
            quality_issues.append("ICP coverage/RMSE threshold failed")
        valid_depth_fraction = observation.depth_filter_metadata["valid_depth_fraction"]
        if valid_depth_fraction < options.min_depth_coverage:
            quality_issues.append(
                "valid depth covers only "
                f"{valid_depth_fraction:.1%} of the eroded object mask "
                f"(< {options.min_depth_coverage:.1%})"
            )
        if (
            observation.silhouette_metadata["touches_image_border"]
            and not options.allow_border_mask
        ):
            quality_issues.append("object mask touches an image border")
        icp_result.accepted = not quality_issues
        if quality_issues:
            print("[ICP] WARNING: " + "; ".join(quality_issues))

        source_center = icp_result.source_center
        scale_factor = icp_result.scale
        vertices = (generated.vertices - source_center) * scale_factor
        aligned_p3d = vertices @ icp_result.rotation.T + icp_result.translation
        metric_to_camera = np.eye(4)
        metric_to_camera[:3, :3] = icp_result.rotation
        metric_to_camera[:3, 3] = icp_result.translation
        raw_to_camera = icp_result.similarity_transform.copy()
        raw_to_camera[:3, 3] -= icp_result.similarity_transform[:3, :3] @ source_center
        return ScaleSolution(
            vertices=vertices,
            aligned_p3d=aligned_p3d,
            scale_factor=scale_factor,
            scale_method="rgbd_bounded_scale_icp",
            metric_to_camera=metric_to_camera,
            raw_to_camera=raw_to_camera,
            icp_result=icp_result,
            quality_issues=quality_issues,
            confidence_failure=(
                options.require_icp_confidence and not icp_result.accepted
            ),
        )

    fallback_scale = (
        generated.predicted_scale if options.scale_mode == "sam3d" else np.array([])
    )
    vertices, scale_factor, scale_method = fallback_metric_local_vertices(
        generated.vertices,
        fallback_scale,
        observation.depth_extent,
    )
    source_center = (
        generated.vertices.min(axis=0) + generated.vertices.max(axis=0)
    ) / 2.0
    raw_to_camera = generated.predicted_similarity
    if raw_to_camera is not None:
        homogeneous = np.column_stack(
            (generated.vertices, np.ones(len(generated.vertices)))
        )
        aligned_p3d = (raw_to_camera @ homogeneous.T).T[:, :3]
        rotation = raw_to_camera[:3, :3]
        uniform = float(np.cbrt(abs(np.linalg.det(rotation))))
        rotation = rotation / max(uniform, 1e-12)
        metric_to_camera = np.eye(4)
        metric_to_camera[:3, :3] = rotation
        metric_to_camera[:3, 3] = (
            raw_to_camera[:3, :3] @ source_center + raw_to_camera[:3, 3]
        )
    else:
        aligned_p3d = vertices + np.median(observation.target_points, axis=0)
        metric_to_camera = np.eye(4)
        metric_to_camera[:3, 3] = np.median(
            observation.target_points,
            axis=0,
        )
    return ScaleSolution(
        vertices=vertices,
        aligned_p3d=aligned_p3d,
        scale_factor=scale_factor,
        scale_method=scale_method,
        metric_to_camera=metric_to_camera,
        raw_to_camera=raw_to_camera,
        icp_result=None,
        quality_issues=quality_issues,
        confidence_failure=False,
    )


def _export_results(
    generated: GeneratedMesh,
    solution: ScaleSolution,
    observation: RgbdObservation,
    options: ReconstructionOptions,
) -> Path:
    raw_vertex_count = len(solution.vertices)
    raw_face_count = len(generated.faces)
    vertices, faces, vertex_colors = simplify_for_tracking(
        solution.vertices,
        generated.faces,
        generated.vertex_colors,
        options.max_faces,
    )
    metric_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    output_dir = options.data_dir / "mesh"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated.mesh.export(output_dir / "model_raw_normalized.ply")
    model_path = output_dir / "model.obj"
    metric_mesh.export(model_path)
    metric_mesh.export(output_dir / "textured_simple.obj")

    aligned_mesh = trimesh.Trimesh(
        vertices=p3d_to_realsense(solution.aligned_p3d),
        faces=generated.faces,
        vertex_colors=generated.vertex_colors,
        process=False,
    )
    aligned_mesh.export(output_dir / "model_aligned_camera.ply")
    depth_cloud = trimesh.points.PointCloud(p3d_to_realsense(observation.target_points))
    depth_cloud.export(output_dir / "depth_object.ply")
    save_registration_overlay(
        observation.image_bgr,
        observation.mask,
        solution.aligned_p3d,
        observation.intrinsics,
        output_dir / "registration_overlay.png",
    )
    np.savetxt(
        output_dir / "metric_mesh_to_camera.txt",
        solution.metric_to_camera,
    )
    if solution.raw_to_camera is not None:
        np.savetxt(
            output_dir / "raw_mesh_to_camera.txt",
            solution.raw_to_camera,
        )

    metadata = {
        "image": str(observation.image_path),
        "depth": str(observation.depth_path),
        "mask": str(observation.mask_path),
        "depth_scale": observation.depth_scale,
        "valid_masked_depth_points": len(observation.target_points),
        "depth_filter": observation.depth_filter_metadata,
        "depth_extent_m": observation.depth_extent.tolist(),
        "metric_mask_silhouette": observation.silhouette_metadata,
        "sam3d_predicted_scale": generated.predicted_scale.reshape(-1).tolist(),
        "sam3d_similarity_transform": (
            generated.predicted_similarity.tolist()
            if generated.predicted_similarity is not None
            else None
        ),
        "applied_scale_factor": solution.scale_factor,
        "scale_method": solution.scale_method,
        "sam3d_seed": options.seed,
        "scale_quality_issues": solution.quality_issues,
        "metric_scale_icp": (
            solution.icp_result.metadata() if solution.icp_result is not None else None
        ),
        "metric_mesh_to_camera": solution.metric_to_camera.tolist(),
        "raw_mesh_to_camera": (
            solution.raw_to_camera.tolist()
            if solution.raw_to_camera is not None
            else None
        ),
        "metric_mesh_extent_m": metric_mesh.extents.tolist(),
        "raw_vertices": int(raw_vertex_count),
        "raw_faces": int(raw_face_count),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"[SAM3D] mesh={model_path} ({len(vertices)} vertices, {len(faces)} faces)")
    print(f"[SAM3D] scale={solution.scale_factor:.6f} via {solution.scale_method}")
    if solution.icp_result is not None:
        metrics = solution.icp_result.metrics
        print(
            "[ICP] "
            f"accepted={solution.icp_result.accepted}, "
            f"coverage={metrics.target_coverage:.3f}, "
            f"target_rmse={metrics.target_rmse_m * 1000:.2f} mm, "
            f"rotation={solution.icp_result.rotation_source}"
        )
    return model_path


def run_reconstruction(
    options: ReconstructionOptions,
    paths: RuntimePaths | None = None,
) -> Path | None:
    """Execute one SAM3D reconstruction job and return the metric OBJ path."""
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3D inference requires a CUDA GPU")
    paths = paths or RuntimePaths.from_environment()
    _configure_sam3d_imports(paths)
    config_path = _resolve_config_path(options, paths)
    inference = _load_inference(config_path, options)
    if options.wait_for_trigger and not _wait_for_trigger():
        return None

    observation = _load_observation(options)
    generated = _run_sam3d(inference, observation, options)
    solution = _recover_scale(generated, observation, options)
    model_path = _export_results(generated, solution, observation, options)
    if solution.confidence_failure:
        raise RuntimeError(
            "Metric ICP confidence check failed after writing diagnostics: "
            + "; ".join(solution.quality_issues)
        )
    return model_path
