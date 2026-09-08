# Architecture

Click-to-Model is organized as a thin orchestration layer around four isolated
stages. Heavy third-party projects stay outside the application package and are
addressed through `RuntimePaths`.

```text
RealSense capture
      │ aligned RGB-D + intrinsics
      ▼
interactive SAM2 ──► first-frame mask
      │
      ▼
SAM3D mesh generation
      │ normalized mesh + predicted Sim(3)
      ▼
bounded metric-scale ICP
      │ centered metric OBJ
      ▼
SPARK-6D / FoundationPose tracking
```

## Source layout

```text
click_to_model/
├── config.py                  # environment-derived external paths
├── dataset.py                 # sequence allocation and RGB-D archiving
├── camera.py                  # RealSense capture and calibration output
├── segmentation.py            # stateful SAM2 session and click UI
├── processes.py               # one-shot/preloaded worker lifecycle
├── tracking.py                # live/offline tracker command construction
├── pipeline.py                # end-to-end orchestration CLI
└── reconstruction/
    ├── cli.py                 # reconstruction-only CLI
    ├── geometry.py            # RGB-D projection, transforms, export helpers
    ├── metric_scale.py        # bounded scale search and rigid ICP
    └── pipeline.py            # SAM3D inference and metric mesh export
```

The three Python files in the repository root are compatibility shims. New code
should import from `click_to_model.*` and invoke `python -m click_to_model`.

## Dependency boundaries

- `config`, `dataset`, and `processes` contain no GPU-model imports.
- RealSense is imported lazily by `camera.capture_realsense`.
- SAM2 is imported only when `Sam2Session.load` is called.
- SAM3D is executed in a separate process so its memory can be reclaimed before
  tracking.
- Registration is deterministic for a fixed mesh, point cloud, and seed.
- SPARK-6D/FoundationPose remain external repositories and are not imported by
  the core package.

CUDA-specific PyTorch, SAM2, SAM3D, and SPARK-6D are supplied by the activated
research environment rather than installed implicitly by `pyproject.toml`.

## Coordinate and unit conventions

- Depth files are RealSense `z16`; `camera_params.json.depth_scale` converts
  them to metres.
- RGB and depth images are aligned and share `cam_K.txt`.
- SAM3D point maps use PyTorch3D camera axes (`-x`, `-y`, `+z` relative to
  RealSense/OpenCV).
- `mesh/model.obj` is centered in object-local coordinates and measured in
  metres.
- `mesh/model_aligned_camera.ply` is diagnostic geometry in RealSense/OpenCV
  camera coordinates.

## Compatibility and legacy code

- `legacy/` contains the first-generation two-environment implementation and
  patch-copy deployment scripts. It is retained for provenance, not active use.
- `sam-3d-objects`, `FoundationPose`, and `segment-anything` are third-party
  submodules. Local changes inside them are independent of the application
  package.
- `dinov2/` and model checkpoints are local heavy assets and stay ignored.
