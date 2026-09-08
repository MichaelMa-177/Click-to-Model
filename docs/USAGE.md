# Usage

## Environment

```bash
cd Click-to-Model
conda activate foundationpose
source scripts/activate_click_to_model.sh
```

The activation script reuses the configured FoundationPose environment and
sets paths for SPARK-6D, SAM2, SAM3D, DINOv2, and Hugging Face caches.

## Online pipeline

```bash
python -m click_to_model \
  --online \
  --distilled \
  --scale-mode icp \
  --require-icp-confidence \
  --save-tracking-video
```

The first connected RealSense is selected by default. Add
`--camera-serial <SERIAL>` only when a particular device must be selected.

Controls:

- Camera window: `s` captures; `q` or Escape cancels.
- SAM2 window: left click adds an object point, right click adds a background
  point, `u` undoes, `c` clears, and Enter/Space accepts.

Use `--camera-fps 30` if USB bandwidth cannot sustain the D415's configured
`640x480@60` stream.

## Existing sequence

```bash
python -m click_to_model \
  --data-dir /path/to/sequence \
  --no-capture --no-gui \
  --mask-file /path/to/first_frame_mask.png
```

An interrupted online initialization can be resumed from the archived
`masks/rgb/` and `masks/depth/` pair.

## Reconstruction only

```bash
python -m click_to_model.reconstruction.cli \
  --data-dir /path/to/sequence \
  --scale-mode icp \
  --require-icp-confidence
```

Useful scale controls:

- `--icp-voxel-size 0.003`: depth-cloud voxel size in metres.
- `--icp-samples 20000`: generated-mesh surface samples.
- `--icp-iterations 60`: maximum rigid ICP iterations per candidate.
- `--icp-mask-erode 3`: mask erosion to reject boundary background.
- `--min-depth-coverage 0.25`: minimum valid-depth fraction inside the mask.
- `--allow-border-mask`: accept an object clipped by the image boundary; this
  weakens scale reliability.
- `--scale-mode sam3d|extent`: ablation fallbacks without bounded scale ICP.

## Tracking backends

SPARK-6D is the default. Use `--tracker foundationpose` for the frame-by-frame
baseline. Compatibility aliases such as `--live-foundationpose` and
`--skip-foundationpose` remain accepted.

The online benchmark separates CUDA-only tracking FPS, steady end-to-end FPS,
and total FPS including first-frame registration. Per-frame rendering is off by
default; use `--save-tracker-render` when visual artifacts are needed.

## Output sequence

```text
sequence/
├── rgb/ and depth/                 # frames consumed by the tracker
├── masks/000000.png                # first-frame segmentation
├── masks/rgb/ and masks/depth/     # archived initialization RGB-D
├── mesh/model.obj                  # centered metric mesh
├── mesh/model_metadata.json        # scale and registration diagnostics
├── mesh/registration_overlay.png
├── debug/ob_in_cam/                # per-frame poses
├── debug/*benchmark.json
├── debug/tracking.mp4              # when requested
├── cam_K.txt
└── camera_params.json
```

## Checks

```bash
bash scripts/check.sh
```

Camera and full GPU inference are intentionally excluded from the fast unit
suite. They should be exercised as explicit hardware integration checks.
