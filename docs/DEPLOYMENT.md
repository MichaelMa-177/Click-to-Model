# Deployment and reproducibility

This repository stores source code and Git submodule revisions. Model weights,
captured RGB-D data, device serial numbers, credentials, compiled extensions,
and machine-specific paths are intentionally excluded.

## 1. System prerequisites

- Linux x86-64 and an NVIDIA GPU with a recent CUDA-capable driver.
- Conda or Mamba, Git, a C/C++ compiler, CMake, and Ninja.
- Intel RealSense D415 plus USB 3.x for the online path.
- Hugging Face access approval for `facebook/sam-3d-objects`.

The staged mesh-only SAM3D path has been smoke-tested on an RTX 4070 SUPER with
12 GB VRAM, Python 3.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124,
PyTorch3D 0.7.9, nvdiffrast 0.4.0, and Kaolin 0.17.0. This is a verified
configuration, not a universal hardware guarantee.

## 2. Clone the repositories

Keep both repositories in the same workspace to use the default relative path:

```bash
mkdir click-to-model-workspace
cd click-to-model-workspace
git clone https://github.com/MichaelMa-177/SPARK-6D.git
git clone --recurse-submodules \
  https://github.com/MichaelMa-177/Click-to-Model.git
git -C SPARK-6D checkout ad4f258e00c6d2f729b6c01df12d7ec6abbfb221
cd Click-to-Model
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Do not download the repository as a ZIP: ZIP archives do not preserve Git
submodule revisions.

## 3. Build one CUDA environment

Create the FoundationPose environment from the adjacent SPARK-6D repository,
then install a CUDA build of PyTorch compatible with the local driver. The
verified profile uses:

```bash
conda env create --file ../SPARK-6D/environment.yml
conda activate foundationpose

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --requirement ../SPARK-6D/requirements.txt

cd ../SPARK-6D
bash build_all_conda.sh
cd ../Click-to-Model
```

Install SAM3D's Python and CUDA dependencies in the same environment by
following [`sam-3d-objects/doc/setup.md`](../sam-3d-objects/doc/setup.md).
When retaining the verified PyTorch 2.6/cu124 stack, build PyTorch3D and
nvdiffrast against that active environment and install the SAM3D checkout
editable. The low-VRAM code path does not require xFormers or FlashAttention:

```bash
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'
python -m pip install --no-deps --editable ./sam-3d-objects
python -m pip install --editable '.[camera,dev]'
```

SAM3D has a broad research dependency set. If an import is still missing, use
its pinned requirement files rather than guessing a version, but prevent pip
from replacing the already selected PyTorch/CUDA build.

## 4. Download model sources and weights

Clone DINOv2 source locally; the enclosing SAM3D checkpoint supplies its DINO
weights in the default configuration:

```bash
git clone https://github.com/facebookresearch/dinov2.git ./dinov2
git -C ./dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

The SAM3D, FoundationPose, and Segment Anything revisions are pinned by this
repository's Git submodule entries. The explicit SPARK-6D and DINOv2 checkouts
above pin the two adjacent/local-source dependencies that are not submodules.

After `hf auth login` and access approval, use the Hugging Face CLI. Re-running
the command resumes verified cached chunks:

```bash
hf download facebook/sam-3d-objects \
  --repo-type model \
  --local-dir ./sam-3d-objects/checkpoints/hf-download \
  --max-workers 4

mv ./sam-3d-objects/checkpoints/hf-download/checkpoints \
  ./sam-3d-objects/checkpoints/hf
```

Install SAM2 under `../SPARK-6D/third_party/sam2` and place
`sam2.1_hiera_small.pt` in its `checkpoints/` directory. Download the two
FoundationPose network checkpoints according to the SPARK-6D README. The final
layout relevant to this application is:

```text
workspace/
├── SPARK-6D/
│   ├── weights/
│   └── third_party/sam2/
│       ├── sam2/
│       └── checkpoints/sam2.1_hiera_small.pt
└── Click-to-Model/
    ├── dinov2/
    └── sam-3d-objects/checkpoints/hf/pipeline.yaml
```

## 5. Activate and validate

```bash
conda activate foundationpose
source scripts/activate_click_to_model.sh
python scripts/doctor.py --strict --camera
bash scripts/check.sh
```

The activation script derives every default from the checkout and active Conda
environment. For a different directory layout, copy `.env.example`, adjust the
variables, source it, and then source the activation script.

## 6. Run

List command-line options without opening hardware:

```bash
python -m click_to_model --help
```

Run the complete online pipeline using the first connected RealSense:

```bash
python -m click_to_model \
  --online \
  --distilled \
  --scale-mode icp \
  --require-icp-confidence \
  --save-tracking-video
```

For multiple cameras, append `--camera-serial <SERIAL>`. To validate SAM3D and
metric registration without the camera or tracker:

```bash
python -m click_to_model.reconstruction.cli \
  --data-dir /path/to/foundationpose_sequence \
  --distilled \
  --scale-mode icp \
  --require-icp-confidence
```

Captured sequences, weights, caches, meshes, videos, and benchmark output are
ignored by Git. See [USAGE.md](USAGE.md) for the dataset contract and runtime
controls.
