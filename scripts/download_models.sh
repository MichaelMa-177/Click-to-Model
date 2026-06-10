#!/usr/bin/env bash
# 下载所有模型权重到正确路径。
#
# 资产清单:
#   1) SAM ViT-H            -> segment-anything/checkpoints/sam_vit_h_4b8939.pth
#   2) SAM-3D HF checkpoints -> sam-3d-objects/checkpoints/hf/
#   3) DinoV2 ViT-L/14 reg4 -> sam-3d-objects/checkpoints/dino_checkpoints/
#   4) FoundationPose       -> FoundationPose/weights/{2023-10-28-18-33-37,2024-01-11-20-02-45}/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

WGET() { wget --no-check-certificate -c "$@"; }

# -----------------------------------------------------------
# 1) SAM ViT-H
# -----------------------------------------------------------
echo "[1/4] SAM ViT-H ..."
SAM_DIR="$REPO_DIR/segment-anything/checkpoints"
mkdir -p "$SAM_DIR"
if [ ! -f "$SAM_DIR/sam_vit_h_4b8939.pth" ]; then
  WGET -P "$SAM_DIR" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
  echo "  [skip] sam_vit_h_4b8939.pth 已存在"
fi

# -----------------------------------------------------------
# 2) SAM-3D HuggingFace checkpoints
#    facebook/sam-3d-objects (Hugging Face Hub)
# -----------------------------------------------------------
echo "[2/4] SAM-3D HF checkpoints ..."
SAM3D_HF_DIR="$REPO_DIR/sam-3d-objects/checkpoints/hf"
mkdir -p "$SAM3D_HF_DIR"
HF_REPO="${SAM3D_HF_REPO:-facebook/sam-3d-objects}"
if [ ! -f "$SAM3D_HF_DIR/pipeline.yaml" ]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_REPO" --local-dir "$SAM3D_HF_DIR" --local-dir-use-symlinks False
  else
    echo "  ⚠️  huggingface-cli 未安装。先 pip install huggingface_hub，然后重跑本步:"
    echo "      pip install huggingface_hub"
    echo "      huggingface-cli download $HF_REPO --local-dir $SAM3D_HF_DIR --local-dir-use-symlinks False"
  fi
else
  echo "  [skip] SAM-3D checkpoints 已存在"
fi

# -----------------------------------------------------------
# 3) DinoV2 ViT-L/14 with registers
# -----------------------------------------------------------
echo "[3/4] DinoV2 ViT-L/14 reg4 ..."
DINO_DIR="$REPO_DIR/sam-3d-objects/checkpoints/dino_checkpoints"
mkdir -p "$DINO_DIR"
if [ ! -f "$DINO_DIR/dinov2_vitl14_reg4_pretrain.pth" ] && \
   [ ! -f "$DINO_DIR/dinov2_vitl14_reg4.pth" ]; then
  WGET -O "$DINO_DIR/dinov2_vitl14_reg4_pretrain.pth" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
else
  echo "  [skip] DinoV2 reg4 已存在"
fi

# -----------------------------------------------------------
# 4) FoundationPose Refiner + Scorer (Google Drive)
# -----------------------------------------------------------
echo "[4/4] FoundationPose 权重 (Google Drive) ..."
FP_WEIGHTS="$REPO_DIR/FoundationPose/weights"
mkdir -p "$FP_WEIGHTS"

if ! command -v gdown >/dev/null 2>&1; then
  echo "  ⚠️  gdown 未安装。先安装:  pip install gdown"
  echo "  或手动下载并解压到 $FP_WEIGHTS/"
  echo "    folder: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
else
  if [ ! -d "$FP_WEIGHTS/2023-10-28-18-33-37" ] || \
     [ ! -d "$FP_WEIGHTS/2024-01-11-20-02-45" ]; then
    gdown --folder \
      "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" \
      -O "$FP_WEIGHTS"
  else
    echo "  [skip] FoundationPose 权重已存在"
  fi
fi

echo "所有权重下载流程执行完毕"
