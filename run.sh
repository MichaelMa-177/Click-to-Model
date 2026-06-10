#!/usr/bin/env bash
set -e

# =========================
# 基础路径
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAM_DIR="${SCRIPT_DIR}/sam-3d-objects"
FP_DIR="${SCRIPT_DIR}/FoundationPose"

# Python 解释器：默认使用本机 conda env，可被环境变量覆盖
SAM_PY="${SAM_PY:-/data/ubuntu_data/miniconda3/envs/sam3d-objects/bin/python}"
FP_PY="${FP_PY:-/data/ubuntu_data/miniconda3/envs/foundationpose/bin/python}"

# =========================
# 自动查找最大编号数据目录
# 默认 ${SCRIPT_DIR}/data_online，可被 DATA_ROOT 环境变量覆盖
# =========================
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data_online}"

# -L 让 find 跟随软链接（DATA_ROOT 可能是软链）
LATEST_ID=$(find -L "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
  | grep -E '^[0-9]+$' \
  | sort -n \
  | tail -1)

# 允许 DATA_ID 环境变量强制覆盖（用于不同结构数据集切换）
LATEST_ID="${DATA_ID:-$LATEST_ID}"

if [ -z "$LATEST_ID" ]; then
  echo "❌ No numeric data folder found in ${DATA_ROOT}"
  exit 1
fi

DATA_DIR="${DATA_ROOT}/${LATEST_ID}"
echo "✅ Using data directory: ${DATA_DIR}"

# =========================
# 子路径（统一接口）
# =========================
RGB_DIR="${DATA_DIR}/rgb"
DEPTH_DIR="${DATA_DIR}/depth"
MESH_DIR="${DATA_DIR}/mesh"
DEBUG_DIR="${DATA_DIR}/debug"
MASK_DIR="${DATA_DIR}/masks"
# =========================
# Stage 1: SAM-3D
# =========================
echo "[1/2] Running SAM-3D stage..."
cd "$SAM_DIR"
$SAM_PY run_sam3d.py \
  --data_dir "$DATA_DIR"

# =========================
# Stage 2: FoundationPose
# =========================
echo "[2/2] Running FoundationPose stage..."
cd "$FP_DIR"
$FP_PY run_fp.py \
  --test_scene_dir "$DATA_DIR" \
  --mesh_file "$MESH_DIR/model.obj" \
  --debug_dir "$DEBUG_DIR" \
  --debug 3

echo "✅ Pipeline finished."
