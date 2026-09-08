#!/usr/bin/env bash
# 创建两个 conda env：sam3d-objects (Python 3.11) + foundationpose (Python 3.9)
# 然后在 sam3d-objects env 里以 develop 模式编译安装 pytorch3d。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 找不到 conda。先安装 Miniforge / Miniconda 并 source 它的初始化脚本。"
  exit 1
fi

# 让脚本里能用 conda activate
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

create_env() {
  local name="$1" yml="$2"
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "  [skip] env $name 已存在"
  else
    echo "  [create] $name from $yml"
    conda env create -f "$yml"
  fi
}

create_env sam3d-objects "$REPO_DIR/legacy/environments/sam3d-objects.yml"
create_env foundationpose "$REPO_DIR/legacy/environments/foundationpose.yml"

# pytorch3d 用源码 develop install (sam3d-objects env)
echo "  [build] pytorch3d (sam-3d-objects/pytorch3d, develop install)"
echo "         首次编译 C++/CUDA 扩展需要 5-15 分钟，且需要 nvcc 12.1"
conda activate sam3d-objects
cd "$REPO_DIR/sam-3d-objects/pytorch3d"
pip install -e . --no-build-isolation
conda deactivate

# pyrealsense2 (demo 主进程用)
echo "  [pip] pyrealsense2 (sam-3d-objects)"
conda activate sam3d-objects
pip install pyrealsense2
conda deactivate

echo "conda env 安装完成"
