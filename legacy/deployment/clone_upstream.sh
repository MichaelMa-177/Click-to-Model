#!/usr/bin/env bash
# 拉取 6 个上游仓库到 Click-to-Model 根目录下，
# 然后用 patches/ 覆盖关键定制脚本。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

clone_if_missing() {
  local dst="$1" url="$2" ref="${3:-}"
  if [ -d "$dst/.git" ]; then
    echo "  [skip] $dst 已存在"
    return
  fi
  echo "  [clone] $url -> $dst"
  git clone --depth 1 "$url" "$dst"
  if [ -n "$ref" ]; then
    git -C "$dst" fetch --depth 1 origin "$ref" || true
    git -C "$dst" checkout "$ref"
  fi
}

# 顶层 5 个
clone_if_missing segment-anything https://github.com/facebookresearch/segment-anything.git
clone_if_missing sam-3d-objects   https://github.com/facebookresearch/sam-3d-objects.git
clone_if_missing FoundationPose   https://github.com/MichaelMa-177/FoundationPose.git
clone_if_missing dinov2           https://github.com/facebookresearch/dinov2.git
clone_if_missing nvdiffrast       https://github.com/NVlabs/nvdiffrast.git

# pytorch3d 嵌套在 sam-3d-objects 下，作为子模块编译
clone_if_missing sam-3d-objects/pytorch3d https://github.com/facebookresearch/pytorch3d.git

# ----- 用 patches/ 覆盖入口脚本（含 ICP 对齐 / argparse 适配）-----
echo "  [patch] 覆盖 sam-3d-objects/run_sam3d.py"
cp "$REPO_DIR/legacy/patches/run_sam3d.py" "$REPO_DIR/sam-3d-objects/run_sam3d.py"
echo "  [patch] 覆盖 FoundationPose/run_fp.py"
cp "$REPO_DIR/legacy/patches/run_fp.py" "$REPO_DIR/FoundationPose/run_fp.py"

echo "上游 clone + 补丁覆盖完成"
