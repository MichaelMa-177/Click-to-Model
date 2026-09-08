#!/usr/bin/env bash
# 一键部署入口：
#   1) clone 6 个上游仓库到 Click-to-Model 下
#   2) 用 patches/ 覆盖定制脚本
#   3) 创建两个 conda env
#   4) 下载所有权重
#
# 用法:
#   bash scripts/setup.sh                # 全跑
#   SKIP_ENVS=1 bash scripts/setup.sh    # 跳过 conda env 创建
#   SKIP_MODELS=1 bash scripts/setup.sh  # 跳过权重下载
#   SKIP_CLONE=1 bash scripts/setup.sh   # 跳过 clone 上游
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==================================================="
echo " Click-to-Model 部署"
echo " REPO_DIR = $REPO_DIR"
echo "==================================================="

if [ "${SKIP_CLONE:-0}" != "1" ]; then
  echo "[1/4] 拉上游仓库 ..."
  bash "$SCRIPT_DIR/clone_upstream.sh"
else
  echo "[1/4] SKIP_CLONE=1, 跳过 clone"
fi

if [ "${SKIP_ENVS:-0}" != "1" ]; then
  echo "[2/4] 创建 conda env ..."
  bash "$SCRIPT_DIR/install_envs.sh"
else
  echo "[2/4] SKIP_ENVS=1, 跳过 conda env"
fi

if [ "${SKIP_MODELS:-0}" != "1" ]; then
  echo "[3/4] 下载模型权重 ..."
  bash "$SCRIPT_DIR/download_models.sh"
else
  echo "[3/4] SKIP_MODELS=1, 跳过权重下载"
fi

echo "[4/4] 完成。运行示例:"
echo "  cd $REPO_DIR && source scripts/activate_click_to_model.sh"
echo "  python -m click_to_model --help"
