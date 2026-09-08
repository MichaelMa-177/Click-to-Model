#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python -B -m unittest discover -s tests/unit -t . -v
python -B - <<'PY'
import ast
from pathlib import Path

for root in (Path("click_to_model"), Path("tests"), Path("tools"), Path("scripts")):
    for path in sorted(root.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in (Path("run_click_to_model.py"), Path("run_sam3d_rgbd.py"), Path("metric_scale_icp.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checks passed")
PY

for script in scripts/*.sh; do
  bash -n "${script}"
done
echo "Repository checks passed"
