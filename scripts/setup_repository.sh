#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the intended Conda environment before running this script." >&2
  exit 2
fi

git submodule sync --recursive
git submodule update --init --recursive
python -m pip install --no-deps --editable .

echo "Repository initialized. Next run:"
echo "  source scripts/activate_click_to_model.sh"
echo "  python scripts/doctor.py --strict"
