#!/usr/bin/env bash

_click_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_workspace_root="$(cd "${_click_repo}/.." && pwd)"
_spark_repo="${SPARK6D_REPO:-${FOUNDATIONPOSE_REPO:-${_workspace_root}/SPARK-6D}}"
_click_previous_dir="${PWD}"
_environment_prefix="${CLICK_TO_MODEL_ENV_PREFIX:-${CONDA_PREFIX:-}}"
_cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}"
_pip_cache="${PIP_CACHE_DIR:-${_cache_root}/pip}"
_hf_cache="${HF_HOME:-${_cache_root}/huggingface}"
_mpl_cache="${MPLCONFIGDIR:-${_cache_root}/matplotlib}"
_torchinductor_cache="${TORCHINDUCTOR_CACHE_DIR:-${_cache_root}/torchinductor}"

if [[ ! -f "${_spark_repo}/scripts/activate_foundationpose.sh" ]]; then
  echo "SPARK-6D not found at ${_spark_repo}. Set SPARK6D_REPO first." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -z "${_environment_prefix}" || ! -x "${_environment_prefix}/bin/python" ]]; then
  echo "Activate the Click-to-Model Conda environment, or set CLICK_TO_MODEL_ENV_PREFIX." >&2
  return 1 2>/dev/null || exit 1
fi

export FOUNDATIONPOSE_ENV_PREFIX="${FOUNDATIONPOSE_ENV_PREFIX:-${_environment_prefix}}"
export FOUNDATIONPOSE_REPO_ROOT="${FOUNDATIONPOSE_REPO_ROOT:-${_spark_repo}}"
source "${_spark_repo}/scripts/activate_foundationpose.sh"
cd "${_click_previous_dir}"
_active_python="$(command -v python)"

export CLICK_TO_MODEL_ROOT="${_click_repo}"
export SPARK6D_REPO="${_spark_repo}"
export FOUNDATIONPOSE_REPO="${_spark_repo}"
export SPARK6D_PY="${SPARK6D_PY:-${FP_PY:-${_active_python}}}"
export FP_PY="${SPARK6D_PY}"
export SAM3D_REPO="${_click_repo}/sam-3d-objects"
export SAM2_REPO="${SAM2_REPO:-${_spark_repo}/third_party/sam2}"
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2_REPO}/checkpoints/sam2.1_hiera_small.pt}"
export SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-${SAM3D_REPO}/checkpoints/hf}"
export SAM3D_DINO_REPO="${SAM3D_DINO_REPO:-${_click_repo}/dinov2}"
export SAM3D_DINO_WEIGHTS="${SAM3D_DINO_WEIGHTS:-${SAM3D_REPO}/checkpoints/dino_checkpoints/dinov2_vitl14_reg4.pth}"
export SAM3D_DINO_SKIP_WEIGHTS="${SAM3D_DINO_SKIP_WEIGHTS:-1}"
export LIDRA_SKIP_INIT=true
export XFORMERS_DISABLED=1
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SPARSE_BACKEND=spconv
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${_cache_root}"
export PIP_CACHE_DIR="${_pip_cache}"
export HF_HOME="${_hf_cache}"
export MPLCONFIGDIR="${_mpl_cache}"
export TORCHINDUCTOR_CACHE_DIR="${_torchinductor_cache}"
export PYTHONPATH="${CLICK_TO_MODEL_ROOT}:${SAM3D_REPO}:${SAM3D_REPO}/notebook:${SAM2_REPO}:${PYTHONPATH:-}"

unset _click_repo _workspace_root _spark_repo _click_previous_dir
unset _environment_prefix _active_python _cache_root _pip_cache _hf_cache
unset _mpl_cache _torchinductor_cache
