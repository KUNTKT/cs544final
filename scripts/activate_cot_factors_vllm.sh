#!/usr/bin/env bash
# Activate the dedicated cot_factors + vLLM environment on the cluster.
#
# Load mamba first (example):
#   module load mamba/latest
#
# Usage (from project root):
#   source ./scripts/activate_cot_factors_vllm.sh
#
# Env prefix lives on scratch (large quota); run pip only after this activation.
# Consider PYTHONNOUSERSITE below to avoid mixing ~/.local with conda.

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba not found: run  module load mamba/latest  then source this script." >&2
  return 1 2>/dev/null || exit 1
fi

ENV_PREFIX="${COT_FACTORS_VLLM_PREFIX:-/scratch/ktang115/envs/cot_factors_vllm}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Env missing or invalid: ${ENV_PREFIX}" >&2
  echo "Create once with:" >&2
  echo "  module load mamba/latest" >&2
  echo "  mamba create -p ${ENV_PREFIX} -c conda-forge python=3.11 pip setuptools wheel -y" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
# Full isolation from ~/.local: install deps into the prefix first, then uncomment:
# export PYTHONNOUSERSITE=1
# unset PIP_USER

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[cot_factors_vllm] PYTHON=${ENV_PREFIX}/bin/python"
echo "[cot_factors_vllm] PYTHONPATH=${PYTHONPATH}"

# --- One-time env creation (paste into terminal; avoid ~/.local mixing) ---
# module load mamba/latest
# mamba create -p /scratch/ktang115/envs/cot_factors_vllm -c conda-forge python=3.11 pip setuptools wheel -y
# export PYTHONNOUSERSITE=1 PIP_NO_USER=1
# /scratch/ktang115/envs/cot_factors_vllm/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# /scratch/ktang115/envs/cot_factors_vllm/bin/pip install 'vllm==0.11.2' --extra-index-url https://download.pytorch.org/whl/cu124
# /scratch/ktang115/envs/cot_factors_vllm/bin/pip install accelerate safetensors pandas pyyaml scipy
# /scratch/ktang115/envs/cot_factors_vllm/bin/pip install -e /scratch/ktang115/cot_factors_study
