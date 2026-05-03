#!/usr/bin/env bash
# Full factorial + StrategyQA (see max_items in configs/vllm_qwen25_7b_factorial_full.yaml) + vLLM + Qwen2.5-7B-Instruct
set -euo pipefail

ROOT="/scratch/ktang115/cot_factors_study"
PYTHON="${PYTHON:-/scratch/ktang115/envs/cot_factors_vllm/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/vllm_qwen25_7b_factorial_full.yaml}"

cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m cot_factors.run \
  --config "$CONFIG" \
  --backend vllm \
  --mode factorial \
  "$@"
