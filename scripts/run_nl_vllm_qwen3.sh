#!/usr/bin/env bash
# N×L grid: vLLM + Qwen3-0.6B
#
# Avoid bare `python`: after `module load mamba`, PATH may point to a non-conda
# interpreter so vllm installed in sspo is not importable.
# Default below uses sspo's absolute path; override with PYTHON if vLLM is elsewhere.
set -euo pipefail

ROOT="/scratch/ktang115/cot_factors_study"
PYTHON="${PYTHON:-/scratch/ktang115/envs/sspo/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Interpreter not found or not executable: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python3 (env with vllm and torch)" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Line-buffered output when redirecting logs (optional)
export PYTHONUNBUFFERED=1

# Fewer tokenizer multiprocessing warnings; can reduce odd stalls on some clusters
export TOKENIZERS_PARALLELISM=false

# If vLLM init or generate still hangs, try (uncomment):
# export VLLM_USE_V1=0

CONFIG="${CONFIG:-$ROOT/configs/nl_vllm_qwen3.yaml}"
EXTRA=("$@")

# Quick import check (disable with SKIP_VLLM_IMPORT_CHECK=1). First vLLM import can be slow.
if [[ "${SKIP_VLLM_IMPORT_CHECK:-0}" != "1" ]]; then
  if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
    echo "Cannot import vllm in $PYTHON." >&2
    echo "If vllm is in another env: export PYTHON=/path/to/that/bin/python" >&2
    echo "Or install: pip install vllm" >&2
    exit 1
  fi
fi

exec "$PYTHON" -m cot_factors.run \
  --config "$CONFIG" \
  --backend vllm \
  --mode nl_grid \
  "${EXTRA[@]}"
