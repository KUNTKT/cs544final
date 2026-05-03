#!/usr/bin/env bash
# Check CUDA / torch with a fixed interpreter (e.g. sspo), not whatever `python` is on PATH
set -euo pipefail
PY="${PYTHON:-/scratch/ktang115/envs/sspo/bin/python}"
echo "Interpreter: $PY"
if [[ ! -x "$PY" ]]; then
  echo "Not found or not executable. Set: export PYTHON=/your/env/bin/python" >&2
  exit 1
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$PY" - <<'PY'
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("device[0]:", torch.cuda.get_device_name(0))
except ModuleNotFoundError as e:
    print("torch not installed:", e)
PY
