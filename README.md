# CoT factors (N, L, R, P) on StrategyQA

Vary prompt structure (steps, verbosity, role, perspective) with **Qwen2.x 7B instruct** (default `Qwen/Qwen2.5-7B-Instruct` in `configs/default.yaml`; set `model.name` to your path or `Qwen/Qwen2-7B-Instruct` if needed). More background: [`docs/RESEARCH_NOTES_CoT_STRUCTURE.md`](./docs/RESEARCH_NOTES_CoT_STRUCTURE.md).

## 1. Install

```bash
cd /path/to/cot_factors_study
pip install -e .
```

Use a CUDA build of PyTorch if you run on GPU ([pytorch.org](https://pytorch.org/)).

## 2. Config

Edit **`configs/default.yaml`** (or pass flags):

- **`data.path`** — StrategyQA JSON (list of items with `question`, boolean `answer`, optional `facts`)
- **`model.name`** — Hub id or local model folder
- **`data.max_items`** — `null` = all; use a small number to debug

## 3. Run

```bash
export PYTHONPATH=/path/to/cot_factors_study   # not needed if you used pip install -e .

# List conditions and count only (no model load)
python3 -m cot_factors.run --dry-run

# Ablation (~8 conditions), default config
python3 -m cot_factors.run --config configs/default.yaml

# One condition; slug = n_*__l_*__r_*__p_*
python3 -m cot_factors.run --mode single --condition n_mid__l_concise__r_student__p_deductive --max-items 100

# Full grid 3×2×3×2 = 36 conditions
python3 -m cot_factors.run --mode factorial --max-items 50

# Optional: add facts to the prompt
python3 -m cot_factors.run --include-facts
```

**vLLM (optional):** install vLLM, set `data.path` / `model.name` in a yaml, then e.g.  
`python3 -m cot_factors.run --config configs/vllm_qwen25_7b_factorial_full.yaml --backend vllm --mode factorial`  
or **`./scripts/run_factorial_qwen25_vllm_full.sh`**. For a small N×L-only test: **`./scripts/run_nl_vllm_qwen3.sh`** (uses its own yaml; override `PYTHON` if vLLM is in another env).

Results go to **`outputs/run_<timestamp>/`** (`predictions__*.jsonl`, `summary.json`).

## 4. Analyze

```bash
python3 -m cot_factors.analyze outputs/run_xxx
python3 -m cot_factors.compare path/to/predictions_A.jsonl path/to/predictions_B.jsonl
```

**Condition ids:** N = `n_low` / `n_mid` / `n_high`; L = `l_concise` / `l_detailed`; R = `r_logician` / `r_student` / `r_expert`; P = `p_deductive` / `p_analytical`.
