# CoT Structural Factors Experiment (N, L, R, P)

On a fixed LLM and **StrategyQA**, we manipulate **step count N, verbosity per step L, role R, and perspective P** via prompts, and measure QA accuracy and (optional) self-consistency. The design matches [`docs/RESEARCH_NOTES_CoT_STRUCTURE.md`](./docs/RESEARCH_NOTES_CoT_STRUCTURE.md) and the course proposal PDF.

## Environment

- Python ≥ 3.9; install CUDA-matched PyTorch from [pytorch.org](https://pytorch.org/).
- Install dependencies:

```bash
cd /scratch/ktang115/cot_factors_study
pip install -e .
# or: pip install -r requirements.txt
```

## Configuration

Edit `configs/default.yaml`:

- `data.path`: StrategyQA JSON (array of objects with `question`, boolean `answer`, `facts`, etc.).
- `model.name`: e.g. `Qwen/Qwen2.5-7B-Instruct` or a local path.
- `data.max_items`: `null` for full data; use `50` for quick debugging.
- `self_consistency`: set `enabled: true` for multi-sample voting.

## Running experiments

From the project root (set `PYTHONPATH`, or use `pip install -e .`):

```bash
export PYTHONPATH=/scratch/ktang115/cot_factors_study

# Dry-run: metadata and condition list only
python3 -m cot_factors.run --dry-run

# Default: ablation (baseline + one-factor sweeps), ~8 conditions
python3 -m cot_factors.run --config configs/default.yaml

# Single condition (slug: four segments joined by __)
python3 -m cot_factors.run --mode single \
  --condition n_mid__l_concise__r_student__p_deductive \
  --max-items 100

# Full factorial grid (3×2×3×2 = 36 conditions; long runtime)
python3 -m cot_factors.run --mode factorial --max-items 50

# Optionally include StrategyQA facts in the prompt
python3 -m cot_factors.run --include-facts
```

### vLLM: N×L grid + Qwen3-0.6B (same env pattern as SSPO)

Requires **vLLM** matched to your CUDA. Script **`run_nl_vllm_qwen3.sh`** defaults to `/scratch/ktang115/envs/sspo/bin/python` so that after `module load mamba`, the `python` on `PATH` does not shadow the conda env where **vllm** is installed. If vLLM lives elsewhere:

```bash
export PYTHON=/path/to/that/env/bin/python
./scripts/run_nl_vllm_qwen3.sh
```

You can also follow [`SSPO/README_SSPOLTO.md`](../SSPO/README_SSPOLTO.md) to activate `sspo`, but run experiments with the script or an explicit Python path:

```bash
cd /scratch/ktang115/cot_factors_study
export PYTHONPATH=/scratch/ktang115/cot_factors_study

# N×L only: 6 conditions (3 step counts × 2 verbosity levels), R=r_student, P=p_deductive fixed
./scripts/run_nl_vllm_qwen3.sh

# Or pass config explicitly; example item cap: --max-items 50
python3 -m cot_factors.run --config configs/nl_vllm_qwen3.yaml --backend vllm --mode nl_grid --max-items 50
```

Config: `configs/nl_vllm_qwen3.yaml` (`backend: vllm`, default model `/scratch/ktang115/models/Qwen3-0.6B`; **default `data.max_items: 500`**: 500 items per N×L condition → 6×500 = 3000 generations; set `null` for full train split). Batch size: `vllm.batch_size`.

**GPU**: vLLM runs on **CUDA**; there is no CPU-only path. On login nodes or jobs without a GPU, `CUDA_VISIBLE_DEVICES` is often empty and `torch.cuda.is_available()` is false. Use your cluster GPU queue (e.g. Slurm `--gres=gpu:1`) or an interactive session with a GPU. Startup prints `[cuda] ...` for a quick check.

**Sanity check**: bare `python` on clusters may not be the conda interpreter (`No module named 'torch'`). Use a fixed path or `export PYTHON=...`:

```bash
./scripts/check_gpu_env.sh
# or
/scratch/ktang115/envs/sspo/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

First load can show **~0% GPU utilization** for a while (disk IO, compilation); that is normal.

**If logs stall**: (1) `tail -f outputs/run_nl500_latest.log` for `Processed prompts`; (2) `configs/nl_vllm_qwen3.yaml` sets **`enforce_eager: true`** by default (slower but more stable); (3) uncomment **`export VLLM_USE_V1=0`** in `run_nl_vllm_qwen3.sh`; (4) use `tmux`/`sbatch` for long jobs.

Outputs: `outputs/run_<UTC>/` with per-sample `predictions__<condition>.jsonl` and `summary.json`.

## Analysis and paired tests

```bash
python3 -m cot_factors.analyze outputs/run_xxx

# McNemar (two prediction jsonl files, paired by qid)
python3 -m cot_factors.compare \
  outputs/run_xxx/predictions__n_mid__l_concise__r_student__p_deductive.jsonl \
  outputs/run_xxx/predictions__n_high__l_concise__r_student__p_deductive.jsonl
```

## Project layout

| Path | Role |
|------|------|
| `cot_factors/prompts.py` | N/L/R/P levels and `StructureCondition` |
| `cot_factors/dataset.py` | Load StrategyQA and subsample |
| `cot_factors/inference.py` | Hugging Face generation and chat template |
| `cot_factors/metrics.py` | Parse `Final answer: Yes/No`, accuracy |
| `cot_factors/run.py` | Main CLI |
| `cot_factors/compare.py` | McNemar paired test |
| `configs/` | YAML experiment configs |
| `scripts/` | vLLM runners, GPU check, run monitor |
| `docs/RESEARCH_NOTES_CoT_STRUCTURE.md` | Research questions and variable design |

## Condition ID cheat sheet

- **N**: `n_low`, `n_mid`, `n_high`
- **L**: `l_concise`, `l_detailed`
- **R**: `r_logician`, `r_student`, `r_expert`
- **P**: `p_deductive`, `p_analytical`

Example single-condition slug: `n_mid__l_concise__r_student__p_deductive`.
