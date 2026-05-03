from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .config import ExperimentConfig, add_common_args, apply_cli_overrides, load_yaml
from .dataset import iter_items, load_strategyqa_json
from .metrics import accuracy, majority_vote_bool, parse_yes_no, vote_agreement
from .prompts import (
    StructureCondition,
    build_user_prompt,
    default_ablation_conditions,
    full_factorial_grid,
    nl_grid_conditions,
    validate_ids,
)


def _print_cuda_diagnostics() -> None:
    """Print whether this process can see a GPU (login nodes often have none; request GPU on Slurm)."""
    import os

    print(f"[cuda] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    try:
        import torch

        ok = torch.cuda.is_available()
        n = torch.cuda.device_count()
        print(f"[cuda] torch.cuda.is_available()={ok}, device_count={n}")
        if ok and n > 0:
            print(f"[cuda] device[0]={torch.cuda.get_device_name(0)}")
        else:
            print(
                "[cuda] No usable CUDA detected; vLLM may not use the GPU. "
                "Run on a GPU node and ensure the job allocated a GPU (e.g. Slurm --gres=gpu)."
            )
    except Exception as e:
        print(f"[cuda] Could not inspect torch/CUDA: {e}")


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_condition(
    *,
    engine: Any,
    cfg: ExperimentConfig,
    cond: StructureCondition,
    items,
    out_path: Path,
    include_facts: bool,
) -> Dict[str, Any]:
    validate_ids(cond)
    slug = cond.slug()
    gold_list: List[bool] = []
    pred_list: List[Optional[bool]] = []

    sc = cfg.self_consistency
    use_sc = sc.enabled and sc.num_samples > 1

    user_prompts = [
        build_user_prompt(
            it.question,
            cond,
            include_facts=include_facts,
            facts=it.facts if include_facts else None,
        )
        for it in items
    ]

    def _gen_kw(sample: bool = False) -> Dict[str, Any]:
        if sample:
            return {
                "max_new_tokens": cfg.generation.max_new_tokens,
                "do_sample": True,
                "temperature": sc.temperature,
                "top_p": sc.top_p,
            }
        return {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "do_sample": cfg.generation.do_sample,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
        }

    raw_texts: List[str] = []

    if use_sc:
        for it, user_prompt in zip(items, tqdm(user_prompts, desc=slug, leave=False)):
            votes: List[Optional[bool]] = []
            texts: List[str] = []
            for _ in range(sc.num_samples):
                text = engine.generate_one(user_prompt, **_gen_kw(sample=True))
                texts.append(text)
                votes.append(parse_yes_no(text))
            pred = majority_vote_bool(votes)
            agreement = vote_agreement(votes)
            raw_text = "\n---SAMPLE---\n".join(texts)
            gold = bool(it.answer)
            gold_list.append(gold)
            pred_list.append(pred)
            row = {
                "qid": it.qid,
                "condition": slug,
                "gold": gold,
                "pred": pred,
                "parse_ok": pred is not None,
                "raw": raw_text,
                "vote_agreement": agreement,
            }
            _write_jsonl(out_path, row)
    else:
        use_batch = cfg.backend == "vllm" and hasattr(engine, "generate_batch")
        if use_batch:
            raw_texts = engine.generate_batch(
                user_prompts,
                batch_size=cfg.vllm.batch_size,
                progress_desc=slug,
                **_gen_kw(sample=False),
            )
        else:
            raw_texts = []
            for user_prompt in tqdm(user_prompts, desc=slug, leave=False):
                raw_texts.append(engine.generate_one(user_prompt, **_gen_kw(sample=False)))

        for it, raw_text in zip(items, raw_texts):
            pred = parse_yes_no(raw_text)
            gold = bool(it.answer)
            gold_list.append(gold)
            pred_list.append(pred)
            row = {
                "qid": it.qid,
                "condition": slug,
                "gold": gold,
                "pred": pred,
                "parse_ok": pred is not None,
                "raw": raw_text,
            }
            _write_jsonl(out_path, row)

    rep = accuracy(gold_list, pred_list)
    summary = {
        "condition": slug,
        "n_items": rep.n,
        "correct": rep.correct,
        "accuracy": rep.accuracy,
        "parse_failed": rep.parse_failed,
    }
    return summary


def parse_conditions(args: argparse.Namespace) -> List[StructureCondition]:
    if args.mode == "single":
        if not args.condition:
            raise SystemExit("--condition is required (e.g. n_mid__l_concise__r_student__p_deductive)")
        return [StructureCondition.from_slug(args.condition)]
    if args.mode == "ablation":
        return default_ablation_conditions()
    if args.mode == "factorial":
        return full_factorial_grid()
    if args.mode == "nl_grid":
        return nl_grid_conditions()
    raise SystemExit(f"Unknown mode: {args.mode}")


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="CoT structure (N,L,R,P) experiments: StrategyQA + Hugging Face causal LM",
    )
    add_common_args(p)
    p.add_argument(
        "--mode",
        choices=["single", "ablation", "factorial", "nl_grid"],
        default="ablation",
        help="single|ablation|factorial|nl_grid (N×L only; R/P fixed to student+deductive)",
    )
    p.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Used when mode=single; format n_*__l_*__r_*__p_*",
    )
    p.add_argument(
        "--include-facts",
        action="store_true",
        help="Include StrategyQA facts in the prompt (default off to reduce confounds)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conditions and item count only; do not load the model",
    )

    args = p.parse_args(argv)

    cfg_path = args.config or Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_yaml(cfg_path)
    apply_cli_overrides(cfg, args)

    if not cfg.data.path:
        raise SystemExit("Set data JSON path in configs/default.yaml or via --data-path")

    data_path = Path(cfg.data.path)
    records = load_strategyqa_json(data_path)
    items = list(
        iter_items(
            records,
            max_items=cfg.data.max_items,
            seed=cfg.data.seed,
            shuffle=True,
        )
    )

    conditions = parse_conditions(args)
    out_root = Path(cfg.output.dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp_utc": ts,
        "config_path": str(cfg_path),
        "data_path": str(data_path),
        "num_items": len(items),
        "backend": cfg.backend,
        "model": cfg.model.name,
        "mode": args.mode,
        "conditions": [c.slug() for c in conditions],
        "include_facts": bool(args.include_facts),
        "self_consistency": cfg.self_consistency.__dict__,
    }
    if cfg.backend == "vllm":
        meta["vllm"] = dataclasses.asdict(cfg.vllm)
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[dry-run] Would run {len(conditions)} conditions on {len(items)} items.")
        return

    if cfg.backend == "vllm":
        from .inference_vllm import VLLMEngine

        _print_cuda_diagnostics()
        print(
            "Loading vLLM model (first load may take minutes: weights, kernels; GPU util can stay ~0 then; "
            "if CUDA is missing below, this node has no GPU—run inside a GPU job)."
        )
        print(f"Loading vLLM model: {cfg.model.name} …")
        engine = VLLMEngine.from_pretrained(
            cfg.model.name,
            vllm_cfg=cfg.vllm,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        print("vLLM engine ready; inference uses GPU when CUDA is available.")
    else:
        from .inference import HFEngine

        print(f"Loading HF model: {cfg.model.name} …")
        engine = HFEngine.from_pretrained(
            cfg.model.name,
            torch_dtype=cfg.model.torch_dtype,
            device_map=cfg.model.device_map,
            trust_remote_code=cfg.model.trust_remote_code,
        )

    summaries: List[Dict[str, Any]] = []
    n_cond = len(conditions)
    n_items = len(items)
    sc = cfg.self_consistency
    per_item_calls = sc.num_samples if (sc.enabled and sc.num_samples > 1) else 1
    total_prompts = n_cond * n_items * per_item_calls

    cond_bar = tqdm(
        conditions,
        desc="overall",
        unit="cond",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} cond [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for cond in cond_bar:
        slug = cond.slug()
        cond_bar.set_postfix(
            slug=(slug[:48] + "…") if len(slug) > 48 else slug,
            refresh=False,
        )
        out_jsonl = run_dir / f"predictions__{slug}.jsonl"
        if out_jsonl.exists():
            out_jsonl.unlink()
        summ = run_condition(
            engine=engine,
            cfg=cfg,
            cond=cond,
            items=items,
            out_path=out_jsonl,
            include_facts=args.include_facts,
        )
        summaries.append(summ)
        tqdm.write(json.dumps(summ, ensure_ascii=False))

    tqdm.write(
        f"{n_cond} conditions × {n_items} items"
        + (f" × {per_item_calls} samples" if per_item_calls > 1 else "")
        + f" ≈ {total_prompts} generations."
    )

    (run_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tqdm.write(f"Done. Output directory: {run_dir}")


if __name__ == "__main__":
    main()
