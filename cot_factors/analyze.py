"""Aggregate summary.json from runs, or recompute metrics from predictions jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import accuracy, parse_yes_no


def summarize_jsonl(path: Path) -> Dict[str, Any]:
    gold: List[bool] = []
    pred: List[Optional[bool]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gold.append(bool(r["gold"]))
            if "pred" in r and r["pred"] is not None:
                pred.append(bool(r["pred"]))
            else:
                pred.append(parse_yes_no(r.get("raw", "")))
    rep = accuracy(gold, pred)
    return {
        "file": str(path),
        "n": rep.n,
        "correct": rep.correct,
        "accuracy": rep.accuracy,
        "parse_failed": rep.parse_failed,
    }


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Analyze predictions jsonl or a run directory")
    p.add_argument("path", type=Path, help="predictions*.jsonl or a run dir containing summary.json")
    args = p.parse_args(argv)

    path: Path = args.path
    if path.is_dir():
        summaries: List[Any] = []
        summ_file = path / "summary.json"
        if summ_file.exists():
            summaries = json.loads(summ_file.read_text(encoding="utf-8"))
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
            return
        for j in sorted(path.glob("predictions__*.jsonl")):
            summaries.append(summarize_jsonl(j))
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    print(json.dumps(summarize_jsonl(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
