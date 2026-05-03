"""McNemar test on paired outcomes for two structure conditions (same items)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from scipy.stats import mcnemar


def load_correct_by_qid(path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r["qid"])
            gold = bool(r["gold"])
            pred = r.get("pred")
            if pred is None:
                ok = False
            else:
                ok = bool(pred) == gold
            out[qid] = ok
    return out


def mcnemar_from_correct(
    a: Dict[str, bool],
    b: Dict[str, bool],
) -> Tuple[int, int, int, int, float, float]:
    """Build 2×2 table from correct/incorrect; rows=A, cols=B."""
    keys = sorted(set(a) & set(b))
    n00 = n01 = n10 = n11 = 0
    for k in keys:
        ca, cb = a[k], b[k]
        if not ca and not cb:
            n00 += 1
        elif not ca and cb:
            n01 += 1
        elif ca and not cb:
            n10 += 1
        else:
            n11 += 1
    table = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=False, correction=True)
    return n00, n01, n10, n11, float(result.statistic), float(result.pvalue)


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="McNemar: compare correctness in two predictions jsonl (paired by qid)")
    p.add_argument("jsonl_a", type=Path, help="Predictions jsonl for condition A")
    p.add_argument("jsonl_b", type=Path, help="Predictions jsonl for condition B")
    args = p.parse_args(argv)

    a = load_correct_by_qid(args.jsonl_a)
    b = load_correct_by_qid(args.jsonl_b)
    n00, n01, n10, n11, stat, pval = mcnemar_from_correct(a, b)
    print(
        json.dumps(
            {
                "contingency_A_rows_B_cols": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
                "discordant_A_wrong_B_right_n01": n01,
                "discordant_A_right_B_wrong_n10": n10,
                "mcnemar_statistic": stat,
                "p_value": pval,
                "paired_qids": len(set(a) & set(b)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
