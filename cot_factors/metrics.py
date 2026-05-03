from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


_FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*:\s*(yes|no|true|false)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse final Yes/No from model output; prefers a 'Final answer:' line."""
    if not text or not text.strip():
        return None
    m = list(_FINAL_ANSWER_RE.finditer(text))
    if m:
        token = m[-1].group(1).lower()
        if token in ("yes", "true"):
            return True
        if token in ("no", "false"):
            return False

    # Fallback: last yes/no token in the full text (weak heuristic)
    lower = text.lower()
    matches = list(re.finditer(r"\b(yes|no)\b", lower))
    if matches:
        tok = matches[-1].group(1)
        return tok == "yes"
    return None


@dataclass
class AccuracyReport:
    n: int
    correct: int
    accuracy: float
    parse_failed: int

    def __str__(self) -> str:
        return (
            f"n={self.n} correct={self.correct} acc={self.accuracy:.4f} parse_failed={self.parse_failed}"
        )


def accuracy(
    gold: Sequence[bool],
    pred: Sequence[Optional[bool]],
) -> AccuracyReport:
    if len(gold) != len(pred):
        raise ValueError("gold and pred length mismatch")
    n = len(gold)
    ok = 0
    bad = 0
    for g, p in zip(gold, pred):
        if p is None:
            bad += 1
            continue
        if bool(g) == bool(p):
            ok += 1
    denom = n - bad
    acc = ok / denom if denom else 0.0
    return AccuracyReport(n=n, correct=ok, accuracy=acc, parse_failed=bad)


def majority_vote_bool(votes: List[Optional[bool]]) -> Optional[bool]:
    ys = sum(1 for v in votes if v is True)
    ns = sum(1 for v in votes if v is False)
    if ys == 0 and ns == 0:
        return None
    if ys >= ns:
        return True
    return False


def vote_agreement(votes: List[Optional[bool]]) -> float:
    """Agreement among valid votes: 1.0 when all votes match."""
    valid = [v for v in votes if v is not None]
    if not valid:
        return 0.0
    first = valid[0]
    return sum(1 for v in valid if v == first) / len(valid)
