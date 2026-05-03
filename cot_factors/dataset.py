from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class QAItem:
    qid: str
    question: str
    answer: bool
    facts: List[str]
    raw: Dict[str, Any]


def load_strategyqa_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of objects")
    return data


def iter_items(
    records: List[Dict[str, Any]],
    *,
    max_items: Optional[int],
    seed: int,
    shuffle: bool = True,
) -> Iterator[QAItem]:
    indices = list(range(len(records)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if max_items is not None:
        indices = indices[: max(0, max_items)]

    for i in indices:
        r = records[i]
        qid = str(r.get("qid", i))
        question = str(r["question"])
        ans = r["answer"]
        if not isinstance(ans, bool):
            raise ValueError(f"qid={qid}: expected boolean answer")
        facts = list(r.get("facts") or [])
        yield QAItem(qid=qid, question=question, answer=ans, facts=facts, raw=r)
