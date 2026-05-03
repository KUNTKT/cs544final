"""CoT structure manipulation: N=steps, L=verbosity per step, R=role, P=perspective. Templates match the proposal PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------- Discrete levels per axis (extensible) ----------

N_LEVELS: Dict[str, str] = {
    "n_low": (
        "Use exactly 1–2 numbered steps in your reasoning.\n"
        "Each step must start with its number like '1.' or '2.'."
    ),
    "n_mid": (
        "Use exactly 3–4 numbered steps in your reasoning.\n"
        "Each step must start with its number like '1.' … '4.'."
    ),
    "n_high": (
        "Use at least 6 numbered steps in your reasoning.\n"
        "Each step must start with its number like '1.' … '6.' (you may add more if needed)."
    ),
}

L_LEVELS: Dict[str, str] = {
    "l_concise": (
        "Each step must be at most one short sentence (roughly one line).\n"
        "Do not add sub-bullets inside a step."
    ),
    "l_detailed": (
        "Each step must be a detailed paragraph: explain assumptions, intermediate facts,\n"
        "and why the step follows from previous steps."
    ),
}

R_LEVELS: Dict[str, str] = {
    "r_logician": (
        "You are a careful logician.\nPrioritize valid inference over intuition."
    ),
    "r_student": (
        "You are a student solving the problem step-by-step on an exam.\n"
        "Be explicit about each move."
    ),
    "r_expert": (
        "You are an expert analyst.\nBe precise and justify non-obvious claims."
    ),
}

P_LEVELS: Dict[str, str] = {
    "p_deductive": (
        "Use low-level deductive reasoning: start from the question and premises,\n"
        "derive the conclusion one small inference at a time."
    ),
    "p_analytical": (
        "Use high-level analytical reasoning: first outline the situation, then structure your reasoning,\n"
        "then state the conclusion (still using the required numbered steps)."
    ),
}

# Closing block (matches paper “Sample prompt” layout; single-line final answer instruction)
_FINAL_ANSWER_BLOCK = (
    "Respond with your reasoning first, then end with exactly one line:\n"
    "Final answer: Yes\n"
    "or\n"
    "Final answer: No\n"
    "Do not use other words than Yes or No on the Final answer line."
)


@dataclass(frozen=True)
class StructureCondition:
    """One experimental condition: one level id per axis (N, L, R, P)."""

    n_id: str
    l_id: str
    r_id: str
    p_id: str

    def slug(self) -> str:
        return f"{self.n_id}__{self.l_id}__{self.r_id}__{self.p_id}"

    @staticmethod
    def from_slug(s: str) -> "StructureCondition":
        parts = s.split("__")
        if len(parts) != 4:
            raise ValueError(f"Invalid condition slug: {s}")
        return StructureCondition(n_id=parts[0], l_id=parts[1], r_id=parts[2], p_id=parts[3])


def build_user_prompt(
    question: str,
    cond: StructureCondition,
    *,
    include_facts: bool = False,
    facts: List[str] | None = None,
) -> str:
    """Build a single user message body (no chat wrapper)."""
    n = N_LEVELS[cond.n_id]
    l_ = L_LEVELS[cond.l_id]
    r = R_LEVELS[cond.r_id]
    p = P_LEVELS[cond.p_id]

    # Instruction block: R, P, N, L — same order as the paper sample (each level may use internal newlines).
    instruction_lines: List[str] = [
        r.strip(),
        p.strip(),
        n.strip(),
        l_.strip(),
    ]
    parts: List[str] = ["\n".join(instruction_lines), "", "Answer the following yes/no question.", ""]
    if include_facts and facts:
        parts.append("Here are some reference facts (may be incomplete):")
        for i, f in enumerate(facts, 1):
            parts.append(f"{i}. {f}")
        parts.append("")
    parts.append(f"Question: {question.strip()}")
    parts.append("")
    parts.append(_FINAL_ANSWER_BLOCK)
    return "\n".join(parts)


def default_ablation_conditions() -> List[StructureCondition]:
    """Baseline plus one-factor variants for quick comparisons."""
    base = StructureCondition("n_mid", "l_concise", "r_student", "p_deductive")
    out = [base]
    for nid in ["n_low", "n_high"]:
        if nid != base.n_id:
            out.append(StructureCondition(nid, base.l_id, base.r_id, base.p_id))
    for lid in ["l_detailed"]:
        out.append(StructureCondition(base.n_id, lid, base.r_id, base.p_id))
    for rid in ["r_logician", "r_expert"]:
        if rid != base.r_id:
            out.append(StructureCondition(base.n_id, base.l_id, rid, base.p_id))
    for pid in ["p_analytical"]:
        if pid != base.p_id:
            out.append(StructureCondition(base.n_id, base.l_id, base.r_id, pid))
    return out


def full_factorial_grid(
    n_ids: Tuple[str, ...] = ("n_low", "n_mid", "n_high"),
    l_ids: Tuple[str, ...] = ("l_concise", "l_detailed"),
    r_ids: Tuple[str, ...] = ("r_logician", "r_student", "r_expert"),
    p_ids: Tuple[str, ...] = ("p_deductive", "p_analytical"),
) -> List[StructureCondition]:
    grid: List[StructureCondition] = []
    for n in n_ids:
        for l_ in l_ids:
            for r in r_ids:
                for p in p_ids:
                    grid.append(StructureCondition(n, l_, r, p))
    return grid


def nl_grid_conditions(
    *,
    r_id: str = "r_student",
    p_id: str = "p_deductive",
) -> List[StructureCondition]:
    """N×L grid only: R and P fixed (default student + deductive), 3×2=6 conditions."""
    grid: List[StructureCondition] = []
    for n in ("n_low", "n_mid", "n_high"):
        for l_ in ("l_concise", "l_detailed"):
            grid.append(StructureCondition(n, l_, r_id, p_id))
    return grid


def validate_ids(cond: StructureCondition) -> None:
    if cond.n_id not in N_LEVELS:
        raise KeyError(f"Unknown N level: {cond.n_id}")
    if cond.l_id not in L_LEVELS:
        raise KeyError(f"Unknown L level: {cond.l_id}")
    if cond.r_id not in R_LEVELS:
        raise KeyError(f"Unknown R level: {cond.r_id}")
    if cond.p_id not in P_LEVELS:
        raise KeyError(f"Unknown P level: {cond.p_id}")
