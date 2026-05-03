# Research Notes: What Affects Chain-of-Thought Accuracy in LLMs?

This note aligns with the course proposal [`cs544_group58_proposal (1).pdf`](../cs544_group58_proposal%20(1).pdf) (add the file next to the repo root if missing). It summarizes how **CoT structure** affects QA performance under a fixed model and dataset.

## 1. Research question

Chain-of-Thought (CoT) improves reasoning in LLMs, but how **internal structure of the reasoning trace** (step count, per-step verbosity, role, perspective) affects **final answer accuracy** is still under-studied with controlled variables. This work varies **only the reasoning structure in the prompt** (holding other factors fixed), measures task accuracy, consistency, and error patterns, and relates **structural factors to reliability**.

## 2. Core hypotheses (from the proposal)

- **Too few or too many steps** can hurt performance: a moderate number of steps is often most stable; very long chains may add noise or error propagation.
- **Per-step verbosity** must be distinguished from step count: longer steps are not always better and may hurt on easier items.
- **Role** (e.g. “expert” vs “student”) and **perspective** (step-by-step deduction vs high-level analysis) may affect **style and consistency** as much as raw accuracy—this must be quantified empirically.

## 3. Formalization: structure tuple \(S = (N, L, R, P)\)

| Symbol | Meaning | Manipulation (examples) |
|--------|---------|-------------------------|
| **N** | Number of reasoning steps | Ask for 1–2, 3–4, 6+ numbered steps |
| **L** | Length / verbosity per step | One short sentence per step vs a detailed paragraph with justification |
| **R** | Reasoning persona | “Careful logician”, “exam student”, “expert analyst”; question text unchanged |
| **P** | Reasoning perspective | Low-level deduction from premises vs high-level outline-then-conclude (still with numbered steps) |

Aside from these structural instructions, prompts are **templated and aligned in wording and format** to limit confounds.

## 4. Experimental setup (high level)

- **Task & data**: Multi-hop QA benchmarks (e.g. StrategyQA, PiQA in the proposal).
- **Model**: One fixed instruction-tuned open model (e.g. Qwen2.5-7B-Instruct), **no fine-tuning**; differences attributed to prompts.
- **Decoding**: Default **greedy** to reduce randomness; optional temperature sampling with multiple generations for **self-consistency**.
- **Fairness**: **Max tokens** per condition as equal as practical to avoid truncation artifacts.

## 5. Metrics

- **Task accuracy** vs gold labels.
- **Consistency** (if using multiple samples): vote agreement, etc.
- **Error analysis**: whether structural changes mainly cause **instability**, **fact grounding errors**, or **decision threshold shifts** (layered error discussion in the proposal).

## 6. Statistics (proposal plan)

- **Factorial analysis** on \(N, L, R, P\): main effects and interactions (e.g. ANOVA where appropriate).
- **Paired tests** across structure configurations (pairing by question to handle dependence).

## 7. Expected directions (for “expected results” in reports)

- Moderate step counts outperform too few or too many; verbosity interacts with difficulty.
- Role and perspective may affect **consistency** as much as **single-shot accuracy**.
- Actionable **CoT structure guidelines** under compute and token budgets.

## 8. References (aligned with the PDF)

1. Wei et al., 2022. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.  
2. Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.

---

*Purpose: turn the PDF proposal into a concise, extensible outline for experiment logs, results, and the final report.*
