"""pass@k and the per-question buckets of the sampling baseline (documentation/rl.md, step 1).
Standard library only: the dashboard feed runs it under Windows Python.

A question's samples: `right` of `scored` answers checked out (`exec_ok` True). Samples the
checker couldn't decide (`exec_ok` None) don't count either way; a question with none decided
is left out, as in the greedy scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import comb
from typing import Any

K_MAX = 8


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased estimate of pass@k from `c` right of `n` samples (Chen et al., 2021):
    1 - C(n-c, k) / C(n, k). With fewer than k decided samples it is the best available
    (1 when any is right)."""
    if n <= 0:
        raise ValueError("no samples")
    if k > n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def counts(sample_items: Sequence[Sequence[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per question id: {"origin", "right", "scored", "seen"} over the sample files' items
    (each a list of `publiceval.score` items)."""
    out: dict[str, dict[str, Any]] = {}
    for items in sample_items:
        for it in items:
            q = out.setdefault(
                it["id"], {"origin": it["origin"], "right": 0, "scored": 0, "seen": 0}
            )
            q["seen"] += 1
            if it.get("exec_ok") is not None:
                q["scored"] += 1
                q["right"] += bool(it["exec_ok"])
    return out


def block(qs: Iterable[dict[str, Any]], samples: int) -> dict[str, Any]:
    """One summary block over questions {"right", "scored", "seen", "greedy"}."""
    qs = list(qs)
    decided = [q for q in qs if q["scored"] > 0]
    greedy = [q["greedy"] for q in qs if q.get("greedy") is not None]
    n = len(decided)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    hist = [0] * (samples + 1)
    for q in decided:
        if q["scored"] == samples:
            hist[q["right"]] += 1
    return {
        "questions": n,
        "greedy_pass1": mean([float(g) for g in greedy]),
        "mean_at8": mean([q["right"] / q["scored"] for q in decided]),
        "pass_at_k": [
            mean([pass_at_k(q["scored"], q["right"], k) for q in decided])
            for k in range(1, samples + 1)
        ],
        "always": sum(1 for q in decided if q["right"] == q["scored"]),
        "sometimes": sum(1 for q in decided if 0 < q["right"] < q["scored"]),
        "never": sum(1 for q in decided if q["right"] == 0),
        "hist": hist,
        "undecided_samples": sum(q["seen"] - q["scored"] for q in qs),
    }


def summarise(
    sample_items: Sequence[Sequence[dict[str, Any]]], greedy: dict[str, bool | None]
) -> dict[str, Any]:
    """{"overall": block, "by_set": {origin: block}, "questions": [...]} over the scored sample
    files so far; `greedy` is the released model's greedy exec_ok per question id."""
    samples = len(sample_items)
    per_q = counts(sample_items)
    for qid, q in per_q.items():
        q["greedy"] = greedy.get(qid)
    sets = sorted({q["origin"] for q in per_q.values()})
    return {
        "samples": samples,
        "overall": block(per_q.values(), samples),
        "by_set": {
            s: block((q for q in per_q.values() if q["origin"] == s), samples) for s in sets
        },
        "questions": [
            {
                "id": qid,
                "origin": q["origin"],
                "greedy": q["greedy"],
                "right": q["right"],
                "scored": q["scored"],
            }  # fmt: skip
            for qid, q in per_q.items()
        ],
    }
