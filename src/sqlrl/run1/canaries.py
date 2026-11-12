"""Run 1's canaries and stop rules (pure functions, standard library only).

`text_canaries` reads a step's rollouts for the signs of reward hacking and of the teacher
leaking into the student (`documentation/run1/design.md`, sections 1 and 5):

- `distinct_frac`, `select_star_frac`: queries with `DISTINCT` / `SELECT *`;
- `multi_call_frac`: replies with more than one tool call;
- `leak_frac`: thoughts that cite a "correct / provided / given / reference solution". SRPO's
  teacher is shown a right sibling's query under the words "Correct solution"; a student that
  starts writing as if it had been given one is copying the teacher's situation, not its skill;
- `query_chars_median`, `resp_tokens_median`.

`stop_reasons` compares the run's history with the stop rules in the config. The relative rules
use the mean of the first five steps as the run's own starting point.
"""

from __future__ import annotations

import re
from statistics import mean, median
from typing import Any

ARG = re.compile(r"<arg_value>(.*?)</arg_value>", re.S)
DISTINCT = re.compile(r"\bdistinct\b", re.I)
SELECT_STAR = re.compile(r"\bselect\s+\*", re.I)
LEAK = re.compile(r"\b(correct|provided|given|reference)\s+(solution|query|answer)\b", re.I)
BASE_STEPS = 5


def query_of(text: str) -> str:
    """The query a rollout called the tool with (its last argument value)."""
    found = ARG.findall(text.split("</think>", 1)[-1])
    return found[-1].strip() if found else ""


def text_canaries(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rollouts), 1)
    queries = [query_of(r["text"]) for r in rollouts]
    with_q = [q for q in queries if q]
    nq = max(len(with_q), 1)
    return {
        "distinct_frac": sum(bool(DISTINCT.search(q)) for q in with_q) / nq,
        "select_star_frac": sum(bool(SELECT_STAR.search(q)) for q in with_q) / nq,
        "multi_call_frac": sum(
            r["text"].split("</think>", 1)[-1].count("<tool_call>") > 1 for r in rollouts
        )
        / n,  # fmt: skip
        "leak_frac": sum(bool(LEAK.search(r["text"].split("</think>", 1)[0])) for r in rollouts)
        / n,  # fmt: skip
        "query_chars_median": float(median(len(q) for q in with_q)) if with_q else 0.0,
        "resp_tokens_median": float(median(len(r["gen_ids"]) for r in rollouts))
        if rollouts
        else 0.0,  # fmt: skip
    }


def _last_all(history: list[dict[str, Any]], key: str, n: int, bad) -> bool:
    vals = [h.get(key) for h in history[-n:]]
    return len(vals) == n and all(v is not None and bad(v) for v in vals)


def stop_reasons(history: list[dict[str, Any]], rules: dict[str, Any]) -> list[str]:
    """Why the run should pause after the last step in `history` (the steps' metrics, in
    order); empty when every canary is inside its bound."""
    if not history:
        return []
    last, out = history[-1], []

    def val(key: str) -> float | None:
        v = last.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    base = history[:BASE_STEPS] if len(history) > BASE_STEPS else None

    def base_mean(key: str) -> float | None:
        vals = [h[key] for h in base or [] if isinstance(h.get(key), (int, float))]
        return mean(vals) if vals else None

    ent, ent0 = val("entropy"), base_mean("entropy")
    if ent is not None and ent0 and ent < rules["entropy_min_frac"] * ent0:
        out.append(f"entropy {ent:.3f} is below {rules['entropy_min_frac']:.0%} of the first "
                   f"steps' {ent0:.3f}")  # fmt: skip
    if _last_all(history, "informative_frac", rules["informative_steps"],
                 lambda v: v < rules["informative_min"]):  # fmt: skip
        out.append(f"informative groups below {rules['informative_min']:.0%} for "
                   f"{rules['informative_steps']} steps")  # fmt: skip
    if _last_all(history, "malformed_frac", rules["malformed_steps"],
                 lambda v: v > rules["malformed_max"]):  # fmt: skip
        out.append(f"malformed calls above {rules['malformed_max']:.0%} for "
                   f"{rules['malformed_steps']} steps")  # fmt: skip
    for key, bound, label in (
        ("mismatch", "mismatch_max", "rollout-vs-trainer log-prob gap"),
        ("kl_sft", "kl_max", "KL to the SFT model"),
        ("leak_frac", "leak_max", "thoughts citing a given solution"),
    ):
        v = val(key)
        if v is not None and v > rules[bound]:
            out.append(f"{label} {v:.4f} is above {rules[bound]}")
    length, length0 = val("resp_tokens_median"), base_mean("resp_tokens_median")
    if length and length0:
        ratio = max(length / length0, length0 / length)
        if ratio > rules["length_ratio"]:
            out.append(f"median response length {length:.0f} against {length0:.0f} at the start")
    d, d0 = val("distinct_frac"), base_mean("distinct_frac")
    if d is not None and d0 is not None and d > d0 + rules["distinct_up"]:
        out.append(f"DISTINCT in {d:.0%} of queries against {d0:.0%} at the start")
    return out
