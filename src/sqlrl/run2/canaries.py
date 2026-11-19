"""Run 2's canaries and stop rules (pure functions, standard library only).

`text_canaries` reads a step's episodes; `stop_reasons` compares the run's history with the stop
rules in the config. Any reason pauses the run for a look: nothing here is tuned around.

The rules, by what they guard (design.md section 7; "the start" is the mean of the run's own
first five steps, so the relative rules begin after step 5):

- **Collapse:** first-turn token entropy (measured on every episode that carries the server's
  log-probabilities, before the update, so it doesn't follow the routing), the number of
  different final queries a group holds, the share of groups with both a right and a wrong
  outcome, the first turn's length.
- **Drift:** first-turn KL to the start adapter as the server measures it, and that reading at
  step 1 (where the true value is zero: an offset too large to read a bound against).
- **The protocol:** episodes with no query, calls an episode, episodes that tried a call too
  many, the same query run twice, episodes too long to train on.
- **The first-try skill:** answers broken by a retry, and the first query's solve rate. Outcome
  credit can teach a careless first query when a retry fixes it.
- **Reward hacking:** final queries the constants rule caught, final queries with a literal read
  off a result, random-fill results that show a reference literal, DISTINCT and `SELECT *`
  habits, first-turn thoughts that cite a given solution (the teacher's situation), the
  duplicate probe.
"""

from __future__ import annotations

import re
from statistics import mean, median
from typing import Any

DISTINCT = re.compile(r"\bdistinct\b", re.I)
SELECT_STAR = re.compile(r"\bselect\s+\*", re.I)
LEAK = re.compile(r"\b(correct|provided|given|reference)\s+(solution|query|answer)\b", re.I)
BASE_STEPS = 5


def _first_thought(ep: dict[str, Any]) -> str:
    """The first turn's thought: the only one the teacher is shown a solution for. In later
    turns "the correct query should be ..." is the model talking about its own retry (seen in
    the no-training test), not the teacher's situation leaking."""
    turns = ep.get("turns") or []
    return (turns[0].get("text") or "").split("</think>", 1)[0] if turns else ""


def turn1_tokens(ep: dict[str, Any]) -> int:
    turns = ep.get("turns") or []
    return int(turns[0]["tokens"]) if turns else 0


def text_canaries(eps: list[dict[str, Any]], max_seq: int | None = None) -> dict[str, Any]:
    n = max(len(eps), 1)
    queries = [ep.get("final_query") or "" for ep in eps]
    with_q = [q for q in queries if q]
    nq = max(len(with_q), 1)
    later = [sum(t["tokens"] for t in (ep.get("turns") or [])[1:]) for ep in eps
             if len(ep.get("turns") or []) > 1]  # fmt: skip
    out = {
        "distinct_frac": sum(bool(DISTINCT.search(q)) for q in with_q) / nq,
        "select_star_frac": sum(bool(SELECT_STAR.search(q)) for q in with_q) / nq,
        "leak_frac": sum(bool(LEAK.search(_first_thought(ep))) for ep in eps) / n,
        "query_chars_median": float(median(len(q) for q in with_q)) if with_q else 0.0,
        "t1_tokens_median": float(median(turn1_tokens(ep) for ep in eps)) if eps else 0.0,
        "later_tokens_median": float(median(later)) if later else None,
        "episode_tokens_mean": mean(sum(ep["gen_mask"]) for ep in eps) if eps else 0.0,
        "think_forced_frac": sum(any(t.get("forced") for t in ep.get("turns") or []) for ep in eps)
        / n,  # fmt: skip
    }
    if max_seq:
        out["too_long_frac"] = sum(len(ep["prompt_ids"]) + len(ep["gen_ids"]) > max_seq
                                   for ep in eps) / n  # fmt: skip
    return out


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
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    base = history[:BASE_STEPS] if len(history) > BASE_STEPS else None

    def base_mean(key: str) -> float | None:
        vals = [h[key] for h in base or [] if isinstance(h.get(key), (int, float))]
        return mean(vals) if vals else None

    after = history[BASE_STEPS:]

    # collapse
    ent, ent0 = val("entropy_t1"), base_mean("entropy_t1")
    if ent is not None and ent0 and ent < rules["entropy_min_frac"] * ent0:
        out.append(f"first-turn entropy {ent:.3f} is below {rules['entropy_min_frac']:.0%} of "
                   f"the first steps' {ent0:.3f}")  # fmt: skip
    d0 = base_mean("distinct_final_mean")
    if d0 and _last_all(after, "distinct_final_mean", rules["diversity_steps"],
                        lambda v: v < rules["diversity_min_frac"] * d0):  # fmt: skip
        out.append(
            f"a group holds fewer than {rules['diversity_min_frac'] * d0:.1f} different "
            f"final queries for {rules['diversity_steps']} steps ({d0:.1f} at the start)"
        )
    if _last_all(history, "informative_frac", rules["informative_steps"],
                 lambda v: v < rules["informative_min"]):  # fmt: skip
        out.append(f"informative groups below {rules['informative_min']:.0%} for "
                   f"{rules['informative_steps']} steps")  # fmt: skip
    length, length0 = val("t1_tokens_median"), base_mean("t1_tokens_median")
    if length and length0:
        ratio = max(length / length0, length0 / length)
        if ratio > rules["length_ratio"]:
            out.append(f"median first-turn length {length:.0f} against {length0:.0f} at the start")

    # drift and the simple bounds
    for key, bound, label in (
        ("kl_start", "kl_start_max", "first-turn KL to the start adapter"),
        ("kl_step1", "kl_step1_max", "the drift reading at step 1 (true value: zero)"),
        ("mismatch", "mismatch_max", "rollout-vs-trainer log-prob gap"),
        ("too_long_frac", "too_long_max", "episodes too long to train on"),
        ("hardcoded_frac", "hardcoded_max", "final queries that return constants"),
        ("look_leak_frac", "look_leak_max", "random-fill results showing a reference literal"),
        ("select_star_frac", "select_star_max", "final queries with SELECT *"),
        ("leak_frac", "leak_max", "thoughts citing a given solution"),
    ):
        v = val(key)
        if v is not None and v > rules[bound]:
            out.append(f"{label} {v:.4f} is above {rules[bound]}")

    # the protocol
    if _last_all(history, "malformed_frac", rules["malformed_steps"],
                 lambda v: v > rules["malformed_max"]):  # fmt: skip
        out.append(f"episodes with no query above {rules['malformed_max']:.0%} for "
                   f"{rules['malformed_steps']} steps")  # fmt: skip
    if _last_all(history, "calls_mean", rules["queries_steps"],
                 lambda v: v > rules["queries_mean_max"]):  # fmt: skip
        out.append(f"more than {rules['queries_mean_max']} calls an episode for "
                   f"{rules['queries_steps']} steps")  # fmt: skip
    # the untrained start model already loops, runs out of calls and breaks some right answers
    # (the no-training test), so these three are measured against the run's own start
    for key, up, steps, label in (
        ("limit_frac", "limit_up", "queries_steps", "episodes that tried a call too many"),
        ("loop_frac", "loops_up", "loops_steps", "episodes that ran the same query twice"),
        ("broken_frac", "broken_up", "broken_steps", "right first answers a retry made wrong"),
    ):
        v0 = base_mean(key)
        if v0 is not None and _last_all(after, key, rules[steps], lambda v: v > v0 + rules[up]):  # noqa: B023
            out.append(f"{label}: above {v0 + rules[up]:.0%} for {rules[steps]} steps "
                       f"({v0:.0%} at the start)")  # fmt: skip

    # the first-try skill
    f0 = base_mean("first_solve_rate")
    if f0 is not None and _last_all(after, "first_solve_rate", rules["first_right_steps"],
                                    lambda v: v < f0 - rules["first_right_down"]):  # fmt: skip
        out.append(f"the first query's solve rate is below {f0 - rules['first_right_down']:.2f} "
                   f"for {rules['first_right_steps']} steps ({f0:.2f} at the start)")  # fmt: skip

    # habits measured against the start
    c, c0 = val("copied_frac"), base_mean("copied_frac")
    if c is not None and c0 is not None and c > c0 + rules["copied_up"]:
        out.append(f"final queries with a literal read off a result: {c:.0%} against {c0:.0%} "
                   "at the start")  # fmt: skip
    d, dd0 = val("distinct_frac"), base_mean("distinct_frac")
    if d is not None and dd0 is not None and d > dd0 + rules["distinct_up"]:
        out.append(f"DISTINCT in {d:.0%} of queries against {dd0:.0%} at the start")
    p0 = base_mean("dup_differ_frac")
    if rules.get("dup_up") and p0 is not None and _last_all(
        after, "dup_differ_frac", rules.get("dup_steps", 3),
        lambda v: v > p0 + rules["dup_up"],
    ):  # fmt: skip
        out.append(f"answers that repeat rows above {p0 + rules['dup_up']:.0%} of the passed "
                   f"CypherBench answers for {rules.get('dup_steps', 3)} steps ({p0:.0%} at "
                   "the start)")  # fmt: skip
    return out
