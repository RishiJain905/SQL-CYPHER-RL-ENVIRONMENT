"""Rewards and group advantages (pure functions; the checker is called by `train_step`).
Run 1's copy of the pilots' module, with `std_floor` added (tests prove the rest is equal).

A rollout's reward comes from execution only (CLAUDE.md -> rules of the RL run): its query
returns the gold query's result on the checker's databases (`exec_ok` True) -> `reward_right`;
a call that ran but answered differently -> `reward_wrong`; no parsable call to the tool ->
`reward_malformed`. An answer the checker can't decide (`exec_ok` None on a parsable call) gets
no reward at all: it is left out of its group's statistics and of the loss.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def reward(item: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    """`item` is the checker's verdict (`publiceval.score`)."""
    if not item.get("json_ok"):
        return cfg["reward_malformed"]
    ok = item.get("exec_ok")
    if ok is None:
        return None
    return cfg["reward_right"] if ok else cfg["reward_wrong"]


def advantages(rollouts: list[dict[str, Any]], mode: str, eps: float = 1e-6,
               std_floor: float = 0.0) -> None:  # fmt: skip
    """Set `adv` on each rollout from its group (same `qid`), in place. "center": r - mean
    (Dr. GRPO); "std": (r - mean) / std (GRPO, DAPO, GSPO). Undecided rollouts (reward None)
    get adv None; a group whose decided rewards are all equal gets 0 everywhere (it carries no
    learning signal).

    `std_floor` (run 1, decisions R8): the divisor is never smaller than this. Dividing by the
    group's own std turns any spread into advantages of size ~1, so a group with no right answer
    and one malformed call (rewards 0 and -0.1, std 0.03) would push its seven wrong answers up
    as hard as a right answer is pushed in a mixed group. With a floor of 0.25 (below the 0.33
    of a group with one right answer in eight, so right-vs-wrong groups are untouched) that
    group's advantages stay in proportion to the 0.1 its rewards differ by."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        groups[r["qid"]].append(r)
    for rs in groups.values():
        vals = [r["reward"] for r in rs if r["reward"] is not None]
        mu = mean(vals) if vals else 0.0
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        for r in rs:
            if r["reward"] is None:
                r["adv"] = None
            elif sd < eps:
                r["adv"] = 0.0
            elif mode == "center":
                r["adv"] = r["reward"] - mu
            elif mode == "std":
                r["adv"] = (r["reward"] - mu) / (max(sd, std_floor) + eps)
            else:
                raise ValueError(f"unknown advantage mode {mode!r}")


def group_stats(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """The step's reward numbers: overall and per dataset."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        groups[r["qid"]].append(r)
    decided = [r for r in rollouts if r["reward"] is not None]
    informative = [rs for rs in groups.values()
                   if len({r["reward"] for r in rs if r["reward"] is not None}) > 1]  # fmt: skip
    by_set: dict[str, list[float]] = defaultdict(list)
    right_by_set: dict[str, list[bool]] = defaultdict(list)
    for r in decided:
        by_set[r["origin"]].append(r["reward"])
        right_by_set[r["origin"]].append(r.get("exec_ok") is True)
    return {
        "reward_mean": mean(r["reward"] for r in decided) if decided else None,
        "reward_by_set": {s: mean(v) for s, v in sorted(by_set.items())},
        "solve_rate": mean(r.get("exec_ok") is True for r in decided) if decided else None,
        "solve_by_set": {s: mean(v) for s, v in sorted(right_by_set.items())},
        "informative_frac": len(informative) / len(groups) if groups else None,
        "undecided_frac": 1 - len(decided) / len(rollouts) if rollouts else None,
        "malformed_frac": mean(not r.get("json_ok") for r in rollouts) if rollouts else None,
        "groups": len(groups),
        "rollouts": len(rollouts),
    }
