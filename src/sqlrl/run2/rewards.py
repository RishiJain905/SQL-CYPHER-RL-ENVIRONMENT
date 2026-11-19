"""Rewards and group advantages of run 2 (pure functions; the checker is called by `scoring`).

An episode gets **one reward, from its final query** (the last one that ran), by execution only:

- right (`exec_ok` True): `reward_right - turn_cost x (calls - 1)`. A right answer on the first
  query pays 1.00, on the third 0.94: the first-try answer stays the best outcome, and a fix is
  still worth nearly as much. An unreadable call uses a call slot, so it costs too;
- wrong: `reward_wrong` (0), whatever it took. Trying again never pays less than giving up;
- no query at all: `reward_malformed`;
- undecided by the checker, or passed by the checker but withheld by the duplicate probe
  (run 1b): no reward, the episode is left out of its group's statistics and of the loss;
- right by the checker but caught by the constants rule (`guards.hardcoded`): 0. The query
  returned the reference's rows without computing them.

Every token the model wrote in the episode shares the episode's advantage (outcome-level
credit). A group is **informative** when it holds a right and a not-right outcome: the turn cost
alone (1.00 against 0.97) doesn't make a group informative.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def outcome(ep: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """right | wrong | malformed, or None when the episode has no reward."""
    if not ep.get("json_ok"):
        return "malformed"
    ok = ep.get("exec_ok")
    if ok is None:
        return None
    if ok and ep.get("dup") == "differ":
        return None
    if ok and cfg.get("hardcode_rule", True) and ep.get("hardcoded"):
        return "wrong"
    return "right" if ok else "wrong"


def reward(ep: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    """`ep` carries the scorer's verdict fields (`scoring.score_episodes`)."""
    o = outcome(ep, cfg)
    if o is None:
        return None
    if o == "malformed":
        return cfg["reward_malformed"]
    if o == "wrong":
        return cfg["reward_wrong"]
    extra = max(int(ep.get("n_calls") or 1) - 1, 0)
    return cfg["reward_right"] - cfg.get("turn_cost", 0.0) * extra


def advantages(eps: list[dict[str, Any]], mode: str, eps_: float = 1e-6,
               std_floor: float = 0.0) -> None:  # fmt: skip
    """Set `adv` on each episode from its group (same `qid`), in place. Run 1's function: "std"
    is (r - mean) / max(std, `std_floor`); undecided episodes get None; a group whose rewards
    are all equal gets 0."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in eps:
        groups[r["qid"]].append(r)
    for rs in groups.values():
        vals = [r["reward"] for r in rs if r["reward"] is not None]
        mu = mean(vals) if vals else 0.0
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        for r in rs:
            if r["reward"] is None:
                r["adv"] = None
            elif sd < eps_:
                r["adv"] = 0.0
            elif mode == "center":
                r["adv"] = r["reward"] - mu
            elif mode == "std":
                r["adv"] = (r["reward"] - mu) / (max(sd, std_floor) + eps_)
            else:
                raise ValueError(f"unknown advantage mode {mode!r}")


def _norm(q: str | None) -> str:
    return " ".join((q or "").strip().rstrip(";").split()).lower()


def group_stats(eps: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """The step's reward numbers: overall and per dataset."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in eps:
        groups[r["qid"]].append(r)
    decided = [r for r in eps if r["reward"] is not None]
    # the solve rate is the checker's own verdict (a withheld or hardcoded answer still passed)
    checked = [r for r in eps if r.get("exec_ok") is not None]
    first = [r for r in eps if r.get("first_ok") is not None]
    outs = {q: {outcome(r, cfg) for r in rs} - {None} for q, rs in groups.items()}
    informative = [q for q, o in outs.items() if "right" in o and len(o) > 1]
    by_set: dict[str, list[float]] = defaultdict(list)
    right_by_set: dict[str, list[bool]] = defaultdict(list)
    for r in decided:
        by_set[r["origin"]].append(r["reward"])
    for r in checked:
        right_by_set[r["origin"]].append(r["exec_ok"] is True)
    probed = [r for r in eps if r.get("dup") in ("same", "differ")]
    out = {
        "reward_mean": mean(r["reward"] for r in decided) if decided else None,
        "reward_by_set": {s: mean(v) for s, v in sorted(by_set.items())},
        "solve_rate": mean(r["exec_ok"] is True for r in checked) if checked else None,
        "solve_by_set": {s: mean(v) for s, v in sorted(right_by_set.items())},
        "first_solve_rate": mean(r["first_ok"] is True for r in first) if first else None,
        "informative_frac": len(informative) / len(groups) if groups else None,
        "undecided_frac": 1 - len(decided) / len(eps) if eps else None,
        "malformed_frac": mean(not r.get("json_ok") for r in eps) if eps else None,
        # how many different answers a group holds: 1.0 means all eight wrote the same query
        "distinct_final_mean": mean(
            len({_norm(r.get("final_query")) for r in rs}) for rs in groups.values()
        )
        if groups
        else None,  # fmt: skip
        "distinct_first_mean": mean(
            len({_norm(r.get("first_query")) for r in rs}) for rs in groups.values()
        )
        if groups
        else None,  # fmt: skip
        "groups": len(groups),
        "rollouts": len(eps),
    }
    if probed:  # of the passed CypherBench answers the probe judged, the share it withheld
        out["dup_differ_frac"] = mean(r["dup"] == "differ" for r in probed)
        out["dup_probed"] = len(probed)
    return out
