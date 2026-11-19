"""What a set of scored episodes says (pure functions, standard library only): the numbers the
evaluations, the no-training tests and the training canaries all report.

Two ways to count a score (design.md section 6):

- **decided**: right / (right + wrong). An answer the checker can't decide, and an episode with
  no query at all, are left out. This is how every earlier number was counted (54.5 is 102 of
  187), so it is the comparable one.
- **strict**: right / all. Undecided and no-query episodes count as wrong. A model can't raise
  this by ending without an answer.

Both are given for the **final** query (the episode's answer) and the **first** query (what a
single-turn model would have answered: the forced single-query number, free of charge).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def _rate(a: int, b: int) -> float | None:
    return a / b if b else None


def counts(eps: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """right / wrong / undecided / no_query for `key` ("exec_ok": final, "first_ok": first)."""
    c = Counter()
    for ep in eps:
        if not ep.get("final_query"):
            c["no_query"] += 1
        elif ep.get(key) is True:
            c["right"] += 1
        elif ep.get(key) is False:
            c["wrong"] += 1
        else:
            c["undecided"] += 1
    n = len(eps)
    return {"right": c["right"], "wrong": c["wrong"], "undecided": c["undecided"],
            "no_query": c["no_query"], "decided": _rate(c["right"], c["right"] + c["wrong"]),
            "strict": _rate(c["right"], n)}  # fmt: skip


def changes(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """What the later turns did to the first answer, over episodes where both are decided."""
    c = Counter()
    for ep in eps:
        a, b = ep.get("first_ok"), ep.get("exec_ok")
        if a is None or b is None:
            continue
        c[("right" if a else "wrong") + "_" + ("right" if b else "wrong")] += 1
    right_first = c["right_right"] + c["right_wrong"]
    wrong_first = c["wrong_right"] + c["wrong_wrong"]
    return {"fixed": c["wrong_right"], "broken": c["right_wrong"],
            "kept_right": c["right_right"], "kept_wrong": c["wrong_wrong"],
            "net": c["wrong_right"] - c["right_wrong"],
            "broken_frac": _rate(c["right_wrong"], right_first),
            "fixed_frac": _rate(c["wrong_right"], wrong_first)}  # fmt: skip


def protocol(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """How the episodes used their turns."""
    n = max(len(eps), 1)
    calls_n = Counter(int(ep.get("n_calls") or 0) for ep in eps)
    after: dict[str, list[bool]] = defaultdict(list)  # the first result's kind -> called again?
    for ep in eps:
        t1 = next((t for t in ep.get("turns") or [] if t.get("t") == 1), None)
        if t1 and t1.get("result"):
            after[t1["result"]].append(int(ep.get("n_calls") or 0) > 1)
    return {
        "calls_mean": mean(int(ep.get("n_calls") or 0) for ep in eps) if eps else None,
        "calls": {str(k): calls_n[k] for k in sorted(calls_n)},
        "ends": dict(Counter(ep.get("end") or "?" for ep in eps).most_common()),
        "limit_frac": sum(ep.get("end") == "limit" for ep in eps) / n,
        "no_query_frac": sum(not ep.get("final_query") for ep in eps) / n,
        "retry_after": {
            k: {"n": len(v), "again": sum(v) / len(v)} for k, v in sorted(after.items())
        },
        "look_error_frac": sum(bool(ep.get("look_error")) for ep in eps) / n,
    }


def with_turns(eps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The episodes that had turns. A dataset kept on run 1's protocol (`single_turn_sets`)
    ends at its first call (`one_query`; `end` is "single", or "no_query" when it made none):
    nothing is shown, so nothing can be retried, looped, copied or leaked. Counting those
    episodes in the rates below would only dilute them, and the stop rules' bounds were set on
    episodes with turns."""
    return [ep for ep in eps if not ep.get("one_query") and ep.get("end") != "single"]


def hacking(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """The signs of an answer that is copied, leaked or looped. The constants rule is read on
    every final query; the others only on episodes that were shown a result."""
    with_q = [ep for ep in eps if ep.get("final_query")]
    nq = max(len(with_q), 1)
    shown = with_turns(with_q)
    ns = max(len(shown), 1)
    fills = [ep for ep in shown if ep["origin"] in ("sqale", "text2cypher")]
    multi = [ep for ep in shown if int(ep.get("n_calls") or 0) > 1]
    return {
        "hardcoded_frac": sum(bool(ep.get("hardcoded")) for ep in with_q) / nq,
        "hardcoded_right": sum(
            bool(ep.get("hardcoded")) and ep.get("exec_ok") is True for ep in with_q
        ),  # fmt: skip
        "copied_frac": sum(bool(ep.get("copied")) for ep in shown) / ns,
        "copied_multi_frac": _rate(sum(bool(ep.get("copied")) for ep in multi), len(multi)),
        "look_leak_frac": _rate(sum(bool(ep.get("look_leak")) for ep in fills), len(fills)),
        "loop_frac": sum(bool(ep.get("loop")) for ep in shown) / ns,
    }


NOTHING = (None, 0, 0.0, "", False, (), [])
_NOTHING_TEXT = {"NULL", "null", "None", "", "0", "0.0", "False", "false", "[]"}


def says_something(rows: Any) -> bool:
    """Do a result's rows say anything? The checker's rule (`querycheck._informative`): some
    value that is not NULL, zero or empty. `0 rows` and a lone `COUNT(*) = 0` both say nothing."""
    return any(v not in NOTHING for r in rows or [] for v in r)


def shown(query: dict[str, Any]) -> str:
    """What the model was shown for a query that ran (an item of an episode's `queries`):
    "error", "something" (a row with a real value) or "nothing" (`0 rows`, or only NULL, zero
    and empty values). Read from the result's text, as the model read it."""
    if query.get("kind") == "error":
        return "error"
    if query.get("kind") != "rows":
        return "nothing"
    lines = [x for x in (query.get("obs") or "").splitlines()
             if x.strip() and not x.startswith("(")]  # fmt: skip
    if lines and lines[0].lstrip().startswith("{"):  # one JSON object a row
        cells: list[Any] = []
        for x in lines:
            try:
                cells += list(json.loads(x).values())
            except (json.JSONDecodeError, AttributeError):
                return "something"  # a row that was cut: it held text
        return "something" if any(v not in NOTHING for v in cells) else "nothing"
    texts = [c.strip() for x in lines[1:] for c in x.split(" | ")]  # a table: header first
    return "something" if any(c not in _NOTHING_TEXT for c in texts) else "nothing"


def _same(a: str | None, b: str | None) -> bool:
    def norm(q: str | None) -> str:
        return " ".join((q or "").strip().rstrip(";").split()).lower()

    return norm(a) == norm(b)


def first_result(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """What the turns did, split by what the **first** query's result showed (episodes with
    turns whose first and final query are both decided). The no-training test's key reading: a
    right first answer is kept when its result shows values and often rewritten when it shows
    nothing, and the pool holds no question where "nothing" is the right result.

    Per class: `kept_right`, `broken`, `fixed`, `kept_wrong`, and how the broken ones broke:
    `ran_out` (the episode tried a call too many, so the last query that ran was not meant as
    the answer) or `rewrote` (it changed the query and stopped)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for ep in with_turns(eps):
        ran = ep.get("queries") or []
        a, b = ep.get("first_ok"), ep.get("exec_ok")
        if not ran or a is None or b is None:
            continue
        c = out[shown(ran[0])]
        c[("kept_right", "broken", "fixed", "kept_wrong")[(0 if a else 2) + (0 if b else 1)]] += 1
        if a and not b:
            ran_out = ep.get("end") == "limit" and not _same(ran[-1]["query"], ran[0]["query"])
            c["ran_out" if ran_out else "rewrote"] += 1
    return {
        k: {
            **{
                f: v[f]
                for f in ("kept_right", "broken", "fixed", "kept_wrong", "ran_out", "rewrote")
            },
            "broken_frac": _rate(v["broken"], v["kept_right"] + v["broken"]),
            "fixed_frac": _rate(v["fixed"], v["fixed"] + v["kept_wrong"]),
        }  # fmt: skip
        for k, v in sorted(out.items())
    }


def returned_metrics(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """The returned gretel questions (the right result says nothing) beside the other gretel
    questions (a result that says nothing means a wrong query). The two pull opposite ways
    after an empty result, so a step's numbers are kept apart: the broken rate on the returned
    ones is what training has to lower without the other's fixed count falling.
    `returned_weak_only_frac`: of the returned answers the checker passes, the share the
    filled copies don't confirm ("it showed nothing, so stop" with a wrong query)."""
    back = [e for e in eps if e.get("kind") == "returned"]
    other = [e for e in eps if e.get("origin") == "gretel" and e.get("kind") != "returned"]
    out: dict[str, Any] = {"returned_n": len(back)}
    for name, sub in (("returned", back), ("gretel_other", other)):
        b = block(sub)
        out.update({
            f"{name}_solve": b["final"]["decided"],
            f"{name}_first_solve": b["first"]["decided"],
            f"{name}_fixed": b["changes"]["fixed"],
            f"{name}_broken": b["changes"]["broken"],
            f"{name}_broken_frac": b["changes"]["broken_frac"],
            f"{name}_calls_mean": b["protocol"]["calls_mean"],
        })  # fmt: skip
    weak = [e for e in back if e.get("exec_weak") is True]
    out["returned_weak_only_frac"] = _rate(sum(e.get("exec_ok") is not True for e in weak),
                                           len(weak))  # fmt: skip
    return out


def block(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """`final` and `first` count every episode; `changes` and `protocol` describe what the turns
    did, so they count the episodes that had turns (`with_turns`)."""
    turned = with_turns(eps)
    return {
        "n": len(eps),
        "with_turns": len(turned),
        "final": counts(eps, "exec_ok"),
        "first": counts(eps, "first_ok"),
        "changes": changes(turned),
        "protocol": protocol(turned),
        "hacking": hacking(eps),
    }


def summarise(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """`block` overall and per dataset."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in eps:
        by[ep["origin"]].append(ep)
    return {"overall": block(eps), "by_set": {s: block(v) for s, v in sorted(by.items())}}


def headline(s: dict[str, Any]) -> dict[str, Any]:
    """The few numbers a log line carries."""
    o = s["overall"]
    return {"final_decided": o["final"]["decided"], "final_strict": o["final"]["strict"],
            "first_decided": o["first"]["decided"], "first_strict": o["first"]["strict"],
            "fixed": o["changes"]["fixed"], "broken": o["changes"]["broken"],
            "calls_mean": o["protocol"]["calls_mean"]}  # fmt: skip
