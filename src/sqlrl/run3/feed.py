"""The run-3 dashboard's feed: phases 0 and 1 as four JSON documents.

    python -m sqlrl.run3.feed            # writes <data>/run3/feed/{state,phase0,train,evals}.json

The page ("Run 3 Warm Start", `documentation/run3/dashboard/`) reads collection `run3` of its
Artifact store, one document a file (document id = file name without `.json`), pushed with
`ArtifactData` as runs 1 and 2's were. Every field may be missing: the page draws a gap.

- `state`: status, phase, the step in progress, events (`<data>/run3/phase1/status.json`,
  written by the orchestrator), the warm start's settings and example counts.
- `phase0`: the anchor, pass@8 by stratum and harness variant, step 40 on the dev set.
- `train`: one row an optimiser step (`phase1/metrics.jsonl`).
- `evals`: each checkpoint's dev set and evaluation set, its behaviour and the gates.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sqlrl.run3 import settings

GATES = {"evalset_min": 57.0, "evalset_p": 0.05, "malformed_max": 0.05, "write_min": 0.90,
         "wrongdb_min": 0.80, "false_decline_max": 0.05, "look_min": 0.80,
         "useful_min": 0.15}  # fmt: skip


def root() -> Path:
    return settings.get().data


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lines(p: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def behaviour(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """The phase-1 behaviour gates' numbers from a dev set's greedy episodes."""
    ans = [r for r in eps if r.get("kind") == "answer"]
    none = [r for r in ans if (r.get("setting") or {}).get("schema") == "none"]
    sub_none = [r for r in none if r.get("ending") == "submit"]

    def share(rs: list[dict[str, Any]], f) -> float | None:
        return round(sum(1 for r in rs if f(r)) / len(rs), 3) if rs else None

    by: dict[str, Any] = {}
    for st in sorted({r["stratum"] for r in eps}):
        rs = [r for r in eps if r["stratum"] == st]
        by[st] = {"n": len(rs), "right": sum(r.get("outcome") == "right" for r in rs)}
    return {
        "n": len(eps),
        "right": sum(r.get("outcome") == "right" for r in ans),
        "answer_n": len(ans),
        "right_rate": share(ans, lambda r: r.get("outcome") == "right"),
        "malformed": share(eps, lambda r: r.get("outcome") in ("malformed", "crash")),
        "write_decline": share(
            [r for r in eps if r.get("kind") == "write"], lambda r: r.get("ending") == "decline"
        ),  # fmt: skip
        "wrongdb_decline": share(
            [r for r in eps if r.get("kind") == "unanswerable"],
            lambda r: r.get("ending") == "decline",
        ),  # fmt: skip
        "false_decline": share(ans, lambda r: r.get("ending") == "decline"),
        "look_when_none": share(sub_none, lambda r: r.get("n_calls", 0) > 0),
        "calls_mean": round(sum(r.get("n_calls", 0) for r in eps) / len(eps), 2) if eps else None,
        "endings": dict(Counter(r.get("ending") for r in eps)),
        "by_stratum": by,
    }


def _paired(base: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    """The dev set question by question (same gid): better / worse and a two-sided sign test."""
    from math import comb

    b = {r["gid"]: r.get("outcome") == "right" for r in base if r.get("kind") == "answer"}
    n = {r["gid"]: r.get("outcome") == "right" for r in new if r.get("kind") == "answer"}
    better = sum(1 for g in n if g in b and n[g] and not b[g])
    worse = sum(1 for g in n if g in b and b[g] and not n[g])
    m = better + worse
    p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(better, worse) + 1))
                               / 2**m)  # fmt: skip
    return {"better": better, "worse": worse, "p": round(p, 4)}


def _evalset(d: Path) -> dict[str, Any] | None:
    s = _json(d / "summary.json")
    if not s:
        return None
    f = s["greedy"]["overall"]["final"]
    vs = ((s.get("vs") or {}).get("run2_step40") or {}).get("greedy")
    return {"right": f["right"], "decided": round(100 * f["decided"], 1),
            "strict": round(100 * f["strict"], 1), "vs_step40": vs}  # fmt: skip


def gates(ev: dict[str, Any], base_dev: list[dict[str, Any]] | None) -> dict[str, Any]:
    es, b = ev.get("evalset"), ev.get("dev") or {}
    out: dict[str, Any] = {}

    def g(name: str, value: Any, ok: bool | None, bound: str) -> None:
        out[name] = {"value": value, "ok": ok, "bound": bound}

    if es:
        vs = es.get("vs_step40") or {}
        worse_sig = bool(vs) and vs["worse"] > vs["better"] and vs["p"] < GATES["evalset_p"]
        g(
            "evalset",
            es["decided"],
            es["decided"] >= GATES["evalset_min"] and not worse_sig,
            f">= {GATES['evalset_min']} and not worse than step 40 (sign test)",
        )
    else:
        g("evalset", None, None, f">= {GATES['evalset_min']}")
    pd = ev.get("dev_vs_step40")
    g("dev_better", pd, (pd["better"] > pd["worse"]) if pd else None, "more right than step 40")
    for key, lim, kind in (("malformed", "malformed_max", "max"), ("write_decline", "write_min",
                           "min"), ("wrongdb_decline", "wrongdb_min", "min"),
                           ("false_decline", "false_decline_max", "max"),
                           ("look_when_none", "look_min", "min")):  # fmt: skip
        v = b.get(key)
        ok = None if v is None else (v <= GATES[lim] if kind == "max" else v >= GATES[lim])
        g(key, v, ok, f"{'<=' if kind == 'max' else '>='} {GATES[lim]}")
    return out


def build() -> dict[str, Any]:
    r = root()  # run 3's data folder
    p0, p1 = r / "phase0", r / "phase1"
    base_eps = _lines(p0 / "dev-step40" / "episodes.jsonl")
    phase0 = {
        "anchor": _evalset(p0 / "anchor-step40-pc"),
        "pass8": _json(p0 / "pass8-step40" / "summary.json"),
        "dev_step40": behaviour(base_eps) if base_eps else None,
        "gates": {"useful_min": GATES["useful_min"]},
    }
    train = {"steps": _lines(p1 / "metrics.jsonl")}
    evals = []
    for ck in sorted(p1.glob("ckpt-*")):
        eps = _lines(p1 / f"eval-{ck.name}" / "dev" / "episodes.jsonl")
        ev: dict[str, Any] = {"ckpt": ck.name, "step": int(ck.name.split("-")[1]),
                              "evalset": _evalset(p0 / f"anchor-ws-{ck.name}"),
                              "dev": behaviour(eps) if eps else None,
                              "dev_vs_step40": (_paired(base_eps, eps)
                                                if eps and base_eps else None)}  # fmt: skip
        ev["gates"] = gates(ev, base_eps)
        evals.append(ev)
    st = _json(p1 / "status.json") or {"status": "measuring", "phase": "0", "events": []}
    state = {**st, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "config": _json(p1 / "config.json"), "collect": _json(p1 / "collect.json"),
             "last_step": train["steps"][-1] if train["steps"] else None,
             "gate_bounds": GATES}  # fmt: skip
    return {"state": state, "phase0": phase0, "train": train, "evals": {"evals": evals}}


def main() -> int:
    out = root() / "feed"
    out.mkdir(parents=True, exist_ok=True)
    docs = build()
    changed = []
    for name, doc in docs.items():
        f = out / f"{name}.json"
        text = json.dumps(doc, default=str)
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            changed.append(name)
        f.write_text(text, encoding="utf-8")
    print(json.dumps({"dir": str(out), "files": list(docs), "changed": changed,
                      "kb": {n: round(len(json.dumps(d, default=str)) / 1024, 1)
                             for n, d in docs.items()}}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
