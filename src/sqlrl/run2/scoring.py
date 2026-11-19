"""Scoring episodes with the checker (inside WSL, CPU): one scorer for the training steps, the
evaluations and the no-training tests.

An episode's answer is its **final query** (the last one that ran), wrapped as a one-call reply
and given to `publiceval.score` unchanged: checker 4, the reference from the original example,
the checker's own databases (never the look-around copy). Its **first query** is scored the same
way: that is what a single-turn model would have answered, so "fixed" (wrong, then right) and
"broken" (right, then wrong) can be counted, and the first-try skill watched.

Per episode, `score_episodes` sets:

- `json_ok`, `exec_ok`, `reason`: the final query's verdict (`exec_ok` None: undecided);
- `first_ok`, `first_reason`: the first query's;
- `dup`: run 1b's second look at passed CypherBench answers (`dupprobe`);
- `hardcoded`: why the final query can't have computed its result (`guards.hardcoded`), or None;
- `copied`: the final query's literals read off an earlier result (`guards.copied_literals`);
- `look_leak`: reference literals that appear in a result and not in the prompt. On the
  random-fill sets (SQaLe, text2cypher) the look-around copy is built without the reference, so
  this stays near zero unless something leaks;
- `loop`: the same query ran twice.

Verdicts are remembered per (question, query): a group's eight episodes often end on the same
query, and each first query is often the final one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlrl.run2 import calls, guards

FIELDS = ("json_ok", "exec_ok", "reason", "first_ok", "first_reason", "dup", "hardcoded",
          "copied", "look_leak", "loop")  # fmt: skip
FILL_SETS = ("sqale", "text2cypher")


def _norm(q: str | None) -> str:
    return " ".join((q or "").strip().rstrip(";").split()).lower()


def prompt_text(row: dict[str, Any]) -> str:
    return "\n".join((m.get("content") or "") for m in row["messages"][:2])


def reference_literals(row: dict[str, Any]) -> set[str]:
    gold = (row.get("meta") or {}).get("gold") or ""
    return guards.query_literals(gold, row["dialect"])


def annotate(ep: dict[str, Any], row: dict[str, Any]) -> None:
    """The fields that need no checker: `hardcoded`, `copied`, `look_leak`, `loop`."""
    ran = ep.get("queries") or []
    final = ep.get("final_query")
    tables = ran[-1].get("tables") if ran else None
    ep["hardcoded"] = guards.hardcoded(row["dialect"], final, tables)
    prompt = prompt_text(row)
    before = [x["obs"] for x in ran[:-1]]
    ep["copied"] = guards.copied_literals(final, row["dialect"], before, prompt)
    shown = "\n".join(x["obs"] for x in ran).lower()
    p = prompt.lower()
    ep["look_leak"] = sorted(s for s in reference_literals(row)
                             if s.lower() not in p and s.lower() in shown)  # fmt: skip
    seen = [_norm(x["query"]) for x in ran]
    ep["loop"] = len(set(seen)) < len(seen)


def score_episodes(eps: list[dict[str, Any]], rows: dict[str, dict[str, Any]],
                   checker: int = 4, dup_probe: bool = True, log=print) -> float:  # fmt: skip
    """Set the verdict fields on every episode in place; returns the seconds spent."""
    from asktrainer import publiceval
    from asktrainer.data.schema import Example
    from asktrainer.settings import get_paths as trainer_paths

    data = publiceval.PublicData(trainer_paths().public)
    exs = {qid: Example.model_validate(row) for qid, row in rows.items()}
    if any(e.dialect == "cypher" for e in exs.values()) and publiceval._graph() is None:
        # without it every Cypher answer would score "undecided" and get no reward, silently
        raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
    t0 = time.time()
    memo: dict[tuple[str, str], dict[str, Any]] = {}

    def verdict(ex, text: str) -> dict[str, Any]:
        key = (ex.id, text)
        if key not in memo:
            memo[key] = publiceval.score(ex, text, data, checker=checker)
        return memo[key]

    # one graph load per CypherBench graph: score grouped by graph
    order = sorted(range(len(eps)), key=lambda i: (eps[i]["origin"],
                                                   exs[eps[i]["qid"]].meta.get("db_id") or "",
                                                   eps[i]["qid"]))  # fmt: skip
    for i in order:
        ep = eps[i]
        ex = exs[ep["qid"]]
        item = verdict(ex, ep["text"])
        ep.update(json_ok=bool(item["json_ok"]), exec_ok=item["exec_ok"], reason=item["reason"])
        first = verdict(ex, ep.get("first_text") or ep["text"])
        ep.update(first_ok=first["exec_ok"] if first["json_ok"] else None,
                  first_reason=first["reason"])  # fmt: skip
        annotate(ep, rows[ep["qid"]])
    if dup_probe:
        from sqlrl.run2 import dupprobe

        log(f"duplicate probe: {dupprobe.mark(eps, exs)}")
    return time.time() - t0


def save(eps: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for ep in eps:
            v = {"qid": ep["qid"], "g": ep["g"], **{k: ep.get(k) for k in FIELDS}}
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load(eps: list[dict[str, Any]], path: Path) -> bool:
    """Read saved verdicts into the episodes; False when the file doesn't cover them all."""
    if not path.exists():
        return False
    got = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = json.loads(line)
            got[(v["qid"], v["g"])] = v
    if not all((ep["qid"], ep["g"]) in got for ep in eps):
        return False
    for ep in eps:
        v = got[(ep["qid"], ep["g"])]
        ep.update({k: v.get(k) for k in FIELDS})
    return True


def read_episodes(path: Path) -> list[dict[str, Any]]:
    """An episode file's episodes, one per (question, g): a later line replaces an earlier one,
    a failed one never replaces a real one."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["qid"], r["g"])
            if "error" not in r or key not in out:
                out[key] = r
    return [out[k] for k in sorted(out)]


def wrap(query: str | None, row: dict[str, Any]) -> str:
    """A query as the reply text the checker scores (for tests and tools)."""
    tool, arg = calls.query_tool(row)
    return calls.as_call(query, tool, arg)
