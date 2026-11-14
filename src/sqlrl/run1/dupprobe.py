"""A second look at the CypherBench answers the checker passed (run 1b, decisions R13).

The checker compares an answer's rows with the reference's on CypherBench's sampled graph. That
graph is sparse: for 82% of run 1's training questions whose reference uses `DISTINCT`, the
reference returns the same rows without it, so an answer that drops it is passed there
(`documentation/results/run1.md`). During run 1 the model's use of `DISTINCT` fell even on the
18% of questions where the graph does show the difference.

The probe runs the reference and the answer once more, on **the same graph with every
relationship stored twice**. Two paths now lead wherever one did. A reference that removes
repeats returns what it returned before; an answer that doesn't returns every row more than
once. The verdicts:

- `same`: the answer still returns the reference's rows;
- `differ`: it doesn't. The answer is right only while no two paths lead to the same place;
- `unknown`: the copy can't judge this question. Either a query failed or tied at a `LIMIT`
  there, or **the reference itself returns other values on the copy** than on the real graph.
  That happens when it counts paths (`count(m)` doubles), and with patterns like
  `(n)-[:targets]->(t)<-[:targets]-(x {name: ...})`: Cypher never uses one relationship twice
  in a pattern, which keeps `n` from being `x` on the real graph, and the twin relationship
  lifts that. So the probe only speaks when the reference's distinct rows are the same on
  both graphs.

Why a doubled graph and not every row twice (`UNWIND [0, 1]` in front of both queries, no
second graph): measured on run 1's 1,045 passed CypherBench answers, that also withholds the
reward from answers that are right for any graph (`WHERE EXISTS { ... }` in place of
`DISTINCT`; a single node looked up by name where the reference's `WITH DISTINCT n` changes
nothing): 104 answers the doubled graph rightly calls `same`.

A `differ` gets **no reward** (`rewards.reward`): not right, not wrong. The copy is artificial
(a real graph has no doubled relationships), so it never makes an answer wrong; it only
withholds a reward the sampled graph can't back. The yardstick (the evaluations) is unchanged:
checker 4 on the sampled graphs.
"""

from __future__ import annotations

from typing import Any


def doubled(graph: tuple[dict, list, dict]) -> tuple[dict, list, dict]:
    """The graph (`cyphercheck.cypherbench_graph`'s nodes, relationships, types) with every
    relationship twice."""
    nodes, rels, types = graph
    return nodes, list(rels) + list(rels), types


def verdict(outcome: str) -> str:
    """`cyphercheck.compare`'s outcome on the doubled graph, as the probe's verdict."""
    if outcome in ("match", "uninformative"):
        return "same"
    return "differ" if outcome == "differ" else "unknown"


def candidates(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rollouts the probe looks at: CypherBench answers the checker passed by running them
    (an answer that is the reference's own text needs no second look)."""
    return [r for r in rollouts if r.get("origin") == "cypherbench" and r.get("exec_ok") is True
            and r.get("reason") != "exact"]  # fmt: skip


def mark(rollouts: list[dict[str, Any]], exs: dict[str, Any], log=print) -> dict[str, int]:
    """Set `dup` ("same" | "differ" | "unknown") on each candidate, graph by graph. Needs the
    checker's Neo4j; leaves it marked as holding no graph, so the next scoring loads its own."""
    from asktrainer import cyphercheck as cc
    from asktrainer import publiceval

    todo = candidates(rollouts)
    counts = {"same": 0, "differ": 0, "unknown": 0}
    if not todo:
        return counts
    db = publiceval._graph()
    if db is None:
        raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
    by_graph: dict[str, dict[str, list[tuple[dict[str, Any], str]]]] = {}
    for r in todo:
        ex = exs[r["qid"]]
        _, q, _ = publiceval.parse_call(ex, r["text"])
        gold = ex.meta.get("gold") or ex.target
        if q:
            by_graph.setdefault(ex.meta.get("db_id") or "", {}).setdefault(gold, []).append((r, q))

    def rows_of(gold: str) -> set | None:
        try:
            return set(db.run(gold)[1])
        except Exception:
            return None

    try:
        for graph, by_gold in sorted(by_graph.items()):
            try:
                data = cc.cypherbench_graph(graph)
                cc.load(db, *data, key="eid", keep_key=True, index=True)
                real = {gold: rows_of(gold) for gold in by_gold}
                cc.load(db, *doubled(data), key="eid", keep_key=True, index=True)
            except Exception as e:  # a graph didn't load: nothing is known about these
                log(f"duplicate probe: graph {graph!r} not loaded ({type(e).__name__})")
                real = {}
            for gold, items in by_gold.items():
                # the copy may judge only when the reference means the same on it
                if real.get(gold) is None or rows_of(gold) != real[gold]:
                    outs = [{"outcome": "reference_changed"}] * len(items)
                else:
                    outs = cc.compare(db, gold, [q for _, q in items])
                for (r, _), o in zip(items, outs, strict=True):
                    r["dup"] = verdict(o["outcome"])
                    counts[r["dup"]] += 1
    finally:
        publiceval._GRAPH["loaded"] = None
    return counts
