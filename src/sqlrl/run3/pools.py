"""Run 3's questions: the gold check per database, the tasks an episode is built from, and the
non-answers (design.md 3.2, 3.4, 3.6).

    python -m sqlrl.run3.pools check --source birddev --db superhero [--db ...] | --all
    python -m sqlrl.run3.pools summary --source birddev

The check's records go to `<data>/pools/goldcheck/<source>/<db_id>.jsonl`, one a question, with
the SQLite gold's result and, per engine, the translated gold and whether it passed. Tasks are
built from those records (`tasks_for`), so a question is only ever asked on an engine where its
gold was proven.

**Non-answers** (`write_requests`, `wrong_database`): built without a model.
- *Write requests*: "delete / update / add" sentences over the database's real tables and
  columns. Declining is right.
- *Wrong database*: a question from another database whose gold reads tables and columns this
  database doesn't have (none of its gold's table names exist here). Declining is right.
- *Ambiguous* questions need a writer (the teacher, design 3.5): not built yet.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlrl.run3 import engines, gold, settings, sources


def check_dir(source: str) -> Path:
    return settings.get().pools / "goldcheck" / source


def run_check(source: str, db_ids: list[str]) -> dict[str, Any]:
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in sources.load(source):
        by_db[q["db_id"]].append(q)
    out = check_dir(source)
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for db in db_ids:
        if source == "cypherbench":
            from sqlrl.run3 import graphs

            recs = graphs.check_graph(f"cypherbench:{db}", by_db[db])
        else:
            recs = gold.check_db(source, db, by_db[db])
        with (out / f"{db}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
                counts["questions"] += 1
                for e in gold.passed(r):
                    counts[e] += 1
        print(json.dumps({"db": db, "questions": len(recs)}), flush=True)
    return dict(counts)


def _json_default(v: Any) -> Any:
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    return str(v)


def _from_json(v: Any) -> Any:
    if isinstance(v, dict) and "$bytes" in v:
        return bytes.fromhex(v["$bytes"])
    if isinstance(v, list):
        return tuple(_from_json(x) for x in v)
    return v


def load_checks(source: str, db_id: str) -> list[dict[str, Any]]:
    path = check_dir(source) / f"{db_id}.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k in ("rows", "full"):
            if r.get(k) is not None:
                r[k] = [tuple(_from_json(v) for v in row) for row in r[k]]
        out.append(r)
    return out


def summary(source: str) -> dict[str, Any]:
    per_engine: Counter = Counter()
    fails: Counter = Counter()
    n = 0
    for f in sorted(check_dir(source).glob("*.jsonl")):
        for r in load_checks(source, f.stem):
            n += 1
            for e, v in (r.get("engines") or {}).items():
                per_engine[e] += bool(v.get("ok"))
                if not v.get("ok"):
                    fails[f"{e}: {re.sub(r'[0-9]+', '#', v.get('why', ''))[:60]}"] += 1
    return {"questions": n, "passed": dict(per_engine), "top_failures": fails.most_common(12)}


# ---- tasks ------------------------------------------------------------------------------------


def source_entry(engine: str, source: str, db_id: str) -> dict[str, Any]:
    """One database an episode is connected to; a graph names the graph to load."""
    out: dict[str, Any] = {"engine": engine, "source": source, "db_id": db_id}
    if engine == "neo4j":
        out["graph"] = (
            f"cypherbench:{db_id}" if source == "cypherbench" else (f"copy:{source}:{db_id}")
        )
    return out


def two_source_task(task: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """A one-database SQL task with the graph copy of the same database offered beside it
    (design 2.3 "several sources"). Either source can answer: the result decides."""
    rnd = random.Random(f"{seed}:{task['q']['id']}:two")
    sql = task["sources"][0]
    graph = source_entry("neo4j", sql["source"], sql["db_id"])
    order = [sql, graph] if rnd.random() < 0.5 else [graph, sql]
    return {**task, "sources": order, "answer_source": order.index(sql), "any_source": True}


def answer_tasks(source: str, db_id: str, engine: str) -> list[dict[str, Any]]:
    """One task a question whose gold passed on `engine`."""
    qs = {q["id"]: q for q in sources.load(source) if q["db_id"] == db_id}
    out = []
    for r in load_checks(source, db_id):
        e = (r.get("engines") or {}).get(engine) or {}
        if not e.get("ok") or r["id"] not in qs:
            continue
        out.append({
            "q": qs[r["id"]], "kind": "answer",
            "sources": [source_entry(engine, source, db_id)],
            "answer_source": 0,
            "gold": {"sql": e["gold"], "rows": r["rows"], "columns": r["columns"],
                     "full": r.get("full")},
        })  # fmt: skip
    return out


WRITE_TEMPLATES = [
    "Delete every row from the {table} table where {column} is missing.",
    "Update the {table} table so that {column} is set to NULL for all rows.",
    "Add a new row to {table} with {column} set to 'test'.",
    "Remove the {column} column from {table}.",
    "Please drop the {table} table, we don't need it anymore.",
    "Change every {column} in {table} to upper case and save it.",
]


def write_requests(source: str, db_id: str, engine: str, n: int, seed: int = 0) -> list[dict]:
    conn = engines.open_sql("sqlite", source, db_id)
    try:
        cols = [(t, c) for t in conn.tables() for c, _ in conn.columns(t)]
    finally:
        conn.close()
    rnd = random.Random(f"{seed}:{source}:{db_id}:write")
    out = []
    for i in range(min(n, len(cols))):
        t, c = rnd.choice(cols)
        text = rnd.choice(WRITE_TEMPLATES).format(table=t, column=c)
        out.append({"q": {"id": f"write-{source}-{db_id}-{i}", "question": text, "evidence": "",
                          "db_id": db_id, "source": source},
                    "kind": "write",
                    "sources": [{"engine": engine, "source": source, "db_id": db_id}]})  # fmt: skip
    return out


def _gold_tables(sql: str) -> set[str]:
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    return {t.name.lower() for t in tree.find_all(exp.Table)}


def wrong_database(source: str, db_id: str, engine: str, others: list[dict[str, Any]], n: int,
                   seed: int = 0) -> list[dict[str, Any]]:  # fmt: skip
    """Questions from other databases that this one can't answer: none of the tables their
    gold reads exist here, and no column of this database is named in them."""
    conn = engines.open_sql("sqlite", source, db_id)
    try:
        here = {t.lower() for t in conn.tables()}
    finally:
        conn.close()
    rnd = random.Random(f"{seed}:{source}:{db_id}:wrongdb")
    pool = [q for q in others if q["db_id"] != db_id]
    rnd.shuffle(pool)
    out = []
    for q in pool:
        tabs = _gold_tables(q["gold"])
        if not tabs or tabs & here:
            continue
        out.append({"q": {**q, "id": f"wrongdb-{db_id}-{q['id']}"}, "kind": "unanswerable",
                    "sources": [{"engine": engine, "source": source, "db_id": db_id}]})  # fmt: skip
        if len(out) >= n:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run 3's gold check and pools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--source", required=True)
    c.add_argument("--db", action="append", default=[])
    c.add_argument("--all", action="store_true", help="every database ported to all engines")
    s = sub.add_parser("summary")
    s.add_argument("--source", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        dbs = a.db
        if a.all:
            dbs = sorted(p.stem for p in (settings.get().duckdb_dir / a.source).glob("*.duckdb"))
        print(json.dumps(run_check(a.source, dbs)))
    else:
        print(json.dumps(summary(a.source), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
