"""The environment's end-to-end smoke: a model plays a few episodes on every engine
(not training, not phase 0's measurement: it proves the pieces work together).

    python -m sqlrl.run3.smoke --url http://127.0.0.1:8091 --source birddev \
        --db superhero --db student_club --per-engine 6 --nonanswers 4 --out <dir>

Greedy, the evaluation's fixed setting, so the episodes are comparable across engines. Writes
`episodes.jsonl` (each with its grade) and `summary.json` to `--out`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sqlrl.run3 import engines, episode, pools, protocol, sources
from sqlrl.run3.policy import LlamaPolicy


def build_tasks(source: str, dbs: list[str], per_engine: int, nonanswers: int,
                seed: int = 0, graphs_: list[str] | None = None,
                two_source: int = 0) -> list[dict[str, Any]]:  # fmt: skip
    rnd = random.Random(seed)
    tasks = []
    for g in graphs_ or []:  # CypherBench questions on its sampled graph
        pool = [
            t for t in pools.answer_tasks("cypherbench", g, "neo4j") if t["gold"]["rows"]
        ]  # informative ones for the smoke
        rnd.shuffle(pool)
        tasks += pool[:per_engine]
    if two_source:  # a SQL question with the graph copy of its database beside it
        pool = [
            pools.two_source_task(t, seed) for t in pools.answer_tasks(source, dbs[0], "postgres")
        ]
        rnd.shuffle(pool)
        tasks += pool[:two_source]
    others = sources.load(source)
    for e in engines.SQL_ENGINES:
        pool = [t for db in dbs for t in pools.answer_tasks(source, db, e)]
        rnd.shuffle(pool)
        tasks += pool[:per_engine]
    for i, db in enumerate(dbs):
        e = engines.SQL_ENGINES[i % len(engines.SQL_ENGINES)]
        tasks += pools.write_requests(source, db, e, nonanswers // 2 or 1, seed)
        tasks += pools.wrong_database(source, db, e, others, nonanswers - nonanswers // 2, seed)
    return tasks


def version_of(engine: str, source: str, db: str) -> str:
    c = engines.open_sql(engine, source, db)
    try:
        return c.version()
    finally:
        c.close()


def play(task: dict[str, Any], url: str, versions: dict[str, str]) -> dict[str, Any]:
    srcs = task["sources"]
    eng = srcs[task.get("answer_source", 0)]["engine"] if len(srcs) == 1 else "two_source"
    setting = protocol.draw_setting([(s["engine"], versions[s["engine"]]) for s in srcs], None)
    pol = LlamaPolicy(url, temperature=0.0)
    try:
        rec = episode.run_episode(task, pol, setting)
        episode.grade(task, rec)
    except Exception as e:  # one broken episode is a finding, not a stop
        rec = {
            "qid": task["q"]["id"],
            "kind": task["kind"],
            "ending": "crash",
            "outcome": "crash",
            "reward": None,
            "error": f"{type(e).__name__}: {e}"[:300],
        }
    rec["engine"] = eng
    return rec


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["engine"] if r["kind"] == "answer" else r["kind"]].append(r)  # engine or kind
    out: dict[str, Any] = {"episodes": len(recs)}
    for k, rs in sorted(by.items()):
        out[k] = {
            "n": len(rs),
            "right": sum(r.get("outcome") == "right" for r in rs),
            "endings": dict(Counter(r.get("ending") for r in rs)),
            "calls_mean": round(sum(r.get("n_calls", 0) for r in rs) / len(rs), 2),
            "last_query_ok": sum(bool(r.get("last_query_ok")) for r in rs),
            "why_wrong": dict(Counter(r.get("why") for r in rs if r.get("outcome") == "wrong")),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--source", default="birddev")
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--per-engine", type=int, default=6)
    ap.add_argument("--nonanswers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--graph", action="append", default=[], help="a CypherBench graph")
    ap.add_argument("--two-source", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        a.source, a.db, a.per_engine, a.nonanswers, graphs_=a.graph, two_source=a.two_source
    )
    # graph episodes last and grouped by graph: Neo4j holds one graph at a time
    tasks.sort(key=lambda t: max((s.get("graph") or "" for s in t["sources"]), default=""))
    versions = {e: version_of(e, a.source, a.db[0]) for e in engines.SQL_ENGINES}
    versions["neo4j"] = engines.Neo4j().version()
    print(json.dumps({"tasks": len(tasks), "versions": versions}), flush=True)
    lock = threading.Lock()
    recs: list[dict[str, Any]] = []
    t0 = time.time()
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(a.workers) as pool:  # fmt: skip
        for rec in pool.map(lambda t: play(t, a.url, versions), tasks):
            with lock:
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(json.dumps({"n": len(recs), "qid": rec["qid"], "engine": rec["engine"],
                                  "ending": rec.get("ending"), "outcome": rec.get("outcome"),
                                  "calls": rec.get("n_calls"), "s": rec.get("seconds")}),
                      flush=True)  # fmt: skip
    summ = summarise(recs)
    summ["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
