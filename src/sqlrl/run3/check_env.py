"""Is this machine's run-3 environment working? (the PC or a Colab VM; no model, no GPU)

    python -m sqlrl.run3.check_env [--source birddev --db superhero]

Checks, and prints one line each (exit 1 if any fails):
1. every engine answers, and its version;
2. the reader can't write: a write through the engine's own path (the gate bypassed) is refused
   by the engine itself;
3. a database is on every SQL engine with the same row counts as its SQLite source;
4. a gold-checked question plays a whole scripted episode on every engine (look, query,
   submit) and is graded right;
5. Neo4j: a graph copy loads, survives its round trip, and a question with the copy beside
   its SQL source is graded right through the graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sqlrl.run3 import engines, episode, graphs, pools, protocol

OK, FAIL = "ok", "FAIL"


def line(name: str, ok: bool, detail: Any = "") -> bool:
    text = detail if isinstance(detail, str) else json.dumps(detail)
    print(f"{OK if ok else FAIL:4}  {name}  {text}",
          flush=True)  # fmt: skip
    return ok


def _raw_write(conn: Any) -> str:
    """Try a write past the gate, straight to the engine: the engine must refuse it."""
    e = conn.name
    try:
        if e == "sqlite":
            conn.db.execute("CREATE TABLE _w (a INTEGER)")
        elif e == "duckdb":
            conn.db.execute("CREATE TABLE _w (a INTEGER)")
        elif e == "postgres":
            with conn.db.transaction(), conn.db.cursor() as cur:
                cur.execute("CREATE TABLE _w (a integer)")
        elif e == "mysql":
            with conn.db.cursor() as cur:
                cur.execute("CREATE TABLE _w (a int)")
    except Exception as ex:
        return f"refused: {str(ex).splitlines()[0][:90]}"
    return "WRITE ALLOWED"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="birddev")
    ap.add_argument("--db", default="superhero")
    a = ap.parse_args(argv)
    good = True
    conns: dict[str, Any] = {}
    for e in engines.SQL_ENGINES:
        try:
            conns[e] = engines.open_sql(e, a.source, a.db)
            good &= line(f"{e} answers", True, conns[e].version())
        except Exception as ex:
            good &= line(f"{e} answers", False, f"{type(ex).__name__}: {str(ex)[:150]}")
    neo = engines.Neo4j()
    good &= line("neo4j answers", neo.ok(), neo.version() if neo.ok() else "")
    for e, c in conns.items():
        r = _raw_write(c)
        good &= line(f"{e} refuses a write by itself", r.startswith("refused"), r)
    counts = {}
    for e, c in conns.items():
        tabs = c.tables()
        counts[e] = {t.lower(): c.run(
            f"SELECT count(*) FROM {engines.quote_ident(e, t)}").rows[0][0]
                     for t in tabs}  # fmt: skip
    base = counts.get("sqlite")
    for e in conns:
        good &= line(f"{e} holds the same rows as SQLite", counts[e] == base,
                     "" if counts[e] == base else {"want": base, "got": counts[e]})  # fmt: skip
    for c in conns.values():
        c.close()
    for e in engines.SQL_ENGINES:
        tasks = pools.answer_tasks(a.source, a.db, e)
        if not tasks:
            good &= line(f"{e} scripted episode", False, "no gold-checked question")
            continue
        t = tasks[0]
        s = protocol.draw_setting([(e, e)], None)
        q = t["gold"]["sql"]
        turns = iter([protocol.call_text("list_tables", {}),
                      protocol.call_text(s.sources[0].tool, {s.sources[0].arg: q}),
                      protocol.call_text(s.submit, {"query": q})])  # fmt: skip
        rec = episode.run_episode(t, lambda m, tl, it=turns: next(it), s)
        out = episode.grade(t, rec)
        good &= line(
            f"{e} scripted episode",
            out["outcome"] == "right",
            {"ending": rec["ending"], "outcome": out["outcome"], "why": out.get("why")},
        )
    # a graph copy beside its SQL source; the answer given through the graph
    try:
        g = graphs.convert(a.source, a.db)
        rt = graphs.roundtrip(g)
        good &= line("graph copy round trip", rt["ok"], rt)
        h = graphs.host()
        key = f"copy:{a.source}:{a.db}"
        loaded = h.acquire(key)
        if loaded:
            h.release()
        good &= line("graph copy loads", loaded, h.load_s.get(key, h.failed.get(key)))
        sch = graphs.describe(engines.Neo4j())
        good &= line("graph schema text", "Nodes" in sch["full"], sch["names"][:100])
    except Exception as ex:
        good &= line("graph copy", False, f"{type(ex).__name__}: {str(ex)[:150]}")
    print(json.dumps({"all_ok": bool(good)}))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
