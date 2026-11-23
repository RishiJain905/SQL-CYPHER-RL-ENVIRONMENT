"""Gold queries on every SQL engine, and the comparison every reward uses (design.md 3.2, 4).

**The comparison is the trainer's checker's** (`querycheck.align`, `same_results`), so run 3
keeps one yardstick with runs 1 and 2: columns in any order, rows in any order unless the gold
sorts them, numbers at the gold's precision. A gold with a top-level `LIMIT` gets the checker's
tie rule: a prediction that keeps other rows tied at the cut is right too (`tie`).

**Types are made comparable first** (`norm`): each engine's driver returns its own types for the
same value (PostgreSQL and MySQL return `Decimal` for an average, dates as `date` objects), so
every value is turned into what SQLite would have returned: numbers as int or float, dates and
times as ISO text, booleans as 0 / 1, binary as bytes.

**A (question, engine) pair is kept only when the gold works there** (`check`): the SQLite gold
is translated with sqlglot, run on the engine, and must give the SQLite result under the
comparison, within the engine's time limit with room to spare. The pool records the engines a
question passed on; nothing else is ever trained on that engine.
"""

from __future__ import annotations

import datetime as dt
import decimal
import time
import uuid
from typing import Any

from sqlrl.run3 import engines, porting, settings

GOLD_TIME_S = 30.0
GOLD_MAX_SHARE = 0.5  # a gold slower than half the episode's limit is dropped for that engine
SQLGLOT_DIALECT = {"duckdb": "duckdb", "postgres": "postgres", "mysql": "mysql",
                   "sqlite": "sqlite"}  # fmt: skip


def norm(v: Any) -> Any:
    """One value as SQLite would have returned it."""
    if v is None or isinstance(v, (str, int, float)):
        return int(v) if isinstance(v, bool) else v
    if isinstance(v, decimal.Decimal):
        return int(v) if v == v.to_integral_value() and v.as_tuple().exponent >= 0 else float(v)
    if isinstance(v, dt.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, (dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (list, tuple)):
        return tuple(norm(x) for x in v)
    return str(v)


def norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(norm(v) for v in r) for r in rows]


# ---- the comparison ---------------------------------------------------------------------------


def verdict(gold_sql: str, gold: list[tuple], pred: list[tuple], cols: list[str] | None = None,
            full: list[tuple] | None = None, dialect: str = "sql") -> str:  # fmt: skip
    """match | tie | differ: a prediction's rows against the gold's (both normalised).
    `full`: the gold's rows without its top-level LIMIT (for the tie rule), `cols`: its column
    names (to find the sort key)."""
    qc, cc = settings.checker()
    mod = cc if dialect == "cypher" else qc
    moved = qc.align(gold, pred)
    if moved is not None:
        if mod.sorts_result(gold_sql) and not _ordered_equal(qc, gold, moved):
            pass  # same rows, wrong order: only a tie can save it
        else:
            return "match"
    if full is not None and tie(gold_sql, gold, pred, full, cols or [], dialect):
        return "tie"
    return "differ"


def _ordered_equal(qc, gold: list[tuple], pred: list[tuple]) -> bool:
    return len(gold) == len(pred) and all(
        qc.same_results([a], [b]) for a, b in zip(gold, pred, strict=True))  # fmt: skip


def tie(gold_sql: str, g: list[tuple], p: list[tuple], full: list[tuple], names: list[str],
        dialect: str = "sql") -> bool:  # fmt: skip
    """The checker's `_tie` with the gold's full result given instead of run: do gold and
    prediction differ only by which rows tied at the gold's LIMIT were kept?"""
    qc, cc = settings.checker()
    mod = cc if dialect == "cypher" else qc
    if len(p) != len(g) or not g or len(full) <= len(g):
        return False
    unl = mod.strip_limit(gold_sql)
    if unl is None:
        return False
    if not mod.sorts_result(unl):
        return qc.contained_in(p, full)
    k, key = len(g), mod._sort_key_column(unl, names)
    if key is None:
        return qc.contained_in(p, full[: k + 1])
    cut = qc._norm_value(full[k - 1][key], 6)
    if qc._norm_value(full[k][key], 6) != cut:
        return False
    above = [r for r in full[:k] if qc._norm_value(r[key], 6) != cut]
    tied = [r for r in full if qc._norm_value(r[key], 6) == cut]
    return qc.contained_in(above, p) and qc.contained_in(p, above + tied)


# ---- translation ------------------------------------------------------------------------------


def translate(sql: str, engine: str) -> str:
    """The SQLite gold in `engine`'s dialect (sqlglot). For PostgreSQL a quoted plain name is
    unquoted, because porting created plain names in lower case (`porting.target_name`) and an
    unquoted name is folded to it; MySQL and DuckDB ignore a name's case."""
    if engine == "sqlite":
        return sql
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="sqlite")
    if engine == "postgres":
        for ident in tree.find_all(exp.Identifier):
            if ident.quoted and porting.SIMPLE.match(ident.name):
                ident.set("quoted", False)
    return tree.sql(dialect=SQLGLOT_DIALECT[engine])


# ---- the check --------------------------------------------------------------------------------


def run_gold(conn, sql: str) -> tuple[engines.Result, list[tuple] | None]:
    """(result, rows without the top-level LIMIT or None) of a gold on an open connection."""
    qc, _ = settings.checker()
    res = conn.run(sql)
    full = None
    unl = qc.strip_limit(engines.strip_end(sql))
    if res.kind != "error" and unl is not None:
        r2 = conn.run(unl)
        if r2.kind != "error" and r2.total is not None:
            full = norm_rows(r2.rows)
    return res, full


def check_db(source: str, db_id: str, questions: list[dict[str, Any]],
             which: tuple[str, ...] = engines.SQL_ENGINES,
             limit_s: float = engines.QUERY_TIME_S) -> list[dict[str, Any]]:  # fmt: skip
    """Every question of one database on every engine of `which` (each must already be
    ported). Each record: the SQLite gold's result and, per engine, the translated gold and
    whether it passed."""
    conns: dict[str, Any] = {}
    for e in which:
        try:
            conns[e] = engines.open_sql(e, source, db_id, time_s=GOLD_TIME_S)
        except Exception as ex:
            conns[e] = ex
    out = []
    for q in questions:
        rec: dict[str, Any] = {"id": q["id"], "engines": {}}
        base = conns.get("sqlite")
        if not isinstance(base, (engines.SQLite,)):
            rec["error"] = "no sqlite connection"
            out.append(rec)
            continue
        t0 = time.monotonic()
        res, full = run_gold(base, q["gold"])
        ms = (time.monotonic() - t0) * 1000
        if res.kind == "error":
            rec["error"] = f"sqlite gold fails: {res.error}"
            out.append(rec)
            continue
        if res.total is None:
            rec["error"] = "sqlite gold returns more than MAX_ROWS rows"
            out.append(rec)
            continue
        gold_rows = norm_rows(res.rows)
        rec.update({"columns": res.columns, "rows": gold_rows, "full": full,
                    "tables": res.tables})  # fmt: skip
        for e in which:
            c = conns[e]
            if isinstance(c, Exception):
                rec["engines"][e] = {"ok": False, "why": f"not open: {str(c)[:100]}"}
                continue
            try:
                sql = translate(q["gold"], e)
            except Exception as ex:
                rec["engines"][e] = {"ok": False, "why": f"translate: {str(ex)[:120]}"}
                continue
            if e == "sqlite":
                r, ems = res, ms
            else:
                t1 = time.monotonic()
                r = c.run(sql)
                ems = (time.monotonic() - t1) * 1000
            if r.kind == "error":
                rec["engines"][e] = {"ok": False, "gold": sql, "why": f"error: {r.error[:160]}"}
                continue
            v = verdict(q["gold"], gold_rows, norm_rows(r.rows), res.columns, full)
            slow = ems > limit_s * 1000 * GOLD_MAX_SHARE
            rec["engines"][e] = {"ok": v in ("match", "tie") and not slow, "gold": sql,
                                 "ms": round(ems, 1),
                                 "why": "slow" if slow else (
                                     "" if v == "match" else v)}  # fmt: skip
        out.append(rec)
    for c in conns.values():
        if not isinstance(c, Exception):
            c.close()
    return out


def passed(rec: dict[str, Any]) -> list[str]:
    return [e for e, v in (rec.get("engines") or {}).items() if v.get("ok")]
