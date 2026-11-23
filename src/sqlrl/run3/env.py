"""The environment side of an episode: the schema text, the helper tools, and what a call's
result looks like to the model (design.md 2.3, 2.4).

Helpers run as ordinary read-only queries through the same connection as the model's own
(`engines.*.run`), so they obey the same limits; the names a helper is given are checked against
the catalog and quoted, never pasted into a query as they came.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from typing import Any

from sqlrl.run3 import engines, porting

CELL_CHARS = 40
SAMPLE_MAX = 20
FIND_MAX = 20


# ---- what a result looks like -----------------------------------------------------------------


def cell(v: Any, style: str) -> Any:
    if v is None:
        return None if style == "json" else "NULL"
    if isinstance(v, bool):
        return v if style == "json" else str(v).lower()
    if isinstance(v, int):
        return v if style == "json" else str(v)
    if isinstance(v, float):
        return v if style == "json" else format(v, ".10g")
    if isinstance(v, (bytes, bytearray, memoryview)):
        s = "x'" + bytes(v[:16]).hex() + "'"
    elif isinstance(v, str):
        s = engines.clean(v)
    else:
        from sqlrl.run3.gold import norm

        n = norm(v)
        if isinstance(n, (int, float)):
            return cell(n, style)
        s = engines.clean(n if isinstance(n, str) else json.dumps(n, default=str))
    return s if len(s) <= CELL_CHARS else s[: CELL_CHARS - 3] + "..."


def render(res: engines.Result, rows: int, style: str, hints: bool) -> str:
    """A tool's reply text (run 2's `render`, plus CSV and the engine's hint lines)."""
    if res.kind == "error":
        out = "Error: " + res.error
        return out + ("\n" + res.hint if hints and res.hint else "")
    if res.kind == "empty":
        return "0 rows"
    shown = res.rows[:rows]
    names = [engines.clean(str(c)) for c in res.columns]
    if style == "json":
        lines = [json.dumps({n: cell(v, style) for n, v in zip(names, r, strict=False)},
                            ensure_ascii=False, default=str) for r in shown]  # fmt: skip
    elif style == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(names)
        for r in shown:
            w.writerow([cell(v, style) for v in r])
        lines = buf.getvalue().rstrip("\n").split("\n")
    elif style == "table":
        lines = [" | ".join(names)] + [" | ".join(str(cell(v, style)) for v in r) for r in shown]
    else:
        raise ValueError(f"unknown result style {style!r}")
    n = len(res.rows) if res.total is None else res.total
    if n > len(shown) or res.total is None:
        of = f"more than {engines.MAX_ROWS:,}" if res.total is None else f"{res.total:,}"
        lines.append(f"({len(shown)} of {of} rows)")
    return "\n".join(lines)


# ---- the schema in the prompt -----------------------------------------------------------------


def _source_fks(source: str, db_id: str) -> dict[str, list[tuple[str, str, str]]]:
    path = engines.sqlite_path(source, db_id)
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        out = {}
        for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            rows = db.execute(f"PRAGMA foreign_key_list({engines.quote_ident('sqlite', t)})")
            out[t.lower()] = [(r[3], r[2], r[4]) for r in rows if r[4]]
        return out
    finally:
        db.close()


def _shown(engine: str, name: str) -> str:
    """A name as it would be written in a query on `engine`: bare when plain, quoted if not."""
    n = porting.target_name(engine, name)
    if porting.SIMPLE.match(n) and (engine != "postgres" or n == n.lower()):
        return n
    return engines.quote_ident(engine, n)


def describe_sql(conn: Any, source: str, db_id: str) -> dict[str, str]:
    """{"full": CREATE TABLE lines in the engine's own types, "names": the table names}. The
    foreign keys are the source's (the ported copies carry none, `porting.py`)."""
    fks = _source_fks(source, db_id)
    lines = []
    tables = conn.tables()
    for t in tables:
        cols = [f"{_shown(conn.name, c)} {typ}".strip() for c, typ in conn.columns(t)]
        for col, other, other_col in fks.get(t.lower(), []):
            cols.append(f"FOREIGN KEY ({_shown(conn.name, col)}) REFERENCES "
                        f"{_shown(conn.name, other)} ({_shown(conn.name, other_col)})")  # fmt: skip
        lines.append(f"CREATE TABLE {_shown(conn.name, t)} ({', '.join(cols)});")
    return {"full": "\n".join(lines),
            "names": "Tables: " + ", ".join(_shown(conn.name, t) for t in tables)}  # fmt: skip


# ---- the helper tools -------------------------------------------------------------------------

_LIST = {
    "sqlite": "SELECT name AS table_name FROM sqlite_master WHERE type IN ('table', 'view') "
              "AND name NOT LIKE 'sqlite_%' ORDER BY name",
    "duckdb": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
              "ORDER BY table_name",
    "postgres": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
                "ORDER BY table_name",
    "mysql": "SELECT table_name AS table_name FROM information_schema.tables "
             "WHERE table_schema = DATABASE() ORDER BY table_name",
}  # fmt: skip
_TEXT = {"sqlite": "TEXT", "duckdb": "VARCHAR", "postgres": "TEXT", "mysql": "CHAR"}


def _lit(engine: str, s: str) -> str:
    s = s.replace("'", "''")
    if engine == "mysql":  # MySQL reads backslash escapes inside literals
        s = s.replace("\\", "\\\\")
    return "'" + s + "'"


class SqlHelpers:
    """The helper tools on one SQL connection."""

    def __init__(self, conn: Any):
        self.conn = conn
        self._tables: list[str] | None = None

    def _table(self, name: str) -> str | None:
        if self._tables is None:
            self._tables = self.conn.tables()
        name = (name or "").strip().strip('"`[]')
        if name in self._tables:
            return name
        low = {t.lower(): t for t in self._tables}
        return low.get(name.lower())

    def list_tables(self) -> engines.Result:
        return self.conn.run(_LIST[self.conn.name])

    def describe_table(self, table: str) -> engines.Result:
        t = self._table(table)
        if t is None:
            return engines.Result.fail(f"no such table: {table}")
        cols = self.conn.columns(t)
        return engines.Result.of(["column", "type"], [(c, typ) for c, typ in cols], 0.0)

    def sample_rows(self, table: str, n: Any = 5) -> engines.Result:
        t = self._table(table)
        if t is None:
            return engines.Result.fail(f"no such table: {table}")
        try:
            k = max(1, min(int(n or 5), SAMPLE_MAX))
        except (TypeError, ValueError):
            k = 5
        q = engines.quote_ident(self.conn.name, t)
        return self.conn.run(f"SELECT * FROM {q} LIMIT {k}")

    def find_values(self, column: str, text: str) -> engines.Result:
        t_name, _, c_name = (column or "").rpartition(".")
        if not t_name:
            return engines.Result.fail("give the column as table.column")
        t = self._table(t_name)
        if t is None:
            return engines.Result.fail(f"no such table: {t_name}")
        cols = {c.lower(): c for c, _ in self.conn.columns(t)}
        c = cols.get(c_name.strip().strip('"`[]').lower())
        if c is None:
            return engines.Result.fail(f"no such column: {column}")
        e = self.conn.name
        qc = engines.quote_ident(e, c)
        pat = "%" + (text or "").lower().replace("%", "").replace("_", "") + "%"
        return self.conn.run(
            f"SELECT DISTINCT {qc} FROM {engines.quote_ident(e, t)} "
            f"WHERE LOWER(CAST({qc} AS {_TEXT[e]})) LIKE {_lit(e, pat)} LIMIT {FIND_MAX}"
        )


def run_helper(h: SqlHelpers, name: str, args: dict[str, Any]) -> engines.Result:
    if name == "list_tables":
        return h.list_tables()
    if name == "describe_table":
        return h.describe_table(str(args.get("table", "")))
    if name == "sample_rows":
        return h.sample_rows(str(args.get("table", "")), args.get("n", 5))
    if name == "find_values":
        return h.find_values(str(args.get("column", "")), str(args.get("text", "")))
    return engines.Result.fail(f"unknown tool {name}")
