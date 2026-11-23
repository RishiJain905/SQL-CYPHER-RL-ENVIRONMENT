"""One SQLite database into DuckDB, PostgreSQL and MySQL (design.md section 3.2).

**Types follow what the column holds, not what it was declared as.** SQLite stores any value in
any column, and real databases (BIRD's especially) declare `DATE` for text, `INTEGER` for
numbers stored as text, and so on. Each column's storage classes are counted first:

| The column holds (ignoring NULLs) | DuckDB | PostgreSQL | MySQL |
|---|---|---|---|
| integers only | BIGINT | BIGINT | BIGINT |
| integers and reals | DOUBLE | DOUBLE PRECISION | DOUBLE |
| text (anything with text) | VARCHAR | TEXT | LONGTEXT |
| blobs only | BLOB | BYTEA | LONGBLOB |
| nothing but NULLs | VARCHAR | TEXT | LONGTEXT |

Text that looks like a date stays text, as in the source: a model that wants a year from it
writes the engine's own way to get one, which is the dialect work run 3 is for.

**Names.** DuckDB keeps every name as it is (it ignores case). PostgreSQL folds an unquoted name
to lower case, so a plain name (`[A-Za-z_][A-Za-z0-9_]*`) is created in lower case, the way
PostgreSQL databases are usually written; a name with a space or a symbol keeps its spelling
and has to be quoted, as it would in any real database. MySQL runs with
`lower_case_table_names=1` (table names ignore case, as on Windows and macOS servers) and keeps
column names as they are (MySQL ignores their case).

The server engines get no keys or constraints: the engines' differences in what they enforce
are not what run 3 trains, and loading is far faster without them. The prompt's schema comes
from the engine's own catalog (`schema.py`).

    python -m sqlrl.run3.porting setup                  # the reader role in both servers
    python -m sqlrl.run3.porting port --source bird --db <db_id> [--engines duckdb,postgres,mysql]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from sqlrl.run3 import engines, settings

SIMPLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TYPES = {
    "int": {"duckdb": "BIGINT", "postgres": "BIGINT", "mysql": "BIGINT"},
    "real": {"duckdb": "DOUBLE", "postgres": "DOUBLE PRECISION", "mysql": "DOUBLE"},
    "text": {"duckdb": "VARCHAR", "postgres": "TEXT", "mysql": "LONGTEXT"},
    "blob": {"duckdb": "BLOB", "postgres": "BYTEA", "mysql": "LONGBLOB"},
}
MYSQL_NAME_MAX = 64
PG_NAME_MAX = 63
BATCH = 20_000


def target_name(engine: str, name: str) -> str:
    """A table's or column's name on `engine` (see the module's note)."""
    if engine == "postgres" and SIMPLE.match(name):
        return name.lower()
    return name


def column_kinds(src: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """(name, int | real | text | blob) for each column, from the values it holds."""
    q = engines.quote_ident("sqlite", table)
    cols = [r[1] for r in src.execute(f"PRAGMA table_info({q})")]
    out = []
    for c in cols:
        qc = engines.quote_ident("sqlite", c)
        kinds = {r[0] for r in src.execute(
            f"SELECT DISTINCT typeof({qc}) FROM {q} WHERE {qc} IS NOT NULL")}  # fmt: skip
        if not kinds or "text" in kinds:
            kind = "text"
        elif kinds == {"integer"}:
            kind = "int"
        elif kinds <= {"integer", "real"}:
            kind = "real"
        elif kinds == {"blob"}:
            kind = "blob"
        else:  # blob mixed with numbers: shown as text everywhere
            kind = "text"
        out.append((c, kind))
    return out


def source_tables(src: sqlite3.Connection) -> list[str]:
    return [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]  # fmt: skip


def _value(v: Any, kind: str) -> Any:
    """A source value as the target column takes it."""
    if v is None:
        return None
    if kind == "text" and not isinstance(v, str):
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
            return repr(v)  # 3.0 stays "3.0", as SQLite's CAST(x AS TEXT) shows it
        return str(v)
    if kind == "real" and isinstance(v, int):
        return float(v)
    return v


def _rows(src: sqlite3.Connection, table: str, kinds: list[tuple[str, str]]):
    q = engines.quote_ident("sqlite", table)
    sel = ", ".join(engines.quote_ident("sqlite", c) for c, _ in kinds)
    cur = src.execute(f"SELECT {sel} FROM {q}")
    while True:
        chunk = cur.fetchmany(BATCH)
        if not chunk:
            return
        yield [tuple(_value(v, k) for v, (_, k) in zip(r, kinds, strict=True)) for r in chunk]


# ---- the server roles -------------------------------------------------------------------------


def setup() -> dict[str, str]:
    """The read-only `run3_reader` in PostgreSQL and MySQL (idempotent)."""
    pg = engines.pg_connect("postgres", user="postgres", read_only=False)
    pg.autocommit = True
    with pg.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (settings.READER,))
        if not cur.fetchone():
            cur.execute(f"CREATE ROLE {settings.READER} LOGIN NOSUPERUSER NOCREATEDB "
                        "NOCREATEROLE")  # fmt: skip
    pg.close()
    my = engines.my_connect(None, user="root", stream=False)
    with my.cursor() as cur:
        cur.execute(f"CREATE USER IF NOT EXISTS '{settings.READER}'@'%'")
    my.close()
    return {"postgres": "ok", "mysql": "ok"}


# ---- one engine -------------------------------------------------------------------------------


def port_duckdb(src_path: Path, out: Path, plan: list[tuple[str, list[tuple[str, str]]]]) -> None:
    import duckdb

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)
    db = duckdb.connect(str(tmp))
    src = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table, kinds in plan:
            t = engines.quote_ident("duckdb", table)
            cols = ", ".join(f"{engines.quote_ident('duckdb', c)} {TYPES[k]['duckdb']}"
                             for c, k in kinds)  # fmt: skip
            db.execute(f"CREATE TABLE {t} ({cols})")
            # through the same file MySQL loads (inserts row by row are slow in DuckDB)
            csv_path = _write_csv(src, table, kinds)
            try:
                spec = ", ".join(f"'c{i}': '{'VARCHAR' if k == 'blob' else TYPES[k]['duckdb']}'"
                                 for i, (_, k) in enumerate(kinds))  # fmt: skip
                sel = ", ".join(f"unhex(c{i})" if k == "blob" else f"c{i}"
                                for i, (_, k) in enumerate(kinds))  # fmt: skip
                db.execute(
                    f"INSERT INTO {t} SELECT {sel} FROM read_csv(?, columns={{{spec}}}, "
                    "header=false, delim=',', quote='\"', escape='\"', nullstr='NULL', "
                    "allow_quoted_nulls=false, strict_mode=true)", [str(csv_path)])  # fmt: skip
            finally:
                csv_path.unlink(missing_ok=True)
        db.execute("CHECKPOINT")
    finally:
        src.close()
        db.close()
    out.unlink(missing_ok=True)
    tmp.rename(out)


def port_postgres(src_path: Path, key: str,
                  plan: list[tuple[str, list[tuple[str, str]]]]) -> None:  # fmt: skip
    name = engines.db_name(key)
    admin = engines.pg_connect("postgres", user="postgres", read_only=False)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        cur.execute(f'CREATE DATABASE "{name}"')
        cur.execute(f'REVOKE ALL ON DATABASE "{name}" FROM PUBLIC')
        cur.execute(f'GRANT CONNECT ON DATABASE "{name}" TO {settings.READER}')
    admin.close()
    db = engines.pg_connect(name, user="postgres", read_only=False)
    src = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table, kinds in plan:
            t = engines.quote_ident("postgres", target_name("postgres", table))
            cols = ", ".join(
                f"{engines.quote_ident('postgres', target_name('postgres', c))} "
                f"{TYPES[k]['postgres']}" for c, k in kinds)  # fmt: skip
            with db.cursor() as cur:
                cur.execute(f"CREATE TABLE {t} ({cols})")
                with cur.copy(f"COPY {t} FROM STDIN") as cp:
                    for chunk in _rows(src, table, kinds):
                        for r in chunk:
                            cp.write_row([_pg_text(v) for v in r])
            db.commit()
        with db.cursor() as cur:
            cur.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
            cur.execute(f"GRANT USAGE ON SCHEMA public TO {settings.READER}")
            cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {settings.READER}")
            cur.execute("ANALYZE")
        db.commit()
    finally:
        src.close()
        db.close()


def _pg_text(v: Any) -> Any:
    # PostgreSQL text can't hold NUL characters (SQLite's can)
    return v.replace("\x00", "") if isinstance(v, str) else v


def _mysql_field(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        return '"' + v.hex() + '"'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    return '"' + str(v).replace('"', '""') + '"'


def _write_csv(src: sqlite3.Connection, table: str, kinds: list[tuple[str, str]]) -> Path:
    """A table as the CSV both DuckDB and MySQL load: every text quoted ("" inside), NULL the
    bare word, blobs as hex, one row a line (a newline inside quotes is part of the value)."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".csv",
                                     delete=False) as f:  # fmt: skip
        for chunk in _rows(src, table, kinds):
            f.writelines(",".join(_mysql_field(v) for v in r) + "\n" for r in chunk)
        return Path(f.name)


def port_mysql(src_path: Path, key: str, plan: list[tuple[str, list[tuple[str, str]]]]) -> None:
    name = engines.db_name(key)
    db = engines.my_connect(None, user="root", stream=False)
    src = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with db.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{name}`")
            cur.execute(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4")
            cur.execute(f"USE `{name}`")
        for table, kinds in plan:
            t = engines.quote_ident("mysql", table)
            cols = ", ".join(f"{engines.quote_ident('mysql', c)} {TYPES[k]['mysql']}"
                             for c, k in kinds)  # fmt: skip
            with db.cursor() as cur:
                cur.execute(f"CREATE TABLE {t} ({cols}) ENGINE=InnoDB")
            # LOAD DATA from a file: far faster than inserts. Fields are quoted, NULL is the
            # bare word, nothing is escaped (ESCAPED BY ''), blobs travel as hex.
            vars_, sets = [], []
            for i, (c, k) in enumerate(kinds):
                if k == "blob":
                    vars_.append(f"@v{i}")
                    sets.append(f"{engines.quote_ident('mysql', c)} = UNHEX(@v{i})")
                else:
                    vars_.append(engines.quote_ident("mysql", c))
            csv_path = _write_csv(src, table, kinds)
            try:
                load = (f"LOAD DATA LOCAL INFILE '{csv_path.as_posix()}' INTO TABLE {t} "
                        "CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY "
                        "'\"' ESCAPED BY '' LINES TERMINATED BY '\\n' "
                        f"({', '.join(vars_)})"
                        + (f" SET {', '.join(sets)}" if sets else ""))  # fmt: skip
                with db.cursor() as cur:
                    cur.execute(load)
                    warnings = cur.execute("SHOW WARNINGS")
                    if warnings:
                        first = cur.fetchone()
                        raise RuntimeError(f"{table}: LOAD DATA warned: {first}")
                db.commit()
            finally:
                csv_path.unlink(missing_ok=True)
        with db.cursor() as cur:
            cur.execute(f"GRANT SELECT ON `{name}`.* TO '{settings.READER}'@'%'")
    finally:
        src.close()
        db.close()


# ---- a whole database -------------------------------------------------------------------------


def plan_of(src_path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    src = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [(t, column_kinds(src, t)) for t in source_tables(src)]
    finally:
        src.close()


def portable(engine: str, plan: list[tuple[str, list[tuple[str, str]]]]) -> str | None:
    """Why a database can't go to `engine` as it is (None: it can)."""
    limit = {"mysql": MYSQL_NAME_MAX, "postgres": PG_NAME_MAX}.get(engine)
    if limit is None:
        return None
    for t, kinds in plan:
        names = [t] + [c for c, _ in kinds]
        if any(len(n.encode("utf-8")) > limit for n in names):
            return f"a name longer than {limit} bytes in {t}"
        if engine == "postgres":
            folded = [target_name("postgres", c) for c, _ in kinds]
            if len(set(folded)) != len(folded):
                return f"two columns of {t} differ only by case"
    if engine in ("postgres", "mysql"):
        folded = [t.lower() for t, _ in plan]
        if len(set(folded)) != len(folded):
            return "two tables differ only by case"
    return None


def port(source: str, db_id: str, which: list[str]) -> dict[str, Any]:
    src_path = engines.sqlite_path(source, db_id)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    plan = plan_of(src_path)
    out: dict[str, Any] = {"source": source, "db_id": db_id, "tables": len(plan), "engines": {}}
    for engine in which:
        why = portable(engine, plan)
        if why:
            out["engines"][engine] = {"ok": False, "why": why}
            continue
        t0 = time.time()
        try:
            if engine == "duckdb":
                port_duckdb(src_path, engines.duckdb_path(source, db_id), plan)
            elif engine == "postgres":
                port_postgres(src_path, engines.db_key(source, db_id), plan)
            elif engine == "mysql":
                port_mysql(src_path, engines.db_key(source, db_id), plan)
            else:
                raise ValueError(engine)
            out["engines"][engine] = {"ok": True, "s": round(time.time() - t0, 1)}
        except Exception as e:
            out["engines"][engine] = {"ok": False, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Port SQLite databases to the other SQL engines")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    p = sub.add_parser("port")
    p.add_argument("--source", required=True)
    p.add_argument("--db", required=True, action="append")
    p.add_argument("--engines", default="duckdb,postgres,mysql")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        print(json.dumps(setup()))
        return 0
    for db_id in a.db:
        print(json.dumps(port(a.source, db_id, a.engines.split(","))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
