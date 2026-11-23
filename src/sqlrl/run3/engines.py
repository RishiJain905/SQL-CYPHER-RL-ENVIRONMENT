"""The five engines a run-3 episode can be connected to, behind one interface
(design.md section 2.1): `run(query) -> Result`, plus the catalog the helper tools read.

| Engine | A database is | Read only by |
|---|---|---|
| SQLite | a `.sqlite` file, opened `mode=ro` | `query_only`, an authorizer (run 2's) |
| DuckDB | a `.duckdb` file | `read_only=True`, `enable_external_access=false` |
| PostgreSQL | a database `r3_<id>` | the `run3_reader` role (SELECT only), read-only transactions |
| MySQL | a database `r3_<id>` | the `run3_reader` user (SELECT only), read-only transactions |
| Neo4j | the one graph loaded now | READ transactions, no procedures but the schema listings |

Every engine: one statement a query, a time limit (`QUERY_TIME_S`), at most `MAX_ROWS` rows
read (more is "more than 10000"), and the engine's own error text: its first line, and its hint
lines kept apart (`Result.hint`) so the rendering can show them or not (2.4).

The model's queries pass a first gate before the engine sees them: the statement kinds each
engine allows for reading and looking around (`READ_KINDS`). The engine's own read-only rule is
the real guard; the gate gives a plain refusal instead of a permission error.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlrl.run3 import guards, settings

QUERY_TIME_S = 5.0
MAX_ROWS = 10_000
MAX_VALUE = 1_000_000  # bytes of one SQLite value (randomblob, zeroblob, group_concat bombs)
CYPHER_MAX_BYTES = 4_000_000
ERR_CHARS = 240

MSG_READ = "only one read-only statement is allowed"
MSG_CYPHER_READ = "only read queries are allowed"
MSG_CALL = "procedure calls are not available"
MSG_TIMEOUT = "the query took too long and was stopped"
MSG_TOO_BIG = "the result is too large to return"
MSG_NO_DB = "the database is not available"

# The statement kinds a model query may start with, per engine: reading and the engine's own
# ways of looking around (PRAGMA table_info, SHOW TABLES, DESCRIBE, ...)
READ_KINDS = {
    "sqlite": ("SELECT", "WITH", "VALUES", "PRAGMA"),
    "duckdb": ("SELECT", "WITH", "VALUES", "FROM", "SHOW", "DESCRIBE", "DESC", "SUMMARIZE",
               "PRAGMA", "TABLE"),
    "postgres": ("SELECT", "WITH", "VALUES", "TABLE"),
    "mysql": ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "TABLE"),
}  # fmt: skip
SQLITE_PRAGMAS = {"table_info", "table_xinfo", "table_list", "index_list", "index_info",
                  "foreign_key_list", "database_list"}  # fmt: skip

CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)
CYPHER_CALL = re.compile(r"\bCALL\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
CALL_OK = ("db.labels", "db.relationshiptypes", "db.propertykeys", "db.schema.")
ENGINES = ("sqlite", "duckdb", "postgres", "mysql", "neo4j")
SQL_ENGINES = ("sqlite", "duckdb", "postgres", "mysql")


@dataclass
class Result:
    """What one query did. `kind`: rows | empty | error."""

    kind: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    total: int | None = None  # None: more than `MAX_ROWS`
    error: str = ""
    hint: str = ""  # the engine's hint / detail lines, if it gave any
    tables: list[str] | None = None  # SQLite: the tables the query read
    ms: float = 0.0

    @staticmethod
    def fail(msg: str, hint: str = "", ms: float = 0.0) -> Result:
        return Result("error", error=first_line(msg), hint=clean(hint)[:ERR_CHARS], ms=ms)

    @staticmethod
    def of(columns: list[str], rows: list[tuple], ms: float, tables=None) -> Result:
        if not rows:
            return Result("empty", columns=columns, total=0, tables=tables, ms=ms)
        total = len(rows) if len(rows) <= MAX_ROWS else None
        return Result("rows", columns, rows[:MAX_ROWS], total, tables=tables, ms=ms)


def clean(s: str) -> str:
    """Text without control characters (newlines and tabs become spaces)."""
    return "".join(" " if ch in "\r\n\t" else ch for ch in s if ch in "\r\n\t" or ch >= " ")


def first_line(msg: str) -> str:
    line = next((x for x in (msg or "").splitlines() if x.strip()), "").strip()
    return clean(line)[:ERR_CHARS] or "the query failed"


def _kind(q: str) -> str:
    m = re.match(r"\s*\(*\s*([A-Za-z]+)", q)
    return m.group(1).upper() if m else ""


def gate(engine: str, q: str) -> str | None:
    """Why a SQL query isn't sent to the engine at all (None: it is)."""
    if not q.strip() or not guards.one_statement(q):
        return MSG_READ
    if _kind(q) not in READ_KINDS[engine]:
        return MSG_READ
    return None


def strip_end(q: str) -> str:
    q = q.strip()
    while q.endswith(";"):
        q = q[:-1].rstrip()
    return q


class _Clock:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def ms(self) -> float:
        return round((time.monotonic() - self.t0) * 1000, 1)


def db_name(db_id: str) -> str:
    """A source database's name in PostgreSQL and MySQL: `r3_` and the id, lower case, safe."""
    return "r3_" + re.sub(r"[^a-z0-9_]", "_", db_id.lower())[:58]


# ---- SQLite -----------------------------------------------------------------------------------

_SQL_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE}  # fmt: skip
_SQL_NO_FUNCTION = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class SQLite:
    """One SQLite database file, read only, for one thread."""

    name = "sqlite"
    dialect = "sql"

    def __init__(self, path: Path | str, time_s: float = QUERY_TIME_S):
        self.path, self.time_s = Path(path), time_s
        uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.db.execute("PRAGMA query_only=ON")
        self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE)
        self._tables = {r[0].lower() for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}  # fmt: skip
        self._reads: set[tuple[str, str]] = set()
        self._read_by_sql: dict[str, list[str]] = {}
        self.db.set_authorizer(self._authorize)

    def version(self) -> str:
        return "SQLite " + sqlite3.sqlite_version

    def _authorize(self, action: int, a1: str | None, a2: str | None, *_: Any) -> int:
        if action == sqlite3.SQLITE_READ:
            self._reads.add((a1 or "", a2 or ""))
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (a2 or "").lower() in _SQL_NO_FUNCTION:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:  # the read-only listings, never a setting
            ok = (a1 or "").lower() in SQLITE_PRAGMAS
            return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in _SQL_ALLOWED else sqlite3.SQLITE_DENY

    def run(self, query: str) -> Result:
        c = _Clock()
        why = gate("sqlite", query)
        if why:
            return Result.fail(why)
        q = strip_end(query)
        self._reads = set()
        deadline = time.monotonic() + self.time_s
        self.db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = self.db.execute(q)
            rows = cur.fetchmany(MAX_ROWS + 1)
            names = [d[0] for d in cur.description or []]
        except sqlite3.Error as e:
            late = time.monotonic() > deadline
            msg = MSG_TIMEOUT if late else (MSG_READ if "not authorized" in str(e) else str(e))
            return Result.fail(msg, ms=c.ms())
        except Exception as e:  # "You can only execute one statement at a time."
            return Result.fail(MSG_READ if "one statement" in str(e) else str(e), ms=c.ms())
        finally:
            self.db.set_progress_handler(None, 0)
        # SQLite asks the authorizer when it prepares a statement, and Python keeps prepared
        # statements: the same text run again reads nothing new (run 2's note)
        if q not in self._read_by_sql:
            self._read_by_sql[q] = sorted({t for t, _ in self._reads if t.lower() in self._tables})
        return Result.of(names, rows, c.ms(), tables=list(self._read_by_sql[q]))

    # the catalog (the harness's own queries, not the model's)
    def tables(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]  # fmt: skip

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[1], r[2] or "") for r in self.db.execute(
            f"PRAGMA table_info({quote_ident('sqlite', table)})")]  # fmt: skip

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        """(column, other table, other column) for each declared foreign key."""
        rows = self.db.execute(f"PRAGMA foreign_key_list({quote_ident('sqlite', table)})")
        return [(r[3], r[2], r[4]) for r in rows]

    def admin(self, q: str) -> tuple[list[str], list[tuple]]:
        cur = self.db.execute(q)
        return [d[0] for d in cur.description or []], cur.fetchmany(MAX_ROWS + 1)

    def close(self) -> None:
        try:
            self.db.close()
        except sqlite3.Error:
            pass


# ---- DuckDB -----------------------------------------------------------------------------------


class DuckDB:
    """One DuckDB file, read only, no file or network access, for one thread."""

    name = "duckdb"
    dialect = "sql"

    def __init__(self, path: Path | str, time_s: float = QUERY_TIME_S):
        import duckdb

        self.time_s = time_s
        self._duckdb = duckdb
        # the same config every time: DuckDB shares one instance per file in a process
        self.db = duckdb.connect(str(path), read_only=True,
                                 config={"enable_external_access": False,
                                         "threads": 2})  # fmt: skip

    def version(self) -> str:
        return "DuckDB " + self._duckdb.__version__

    def run(self, query: str) -> Result:
        c = _Clock()
        why = gate("duckdb", query)
        if why:
            return Result.fail(why)
        timer = threading.Timer(self.time_s, self.db.interrupt)
        timer.start()
        try:
            cur = self.db.execute(strip_end(query))
            names = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1) if cur.description else []
        except self._duckdb.InterruptException:
            return Result.fail(MSG_TIMEOUT, ms=c.ms())
        except self._duckdb.Error as e:
            if not timer.is_alive() and "interrupt" in str(e).lower():
                return Result.fail(MSG_TIMEOUT, ms=c.ms())
            lines = [x for x in str(e).splitlines() if x.strip()]
            msg = lines[0] if lines else str(e)
            if isinstance(e, self._duckdb.PermissionException) or "read-only" in msg:
                msg = MSG_READ
            return Result.fail(msg, " ".join(lines[1:3]), ms=c.ms())
        finally:
            timer.cancel()
        return Result.of(names, [tuple(r) for r in rows], c.ms())

    def tables(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
            "ORDER BY table_name").fetchall()]  # fmt: skip

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.db.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
            [table]).fetchall()]  # fmt: skip

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        return []  # ported without constraints (porting.py): the prompt reads SQLite's

    def admin(self, q: str) -> tuple[list[str], list[tuple]]:
        cur = self.db.execute(q)
        return [d[0] for d in cur.description or []], cur.fetchmany(MAX_ROWS + 1)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


# ---- PostgreSQL -------------------------------------------------------------------------------


def pg_connect(db: str, user: str | None = None, read_only: bool = True, time_s: float = 0):
    """A psycopg connection to database `db` (as `run3_reader` by default)."""
    import psycopg

    opts = []
    if read_only:
        opts.append("-c default_transaction_read_only=on")
    if time_s:
        opts.append(f"-c statement_timeout={int(time_s * 1000)}")
    opts.append("-c idle_in_transaction_session_timeout=120000")
    conninfo = settings.get().engine("postgres")
    return psycopg.connect(conninfo, dbname=db, user=user or settings.READER,
                           options=" ".join(opts), connect_timeout=10)  # fmt: skip


class Postgres:
    """One database `r3_<id>` as `run3_reader`, for one thread."""

    name = "postgres"
    dialect = "sql"

    def __init__(self, key: str, time_s: float = QUERY_TIME_S):
        self.dbname, self.time_s = db_name(key), time_s
        self.db = pg_connect(self.dbname, time_s=time_s)
        self._n = 0

    def version(self) -> str:
        v = self.db.info.server_version  # e.g. 160004
        return f"PostgreSQL {v // 10000}.{v % 100}"

    def run(self, query: str) -> Result:
        import psycopg

        c = _Clock()
        why = gate("postgres", query)
        if why:
            return Result.fail(why)
        self._n += 1
        try:
            with self.db.transaction():
                # a server-side cursor: a huge result is never sent whole
                with self.db.cursor(name=f"q{self._n}") as cur:
                    cur.execute(strip_end(query))
                    rows = cur.fetchmany(MAX_ROWS + 1)
                    names = [d.name for d in cur.description or []]
        except psycopg.errors.QueryCanceled:
            return Result.fail(MSG_TIMEOUT, ms=c.ms())
        except (psycopg.errors.ReadOnlySqlTransaction, psycopg.errors.InsufficientPrivilege):
            return Result.fail(MSG_READ, ms=c.ms())
        except psycopg.Error as e:
            d = e.diag
            hint = " ".join(x for x in (d.message_detail, d.message_hint and
                                        "HINT: " + d.message_hint) if x)  # fmt: skip
            if self.db.closed:
                self.db = pg_connect(self.dbname, time_s=self.time_s)
            return Result.fail(d.message_primary or str(e), hint, ms=c.ms())
        return Result.of(names, [tuple(r) for r in rows], c.ms())

    def _admin_rows(self, q: str, args: tuple = ()) -> list[tuple]:
        with self.db.transaction(), self.db.cursor() as cur:
            cur.execute(q, args)
            return cur.fetchall()

    def tables(self) -> list[str]:
        return [r[0] for r in self._admin_rows(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
            "ORDER BY table_name")]  # fmt: skip

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self._admin_rows(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
            (table,))]  # fmt: skip

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        return []

    def admin(self, q: str) -> tuple[list[str], list[tuple]]:
        with self.db.transaction(), self.db.cursor() as cur:
            cur.execute(q)
            return [d.name for d in cur.description or []], cur.fetchmany(MAX_ROWS + 1)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


# ---- MySQL ------------------------------------------------------------------------------------


def my_connect(db: str | None, user: str | None = None, time_s: float = 0, stream: bool = True):
    """A PyMySQL connection (as `run3_reader` by default). `stream`: rows are read as they come
    (a huge result is never held whole)."""
    import pymysql
    import pymysql.cursors

    host, _, port = settings.get().engine("mysql").partition(":")
    conn = pymysql.connect(
        host=host, port=int(port or 3306), user=user or settings.READER, database=db,
        charset="utf8mb4", connect_timeout=10, local_infile=user is not None,
        cursorclass=pymysql.cursors.SSCursor if stream else pymysql.cursors.Cursor,
        autocommit=True,
    )  # fmt: skip
    if time_s:
        with conn.cursor() as cur:
            cur.execute(f"SET SESSION max_execution_time = {int(time_s * 1000)}")
            cur.execute("SET SESSION TRANSACTION READ ONLY")
    return conn


class MySQL:
    """One database `r3_<id>` as `run3_reader`, for one thread."""

    name = "mysql"
    dialect = "sql"

    def __init__(self, key: str, time_s: float = QUERY_TIME_S):
        self.dbname, self.time_s = db_name(key), time_s
        self.db: Any = None
        self._open()

    def _open(self) -> None:
        self.db = my_connect(self.dbname, time_s=self.time_s)

    def version(self) -> str:
        return "MySQL " + self.db.get_server_info().split("-")[0]

    def _drop(self, cur: Any = None) -> None:
        """Close without reading the rest of a cut result (PyMySQL would read it all, and its
        cursor and result try again when they are collected: both are told it is over)."""
        for res in (getattr(cur, "_result", None), getattr(self.db, "_result", None)):
            if res is not None:
                res.unbuffered_active = False
        if cur is not None:
            cur._result = None
        try:
            self.db._result = None
            self.db._force_close()
        except Exception:
            pass
        self.db = None

    def run(self, query: str) -> Result:
        import pymysql

        c = _Clock()
        why = gate("mysql", query)
        if why:
            return Result.fail(why)
        if self.db is None:
            self._open()
        cut = False
        cur = None
        try:
            cur = self.db.cursor()
            cur.execute(strip_end(query))
            names = [d[0] for d in cur.description or []]
            rows = list(cur.fetchmany(MAX_ROWS + 1)) if cur.description else []
            cut = len(rows) > MAX_ROWS
            if not cut:
                cur.close()
        except pymysql.MySQLError as e:
            code = e.args[0] if e.args else 0
            msg = e.args[1] if len(e.args) > 1 else str(e)
            self._drop(cur)
            if code == 3024:  # max_execution_time exceeded
                return Result.fail(MSG_TIMEOUT, ms=c.ms())
            if code in (1142, 1792, 1044, 1227):  # denied / read-only transaction
                return Result.fail(MSG_READ, ms=c.ms())
            return Result.fail(f"({code}) {msg}" if code else msg, ms=c.ms())
        if cut:
            self._drop(cur)
        return Result.of(names, [tuple(r) for r in rows], c.ms())

    def _admin_rows(self, q: str, args: tuple = ()) -> list[tuple]:
        if self.db is None:
            self._open()
        with self.db.cursor() as cur:
            cur.execute(q, args)
            return list(cur.fetchall())

    def tables(self) -> list[str]:
        return [r[0] for r in self._admin_rows(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s "
            "ORDER BY table_name", (self.dbname,))]  # fmt: skip

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self._admin_rows(
            "SELECT column_name, column_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (self.dbname, table))]  # fmt: skip

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        return []

    def admin(self, q: str) -> tuple[list[str], list[tuple]]:
        if self.db is None:
            self._open()
        with self.db.cursor() as cur:
            cur.execute(q)
            names = [d[0] for d in cur.description or []]
            rows = list(cur.fetchmany(MAX_ROWS + 1))
        return names, rows

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                self._drop()


# ---- Neo4j ------------------------------------------------------------------------------------


def cypher_block(query: str) -> str | None:
    """Why a Cypher query isn't sent to the graph at all (None: it is). Run 2's rule."""
    if not query.strip() or not guards.one_statement(query, backslash=True):
        return MSG_CYPHER_READ
    bare = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`", "''", query)
    if CYPHER_WRITE.search(bare):
        return MSG_CYPHER_READ
    for m in CYPHER_CALL.finditer(bare):
        if not m.group(1).lower().startswith(CALL_OK):
            return MSG_CALL
    return None


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


class Neo4j:
    """The run's Neo4j (Community: one database, so one graph at a time; `graphs.GraphHost`
    decides which). `run` sends a model query as a READ transaction with a server time limit."""

    name = "neo4j"
    dialect = "cypher"

    def __init__(self, url: str | None = None, time_s: float = QUERY_TIME_S):
        self.url = (url or settings.get().engine("neo4j")).rstrip("/")
        self.time_s = time_s

    def _post(self, statements: list[dict[str, Any]], read: bool, limit: int | None = None,
              timeout_ms: int | None = None) -> dict[str, Any]:  # fmt: skip
        body = json.dumps({"statements": statements}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if read:
            headers["access-mode"] = "READ"
        if timeout_ms:
            headers["max-execution-time"] = str(timeout_ms)
        req = urllib.request.Request(f"{self.url}/db/neo4j/tx/commit", data=body, headers=headers)
        wait = (timeout_ms or 600_000) / 1000 + 20
        with urllib.request.urlopen(req, timeout=wait) as r:
            raw = r.read(limit + 1) if limit else r.read()
        if limit and len(raw) > limit:
            raise OverflowError(MSG_TOO_BIG)
        return json.loads(raw)

    def version(self) -> str:
        try:
            out = self._post([{"statement": "CALL dbms.components() YIELD versions "
                                            "RETURN versions[0]"}], read=True)  # fmt: skip
            return "Neo4j " + str(out["results"][0]["data"][0]["row"][0])
        except Exception:
            return "Neo4j 5"

    def ok(self) -> bool:
        try:
            return self.admin("RETURN 1")[1] == [(1,)]
        except Exception:
            return False

    def run(self, query: str) -> Result:
        c = _Clock()
        why = cypher_block(query)
        if why:
            return Result.fail(why)
        st = {"statement": strip_end(query), "parameters": {}, "resultDataContents": ["row"]}
        try:
            out = self._post([st], read=True, limit=CYPHER_MAX_BYTES,
                             timeout_ms=int(self.time_s * 1000))  # fmt: skip
        except OverflowError:
            return Result.fail(MSG_TOO_BIG, ms=c.ms())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            return Result.fail(MSG_TIMEOUT if "timed out" in str(e).lower() else MSG_NO_DB,
                               ms=c.ms())  # fmt: skip
        if out.get("errors"):
            e = out["errors"][0]
            code, msg = e.get("code", ""), e.get("message", "")
            if "TransactionTimedOut" in code or "Terminated" in code:
                return Result.fail(MSG_TIMEOUT, ms=c.ms())
            if "AccessMode" in code or "Forbidden" in code:
                return Result.fail(MSG_CYPHER_READ, ms=c.ms())
            lines = [x for x in msg.splitlines() if x.strip()]
            return Result.fail(lines[0] if lines else code, " ".join(lines[1:2]), ms=c.ms())
        res = (out.get("results") or [{}])[0]
        rows = [tuple(_hashable(v) for v in (d.get("row") or ())) for d in res.get("data") or []]
        return Result.of(list(res.get("columns") or []), rows, c.ms())

    def admin(self, q: str, params: dict[str, Any] | None = None,
              read: bool = True) -> tuple[list[str], list[tuple]]:  # fmt: skip
        """The harness's own query (raises on an error)."""
        out = self._post([{"statement": q, "parameters": params or {},
                           "resultDataContents": ["row"]}], read=read)  # fmt: skip
        if out.get("errors"):
            e = out["errors"][0]
            raise RuntimeError(f"{e.get('code')}: {e.get('message', '')[:300]}")
        res = out["results"][0]
        return res["columns"], [tuple(_hashable(v) for v in d["row"]) for d in res["data"]]

    def close(self) -> None:
        pass


# ---- one place to open any of them ------------------------------------------------------------


def quote_ident(engine: str, name: str) -> str:
    if engine == "mysql":
        return "`" + name.replace("`", "``") + "`"
    if engine == "neo4j":
        return "`" + name.replace("`", "``") + "`"
    return '"' + name.replace('"', '""') + '"'


def sqlite_path(source: str, db_id: str) -> Path:
    return settings.get().dbs / source / f"{db_id}.sqlite"


def duckdb_path(source: str, db_id: str) -> Path:
    return settings.get().duckdb_dir / source / f"{db_id}.duckdb"


def open_sql(engine: str, source: str, db_id: str, time_s: float = QUERY_TIME_S):
    """A model-facing connection to a SQL database on `engine`."""
    if engine == "sqlite":
        return SQLite(sqlite_path(source, db_id), time_s)
    if engine == "duckdb":
        return DuckDB(duckdb_path(source, db_id), time_s)
    if engine == "postgres":
        return Postgres(db_key(source, db_id), time_s)
    if engine == "mysql":
        return MySQL(db_key(source, db_id), time_s)
    raise ValueError(f"not a SQL engine: {engine!r}")


def db_key(source: str, db_id: str) -> str:
    """The id a server engine knows a database by (two sources can share a `db_id`)."""
    return f"{source}_{db_id}"
