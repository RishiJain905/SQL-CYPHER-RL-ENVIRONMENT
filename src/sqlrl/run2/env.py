"""The environment of a multi-turn episode: the database the model looks at, how its queries
run there, and what comes back (Windows, next to llama-server; the standard library plus the
trainer's checker modules, which are standard library too).

**Which database the model looks at** (decisions R15, design.md section 3):

| Dataset | Look-around copy | The reward is decided on |
|---|---|---|
| gretel | the dataset's own rows (`sql_context`) | the same rows |
| SQaLe | the schema with random rows (from the schema and the question) | the checker's fills |
| CypherBench | the benchmark's sampled graph | the same graph |
| text2cypher | a random graph (from the prompt's schema and the question) | the checker's graphs |

The checker's fills plant the reference query's literals into the data. A model that looked at
one would see values lifted from the answer's filter. So the look-around copies are built here,
by the checker's fill code but **without the reference**: the builder gets the schema and the
question, and seeds the checker never uses. `public_row` is the gate: it drops the reference
query and the reference reply before a question reaches any of this.

**Read only, enforced by the engine:**
- SQLite: `PRAGMA query_only`, and an authorizer that allows reading and nothing else (no write,
  no PRAGMA, no ATTACH, no temp table). The authorizer also reports which tables a query read,
  which `guards.hardcoded` uses. A 2 s limit (the checker's), a cap on rows and on value size.
- Neo4j: a read transaction (`access-mode: READ`), a server-side time limit per query, a cap on
  the response's size while it is read, and no procedure calls except schema listings.

**What comes back** (`render`): column names, the first rows, cells cut at 40 characters, and a
count when rows were cut; `0 rows`; or `Error: ` and the engine's own message (first line, 200
characters). Training varies the row cap and the table's form per episode (decisions R5: what
other harnesses won't fix is randomised); evaluation uses one setting.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Any

from sqlrl.run2 import calls

# What the model is shown for a query: which copy it looks at, how the copy is filled (rows,
# planting, seeds), the limits, and the result's text. Raise it with ANY change to those: an
# evaluation records it (`episodes.claim`), so an evaluation made on other look-around data is
# never continued, and never used as a run's start reference.
LOOK_VERSION = 1

MAX_ROWS = 10_000  # rows fetched to count a result (the checker's cap)
SQL_TIME_S = 2.0  # the checker's limit
SQL_MAX_VALUE = 1_000_000  # bytes of one value (randomblob, zeroblob, group_concat bombs)
CYPHER_TIME_MS = 5_000
CYPHER_MAX_BYTES = 4_000_000
ERR_CHARS = 200
CELL_CHARS = 40
EVAL_OBS = {"rows": 10, "style": "table"}
SEED_BASE = 1_000_003  # far above every seed the checker fills with (11 ... 112)
MAX_LITERALS = 12
PLANT_P = 0.3  # how often a question's value lands in a property of the random graph
FILL_ROWS = 60  # rows a table of the random SQL fill
FILL_LIT_P = 0.6  # how often a number or date is one the question names (or next to it)
FILL_TEXT_P = 0.75  # how often a text value is one the question names

SQL_READ = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)
CYPHER_CALL = re.compile(r"\bCALL\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
CALL_OK = ("db.labels", "db.relationshiptypes", "db.propertykeys", "db.schema.")

MSG_SQL_READ = "only one read-only SELECT statement is allowed"
MSG_CYPHER_READ = "only read queries are allowed"
MSG_CALL = "procedure calls are not available"
MSG_TIMEOUT = "the query took too long and was stopped"
MSG_TOO_BIG = "the result is too large to return"
MSG_NO_DB = "the database is not available"


def checker():
    """The trainer's `querycheck` and `cyphercheck`. Only their fill and load code is used here,
    never their seeds and never with a reference query."""
    from sqlrl.paths import get_paths

    src = str(get_paths().trainer_repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from asktrainer import cyphercheck, querycheck

    return querycheck, cyphercheck


# ---- what the rollout side may know about a question ------------------------------------------


def public_row(row: dict[str, Any], script: str | None = None) -> dict[str, Any]:
    """A question as the rollout side sees it: the two prompt messages, the tools, the query
    tool's name and argument, the dialect and the database. The reference query (`meta.gold`)
    and the reference reply (`messages[2:]`) stop here."""
    tool, arg = calls.query_tool(row)
    meta = row.get("meta") or {}
    origin = row["origin"]
    if origin in ("gretel", "sqale") and not script:
        # without it every look would answer "not available", silently: a missing sidecar row
        raise ValueError(f"{row['id']}: no database script (python -m sqlrl.run2.pool build-db)")
    if origin == "gretel":
        db: dict[str, Any] = {"kind": "sqlite_rows", "script": script}
    elif origin == "sqale":
        db = {"kind": "sqlite_fill", "script": script}
    elif origin == "cypherbench":
        db = {"kind": "graph", "graph": meta.get("db_id") or ""}
    elif origin == "text2cypher":
        db = {"kind": "graph_fill"}
    else:
        raise ValueError(f"{row['id']}: no environment for the dataset {origin!r}")
    return {
        "id": row["id"],
        "origin": origin,
        "dialect": row["dialect"],
        "messages": [
            {"role": m["role"], "content": m.get("content") or ""} for m in row["messages"][:2]
        ],  # fmt: skip
        "tools": row["tools"],
        "tool": tool,
        "arg": arg,
        "thinking": row.get("rl_thinking") or row.get("thinking") or "high",
        "prompt_tokens": row.get("prompt_tokens"),
        "db": db,
    }


# ---- the question's own words, for the random fills -------------------------------------------

_QUOTED = re.compile(
    r"(?<![\w])'([^'\n]{1,80})'(?![\w])|\"([^\"\n]{1,80})\"|“([^”\n]{1,80})”|‘([^’\n]{1,80})’"
)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.#-])-?\d+(?:\.\d+)?(?![\w.-])")
_CODE = re.compile(r"(?<![\w])[#@]?[A-Za-z_]*[\w.-]*\d[\w.-]*|(?<![\w])[#@]\w+")
_WORD = r"[A-Z][\w&.-]*(?:['’][A-Za-z]+)?"
_PROPER = re.compile(
    rf"\b{_WORD}(?:\s+(?:(?:of|the|and|de|del|la|van|von|der|for|in)\s+)?{_WORD})*"
)
_POSSESSIVE = re.compile(r"['’]s$")
_SENTENCE_START = re.compile(r"(?:^|[.?!]\s+)$")
_LEAD = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "list", "show",
    "find", "give", "get", "identify", "retrieve", "provide", "return", "calculate", "count",
    "determine", "display", "name", "tell", "can", "could", "would", "is", "are", "was", "were",
    "do", "does", "did", "the", "a", "an", "in", "for", "of", "on", "if", "please", "total",
    "average", "number", "i", "we", "all", "each", "every", "among", "between", "from", "by",
    "with", "to", "at", "as", "and", "or", "there", "these", "those", "this", "that", "it",
    "write", "select", "fetch", "compare", "rank", "sort", "group", "has", "have", "any",
}  # fmt: skip


def question_literals(question: str) -> tuple[list[str], list[float]]:
    """(text values, numbers) a person named in the question: quoted phrases, proper names,
    codes with digits or a leading `#` / `@`, dates and numbers. These are what the random fills
    are seeded with, so a filter on a value the question names can match something."""
    q = " ".join((question or "").split())
    strs: list[str] = []

    def add(s: str) -> None:
        s = _POSSESSIVE.sub("", s.strip().strip(".,;:?!()'’\"").strip())
        if 0 < len(s) <= 80 and s not in strs:
            strs.append(s)

    for m in _QUOTED.finditer(q):
        add(next(g for g in m.groups() if g is not None))
    for m in _DATE.finditer(q):
        add(m.group(0))
    for m in _CODE.finditer(q):
        tok = m.group(0).strip(".")
        if not _NUMBER.fullmatch(tok) and not _DATE.fullmatch(tok):
            add(tok)
    for m in _PROPER.finditer(q):
        words = m.group(0).split()
        while words and words[0].lower().strip("'’.,") in _LEAD:
            words = words[1:]
        while words and words[-1].lower() in ("of", "the", "and", "for", "in"):
            words = words[:-1]
        if not words or not words[0][:1].isupper():
            continue
        # one capitalised word that only opens a sentence ("Earliest ...") names nothing
        opens = len(words) == 1 and _SENTENCE_START.search(q[: m.start()]) is not None
        if not (opens and len(m.group(0).split()) == 1 and words[0].isalpha()):
            add(" ".join(words))
    nums: list[float] = []
    for m in _NUMBER.finditer(_DATE.sub(" ", q)):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        if abs(v) < 10**9 and v not in nums:
            nums.append(v)
    return strs[:MAX_LITERALS], nums[:MAX_LITERALS]


def look_seed(qid: str) -> int:
    """The seed of a question's random fill: fixed per question, never one of the checker's."""
    return SEED_BASE + zlib.crc32(qid.encode("utf-8"))


# ---- results ----------------------------------------------------------------------------------


@dataclass
class Result:
    """What one query did. `kind`: rows | empty | error."""

    kind: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    total: int | None = None  # None: more than `MAX_ROWS`
    error: str = ""
    tables: list[str] | None = None  # SQL: the tables the query read
    ms: float = 0.0

    @staticmethod
    def fail(msg: str, ms: float = 0.0) -> Result:
        return Result("error", error=first_line(msg), ms=ms)


def first_line(msg: str) -> str:
    line = next((x for x in (msg or "").splitlines() if x.strip()), "").strip()
    return clean(line)[:ERR_CHARS] or "the query failed"


def clean(s: str) -> str:
    """Text without control characters (newlines and tabs become spaces)."""
    return "".join(" " if ch in "\r\n\t" else ch for ch in s if ch in "\r\n\t" or ch >= " ")


def cell(v: Any, style: str) -> Any:
    """One value as the model reads it: cut at `CELL_CHARS`, nested values as compact JSON."""
    if v is None:
        return None if style == "json" else "NULL"
    if isinstance(v, bool):
        return v if style == "json" else str(v).lower()
    if isinstance(v, int):
        return v if style == "json" else str(v)
    if isinstance(v, float):
        return v if style == "json" else format(v, ".10g")
    if isinstance(v, (bytes, bytearray)):
        s = "x'" + bytes(v[:16]).hex() + "'"
    elif isinstance(v, str):
        s = clean(v)
    else:
        s = clean(json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=str))
    return s if len(s) <= CELL_CHARS else s[: CELL_CHARS - 3] + "..."


def render(res: Result, rows: int, style: str) -> str:
    """The tool's reply text for a result."""
    if res.kind == "error":
        return "Error: " + res.error
    if res.kind == "empty":
        return "0 rows"
    shown = res.rows[:rows]
    names = [clean(str(c)) for c in res.columns]
    if style == "json":
        lines = [json.dumps({n: cell(v, style) for n, v in zip(names, r, strict=False)},
                            ensure_ascii=False) for r in shown]  # fmt: skip
    elif style == "table":
        lines = [" | ".join(names)] + [" | ".join(str(cell(v, style)) for v in r) for r in shown]
    else:
        raise ValueError(f"unknown result style {style!r}")
    n = len(res.rows) if res.total is None else res.total
    if n > len(shown) or res.total is None:
        of = f"more than {MAX_ROWS}" if res.total is None else str(res.total)
        lines.append(f"({len(shown)} of {of} rows)")
    return "\n".join(lines)


def obs_setting(cfg: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """An episode's result format: one of the training settings by `seed`, or (seed None) the
    evaluation's fixed one."""
    o = cfg.get("obs") or {}
    if seed is None:
        return dict(o.get("eval") or EVAL_OBS)
    rnd = random.Random(seed)
    return {"rows": rnd.choice(list(o.get("rows") or [EVAL_OBS["rows"]])),
            "style": rnd.choice(list(o.get("styles") or [EVAL_OBS["style"]]))}  # fmt: skip


# ---- SQLite -----------------------------------------------------------------------------------

_SQL_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE}  # fmt: skip
_SQL_NO_FUNCTION = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class SqlLook:
    """A question's SQLite database for one worker thread (built and read on that thread)."""

    def __init__(self, pub: dict[str, Any]):
        qc, _ = checker()
        db_info = pub["db"]
        self.error: str | None = None
        self._reads: set[tuple[str, str]] = set()
        self._tables: set[str] = set()
        self._read_by_sql: dict[str, list[str]] = {}
        self.db: sqlite3.Connection | None = None
        try:
            if not db_info.get("script"):
                raise sqlite3.Error("no schema")
            if db_info["kind"] == "sqlite_fill":
                # the checker's fill code with the question's own values planted often, so a
                # filter on a value the question names finds rows (measured on the dev slice:
                # `phase0_score lookcheck`)
                lits = question_literals(pub["messages"][1]["content"])
                self.db = qc.fuzz_db(
                    qc._creates(db_info["script"]),
                    look_seed(pub["id"]),
                    lits,
                    rows=FILL_ROWS,
                    lit_p=FILL_LIT_P,
                    text_p=FILL_TEXT_P,
                )
            else:
                self.db = qc._connect(db_info["script"])
            self._tables = {r[0].lower() for r in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}  # fmt: skip
            self.db.commit()
            self.db.execute("PRAGMA query_only=ON")
            self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, SQL_MAX_VALUE)
            self.db.set_authorizer(self._authorize)
        except Exception as e:  # the schema doesn't build: every query gets "not available"
            self.error = f"{type(e).__name__}: {str(e)[:120]}"
            self.close()

    def _authorize(self, action: int, a1: str | None, a2: str | None, *_: Any) -> int:
        if action == sqlite3.SQLITE_READ:
            self._reads.add((a1 or "", a2 or ""))
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (a2 or "").lower() in _SQL_NO_FUNCTION:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in _SQL_ALLOWED else sqlite3.SQLITE_DENY

    def run(self, query: str) -> Result:
        t0 = time.monotonic()

        def ms() -> float:
            return round((time.monotonic() - t0) * 1000, 1)

        if self.db is None:
            return Result.fail(MSG_NO_DB)
        q = query.strip().rstrip(";").strip()
        if not SQL_READ.match(q):
            return Result.fail(MSG_SQL_READ)
        self._reads = set()
        deadline = time.monotonic() + SQL_TIME_S
        self.db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = self.db.execute(q)
            rows = cur.fetchmany(MAX_ROWS + 1)
            names = [d[0] for d in cur.description or []]
        except sqlite3.Error as e:
            late = time.monotonic() > deadline
            msg = MSG_TIMEOUT if late else (MSG_SQL_READ if "not authorized" in str(e) else str(e))
            return Result.fail(msg, ms())
        except Exception as e:  # e.g. "You can only execute one statement at a time."
            return Result.fail(MSG_SQL_READ if "one statement" in str(e) else str(e), ms())
        finally:
            self.db.set_progress_handler(None, 0)
        # SQLite asks the authorizer when it prepares a statement, and Python keeps prepared
        # statements: the same text run again reads nothing new. So a query's tables are
        # remembered from its first run (a repeated right answer must not look like constants)
        if q not in self._read_by_sql:
            self._read_by_sql[q] = sorted({t for t, _ in self._reads if t.lower() in self._tables})
        tables = list(self._read_by_sql[q])
        if not rows:
            return Result("empty", columns=names, total=0, tables=tables, ms=ms())
        total = len(rows) if len(rows) <= MAX_ROWS else None
        return Result("rows", names, rows[:MAX_ROWS], total, tables=tables, ms=ms())

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            except sqlite3.Error:
                pass
            self.db = None


# ---- Neo4j ------------------------------------------------------------------------------------


def cypher_block(query: str) -> str | None:
    """Why a Cypher query isn't sent to the graph at all (None: it is)."""
    bare = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`", "''", query)
    if CYPHER_WRITE.search(bare):
        return MSG_CYPHER_READ
    for m in CYPHER_CALL.finditer(bare):
        if not m.group(1).lower().startswith(CALL_OK):
            return MSG_CALL
    return None


class GraphHost:
    """The one Neo4j every Cypher question shares (the checker's: nothing else is in it), and
    the graph it holds now. `ensure` loads a question's graph; the scheduler (`episodes.Plan`)
    makes sure no episode of another graph is running when it does."""

    def __init__(self, url: str | None = None):
        _, cc = checker()
        self.cc = cc
        self.db = cc.Neo4j(url or cc.NEO4J_URL)
        self.url = self.db.url
        self.loaded: tuple[str, str] | None = None
        self.failed: dict[tuple[str, str], str] = {}
        self.load_s: dict[str, float] = {}
        self._lock = threading.Lock()

    def ok(self) -> bool:
        return self.db.ok()

    @staticmethod
    def key(pub: dict[str, Any]) -> tuple[str, str]:
        """The graph a question needs: one per CypherBench graph, one per text2cypher question."""
        if pub["db"]["kind"] == "graph":
            return ("graph", pub["db"]["graph"])
        return ("fill", pub["id"])

    def ensure(self, pub: dict[str, Any]) -> None:
        key = self.key(pub)
        with self._lock:
            if self.loaded == key or key in self.failed:
                return
            self.loaded = None
            t0 = time.time()
            try:
                if key[0] == "graph":
                    self.cc.load(self.db, *self.cc.cypherbench_graph(key[1]), key="eid",
                                 keep_key=True, index=True)  # fmt: skip
                else:
                    schema = self.cc.parse_schema(pub["messages"][0]["content"])
                    if not schema["nodes"]:
                        raise ValueError("no schema in the prompt")
                    strs, nums = question_literals(pub["messages"][1]["content"])
                    by_prop: dict[str, list[Any]] = {}
                    props = [p for ps in schema["nodes"].values() for p in ps.items()]
                    props += [p for *_, ps in schema["rels"] for p in ps.items()]
                    for name, kind in props:
                        k = kind.upper()
                        vals: list[Any] = (
                            [int(n) if n == int(n) else n for n in nums]
                            if k.startswith("INT") or k == "FLOAT" else list(strs)
                        )  # fmt: skip
                        if vals:
                            by_prop[name] = vals
                    lits = {"by_prop": by_prop, "strs": strs, "nums": nums}
                    self.cc.load(self.db, *self.cc.fuzz_graph(schema, look_seed(pub["id"]), lits,
                                                              plant_p=PLANT_P))  # fmt: skip
                self.loaded = key
            except Exception as e:
                self.failed[key] = f"{type(e).__name__}: {str(e)[:120]}"
            self.load_s[f"{key[0]}:{key[1]}"] = round(time.time() - t0, 2)

    def run(self, pub: dict[str, Any], query: str) -> Result:
        t0 = time.monotonic()

        def ms() -> float:
            return round((time.monotonic() - t0) * 1000, 1)

        key = self.key(pub)
        if key in self.failed or self.loaded != key:
            return Result.fail(MSG_NO_DB)
        why = cypher_block(query)
        if why:
            return Result.fail(why)
        body = json.dumps(
            {"statements": [{"statement": query, "parameters": {}, "resultDataContents": ["row"]}]}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/db/{self.db.db}/tx/commit", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "access-mode": "READ", "max-execution-time": str(CYPHER_TIME_MS)},
        )  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=CYPHER_TIME_MS / 1000 + 20) as r:
                raw = r.read(CYPHER_MAX_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return Result.fail(MSG_TIMEOUT if "timed out" in str(e).lower() else MSG_NO_DB, ms())
        if len(raw) > CYPHER_MAX_BYTES:
            return Result.fail(MSG_TOO_BIG, ms())
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            return Result.fail(MSG_NO_DB, ms())
        if out.get("errors"):
            e = out["errors"][0]
            code, msg = e.get("code", ""), e.get("message", "")
            if "TransactionTimedOut" in code or "Terminated" in code:
                msg = MSG_TIMEOUT
            elif "AccessMode" in code or "Forbidden" in code:
                msg = MSG_CYPHER_READ
            return Result.fail(msg, ms())
        res = (out.get("results") or [{}])[0]
        rows = [tuple(d.get("row") or ()) for d in res.get("data") or []]
        names = list(res.get("columns") or [])
        if not rows:
            return Result("empty", columns=names, total=0, ms=ms())
        total = len(rows) if len(rows) <= MAX_ROWS else None
        return Result("rows", names, rows[:MAX_ROWS], total, ms=ms())


# ---- one question's look-around database --------------------------------------------------------


class NoLook:
    """What a one-query question gets: nothing is ever run in front of the model."""

    error = None

    def run(self, query: str) -> Result:
        raise RuntimeError("a one-query question is never run on a look-around copy")

    def close(self) -> None:
        pass


class Look:
    """The database of one question, for one worker thread: `run(query)` -> `Result`."""

    def __init__(self, pub: dict[str, Any], graphs: GraphHost | None = None):
        self.pub = pub
        self.sql: SqlLook | None = None
        self.graphs = graphs
        if pub["dialect"] == "cypher":
            if graphs is None:
                raise RuntimeError("a Cypher question needs the graph host")
        else:
            self.sql = SqlLook(pub)

    @property
    def error(self) -> str | None:
        if self.sql is not None:
            return self.sql.error
        return self.graphs.failed.get(self.graphs.key(self.pub))

    def run(self, query: str) -> Result:
        if self.sql is not None:
            return self.sql.run(query)
        return self.graphs.run(self.pub, query)

    def close(self) -> None:
        if self.sql is not None:
            self.sql.close()
