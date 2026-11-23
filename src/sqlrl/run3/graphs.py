"""Neo4j for run 3: graph copies of the SQL databases, the CypherBench graphs, and the helper
tools and schema text on a graph (design.md 2.1, 3.3).

**One graph at a time.** Neo4j Community holds one database, so `GraphHost.ensure(key)` wipes it
and loads the graph an episode needs (the trainer's `cyphercheck.load`); episodes are grouped by
graph so this happens rarely. Keys: `cypherbench:<name>` (the benchmark's sampled graph, as runs
1 and 2 used) and `copy:<source>:<db_id>` (a SQL database as a graph).

**A SQL database as a property graph** (`convert`), by fixed rules:
- a table's row is a node, labelled with the table's name in CamelCase, its columns as
  properties (a name with a space or symbol gets `_` instead);
- a declared foreign key becomes a relationship from the row to the row it points at, typed from
  the column (`publisher_id` -> `PUBLISHER`); the column itself stays a property only if some
  value doesn't resolve to exactly one row;
- a table that only joins two others (two foreign keys, at most two other columns, nothing
  pointing at it) becomes relationships of its name, its other columns their properties.

`roundtrip` reads the converted graph back into tables and compares them with the source, row
for row (multisets), before anything is loaded: a copy that loses or invents a row is refused.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlrl.run3 import engines, settings

MAX_COPY_ROWS = 300_000  # databases larger than this aren't copied (design 3.3: start small)
JOIN_EXTRA_COLS = 2


def label_of(table: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", table)
    s = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return s if s and not s[0].isdigit() else "T" + s


def prop_of(column: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", column).strip("_")
    return s if s and not s[0].isdigit() else "c_" + s


def rel_of(column: str, target: str) -> str:
    base = re.sub(r"(?i)[_ ]?id$", "", column).strip("_ ") or target
    return re.sub(r"[^0-9A-Za-z]+", "_", base).upper().strip("_") or "LINKS_TO"


# ---- SQL -> graph -----------------------------------------------------------------------------


def _schema(src: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    tabs: dict[str, dict[str, Any]] = {}
    for (t,) in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT "
                            "LIKE 'sqlite_%' ORDER BY name"):  # fmt: skip
        q = engines.quote_ident("sqlite", t)
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({q})")]
        fks = [(r[3], r[2], r[4]) for r in src.execute(f"PRAGMA foreign_key_list({q})") if r[4]]
        tabs[t] = {"cols": cols, "fks": fks}
    # foreign keys to tables or columns that don't exist are dropped
    low = {t.lower(): t for t in tabs}
    for info in tabs.values():
        ok = []
        for col, other, ocol in info["fks"]:
            o = low.get(other.lower())
            if o and col in info["cols"] and ocol.lower() in (c.lower() for c in tabs[o]["cols"]):
                ocol = next(c for c in tabs[o]["cols"] if c.lower() == ocol.lower())
                ok.append((col, o, ocol))
        info["fks"] = ok
    return tabs


def _is_join(t: str, tabs: dict[str, dict[str, Any]]) -> bool:
    info = tabs[t]
    if len(info["fks"]) != 2 or len({f[0] for f in info["fks"]}) != 2:
        return False
    if any(o == t for _, o, _ in info["fks"]):
        return False
    pointed_at = any(o == t for x in tabs.values() for _, o, _ in x["fks"])
    extra = [c for c in info["cols"] if c not in {f[0] for f in info["fks"]}]
    return not pointed_at and len(extra) <= JOIN_EXTRA_COLS


def convert(source: str, db_id: str) -> dict[str, Any]:
    """The graph of a SQL database: {"nodes", "rels", "types", "plan", "rows"} where `plan`
    records every decision (for the round trip and the docs)."""
    path = engines.sqlite_path(source, db_id)
    src = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tabs = _schema(src)
        total = sum(src.execute(f"SELECT count(*) FROM {engines.quote_ident('sqlite', t)}")
                    .fetchone()[0] for t in tabs)  # fmt: skip
        if total > MAX_COPY_ROWS:
            raise ValueError(f"{total} rows: more than {MAX_COPY_ROWS} (not copied)")
        rows = {t: [tuple(r) for r in src.execute(
            f"SELECT {', '.join(engines.quote_ident('sqlite', c) for c in tabs[t]['cols'])} "
            f"FROM {engines.quote_ident('sqlite', t)}")] for t in tabs}  # fmt: skip
    finally:
        src.close()
    joins = {t for t in tabs if _is_join(t, tabs)}
    # index every referenced column: value -> the node keys of rows holding it
    index: dict[tuple[str, str], dict[Any, list[str]]] = {}
    for info in tabs.values():
        for _, o, oc in info["fks"]:
            if (o, oc) not in index and o not in joins:
                i = tabs[o]["cols"].index(oc)
                idx: dict[Any, list[str]] = defaultdict(list)
                for n, r in enumerate(rows[o]):
                    if r[i] is not None:
                        idx[r[i]].append(f"{o}:{n}")
                index[(o, oc)] = idx

    def resolves(t: str, col: str, o: str, oc: str) -> bool:
        i = tabs[t]["cols"].index(col)
        idx = index.get((o, oc)) or {}
        return all(v is None or len(idx.get(v, ())) == 1 for v in (r[i] for r in rows[t]))

    plan: dict[str, Any] = {"labels": {}, "edges": [], "joins": {}}
    nodes: dict[str, list[dict[str, Any]]] = {}
    rels: list[tuple[str, str, str, str, str, dict[str, Any]]] = []
    for t, info in tabs.items():
        if t in joins:
            continue
        dropped = set()
        for col, o, oc in info["fks"]:
            if o in joins:
                continue
            typ = rel_of(col, o)
            keep = not resolves(t, col, o, oc)
            plan["edges"].append({"table": t, "column": col, "to": o, "to_column": oc,
                                  "type": typ, "kept_as_property": keep})  # fmt: skip
            if not keep:
                dropped.add(col)
            i = info["cols"].index(col)
            for n, r in enumerate(rows[t]):
                hit = index[(o, oc)].get(r[i]) if r[i] is not None else None
                if hit and len(hit) == 1:
                    rels.append((typ, label_of(t), f"{t}:{n}", label_of(o), hit[0], {}))
        props = [(c, prop_of(c)) for c in info["cols"] if c not in dropped]
        plan["labels"][t] = {"label": label_of(t), "props": dict(props)}
        nodes[label_of(t)] = [
            {"_k": f"{t}:{n}", **{p: r[info["cols"].index(c)] for c, p in props
                                  if r[info["cols"].index(c)] is not None}}
            for n, r in enumerate(rows[t])]  # fmt: skip
    for t in joins:
        # the relationship runs from the row its first key column points at
        (ca, oa, oca), (cb, ob, ocb) = sorted(
            tabs[t]["fks"], key=lambda f: tabs[t]["cols"].index(f[0])
        )
        extra = [c for c in tabs[t]["cols"] if c not in (ca, cb)]
        typ = re.sub(r"[^0-9A-Za-z]+", "_", t).upper()
        plan["joins"][t] = {"type": typ, "from": oa, "to": ob, "props": {c: prop_of(c)
                                                                       for c in extra}}  # fmt: skip
        ia, ib = tabs[t]["cols"].index(ca), tabs[t]["cols"].index(cb)
        for r in rows[t]:
            ha = index[(oa, oca)].get(r[ia]) if r[ia] is not None else None
            hb = index[(ob, ocb)].get(r[ib]) if r[ib] is not None else None
            if ha and hb and len(ha) == 1 and len(hb) == 1:
                p = {prop_of(c): r[tabs[t]["cols"].index(c)] for c in extra
                     if r[tabs[t]["cols"].index(c)] is not None}  # fmt: skip
                rels.append((typ, label_of(oa), ha[0], label_of(ob), hb[0], p))
    return {"nodes": nodes, "rels": rels, "types": {}, "plan": plan, "rows": rows,
            "tabs": tabs, "joins": sorted(joins)}  # fmt: skip


def roundtrip(g: dict[str, Any]) -> dict[str, Any]:
    """Rebuild every source table from the graph (in memory) and compare as multisets."""
    tabs, rows, plan = g["tabs"], g["rows"], g["plan"]
    by_key: dict[str, dict[str, Any]] = {}
    for label_rows in g["nodes"].values():
        for n in label_rows:
            by_key[n["_k"]] = n
    out_edges: dict[tuple[str, str], str] = {}
    for typ, _a, ka, _b, kb, _p in g["rels"]:
        out_edges[(ka, typ)] = kb
    bad = []
    for t, info in tabs.items():
        if t in g["joins"]:
            j = plan["joins"][t]
            (ca, oa, oca), (cb, ob, ocb) = sorted(
                info["fks"], key=lambda f: info["cols"].index(f[0])
            )
            back = []
            for typ, _a, ka, _b, kb, p in g["rels"]:
                if typ != j["type"]:
                    continue
                vals = {ca: by_key[ka].get(prop_of(oca)), cb: by_key[kb].get(prop_of(ocb))}
                vals.update({c: p.get(pp) for c, pp in j["props"].items()})
                back.append(tuple(vals.get(c) for c in info["cols"]))
            want = [r for r in rows[t]]
        else:
            lab = plan["labels"][t]
            edges = [e for e in plan["edges"] if e["table"] == t and not e["kept_as_property"]]
            back = []
            for n in range(len(rows[t])):
                node = by_key[f"{t}:{n}"]
                vals = {c: node.get(p) for c, p in lab["props"].items()}
                for e in edges:
                    target = out_edges.get((f"{t}:{n}", e["type"]))
                    vals[e["column"]] = (by_key[target].get(prop_of(e["to_column"]))
                                         if target else None)  # fmt: skip
                back.append(tuple(vals.get(c) for c in info["cols"]))
            want = rows[t]
        if Counter(back) != Counter(want):
            bad.append(t)
    return {"ok": not bad, "tables_differ": bad}


# ---- the host ---------------------------------------------------------------------------------


class GraphHost:
    """The run's Neo4j and the graph it holds now."""

    def __init__(self, url: str | None = None):
        _, cc = settings.checker()
        self.cc = cc
        self.url = url or settings.get().engine("neo4j")
        self.db = cc.Neo4j(self.url)
        self.loaded: str | None = None
        self.failed: dict[str, str] = {}
        self.load_s: dict[str, float] = {}
        self._lock = threading.Lock()
        self._free = threading.Condition()
        self._users = 0  # episodes using the loaded graph now

    def acquire(self, key: str) -> bool:
        """Hold graph `key` for one episode: waits while other episodes use another graph, then
        loads it if needed. Every `acquire` that returns True is matched by a `release`."""
        with self._free:
            while self._users and self.loaded != key:
                self._free.wait()
            if not self.ensure(key):
                return False
            self._users += 1
            return True

    def release(self) -> None:
        with self._free:
            self._users -= 1
            self._free.notify_all()

    def ensure(self, key: str) -> bool:
        with self._lock:
            if self.loaded == key:
                return True
            if key in self.failed:
                return False
            self.loaded = None
            t0 = time.time()
            try:
                kind, _, rest = key.partition(":")
                if kind == "cypherbench":
                    self.cc.load(self.db, *self.cc.cypherbench_graph(rest, cypherbench_root()),
                                 key="eid", keep_key=True, index=True)  # fmt: skip
                elif kind == "copy":
                    source, _, db_id = rest.partition(":")
                    g = convert(source, db_id)
                    rt = roundtrip(g)
                    if not rt["ok"]:
                        raise ValueError(f"round trip differs: {rt['tables_differ']}")
                    self.cc.load(self.db, g["nodes"], g["rels"], g["types"], key="_k",
                                 index=True)  # fmt: skip
                else:
                    raise ValueError(f"unknown graph key {key!r}")
                self.loaded = key
            except Exception as e:
                self.failed[key] = f"{type(e).__name__}: {str(e)[:200]}"
            self.load_s[key] = round(time.time() - t0, 2)
            return self.loaded == key


_HOST: GraphHost | None = None
_HOST_LOCK = threading.Lock()


def host() -> GraphHost:
    """The process's one graph host (one Neo4j, one graph at a time)."""
    global _HOST
    with _HOST_LOCK:
        if _HOST is None:
            _HOST = GraphHost()
        return _HOST


def cypherbench_questions(split: str = "train") -> list[dict[str, Any]]:
    import json

    d = json.loads((cypherbench_root() / f"{split}.json").read_text(encoding="utf-8"))
    return [{"id": f"cypherbench-{r['qid']}", "source": "cypherbench", "split": split,
             "db_id": r["graph"], "question": r["nl_question"].strip(), "evidence": "",
             "gold": r["gold_cypher"].strip(), "difficulty": None} for r in d]  # fmt: skip


def check_graph(key: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gold records for Cypher questions on graph `key`, in the SQL check's shape: the gold's
    result on the graph the episode will see (for CypherBench, the sampled graph, as runs 1 and
    2 scored on). A gold that fails or times out there is not kept."""
    from sqlrl.run3 import gold

    h = host()
    if not h.acquire(key):
        raise RuntimeError(f"{key}: {h.failed.get(key)}")
    conn = engines.Neo4j(time_s=gold.GOLD_TIME_S)
    out = []
    try:
        _, cc = settings.checker()
        for q in questions:
            res = conn.run(q["gold"])
            rec: dict[str, Any] = {"id": q["id"], "engines": {}}
            if res.kind == "error" or res.total is None:
                rec["engines"]["neo4j"] = {"ok": False, "gold": q["gold"],
                                           "why": res.error or "too many rows"}  # fmt: skip
                out.append(rec)
                continue
            full = None
            unl = cc.strip_limit(q["gold"])
            if unl is not None:
                r2 = conn.run(unl)
                if r2.kind != "error" and r2.total is not None:
                    full = gold.norm_rows(r2.rows)
            slow = res.ms > engines.QUERY_TIME_S * 1000 * gold.GOLD_MAX_SHARE
            rec.update({"columns": res.columns, "rows": gold.norm_rows(res.rows), "full": full})
            rec["engines"]["neo4j"] = {"ok": not slow, "gold": q["gold"], "ms": res.ms,
                                       "why": "slow" if slow else "",
                                       "empty": res.kind == "empty"}  # fmt: skip
            out.append(rec)
    finally:
        h.release()
    return out


def cypherbench_root() -> Path:
    """The CypherBench download: `<data>/raw/cypherbench` if it is there, else the checker's."""
    own = settings.get().raw / "cypherbench"
    if (own / "graphs").exists():
        return own
    _, cc = settings.checker()
    return cc.CYPHERBENCH


# ---- the schema text and the helpers on a graph -----------------------------------------------

_TYPE = {"Long": "INTEGER", "Integer": "INTEGER", "Double": "FLOAT", "Float": "FLOAT",
         "Boolean": "BOOLEAN", "Date": "DATE", "LocalDateTime": "DATETIME",
         "DateTime": "DATETIME"}  # fmt: skip


def _ptype(types: Any) -> str:
    ts = types if isinstance(types, (list, tuple)) else [types]
    for t in ts:
        t = str(t).replace("Array", "")
        if t in _TYPE:
            return _TYPE[t]
        if "Array" in str(ts):
            return "LIST"
    return ""


def describe(conn: engines.Neo4j) -> dict[str, str]:
    """The run-2 schema text (labels with typed properties, relationship patterns)."""
    _, nrows = conn.admin("CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, "
                          "propertyTypes RETURN nodeLabels, propertyName, "
                          "propertyTypes")  # fmt: skip
    props: dict[str, list[str]] = defaultdict(list)
    for labels, name, types in nrows:
        lab = labels[0] if labels else ""
        if name and name not in ("eid", "_k"):
            t = _ptype(types)
            props[lab].append(f"{name}: {t}" if t else name)
        elif lab not in props:
            props[lab] = []
    _, rrows = conn.admin("MATCH (a)-[r]->(b) WITH labels(a)[0] AS a, type(r) AS t, "
                          "labels(b)[0] AS b, keys(r) AS k RETURN a, t, b, collect(DISTINCT k) "
                          "LIMIT 500")  # fmt: skip
    lines = ["Nodes (properties are STRING unless typed):"]
    for lab in sorted(props):
        lines.append(f"- {lab} {{{', '.join(sorted(props[lab]))}}}")
    lines.append("Relationships:")
    for a, t, b, ks in sorted(rrows):
        keys = sorted({k for kk in ks for k in kk})
        lines.append(f"- ({a})-[:{t}{' {' + ', '.join(keys) + '}' if keys else ''}]->({b})")
    return {"full": "\n".join(lines),
            "names": "Node labels: " + ", ".join(sorted(props)) + "\nRelationship types: "
                     + ", ".join(sorted({t for _, t, _, _ in rrows}))}  # fmt: skip


def _label(conn: engines.Neo4j, name: str) -> str | None:
    _, rows = conn.admin("CALL db.labels() YIELD label RETURN label")
    labels = [r[0] for r in rows]
    name = (name or "").strip().strip("`:")
    if name in labels:
        return name
    return {x.lower(): x for x in labels}.get(name.lower())


def run_helper(conn: engines.Neo4j, name: str, args: dict[str, Any]) -> engines.Result:
    q = engines.quote_ident
    try:
        if name == "list_tables":
            cols, rows = conn.admin(
                "CALL db.labels() YIELD label RETURN 'label' AS kind, label AS name UNION ALL "
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN 'relationship' AS kind, relationshipType AS name")  # fmt: skip
            return engines.Result.of(cols, rows, 0.0)
        label = _label(conn, str(args.get("table", "")))
        if name in ("describe_table", "sample_rows") and label is None:
            return engines.Result.fail(f"no such label: {args.get('table')}")
        if name == "describe_table":
            cols, rows = conn.admin(
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, "
                "propertyTypes WHERE $l IN nodeLabels AND propertyName IS NOT NULL "
                "RETURN propertyName AS property, propertyTypes AS types",
                {"l": label})  # fmt: skip
            return engines.Result.of(cols, rows, 0.0)
        if name == "sample_rows":
            try:
                k = max(1, min(int(args.get("n") or 5), 20))
            except (TypeError, ValueError):
                k = 5
            _, rows = conn.admin(f"MATCH (n:{q('neo4j', label)}) RETURN properties(n) AS node "
                                 f"LIMIT {k}")  # fmt: skip
            maps = [{a: b for a, b in r[0] if a not in ("eid", "_k")} for r in rows]
            keys = sorted({x for m in maps for x in m})
            return engines.Result.of(keys, [tuple(m.get(x) for x in keys) for m in maps], 0.0)
        if name == "find_values":
            lab, _, prop = str(args.get("column", "")).rpartition(".")
            label = _label(conn, lab)
            if label is None or not prop:
                return engines.Result.fail("give the property as Label.property")
            cols, rows = conn.admin(
                f"MATCH (n:{q('neo4j', label)}) WHERE toLower(toString(n.{q('neo4j', prop)})) "
                f"CONTAINS toLower($t) RETURN DISTINCT n.{q('neo4j', prop)} AS value LIMIT 20",
                {"t": str(args.get("text", ""))})  # fmt: skip
            return engines.Result.of(cols, rows, 0.0)
    except Exception as e:
        return engines.Result.fail(str(e))
    return engines.Result.fail(f"unknown tool {name}")
