"""Run 2's guards (`sqlrl/run2/guards.py`, copied: run 2 stays frozen), plus `one_statement`.

Guards against an answer that is copied instead of computed (pure functions, standard
library only; decisions R15, `documentation/run2/design.md` section 4).

In a multi-turn episode the model sees rows. On gretel and CypherBench those are the rows the
reward is decided on, so it can run an exploring query, read the answer and then write
`SELECT 'Alice', 42`, which returns the reference's rows without answering anything.

- `hardcoded`: the rule. A final query that reads no table or graph pattern, or whose result
  columns are all literals, earns nothing. It catches the plain form of the trick, not every
  form (`SELECT 'Alice', (SELECT 42) FROM t` passes).
- `copied_literals`: the canary for the rest. The literals of the final query that the model
  could only have read from an earlier result: they appear in a result shown before it and not
  in the prompt. Reading how a value is spelled is legitimate, so this is watched as a rate
  against the run's start, not forbidden.
"""

from __future__ import annotations

import re

_SQL_LITERAL = (r"(?:'(?:[^']|'')*'|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NULL|TRUE|FALSE|"
                r"CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP)")  # fmt: skip
_SQL_ALIAS = r"(?:\s+(?:AS\s+)?(?:\w+|\"[^\"]*\"|`[^`]*`|\[[^\]]*\]))?"
SQL_LITERAL_ITEM = re.compile(rf"^\s*{_SQL_LITERAL}{_SQL_ALIAS}\s*$", re.IGNORECASE)
_CY_SCALAR = r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[-+]?\d+(?:\.\d+)?|true|false|null)"
CY_SCALAR = re.compile(rf"^\s*{_CY_SCALAR}\s*$", re.IGNORECASE)
_CY_ALIAS = re.compile(r"\s+AS\s+(?:\w+|`[^`]*`)\s*$", re.IGNORECASE)
SQL_COMPOUND = re.compile(r"\b(?:UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b", re.IGNORECASE)
SQL_AFTER_LIST = re.compile(r"\b(?:FROM|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|WINDOW)\b",
                            re.IGNORECASE)  # fmt: skip
CY_AFTER_RETURN = re.compile(r"\b(?:ORDER\s+BY|SKIP|LIMIT|UNION)\b", re.IGNORECASE)
SQ = re.compile(r"'((?:[^']|'')*)'")
DQ = re.compile(r'"((?:[^"\\]|\\.)*)"')
NUM = re.compile(r"(?<![\w.])(\d{3,}(?:\.\d+)?)(?![\w.])")


def mask(q: str, backslash: bool = False) -> str:
    """`q` with the same length, where everything inside quotes becomes `x` and everything inside
    brackets (any depth) becomes a space: what is left is the query's own top level.
    `backslash`: a backslash escapes the next character inside quotes (Cypher, not SQL)."""
    out, depth, quote = [], 0, ""
    i = 0
    while i < len(q):
        ch = q[i]
        if quote:
            if ch == "\\" and backslash and i + 1 < len(q):
                out += ["x", "x"] if depth == 0 else [" ", " "]
                i += 2
                continue
            if ch == quote:
                if quote == "'" and not backslash and q[i + 1 : i + 2] == "'":  # SQL's ''
                    out += ["x", "x"] if depth == 0 else [" ", " "]
                    i += 2
                    continue
                quote = ""
                out.append(ch if depth == 0 else " ")
            else:
                out.append("x" if depth == 0 else " ")
        elif ch in "'\"`":
            quote = ch
            out.append(ch if depth == 0 else " ")
        elif ch in "([{":
            out.append(ch if depth == 0 else " ")
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
            out.append(ch if depth == 0 else " ")
        else:
            out.append(ch if depth == 0 else " ")
        i += 1
    return "".join(out)


def one_statement(q: str, backslash: bool = False) -> bool:
    """Is `q` a single statement: no `;` outside quotes and brackets except at its end? (DuckDB
    runs every statement of a string it is given; the other engines refuse, each its own way.)"""
    body = q.strip()
    while body.endswith(";"):
        body = body[:-1].rstrip()
    return ";" not in mask(body, backslash)


def split_top(q: str, m: str | None = None, backslash: bool = False) -> list[str]:
    """`q` cut at its top-level commas (not those inside quotes or brackets)."""
    m = mask(q, backslash) if m is None else m
    parts, last = [], 0
    for i, ch in enumerate(m):
        if ch == ",":
            parts.append(q[last:i])
            last = i + 1
    parts.append(q[last:])
    return parts


def sql_literal_select(q: str) -> bool:
    """Does every part of the outermost SELECT (each side of a UNION) return only literals?"""
    q = q.strip().rstrip(";").strip()
    m = mask(q)
    cuts = [0] + [x for mt in SQL_COMPOUND.finditer(m) for x in (mt.start(), mt.end())] + [len(q)]
    parts = [(q[cuts[i] : cuts[i + 1]], m[cuts[i] : cuts[i + 1]]) for i in range(0, len(cuts), 2)]
    seen = False
    for text, mk in parts:
        sel = list(re.finditer(r"\bSELECT\b(?:\s+(?:DISTINCT|ALL)\b)?", mk, re.IGNORECASE))
        if not sel:
            return False
        start = sel[-1].end()  # a WITH's own SELECTs sit inside brackets: masked away
        end = SQL_AFTER_LIST.search(mk, start)
        stop = end.start() if end else len(text)
        items = split_top(text[start:stop], mk[start:stop])
        if not items or not all(SQL_LITERAL_ITEM.match(it) for it in items):
            return False
        seen = True
    return seen


def _cypher_literal(expr: str) -> bool:
    e = _CY_ALIAS.sub("", expr).strip()
    if CY_SCALAR.match(e):
        return True
    if e.startswith("[") and e.endswith("]"):
        inner = e[1:-1]
        return not inner.strip() or all(_cypher_literal(x)
                                        for x in split_top(inner, backslash=True))  # fmt: skip
    return False


def cypher_reads_graph(q: str) -> bool:
    """Does the query look at the graph at all (a MATCH or a CALL outside strings)?"""
    return bool(re.search(r"\b(?:MATCH|CALL)\b", _unquoted(q), re.IGNORECASE))


def _unquoted(q: str) -> str:
    """`q` with quoted text blanked and brackets kept (unlike `mask`)."""
    out, quote, i = [], "", 0
    while i < len(q):
        ch = q[i]
        if quote:
            if ch == "\\" and i + 1 < len(q):
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"`":
            quote = ch
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def cypher_literal_return(q: str) -> bool:
    """Does the query's last RETURN list only literals?"""
    q = q.strip().rstrip(";").strip()
    m = mask(q, backslash=True)
    rets = list(re.finditer(r"\bRETURN\b(?:\s+DISTINCT\b)?", m, re.IGNORECASE))
    if not rets:
        return False
    start = rets[-1].end()
    end = CY_AFTER_RETURN.search(m, start)
    stop = end.start() if end else len(q)
    items = split_top(q[start:stop], m[start:stop])
    return bool(items) and all(_cypher_literal(it) for it in items)


def hardcoded(dialect: str, query: str | None, tables: list[str] | None = None) -> str | None:
    """Why the final query can't have computed its result from the data (None: no reason found).
    `tables`: for SQL, the tables it read when the environment ran it (None when unknown)."""
    if not query:
        return None
    if dialect == "cypher":
        if not cypher_reads_graph(query):
            return "no_match"
        return "constants" if cypher_literal_return(query) else None
    if tables is not None and not tables:
        return "no_table"
    return "constants" if sql_literal_select(query) else None


def query_literals(query: str, dialect: str) -> set[str]:
    """The text values (3 characters or more) and the numbers (3 digits or more) a query names."""
    out = {s.replace("''", "'").strip().strip("%").strip() for s in SQ.findall(query)}
    if dialect == "cypher":
        out |= {s.strip() for s in DQ.findall(query)}
    out = {s for s in out if len(s) >= 3}
    bare = SQ.sub("''", query)
    if dialect == "cypher":
        bare = DQ.sub('""', bare)
    return out | set(NUM.findall(bare))


def copied_literals(query: str | None, dialect: str, seen_before: list[str],
                    prompt: str) -> list[str]:  # fmt: skip
    """The final query's literals that appear in a result shown before it (`seen_before`: the
    earlier observations' text) and nowhere in the prompt (system and user messages)."""
    if not query or not seen_before:
        return []
    p = prompt.lower()
    shown = "\n".join(seen_before).lower()
    return sorted(s for s in query_literals(query, dialect)
                  if s.lower() not in p and s.lower() in shown)  # fmt: skip
