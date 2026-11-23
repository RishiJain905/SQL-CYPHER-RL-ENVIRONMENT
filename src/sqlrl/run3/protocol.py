"""What the model is told and how its replies are read (design.md 2.2-2.4; standard library).

**An episode's setting** (`Setting`, drawn per episode from a seed in training, fixed in
evaluation): the schema shown (`full`, `names`, `none`), the tool set (`query`, `helpers`), the
tools' spellings, the result format (rows shown, table / json / csv), whether error hints are
shown, the call limit.

**The tools.** One query tool a source; with `helpers` also `list_tables`, `describe_table`,
`sample_rows` and `find_values`; and always the three endings:

| Ending | Arguments | Means |
|---|---|---|
| `submit_answer` | the query (and the source, when there are several) | this query is the answer |
| `decline` | a reason | the database can't answer this, or it asks for a change |
| `clarify` | a question for the user | the question could mean two things |

**Reading a reply** (`read_call`): the model's chat format is
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` after `</think>`
(the checker's own patterns, run 2's `calls.py`). One call a turn. A reply with no call ends the
episode without an ending (malformed): the endings are the only way to stop.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

THINK_END = "</think>"
RAW_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
RAW_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
MARKUP = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
          THINK_END, "<think>")  # fmt: skip

# Spellings a tool's role is drawn from (design 2.3: the model learns the role, not the name)
QUERY_NAMES = {
    "sql": [("run_query", "query"), ("execute_sql", "sql"), ("query_database", "query"),
            ("run_sql", "sql"), ("sql_query", "query")],
    "cypher": [("query_graph", "cypher"), ("run_cypher", "query"), ("execute_cypher", "cypher"),
               ("cypher_query", "query")],
}  # fmt: skip
SUBMIT_NAMES = ["submit_answer", "submit_query", "final_answer"]
DECLINE_NAMES = ["decline", "cannot_answer"]
CLARIFY_NAMES = ["clarify", "ask_clarification"]
HELPERS = ("list_tables", "describe_table", "sample_rows", "find_values")

ENGINE_TITLE = {"sqlite": "SQLite", "duckdb": "DuckDB", "postgres": "PostgreSQL",
                "mysql": "MySQL", "neo4j": "Neo4j"}  # fmt: skip
LANGUAGE = {"sqlite": "SQLite SQL", "duckdb": "DuckDB SQL", "postgres": "PostgreSQL SQL",
            "mysql": "MySQL SQL", "neo4j": "Neo4j Cypher"}  # fmt: skip

EVAL_SETTING = {"schema": "full", "tools": "helpers", "rows": 10, "style": "table",
                "hints": True, "max_calls": 8}  # fmt: skip


@dataclass
class Source:
    """One database an episode is connected to."""

    engine: str  # sqlite | duckdb | postgres | mysql | neo4j
    version: str
    tool: str  # the query tool's name
    arg: str  # its argument's name
    label: str = ""  # how the prompt names it when there are several

    @property
    def dialect(self) -> str:
        return "cypher" if self.engine == "neo4j" else "sql"


@dataclass
class Setting:
    schema: str = "full"  # full | names | none
    tools: str = "helpers"  # query | helpers
    rows: int = 10
    style: str = "table"  # table | json | csv
    hints: bool = True
    max_calls: int = 8
    submit: str = "submit_answer"
    decline: str = "decline"
    clarify: str = "clarify"
    sources: list[Source] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return asdict(self)


def draw_setting(engine_sources: list[tuple[str, str]], seed: int | None,
                 cfg: dict[str, Any] | None = None) -> Setting:  # fmt: skip
    """An episode's setting. `engine_sources`: (engine, version) per source. `seed` None: the
    evaluation's fixed setting (first spellings)."""
    cfg = cfg or {}
    rnd = random.Random(seed)
    fixed = seed is None
    s = Setting(**({k: v for k, v in EVAL_SETTING.items()} if fixed else {
        "schema": rnd.choices(["full", "names", "none"], weights=cfg.get("schema_w", [5, 3, 2]))[0],
        "tools": rnd.choices(["query", "helpers"], weights=cfg.get("tools_w", [1, 1]))[0],
        "rows": rnd.choice(cfg.get("rows", [5, 10, 20, 50])),
        "style": rnd.choice(cfg.get("styles", ["table", "json", "csv"])),
        "hints": rnd.random() < cfg.get("hints_p", 0.5),
        "max_calls": int(cfg.get("max_calls", 8)),
    }))  # fmt: skip
    if not fixed:
        s.submit = rnd.choice(SUBMIT_NAMES)
        s.decline = rnd.choice(DECLINE_NAMES)
        s.clarify = rnd.choice(CLARIFY_NAMES)
    used: set[str] = set()
    for i, (engine, version) in enumerate(engine_sources):
        dialect = "cypher" if engine == "neo4j" else "sql"
        names = QUERY_NAMES[dialect]
        name, arg = names[0] if fixed else rnd.choice(names)
        if len(engine_sources) > 1:  # one tool a source: its name says which
            name = f"{name}_{i + 1}"
        while name in used:
            name += "_b"
        used.add(name)
        s.sources.append(Source(engine, version, name, arg, label=f"database {i + 1}"))
    return s


# ---- the tools as the chat template takes them --------------------------------------------------


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}  # fmt: skip


def tools(s: Setting) -> list[dict[str, Any]]:
    out = []
    several = len(s.sources) > 1
    for src in s.sources:
        where = f" ({src.label})" if several else ""
        out.append(_fn(src.tool, f"Run one read-only {LANGUAGE[src.engine]} query on the "
                       f"{ENGINE_TITLE[src.engine]} database{where} and get the rows back.",
                       {src.arg: {"type": "string", "description": "The query text."}},
                       [src.arg]))  # fmt: skip
    if s.tools == "helpers":
        src_prop = ({"source": {"type": "string", "description": "Which database: " + ", ".join(
            x.tool for x in s.sources)}} if several else {})  # fmt: skip
        req = ["source"] if several else []
        out += [
            _fn("list_tables", "List the tables (or, in a graph, the node labels and relationship "
                "types).", dict(src_prop), list(req)),
            _fn("describe_table", "The columns of a table and their types (or a label's "
                "properties).", {**src_prop, "table": {"type": "string"}}, req + ["table"]),
            _fn("sample_rows", "A few rows of a table (or nodes of a label).",
                {**src_prop, "table": {"type": "string"}, "n": {"type": "integer"}},
                req + ["table"]),
            _fn("find_values", "Values of a column that contain the text, to see how the data "
                "spells something (column as table.column, or Label.property).",
                {**src_prop, "column": {"type": "string"}, "text": {"type": "string"}},
                req + ["column", "text"]),
        ]  # fmt: skip
    sub_props = {"query": {"type": "string", "description": "The query that answers the question."}}
    sub_req = ["query"]
    if several:
        sub_props = {"source": {"type": "string", "description": "The query tool it runs on."},
                     **sub_props}  # fmt: skip
        sub_req = ["source", "query"]
    out += [
        _fn(s.submit, "Give your final answer: the query whose result answers the question. "
            "This ends your turn.", sub_props, sub_req),
        _fn(s.decline, "Say the question can't be answered from this database, or that it asks "
            "for a change to the data. This ends your turn.",
            {"reason": {"type": "string"}}, ["reason"]),
        _fn(s.clarify, "Ask the user which of two or more meanings they intend, naming them. "
            "This ends your turn.", {"question": {"type": "string"}}, ["question"]),
    ]  # fmt: skip
    return out


# ---- the prompt -------------------------------------------------------------------------------

RULES = (
    "Rules:\n"
    "- Queries are read only: one statement each (SELECT or WITH ... SELECT; MATCH ... RETURN "
    "for a graph). No comments. A query runs as written, so use literal values, not parameters.\n"
    "- Write the query in the database's own dialect.\n"
    "- Return named columns: the names and the numbers the answer needs. Never return whole rows "
    "of every column unless the question asks for them.\n"
    "- Match names and values as the data spells them.\n"
    "- Return exactly the rows the question asks for, in its shape: one row for a single answer; "
    'one row per entity for "which ...", "every", "list"; the top N for "top N".\n'
)


def ending_rules(s: Setting) -> str:
    return (
        f"You have {s.max_calls} tool calls before you must end. End with exactly one of:\n"
        f"- `{s.submit}`: the query that answers the question (a result with no rows can be the "
        "right answer);\n"
        f"- `{s.decline}`: the database doesn't hold what the question asks for, or the question "
        "asks to change the data;\n"
        f"- `{s.clarify}`: the question could mean two different things; name them.\n"
        "Look at the data only as much as you need to."
    )


def system_prompt(s: Setting, schemas: list[dict[str, Any]]) -> str:
    """`schemas`: per source, {"full": text, "names": text} (`schema.describe`)."""
    several = len(s.sources) > 1
    if several:
        head = ("You answer questions about the databases below. Use the tools to query them; "
                "each query tool runs on one database.\n")  # fmt: skip
    else:
        src = s.sources[0]
        kind = "graph database" if src.dialect == "cypher" else "database"
        head = (
            f"You answer questions about a {src.version} {kind} by calling tools: "
            f"`{src.tool}` runs one read-only {LANGUAGE[src.engine]} query.\n"
        )
    parts = [head]
    for src, sch in zip(s.sources, schemas, strict=True):
        title = f"{src.label}: {src.version}, tool `{src.tool}`" if several else ""
        if s.schema == "full":
            body = sch["full"]
        elif s.schema == "names":
            body = sch["names"] + "\n(Only the names are listed: look up the columns you need.)"
        else:
            body = "(The schema isn't shown: look it up with the tools or a catalog query.)"
        parts.append((title + "\n" if title else "") + body + "\n")
    parts.append(RULES)
    parts.append(ending_rules(s))
    return "\n".join(parts)


def user_message(q: dict[str, Any], with_evidence: bool) -> str:
    text = q["question"]
    if with_evidence and q.get("evidence"):
        text += f"\n(Note: {q['evidence']})"
    return text


# ---- reading a reply --------------------------------------------------------------------------


def answer_part(text: str) -> str:
    return text.split(THINK_END, 1)[1] if THINK_END in text else text


@dataclass
class Call:
    kind: str  # ok | none | unreadable
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""


def read_call(text: str, s: Setting) -> Call:
    """The one call a turn's text makes, by the setting's tool names."""
    body = answer_part(text)
    calls = RAW_CALL.findall(body)
    if not calls:
        return Call("unreadable", why="cut") if "<tool_call>" in body else Call("none")
    if len(calls) > 1:
        return Call("unreadable", why="several_calls")
    name = calls[0].split("<arg_key>", 1)[0].strip()
    known = {t["function"]["name"] for t in tools(s)}
    if name not in known:
        return Call("unreadable", name=name, why="unknown_tool")
    args: dict[str, Any] = {}
    for k, v in RAW_ARG.findall(calls[0]):
        v = v.strip()
        if any(m in v for m in MARKUP):
            return Call("unreadable", name=name, why="markup")
        if k.strip() == "n":
            try:
                v = int(json.loads(v))
            except (ValueError, TypeError):
                return Call("unreadable", name=name, why="bad_argument")
        args[k.strip()] = v
    return Call("ok", name=name, args=args)


def call_text(name: str, args: dict[str, Any]) -> str:
    """A call in the model's own format (for scripted policies and tests)."""
    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args.items())
    return f"{THINK_END}<tool_call>{name}{body}</tool_call>"


def tool_call_message(c: Call, text: str) -> dict[str, Any]:
    """The assistant turn as the chat template takes it back."""
    think = text.split(THINK_END, 1)[0].replace("<think>", "").strip() if THINK_END in text else ""
    return {"role": "assistant", "content": "", "reasoning_content": think,
            "tool_calls": [{"type": "function",
                            "function": {"name": c.name, "arguments": c.args}}]}  # fmt: skip
