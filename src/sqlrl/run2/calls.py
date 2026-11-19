"""Reading a turn's tool call, and writing a query back as a call (standard library only).

One reading for both sides: the environment runs what `read_call` returns, and the scorer gets
that same query wrapped by `as_call`, in the text form the checker parses
(`publiceval.parse_call` through `querycheck.calls_from_message`). The patterns are the
checker's own; `tests/run2` proves the two readings agree.

A query that carries the call's own markup (`</arg_value>`, `</tool_call>`, `</think>`) can't be
wrapped without changing what the checker reads, so it is unreadable here.
"""

from __future__ import annotations

import re

THINK_END = "</think>"
RAW_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
RAW_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
MARKUP = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
          THINK_END, "<think>")  # fmt: skip


def answer_part(text: str) -> str:
    """What follows the thought (the whole text when there is no `</think>`)."""
    return text.split(THINK_END, 1)[1] if THINK_END in text else text


def read_call(text: str, tool: str, arg: str) -> tuple[str, str | None, str]:
    """(kind, query, why) for one turn's text.

    - ("none", None, "no_call"): a plain reply, no call at all;
    - ("ok", query, ""): one call to `tool` with a non-empty `arg`;
    - ("unreadable", None, why): something that starts a call but can't be run: `cut` (no closing
      tag), `several_calls`, `wrong_tool`, `bad_argument`, `markup` (see the module's note).
    """
    body = answer_part(text)
    calls = RAW_CALL.findall(body)
    if not calls:
        return ("unreadable", None, "cut") if "<tool_call>" in body else ("none", None, "no_call")
    if len(calls) > 1:
        return "unreadable", None, "several_calls"
    name = calls[0].split("<arg_key>", 1)[0].strip()
    if name != tool:
        return "unreadable", None, "wrong_tool"
    args = {k.strip(): v for k, v in RAW_ARG.findall(calls[0])}
    q = args.get(arg)
    if not isinstance(q, str) or not q.strip():
        return "unreadable", None, "bad_argument"
    if any(m in q for m in MARKUP):
        return "unreadable", None, "markup"
    return "ok", q, ""


def as_call(query: str | None, tool: str, arg: str) -> str:
    """The reply text of a model that calls `tool` once with `query` (empty: no call), in the
    form the checker reads. An episode's answer is scored as this text."""
    if not query:
        return THINK_END
    return (f"{THINK_END}<tool_call>{tool}<arg_key>{arg}</arg_key>"
            f"<arg_value>{query}</arg_value></tool_call>")  # fmt: skip


def query_tool(row: dict) -> tuple[str, str]:
    """(name, argument) of the question's query tool, from the reference call. Only the tool's
    identity is read: the reference query itself never leaves this function."""
    fn = row["messages"][-1]["tool_calls"][0]["function"]
    args = fn["arguments"]
    if isinstance(args, str):
        import json

        args = json.loads(args)
    return fn["name"], next(iter(args))
