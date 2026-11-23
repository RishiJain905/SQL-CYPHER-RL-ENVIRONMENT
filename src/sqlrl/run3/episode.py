"""One episode, start to end, and its reward (design.md 2.2, 4).

A **task** is a question with the databases the episode is connected to:

    {"q": <pool record>, "kind": "answer" | "unanswerable" | "write" | "ambiguous",
     "sources": [{"engine", "source", "db_id"}], "answer_source": 0,
     "gold": {"sql", "rows", "columns", "full"}, "clarify_terms": [...]}

A **policy** is any `policy(messages, tools) -> text`: the model behind llama-server or vLLM
(`policy.py`), or a scripted one in the tests. The text is the model's raw turn, `<think>` and
the tool call included.

**The reward** (design 4), one an episode, from how it ended:

| Ending | Reward |
|---|---|
| `submit`, right | 1 less 0.02 a call after the second (at least 0.9) |
| `submit`, wrong, or caught by the constants rule | 0 |
| `decline` on an unanswerable question or a write request | 1 |
| `clarify` naming every meaning the question has | 1 |
| `decline` / `clarify` where the question was clear (counted) | 0 |
| no ending, a malformed one, or tool calls after the last one allowed | -0.1 |

`last_query_ok` records whether the last query the model ran (with a query tool) would have been
right: a diagnostic of what the endings cost an untrained model, never a reward.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from sqlrl.run3 import engines, env, gold, guards, protocol

REWARD = {"right": 1.0, "wrong": 0.0, "malformed": -0.1, "call_cost": 0.02, "free_calls": 2,
          "floor": 0.9}  # fmt: skip
Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]]], str]


class Connections:
    """The open connections of one episode, one per source, opened on first use."""

    def __init__(self, sources: list[dict[str, Any]], time_s: float = engines.QUERY_TIME_S):
        self.sources, self.time_s = sources, time_s
        self._conns: dict[int, Any] = {}
        self._helpers: dict[int, env.SqlHelpers] = {}
        self._held = 0  # graphs this episode holds on the host

    def conn(self, i: int) -> Any:
        if i not in self._conns:
            s = self.sources[i]
            if s["engine"] == "neo4j":
                from sqlrl.run3 import graphs

                h = graphs.host()
                if not h.acquire(s["graph"]):
                    raise RuntimeError(f"graph {s['graph']}: {h.failed.get(s['graph'])}")
                self._held += 1
                self._conns[i] = engines.Neo4j(time_s=self.time_s)
            else:
                self._conns[i] = engines.open_sql(s["engine"], s["source"], s["db_id"],
                                                  self.time_s)  # fmt: skip
        return self._conns[i]

    def helpers(self, i: int) -> env.SqlHelpers:
        if i not in self._helpers:
            self._helpers[i] = env.SqlHelpers(self.conn(i))
        return self._helpers[i]

    def close(self) -> None:
        for c in self._conns.values():
            c.close()
        self._conns.clear()
        self._helpers.clear()
        if self._held:
            from sqlrl.run3 import graphs

            for _ in range(self._held):
                graphs.host().release()
            self._held = 0


def schemas_for(task: dict[str, Any], conns: Connections) -> list[dict[str, str]]:
    out = []
    for i, s in enumerate(task["sources"]):
        if s["engine"] == "neo4j":
            from sqlrl.run3 import graphs

            out.append(graphs.describe(conns.conn(i)))
        else:
            out.append(env.describe_sql(conns.conn(i), s["source"], s["db_id"]))
    return out


def _source_index(s: protocol.Setting, args: dict[str, Any], tool: str | None) -> int | None:
    if tool is not None:
        for i, src in enumerate(s.sources):
            if src.tool == tool:
                return i
        return None
    if len(s.sources) == 1:
        return 0
    want = str(args.get("source", "")).strip()
    for i, src in enumerate(s.sources):
        if want in (src.tool, src.label):
            return i
    return None


def run_episode(task: dict[str, Any], policy: Policy, setting: protocol.Setting,
                with_evidence: bool = True) -> dict[str, Any]:  # fmt: skip
    """Play one episode; returns its record (turns, ending, the submitted query)."""
    conns = Connections(task["sources"])
    t0 = time.time()
    rec: dict[str, Any] = {"qid": task["q"]["id"], "kind": task["kind"],
                           "setting": setting.record(), "turns": [], "ending": None,
                           "n_calls": 0}  # fmt: skip
    try:
        messages = [
            {
                "role": "system",
                "content": protocol.system_prompt(setting, schemas_for(task, conns)),
            },
            {"role": "user", "content": protocol.user_message(task["q"], with_evidence)},
        ]
        tools = protocol.tools(setting)
        rec["prompt"] = messages
        last_query: tuple[int, str] | None = None
        while True:
            text = policy(messages, tools)
            call = protocol.read_call(text, setting)
            turn: dict[str, Any] = {"text": text, "call": call.name, "args": call.args}
            rec["turns"].append(turn)
            if call.kind == "none":
                rec["ending"] = "none"
                break
            if call.kind == "ok" and call.name in (setting.submit, setting.decline,
                                                   setting.clarify):  # fmt: skip
                rec["ending"] = {setting.submit: "submit", setting.decline: "decline",
                                 setting.clarify: "clarify"}[call.name]  # fmt: skip
                rec["ending_args"] = call.args
                if rec["ending"] == "submit":
                    rec["submit_source"] = _source_index(setting, call.args, None)
                    rec["final_query"] = str(call.args.get("query", ""))
                break
            if rec["n_calls"] >= setting.max_calls:
                rec["ending"] = "over_limit"
                break
            rec["n_calls"] += 1
            if call.kind == "unreadable":
                obs = f"Error: the call could not be read ({call.why})"
            else:
                i = _source_index(setting, call.args, call.name if call.name not in
                                  protocol.HELPERS else None)  # fmt: skip
                if i is None:
                    obs = "Error: no such source"
                elif call.name in protocol.HELPERS:
                    if task["sources"][i]["engine"] == "neo4j":
                        from sqlrl.run3 import graphs

                        res = graphs.run_helper(conns.conn(i), call.name, call.args)
                    else:
                        res = env.run_helper(conns.helpers(i), call.name, call.args)
                    obs = env.render(res, setting.rows, setting.style, setting.hints)
                else:
                    src = setting.sources[i]
                    q = str(call.args.get(src.arg, ""))
                    res = conns.conn(i).run(q)
                    last_query = (i, q)
                    turn["result"] = {"kind": res.kind, "rows": res.total, "ms": res.ms}
                    obs = env.render(res, setting.rows, setting.style, setting.hints)
            left = setting.max_calls - rec["n_calls"]
            if left == 0:
                obs += (f"\n(No tool calls left: end now with `{setting.submit}`, "
                        f"`{setting.decline}` or `{setting.clarify}`.)")  # fmt: skip
            turn["obs"] = obs
            messages.append(protocol.tool_call_message(call, text) if call.kind == "ok" else
                            {"role": "assistant",
                             "content": protocol.answer_part(text)})  # fmt: skip
            messages.append({"role": "tool", "content": obs})
        if last_query:
            rec["last_query"] = {"source": last_query[0], "query": last_query[1]}
    finally:
        conns.close()
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


# ---- grading ----------------------------------------------------------------------------------


def judge_query(task: dict[str, Any], i: int | None, query: str) -> dict[str, Any]:
    """Run `query` on source `i` and compare it with the gold: {"verdict", "why"}."""
    if i is None:
        return {"verdict": "wrong", "why": "unknown_source"}
    if i != task.get("answer_source", 0) and not task.get("any_source"):
        return {"verdict": "wrong", "why": "other_source"}
    s = task["sources"][i]
    conns = Connections(task["sources"])
    try:
        res = conns.conn(i).run(query)
    finally:
        conns.close()
    if res.kind == "error":
        return {"verdict": "wrong", "why": "error", "error": res.error}
    dialect = "cypher" if s["engine"] == "neo4j" else "sql"
    why = guards.hardcoded(dialect, query, res.tables)
    g = task["gold"]
    v = gold.verdict(g["sql"], g["rows"], gold.norm_rows(res.rows), g.get("columns"),
                     g.get("full"), dialect)  # fmt: skip
    if v in ("match", "tie") and why:
        return {"verdict": "wrong", "why": f"hardcoded:{why}"}
    return {"verdict": "right" if v in ("match", "tie") else "wrong", "why": v,
            "rows": res.total}  # fmt: skip


def grade(task: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """The episode's outcome and reward (in place on `rec`, and returned)."""
    kind, ending = task["kind"], rec["ending"]
    out: dict[str, Any] = {}
    if ending in ("none", "over_limit", None):
        out = {"outcome": "malformed", "reward": REWARD["malformed"]}
    elif ending == "submit":
        if kind != "answer":
            out = {"outcome": "wrong", "why": f"submitted_on_{kind}"}
        else:
            j = judge_query(task, rec.get("submit_source"), rec.get("final_query") or "")
            out = {"outcome": j["verdict"], **{k: v for k, v in j.items() if k != "verdict"}}
    elif ending == "decline":
        out = {"outcome": "right" if kind in ("unanswerable", "write") else "wrong",
               "why": "" if kind in ("unanswerable", "write") else "false_decline"}  # fmt: skip
    elif ending == "clarify":
        if kind == "ambiguous":
            text = str((rec.get("ending_args") or {}).get("question", "")).lower()
            named = all(t.lower() in text for t in task.get("clarify_terms") or [])
            out = {"outcome": "right" if named else "wrong",
                   "why": "" if named else "meanings_not_named"}  # fmt: skip
        else:
            out = {"outcome": "wrong", "why": "false_clarify"}
    if "reward" not in out:
        if out["outcome"] == "right":
            extra = max(rec["n_calls"] - REWARD["free_calls"], 0)
            out["reward"] = max(REWARD["right"] - REWARD["call_cost"] * extra, REWARD["floor"])
        else:
            out["reward"] = REWARD["wrong"]
    # the diagnostic: was the last query it ran right?
    lq = rec.get("last_query")
    if kind == "answer" and lq:
        out["last_query_ok"] = judge_query(task, lq["source"], lq["query"])["verdict"] == "right"
    rec.update(out)
    return out
