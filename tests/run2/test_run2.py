"""Run 2's pure parts (CPU): the prompt, the call reader, the environment and its guards, the
episode loop against a scripted server, the scheduler's lane, the rewards, the canaries, the
settings, the summaries, and the copied modules against run 1's."""

from __future__ import annotations

import json
import os
import random
import threading

import pytest
import torch

from sqlrl.run1 import losses as l1
from sqlrl.run1 import rewards as rewards1
from sqlrl.run2 import (
    calls,
    canaries,
    env,
    episodes,
    eval_score,
    guards,
    prompt,
    rewards,
    scoring,
    summary,
)
from sqlrl.run2 import config as rcfg
from sqlrl.run2 import losses as l2
from sqlrl.run2 import pool as rpool
from sqlrl.run2.train_step import minibatches, turn_masks

SFT_SYSTEM = (
    "You answer questions about the database below by calling the `run_query` tool with ONE "
    "read-only SQLite SQL query.\n\nCREATE TABLE t (id INT, name TEXT);\n\nExamples (tested "
    "queries on this database):\n(none)\n\nRules:\n- One statement, read only (SELECT or WITH "
    "... SELECT). No comments.\n- If the database can't answer the question, don't call the "
    "tool: say why in one short sentence."
)
SCRIPT = (
    "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, n INT);"
    "INSERT INTO t VALUES (1, 'Alice', 42), (2, 'Bob', 7), (3, 'Cy', 7);"
)


def row(
    origin="gretel",
    dialect="sqlite",
    gold="SELECT name FROM t WHERE n = 42",
    question="Who has 42?",
    system=SFT_SYSTEM,
) -> dict:
    fn = {"name": "run_query", "arguments": {"query": gold}}
    return {
        "id": f"{origin}-1",
        "origin": origin,
        "dialect": dialect,
        "rl_thinking": "high",
        "tools": [{"type": "function", "function": {"name": "run_query"}}],
        "meta": {"gold": gold, "db_id": "art"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "secret thought",
                "tool_calls": [{"type": "function", "function": fn}],
            },
        ],
    }


CFG = rcfg.new_config()
RQ = ("run_query", "query")


# ---- the prompt ------------------------------------------------------------------------------
def test_the_prompt_keeps_sft_wording_except_the_one_query_rule():
    out = prompt.system_prompt(SFT_SYSTEM, "turns", 3)
    assert "ONE read-only" not in out and "one read-only SQLite SQL query at a time." in out
    assert "you have 3 calls in all" in out and "The last query you ran is your answer." in out
    # everything else is the SFT prompt, in order
    assert out.index("CREATE TABLE t") < out.index("After each call") < out.index("If the datab")
    assert prompt.system_prompt(SFT_SYSTEM, "sft", 3) == SFT_SYSTEM
    assert "check the result against the question" in prompt.system_prompt(SFT_SYSTEM, "check", 3)
    assert "check the result" not in out


def test_a_prompt_that_is_not_the_sft_one_is_refused():
    with pytest.raises(ValueError):
        prompt.system_prompt("You are a helpful assistant.", "turns", 3)
    with pytest.raises(ValueError):
        prompt.system_prompt(SFT_SYSTEM, "nope", 3)


def test_a_single_turn_dataset_keeps_run_1s_protocol():
    cfg = rcfg.new_config({"single_turn_sets": ["sqale"]})
    assert prompt.protocol(cfg, "sqale") == ("sft", 1)
    assert prompt.protocol(cfg, "gretel") == ("turns", 3)
    assert prompt.episode_messages(row("sqale"), cfg)[0]["content"] == SFT_SYSTEM
    assert "at a time" in prompt.episode_messages(row("gretel"), cfg)[0]["content"]


# ---- reading and writing calls ---------------------------------------------------------------
def call(q: str, tool="run_query", arg="query") -> str:
    body = f"<arg_key>{arg}</arg_key><arg_value>{q}</arg_value>"
    return f"t</think><tool_call>{tool}{body}</tool_call>"


def test_read_call_kinds():
    assert calls.read_call(call("SELECT 1"), *RQ) == ("ok", "SELECT 1", "")
    assert calls.read_call("t</think>It can't be answered.", *RQ)[0] == "none"
    cut = "t</think><tool_call>run_query<arg_key>query"
    assert calls.read_call(cut, *RQ) == ("unreadable", None, "cut")
    two = call("SELECT 1") + call("SELECT 2").split("</think>")[1]
    assert calls.read_call(two, *RQ)[2] == "several_calls"
    assert calls.read_call(call("SELECT 1", tool="send_email"), *RQ)[2] == "wrong_tool"
    assert calls.read_call(call("SELECT 1", arg="sql"), *RQ)[2] == "bad_argument"
    assert calls.read_call(call("  "), *RQ)[2] == "bad_argument"
    # a query that carries the call's own markup can't be wrapped for the checker
    assert calls.read_call(call("SELECT '<arg_key>'"), *RQ)[0] == "unreadable"
    assert calls.read_call(call("SELECT '<tool_call>'"), *RQ)[0] == "unreadable"
    # a closing tag inside the query ends it early, for the environment and the checker alike
    kind, q, _ = calls.read_call(call("SELECT '</arg_value>'"), *RQ)
    assert (kind, q) == ("ok", "SELECT '")
    assert calls.read_call(calls.as_call(q, *RQ), *RQ)[1] == q
    # a call inside the thought is not a call
    assert calls.read_call("<tool_call>x</tool_call></think>no", *RQ)[0] == "none"


def test_the_checker_reads_a_wrapped_query_as_the_environment_did():
    qc, _ = env.checker()
    for q in ["SELECT name FROM t", "MATCH (n) RETURN n.name AS x", "SELECT 'a<b' FROM t\nWHERE 1"]:
        text = calls.as_call(q, *RQ)
        assert calls.read_call(text, *RQ) == ("ok", q, "")
        got = qc.calls_from_message({"content": text.split("</think>", 1)[1]})
        assert got == [("run_query", {"query": q})]
    assert qc.calls_from_message({"content": calls.as_call(None, *RQ)}) == []
    assert calls.query_tool(row()) == RQ


# ---- the guards ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q, expect",
    [
        ("SELECT 'Alice', 42", True),
        ("SELECT 'Alice' AS name, 42 AS n FROM t LIMIT 1", True),
        ("select 'a' union all select 'b'", True),
        ("WITH c AS (SELECT name FROM t) SELECT 'Alice' FROM c", True),
        ("SELECT name, 42 FROM t", False),
        ("SELECT COUNT(*) FROM t", False),
        ("SELECT 'it''s, fine' AS x, name FROM t", False),
        ("SELECT (SELECT 42) FROM t", False),
        ("SELECT name FROM t WHERE name = 'Alice'", False),
    ],
)
def test_sql_literal_select(q, expect):
    assert guards.sql_literal_select(q) is expect


@pytest.mark.parametrize(
    "q, reads, lit",
    [
        ("RETURN 'Alice' AS name, 42", False, True),
        ("UNWIND ['a', 'b'] AS x RETURN x", False, False),
        ("MATCH (n:Person) RETURN 'Alice' AS name LIMIT 1", True, True),
        ("MATCH (n:Person {name: 'Alice'}) RETURN n.name, 42", True, False),
        ("MATCH (n) RETURN ['a', 1] AS xs", True, True),
        ("MATCH (n) WHERE n.t = 'RETURN 1' RETURN count(n)", True, False),
        ("WITH 'MATCH' AS s RETURN s", False, False),
    ],
)
def test_cypher_guards(q, reads, lit):
    assert guards.cypher_reads_graph(q) is reads
    assert guards.cypher_literal_return(q) is lit


def test_hardcoded_and_copied_literals():
    assert guards.hardcoded("sqlite", "SELECT 'Alice', 42", []) == "no_table"
    assert guards.hardcoded("sqlite", "SELECT 'Alice' FROM t LIMIT 1", ["t"]) == "constants"
    assert guards.hardcoded("sqlite", "SELECT COUNT(*) FROM t", ["t"]) is None
    assert guards.hardcoded("sqlite", "SELECT name FROM t", None) is None  # tables unknown
    assert guards.hardcoded("cypher", "RETURN 'Alice'") == "no_match"
    assert guards.hardcoded("cypher", "MATCH (n) RETURN 'Alice'") == "constants"
    assert guards.hardcoded("cypher", "MATCH (n) RETURN n.name") is None
    assert guards.hardcoded("sqlite", None, []) is None
    seen = ["name | n\nAlice Cooper | 42\nBob | 1984"]
    # read off a result and not in the prompt: copied; named by the question: not
    q = "SELECT n FROM t WHERE name = 'Alice Cooper' AND n > 1984"
    assert guards.copied_literals(q, "sqlite", seen, "Who is Bob?") == ["1984", "Alice Cooper"]
    q = "SELECT n FROM t WHERE name = 'Bob'"
    assert guards.copied_literals(q, "sqlite", seen, "Who is Bob?") == []
    assert guards.copied_literals("SELECT 'Alice Cooper'", "sqlite", [], "x") == []


def test_mask_and_split_top_respect_quotes_and_brackets():
    got = guards.split_top("a, f(b, c), 'x, y', [1, 2]")
    assert got == ["a", " f(b, c)", " 'x, y'", " [1, 2]"]
    assert guards.split_top(r"'it\'s, ok', b", backslash=True) == [r"'it\'s, ok'", " b"]
    assert guards.split_top("'it''s, ok', b") == ["'it''s, ok'", " b"]
    assert len(guards.mask("SELECT (a, 'b') FROM t")) == len("SELECT (a, 'b') FROM t")


# ---- the environment -------------------------------------------------------------------------
def pub_of(r: dict, script: str | None = SCRIPT) -> dict:
    return env.public_row(r, script)


def test_public_row_drops_the_reference():
    r = row(gold="SELECT name FROM t WHERE n = 424242")
    pub = pub_of(r)
    blob = json.dumps(pub)
    assert "424242" not in blob and "secret thought" not in blob and "gold" not in blob
    assert pub["tool"] == "run_query" and pub["arg"] == "query"
    assert [m["role"] for m in pub["messages"]] == ["system", "user"]
    assert env.public_row(row("cypherbench", "cypher"))["db"] == {"kind": "graph", "graph": "art"}
    assert env.public_row(row("text2cypher", "cypher"))["db"] == {"kind": "graph_fill"}
    with pytest.raises(ValueError):
        env.public_row(row("spider"))
    with pytest.raises(ValueError, match="no database script"):
        env.public_row(r, None)  # a missing sidecar row is not a silent "not available"
    # what marks a returned question stays on the scoring side
    back = {**r, "kind": "returned", "nothing": "no_rows+empty_tables", "set_index": 7}
    assert pub_of(back) == pub
    assert "returned" not in json.dumps(pub_of(back)) and "nothing" not in pub_of(back)


def test_the_strict_check_tells_a_right_query_from_one_that_is_merely_empty():
    """On an empty database every query returns nothing, so the checker's own comparison passes
    them all. The filled copies are what the returned questions' reward stands on."""
    env.checker()  # puts the trainer's checker on the path
    script = "CREATE TABLE cars (id INTEGER PRIMARY KEY, make TEXT, year INTEGER, price REAL);"
    gold = "SELECT make FROM cars WHERE year = 2021 AND price > 30000"
    assert rpool.strict_check("select make FROM cars WHERE price > 30000 AND year = 2021",
                              gold, script) == "pass"  # fmt: skip
    for wrong in ("SELECT make FROM cars WHERE year = 2020 AND price > 30000",
                  "SELECT make FROM cars WHERE year = 2021",
                  "SELECT make FROM cars WHERE 1 = 0",
                  "SELECT id FROM cars WHERE year = 2021 AND price > 30000"):  # fmt: skip
        assert rpool.strict_check(wrong, gold, script) == "fail", wrong


def test_a_repeated_query_still_reports_its_tables():
    # SQLite keeps prepared statements, so the second run of the same text asks the authorizer
    # nothing: without the memory, a right answer written twice would look like constants
    look = env.Look(pub_of(row()))
    for _ in range(3):
        assert look.run("SELECT name FROM T WHERE n = 42").tables == ["t"]
        assert look.run("SELECT 'Alice'").tables == []
    assert (
        guards.hardcoded("sqlite", "SELECT name FROM T", look.run("SELECT name FROM T").tables)
        is None
    )


def test_sqlite_is_read_only_and_reports_what_it_read():
    look = env.Look(pub_of(row()))
    assert look.error is None
    r = look.run("SELECT name FROM t WHERE n = 42;")
    assert (r.kind, r.rows, r.total, r.tables) == ("rows", [("Alice",)], 1, ["t"])
    assert look.run("SELECT 'Alice', 42").tables == []
    assert look.run("SELECT COUNT(*) FROM t").tables == ["t"]
    assert look.run("SELECT name FROM t WHERE n = 1").kind == "empty"
    for q in [
        "INSERT INTO t VALUES (9, 'x', 1)",
        "DELETE FROM t",
        "PRAGMA table_info(t)",
        "ATTACH ':memory:' AS x",
        "CREATE TEMP TABLE z (a)",
        "WITH c AS (SELECT 1) DELETE FROM t",
        "SELECT name FROM t; DROP TABLE t",
        "SELECT load_extension('x')",
    ]:
        assert look.run(q).kind == "error", q
    assert look.run("SELECT COUNT(*) FROM t").rows == [(3,)]  # nothing was changed
    assert look.run("SELECT nope FROM t").error == "no such column: nope"
    look.close()
    assert look.run("SELECT 1").error == env.MSG_NO_DB


def test_a_runaway_query_is_stopped(monkeypatch):
    monkeypatch.setattr(env, "SQL_TIME_S", 0.2)
    look = env.Look(pub_of(row()))
    q = "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) SELECT COUNT(*) FROM r"
    assert look.run(q).error == env.MSG_TIMEOUT
    assert look.run("SELECT COUNT(*) FROM t").rows == [(3,)]  # and the database still answers


def test_a_schema_that_does_not_build_fails_every_query_softly():
    look = env.Look(pub_of(row(), "CREATE TABLE ("))
    assert look.error and look.run("SELECT 1").error == env.MSG_NO_DB


def test_the_random_fill_uses_the_question_and_never_the_reference():
    r = row(
        "sqale",
        gold="SELECT name FROM t WHERE name = 'ZZZ_SECRET' AND n = 987654",
        question="Which rows are named 'Marlowe' with more than 5150 points?",
    )
    ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, n INT, d DATE);"
    look = env.Look(pub_of(r, ddl))
    names = {x[0] for x in look.run("SELECT name FROM t").rows}
    ns = {x[0] for x in look.run("SELECT n FROM t").rows}
    assert "Marlowe" in names and 5150 in ns
    assert "ZZZ_SECRET" not in names and 987654 not in ns
    assert look.run("SELECT COUNT(*) FROM t").rows[0][0] == env.FILL_ROWS
    # the same question always gets the same copy, and never one of the checker's seeds
    again = env.Look(pub_of(r, ddl))
    assert again.run("SELECT * FROM t").rows == look.run("SELECT * FROM t").rows
    qc, cc = env.checker()
    theirs = {*qc.FUZZ_SEEDS, *qc.VARIANT_SEEDS, *qc.RESCUE_SEEDS, qc.ORDER_SEED}
    theirs |= {*cc.FUZZ_SEEDS, *cc.VARIANT_SEEDS, *cc.RESCUE_SEEDS, cc.ORDER_SEED}
    assert env.look_seed("any-id") > max(theirs)


def test_question_literals():
    strs, nums = env.question_literals(
        "Find games on 'PlayStation 5' without 'Hotseat' for Tapscott Nobbs' account "
        "VCH-00022's document after 2020-05-01, in King county, with 42 points."
    )
    want = {"PlayStation 5", "Hotseat", "Tapscott Nobbs", "VCH-00022", "2020-05-01", "King"}
    assert want <= set(strs)
    assert not any(s.endswith("'") or s.endswith("'s") for s in strs)
    assert 42.0 in nums and 2020.0 not in nums
    assert env.question_literals("Earliest gate 42 starter's email?") == ([], [42.0])
    assert env.question_literals("") == ([], [])


def test_render_table_json_empty_error_and_cuts():
    res = env.Result("rows", ["name", "n"], [("Alice", 1), ("B" * 60, None), ("C\nD|E", 2.5)], 3)
    table = env.render(res, 2, "table").splitlines()
    assert table[0] == "name | n" and table[1] == "Alice | 1"
    assert table[2] == "B" * 37 + "... | NULL" and table[3] == "(2 of 3 rows)"
    assert env.render(res, 10, "table").splitlines()[-1] == "C D|E | 2.5"  # no count: not cut
    js = env.render(res, 10, "json").splitlines()
    assert json.loads(js[0]) == {"name": "Alice", "n": 1} and json.loads(js[1])["n"] is None
    big = env.Result("rows", ["a"], [(1,)] * 3, None)
    assert env.render(big, 2, "table").splitlines()[-1] == f"(2 of more than {env.MAX_ROWS} rows)"
    assert env.render(env.Result("empty"), 10, "table") == "0 rows"
    err = env.Result.fail("no such column: x\n  at line 2\n" + "z" * 500)
    assert env.render(err, 10, "json") == "Error: no such column: x"
    assert len(env.Result.fail("q" * 900).error) == env.ERR_CHARS
    assert env.cell({"a": [1, 2]}, "table") == '{"a":[1,2]}'
    with pytest.raises(ValueError):
        env.render(res, 2, "xml")


GRAPH_SYSTEM = (
    "You answer questions about the graph database below by calling the `run_query` tool with "
    "ONE read-only Neo4j Cypher query.\n\nNodes (label {properties}):\n"
    "- Person {name: STRING, age: INTEGER}\n- City {name: STRING}\n\nRelationships:\n"
    "- (Person)-[:LIVES_IN]->(City)\n\nExamples (tested queries on this database):\n(none)\n\n"
    "Rules:\n- One statement, read only (MATCH ... RETURN).\n- If the database can't answer "
    "the question, don't call the tool: say why in one short sentence."
)


@pytest.mark.skipif(
    not os.environ.get("RUN2_NEO4J_TESTS"),
    reason="loads a graph into the checker's Neo4j: set RUN2_NEO4J_TESTS=1, and never while a "
    "run or an evaluation is using it",
)
def test_the_graph_is_filled_from_the_question_read_only_and_bounded(monkeypatch):
    graphs = env.GraphHost()
    if not graphs.ok():
        pytest.skip("the checker's Neo4j isn't running")
    gold = "MATCH (p:Person {name: 'ZZZ_SECRET'}) RETURN p.age"
    r = row("text2cypher", "cypher", gold, "How old is 'Marlowe', who is 5150?", GRAPH_SYSTEM)
    pub = env.public_row(r)
    graphs.ensure(pub)
    look = env.Look(pub, graphs)
    assert look.error is None
    names = {x[0] for x in look.run("MATCH (p:Person) RETURN p.name").rows}
    assert "Marlowe" in names and "ZZZ_SECRET" not in names  # the question's value, not the gold's
    assert look.run("MATCH (p:Person) SET p.x = 1 RETURN p").error == env.MSG_CYPHER_READ
    assert look.run("MATCH (p:Person) RETRUN p").error.startswith("Invalid input")
    assert look.run("MATCH (p:Person {name: 'nobody'}) RETURN p.name").kind == "empty"
    # the server refuses a write by itself, with our own guard switched off
    monkeypatch.setattr(env, "cypher_block", lambda q: None)
    assert look.run("MATCH (p:Person) SET p.x = 1 RETURN p").kind == "error"
    assert look.run("MATCH (p:Person) WHERE p.x IS NOT NULL RETURN count(p)").rows == [(0,)]
    # a question whose graph isn't the loaded one gets no answer from this one
    other = dict(pub, id="another-question")
    assert graphs.run(other, "MATCH (n) RETURN n").error == env.MSG_NO_DB


def test_obs_setting_varies_in_training_and_is_fixed_in_evaluation():
    seen = {json.dumps(env.obs_setting(CFG, s), sort_keys=True) for s in range(200)}
    assert len(seen) == len(CFG["obs"]["rows"]) * len(CFG["obs"]["styles"])
    assert env.obs_setting(CFG, None) == CFG["obs"]["eval"]
    assert env.obs_setting(CFG, 7) == env.obs_setting(CFG, 7)


def test_cypher_block():
    assert env.cypher_block("MATCH (n) RETURN n") is None
    assert env.cypher_block("MATCH (n) WHERE n.t = 'CREATE x' RETURN n") is None
    assert env.cypher_block("MATCH (n) SET n.x = 1 RETURN n") == env.MSG_CYPHER_READ
    assert env.cypher_block("LOAD CSV FROM 'file:///x' AS r RETURN r") == env.MSG_CYPHER_READ
    assert env.cypher_block("CALL dbms.listConfig() YIELD name RETURN name") == env.MSG_CALL
    assert env.cypher_block("CALL db.labels()") is None
    assert env.cypher_block("MATCH (n) CALL { WITH n RETURN 1 AS x } RETURN x") is None


# ---- the episode loop against a scripted server ------------------------------------------------
class FakeServer:
    """A server that replays `turns`: (thought, answer text, closed with the end token)."""

    think_end_id, eos_id, THINK = 9001, 9002, 9003

    def __init__(self, turns):
        self.turns, self.i, self.seen = list(turns), 0, []
        self.tool_open, self.tool_close = [9100], [9101, self.eos_id, self.THINK]

    def prompt_ids(self, pub, cfg):
        return [1, 2, self.THINK]

    def observation_ids(self, text):
        return self.tool_open + [ord(c) for c in text] + self.tool_close

    def complete(self, ids, n, sampling, seed, slot, stop=None, logprobs=False):
        self.seen.append((list(ids), seed))
        thought, answer, closed = self.turns[self.i]
        if stop:
            toks = [ord(c) for c in thought] + [self.think_end_id]
            out = {"tokens": toks, "content": thought}
        else:
            self.i += 1
            toks = [ord(c) for c in answer] + ([self.eos_id] if closed else [])
            out = {"tokens": toks, "content": answer, "stop_type": "eos" if closed else "limit"}
        if logprobs:
            out["completion_probabilities"] = [{"logprob": -0.5}] * len(toks)
        return out


class FakeLook:
    error = None

    def __init__(self, results):
        self.results, self.ran = list(results), []

    def run(self, q):
        self.ran.append(q)
        return self.results[len(self.ran) - 1]


def tc(q: str) -> str:
    return f"<tool_call>run_query<arg_key>query</arg_key><arg_value>{q}</arg_value></tool_call>"


PUB = {"id": "q1", "origin": "gretel", "dialect": "sqlite", "tool": "run_query", "arg": "query"}
OBS = {"rows": 10, "style": "table"}
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
ROWS = env.Result("rows", ["name"], [("Alice",)], 1, tables=["t"])
ERR = env.Result.fail("no such column: nam")


def run_episode(turns, results, cfg=CFG, **kw):
    srv, look = FakeServer(turns), FakeLook(results)
    ep = episodes.episode(srv, PUB, look, 0, 1000, cfg, 3, SAMPLING, OBS, **kw)
    return ep, srv, look


def test_an_episode_fixes_its_query_and_stops_with_a_reply():
    turns = [
        ("try", tc("SELECT nam FROM t"), True),
        ("fix", tc("SELECT name FROM t"), True),
        ("", "Alice.", True),
    ]
    ep, srv, look = run_episode(turns, [ERR, ROWS], logprobs=True)
    assert look.ran == ["SELECT nam FROM t", "SELECT name FROM t"]
    assert (ep["end"], ep["n_calls"]) == ("reply", 2)
    assert ep["first_query"] == "SELECT nam FROM t" and ep["final_query"] == "SELECT name FROM t"
    assert ep["text"] == calls.as_call("SELECT name FROM t", *RQ)
    assert [t["result"] for t in ep["turns"][:2]] == ["error", "rows"]
    assert ep["turns"][0]["obs"] == "Error: no such column: nam"
    # one growing sequence: masks and turns line up with the tokens, results are never trained
    n = len(ep["gen_ids"])
    assert len(ep["gen_mask"]) == len(ep["gen_turn"]) == len(ep["behavior_lp"]) == n
    obs1 = srv.observation_ids("Error: no such column: nam")
    at = ep["turns"][0]["tokens"]
    assert ep["gen_ids"][at : at + len(obs1)] == obs1
    assert ep["gen_mask"][at : at + len(obs1)] == [0] * len(obs1)
    assert ep["behavior_lp"][at : at + len(obs1)] == [None] * len(obs1)
    assert sum(ep["gen_mask"]) == sum(t["tokens"] for t in ep["turns"])
    assert ep["gen_turn"][0] == 1 and ep["gen_turn"][-1] == 3 and ep["gen_turn"][at] == 1
    # every request continued the same context, with a seed of its own
    assert srv.seen[-1][0] == ep["prompt_ids"] + ep["gen_ids"][: -ep["turns"][2]["tokens"] + 1]
    assert len({seed for _, seed in srv.seen}) == len(srv.seen)


def test_the_call_after_the_last_is_not_run():
    turns = [("", tc(f"SELECT {i}"), True) for i in range(1, 5)]
    ep, _, look = run_episode(turns, [ROWS, ROWS, ROWS])
    assert len(look.ran) == 3 and (ep["end"], ep["n_calls"]) == ("limit", 3)
    assert ep["final_query"] == "SELECT 3" and len(ep["turns"]) == 4


def test_no_query_an_unreadable_call_and_a_cut_turn():
    ep, _, _ = run_episode([("", "It can't be answered.", True)], [])
    assert (ep["end"], ep["final_query"], ep["text"]) == ("no_query", None, "</think>")
    # an unreadable call comes back as an error and uses a call slot
    turns = [("", tc("a") + tc("b"), True), ("", tc("SELECT 1"), True), ("", "done", True)]
    ep, _, look = run_episode(turns, [ROWS])
    assert look.ran == ["SELECT 1"] and ep["n_calls"] == 2 and ep["end"] == "reply"
    assert ep["turns"][0]["obs"] == "Error: one tool call at a time"
    # a turn that ran out of tokens ends the episode; an earlier query still stands
    ep, _, look = run_episode([("", tc("SELECT 1"), True), ("", tc("SELECT 2"), False)], [ROWS])
    assert (ep["end"], ep["final_query"], look.ran) == ("cut", "SELECT 1", ["SELECT 1"])


def test_a_single_turn_dataset_shows_nothing_and_takes_the_first_call():
    cfg = rcfg.new_config({"single_turn_sets": ["gretel"]})
    ep, _, look = run_episode([("", tc("SELECT 1"), True)], [], cfg)
    assert look.ran == []
    assert (ep["end"], ep["final_query"], ep["n_calls"]) == ("single", "SELECT 1", 1)
    assert sum(ep["gen_mask"]) == len(ep["gen_ids"])
    # one turn only: the trainer's later-turn mask is empty, the first turn is every token
    n1, m1, m2 = turn_masks(ep)
    assert n1 == len(ep["gen_ids"]) and sum(m1) == n1 and sum(m2) == 0
    # a call that can't be read ends it too: no query, so the checker calls it malformed
    ep, _, look = run_episode([("", tc("a") + tc("b"), True)], [], cfg)
    assert look.ran == [] and (ep["end"], ep["final_query"], ep["n_calls"]) == ("single", None, 0)
    assert ep["text"] == "</think>" and "obs" not in ep["turns"][0]
    # a plain reply is "no query", as with turns
    ep, _, _ = run_episode([("", "It can't be answered.", True)], [], cfg)
    assert (ep["end"], ep["final_query"], ep["one_query"]) == ("no_query", None, True)
    assert summary.with_turns([ep]) == []  # still not an episode with turns
    assert run_episode([("", "No.", True)], [])[0]["one_query"] is False


def test_the_random_fill_datasets_keep_one_query_and_the_real_data_ones_have_turns():
    assert CFG["single_turn_sets"] == ["sqale", "text2cypher"]
    for origin in ("sqale", "text2cypher"):
        assert prompt.protocol(CFG, origin) == ("sft", 1)
    for origin in ("gretel", "cypherbench"):
        assert prompt.protocol(CFG, origin) == ("turns", 3)
    # a one-query question gets no look-around database at all
    with pytest.raises(RuntimeError):
        env.NoLook().run("SELECT 1")


def test_the_keep_wording_adds_one_sentence_to_the_turn_rule():
    turns = prompt.system_prompt(SFT_SYSTEM, "turns", 3)
    keep = prompt.system_prompt(SFT_SYSTEM, "keep", 3)
    assert keep.replace(prompt.KEEP_HINT, "") == turns and prompt.KEEP_HINT in keep
    assert "no rows can be the correct result" in keep


def test_an_evaluation_folder_keeps_one_protocol(tmp_path):
    out = tmp_path / "eval" / "episodes.jsonl"
    proto = episodes.claim(out, CFG)
    assert proto["single_turn_sets"] == ["sqale", "text2cypher"] and proto["history"] == "full"
    assert episodes.same_protocol(out.parent, CFG)
    every = rcfg.new_config({"single_turn_sets": []})
    assert not episodes.same_protocol(out.parent, every)
    assert not episodes.same_protocol(out.parent, CFG, "stripped")
    assert not episodes.same_protocol(tmp_path / "nothing", CFG)
    episodes.claim(out, every)  # nothing generated yet: the folder can still change hands
    out.write_text('{"qid": "a", "g": 0}\n', encoding="utf-8")
    episodes.claim(out, every)  # the same protocol continues its own file
    fewer = rcfg.new_config({"single_turn_sets": [], "max_queries": 2})
    hinted = rcfg.new_config({"single_turn_sets": [], "prompt_variant": "check"})
    for other in (CFG, fewer, hinted):
        with pytest.raises(RuntimeError, match="another protocol"):
            episodes.claim(out, other)
    with pytest.raises(RuntimeError, match="another protocol"):
        episodes.claim(out, every, "stripped")
    # other look-around data is another protocol: a reference made before a fill changed
    assert episodes.same_protocol(out.parent, every)
    old = env.LOOK_VERSION
    try:
        env.LOOK_VERSION = old + 1
        assert not episodes.same_protocol(out.parent, every)
        with pytest.raises(RuntimeError, match="another protocol"):
            episodes.claim(out, every)
    finally:
        env.LOOK_VERSION = old
    # a record written before the version existed is version 1
    rec = out.parent / "protocol.json"
    saved = json.loads(rec.read_text(encoding="utf-8"))
    assert saved.pop("look") == 1
    rec.write_text(json.dumps(saved), encoding="utf-8")
    assert episodes.same_protocol(out.parent, every)
    # episodes with no record at all (made before the record existed) are refused too
    (out.parent / "protocol.json").unlink()
    with pytest.raises(RuntimeError, match="none recorded"):
        episodes.claim(out, every)


def test_a_full_slot_ends_the_episode():
    cfg = rcfg.new_config({"slot_ctx": 3 + episodes.CTX_MARGIN + episodes.MIN_ROOM - 1})
    ep, _, _ = run_episode([("", tc("SELECT 1"), True)], [ROWS], cfg)
    assert ep["end"] == "context" and ep["turns"] == []


def test_stripped_history_drops_the_thought_from_what_the_model_reads():
    turns = [("ZZZ", tc("SELECT 1"), True), ("", "ok", True)]
    ep, srv, _ = run_episode(turns, [ROWS], history="stripped")
    last_ctx = srv.seen[-1][0]
    assert ord("Z") not in last_ctx  # the thought is gone from what the model reads
    assert ord("Z") in ep["gen_ids"]  # (the record still has what was sampled)
    assert last_ctx[2] == srv.think_end_id  # `<think>` became `</think>`


# ---- the scheduler's lane ------------------------------------------------------------------
def test_plan_runs_one_graph_at_a_time_and_caps_cypher():
    groups = [(q, list(range(8))) for q in range(12)]
    key = dict.fromkeys(range(12))
    key.update({0: ("graph", "a"), 1: ("graph", "a"), 2: ("graph", "b"), 3: ("fill", "x")})
    plan = episodes.Plan(groups, key, cypher_slots=4)
    lock, flying, got, bad = threading.Lock(), {}, [], []

    def worker():
        cur = None
        while (job := plan.take(cur)) is not None:
            if job == episodes.WAIT:
                continue
            q = cur = job[0]
            with lock:
                got.append(job[:2])
                if key[q] is not None:
                    flying[key[q]] = flying.get(key[q], 0) + 1
                    live = [k for k, v in flying.items() if v > 0]
                    if len(live) > 1 or sum(flying.values()) > 4:
                        bad.append(dict(flying))
            if key[q] is not None:
                with lock:
                    flying[key[q]] -= 1
            plan.done(q)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [(q, g) for q in range(12) for g in range(8)]
    assert not bad
    assert plan.switches == 3  # graph a (two questions together), graph b, the fill


def test_plan_waits_when_only_another_graphs_work_is_left():
    key = {0: ("graph", "a"), 1: ("graph", "b")}
    plan = episodes.Plan([(0, [0, 1]), (1, [0])], key, cypher_slots=8)
    assert plan.take(None) == (0, 0, False)
    assert plan.take(None) == (0, 1, True)
    assert plan.take(None) == episodes.WAIT  # graph b must wait for graph a's episodes
    plan.done(0)
    assert plan.take(None) == episodes.WAIT
    plan.done(0)
    assert plan.take(None) == (1, 0, False)
    plan.done(1)
    assert plan.take(None) is None


def test_group_order_and_seeds():
    pubs = [
        {"origin": "gretel", "db": {}},
        {"origin": "cypherbench", "db": {"graph": "b"}},
        {"origin": "text2cypher", "db": {}},
        {"origin": "cypherbench", "db": {"graph": "a"}},
    ]
    assert episodes.group_order(pubs) == [3, 1, 2, 0]
    seeds = {
        episodes.seed_for(s, q, g) + 2 * t + h
        for s in (1, 2)
        for q in range(32)
        for g in range(8)
        for t in range(4)
        for h in (0, 1)
    }
    assert len(seeds) == 2 * 32 * 8 * 4 * 2


def test_eval_and_train_specs():
    ev = episodes.eval_spec(CFG)
    assert ev(0, 0)["sampling"]["temperature"] == 0.0
    assert ev(0, 1)["sampling"]["temperature"] == 1.0
    assert ev(3, 0)["obs"] == CFG["obs"]["eval"] and not ev(3, 0)["logprobs"]
    tr = episodes.train_spec(CFG, 5)
    assert tr(1, 0)["logprobs"] and not tr(1, 1)["logprobs"]
    assert tr(1, 0)["seed"] != tr(1, 1)["seed"] != ev(1, 1)["seed"]


# ---- rewards ---------------------------------------------------------------------------------
def ep(qid="q", g=0, ok=True, calls_n=1, json_ok=True, **kw):
    return {
        "qid": qid,
        "g": g,
        "origin": "gretel",
        "json_ok": json_ok,
        "exec_ok": ok,
        "n_calls": calls_n,
        "final_query": "SELECT 1" if json_ok else None,
        **kw,
    }


def test_reward_rules():
    assert rewards.reward(ep(), CFG) == 1.0
    assert rewards.reward(ep(calls_n=2), CFG) == pytest.approx(0.97)
    assert rewards.reward(ep(calls_n=3), CFG) == pytest.approx(0.94)
    assert rewards.reward(ep(ok=False, calls_n=3), CFG) == 0.0  # a wrong answer never pays less
    assert rewards.reward(ep(ok=None, json_ok=False), CFG) == -0.1  # no query at all
    assert rewards.reward(ep(ok=None), CFG) is None  # undecided
    assert rewards.reward(ep(dup="differ"), CFG) is None  # withheld by the duplicate probe
    assert rewards.reward(ep(hardcoded="constants"), CFG) == 0.0  # right rows, not computed
    assert rewards.reward(ep(ok=False, hardcoded="constants"), CFG) == 0.0
    off = rcfg.new_config({"hardcode_rule": False})
    assert rewards.reward(ep(hardcoded="constants"), off) == 1.0


def test_the_design_pages_group():
    eps = [
        ep(g=0),
        ep(g=1, ok=False),
        ep(g=2, calls_n=2),
        ep(g=3, ok=False, calls_n=2),
        ep(g=4, calls_n=3),
        ep(g=5),
        ep(g=6, ok=None, json_ok=False),
        ep(g=7, ok=False),
    ]
    for e in eps:
        e["reward"] = rewards.reward(e, CFG)
    rewards.advantages(eps, "std", std_floor=0.25)
    assert [round(e["reward"], 2) for e in eps] == [1.0, 0.0, 0.97, 0.0, 0.94, 1.0, -0.1, 0.0]
    want = [1.04, -0.95, 0.98, -0.95, 0.92, 1.04, -1.15, -0.95]
    assert [round(e["adv"], 2) for e in eps] == want
    l2.route(eps)
    routes = ["grpo", "sdpo", "grpo", "sdpo", "grpo", "grpo", "sdpo", "sdpo"]
    assert [e["route"] for e in eps] == routes
    assert {e["teacher_g"] for e in eps if e["route"] == "sdpo"} == {0}  # right the first time


def test_informative_means_right_and_not_right():
    all_right = [ep("a", 0), ep("a", 1, calls_n=3)]  # the turn cost alone is no signal
    mixed = [ep("b", 0), ep("b", 1, ok=False)]
    wrong_and_none = [ep("c", 0, ok=False), ep("c", 1, ok=None, json_ok=False)]
    eps = all_right + mixed + wrong_and_none
    for e in eps:
        e["reward"] = rewards.reward(e, CFG)
    st = rewards.group_stats(eps, CFG)
    assert st["informative_frac"] == pytest.approx(1 / 3)
    assert st["solve_rate"] == pytest.approx(3 / 5)
    assert st["malformed_frac"] == pytest.approx(1 / 6)
    assert st["distinct_final_mean"] == pytest.approx((1 + 1 + 2) / 3)


def test_advantages_are_run_1s():
    rnd = random.Random(3)
    a = [
        {"qid": f"q{i // 8}", "g": i % 8, "reward": rnd.choice([1.0, 0.97, 0.0, -0.1, None])}
        for i in range(64)
    ]
    b = [dict(x) for x in a]
    rewards.advantages(a, "std", std_floor=0.25)
    rewards1.advantages(b, "std", std_floor=0.25)
    assert [x["adv"] for x in a] == [x["adv"] for x in b]


# ---- the copied losses -----------------------------------------------------------------------
def test_losses_are_run_1s():
    torch.manual_seed(0)
    lp, old = -torch.rand(12), -torch.rand(12)
    mask = torch.tensor([1] * 10 + [0, 0])
    mcfg = CFG["method_cfg"]
    a, sa = l2.sequence_loss(mcfg, lp, old, mask, 0.7, 10.0)
    b, sb = l1.sequence_loss(mcfg, lp, old, mask, 0.7, 10.0)
    assert torch.equal(a, b) and sa == sb
    hs, ht, head = torch.randn(9, 16), torch.randn(9, 16), torch.randn(50, 16)
    assert torch.equal(l2.topk_jsd(hs, ht, head, 10), l1.topk_jsd(hs, ht, head, 10))
    assert torch.equal(l2.topk_kl(hs, ht, head, 10), l1.topk_kl(hs, ht, head, 10))
    seqs = [{"qid": "a", "gen_mask": [1, 1, 0]}, {"qid": "b", "gen_mask": [1]}]
    assert l2.norm_for("token_mean", seqs, CFG) == l1.norm_for("token_mean", seqs, CFG) == 3.0
    # routing is run 1's when every episode took one call
    eps = [{"qid": "q", "g": g, "reward": r} for g, r in enumerate([0.0, 1.0, None, 1.0])]
    one, two = [dict(e) for e in eps], [dict(e) for e in eps]
    l1.route(one)
    l2.route(two)
    assert one == two


def test_turn_masks_split_the_first_turn_from_the_rest():
    # turn 1 (with a forced token), a result block, turn 2
    e = {"turns": [{"tokens": 3}, {"tokens": 2}], "gen_mask": [1, 0, 1, 0, 0, 1, 1]}
    n1, m1, m2 = turn_masks(e)
    assert n1 == 3 and m1 == [1, 0, 1, 0, 0, 0, 0] and m2 == [0, 0, 0, 0, 0, 1, 1]
    assert [a + b for a, b in zip(m1, m2, strict=True)] == e["gen_mask"]


def test_minibatches_keep_groups_whole():
    seqs = [{"qid": f"q{i % 5}", "g": i} for i in range(20)]
    mbs = minibatches(seqs, 2, 7)
    assert sum(len(b) for b in mbs) == 20
    assert not {s["qid"] for s in mbs[0]} & {s["qid"] for s in mbs[1]}


# ---- the settings ----------------------------------------------------------------------------
SETS = ("cypherbench", "gretel", "sqale", "text2cypher")


def fake_pool(n=400, back=604):
    out = {s: [{"id": f"{s}-{i}", "origin": s, "set_index": i} for i in range(n)] for s in SETS}
    out["gretel_returned"] = [{"id": f"back-{i}", "origin": "gretel", "set_index": i,
                               "kind": "returned"} for i in range(back)]  # fmt: skip
    return out


def idx(r: dict) -> int:
    return int(r["id"].split("-")[1])


def test_step_questions_skip_run_1bs_and_never_reach_the_held_back_ones():
    pool = fake_pool(1000)
    cfg = rcfg.new_config()
    first = rcfg.step_questions(pool, cfg, 1)
    assert len(first) == 32
    # run 1b used indices below 40 x its mix; run 2 starts right after
    plain = [r for r in first if r["origin"] == "gretel" and r.get("kind") != "returned"]
    back = [r for r in first if r.get("kind") == "returned"]
    assert (len(plain), len(back)) == (7, 3)  # 3 of gretel's 10 are returned questions
    assert min(idx(r) for r in plain) == 400 and [idx(r) for r in back] == [0, 1, 2]
    assert min(idx(r) for r in first if r["origin"] == "text2cypher") == 160
    every = [r for k in range(1, 41) for r in rcfg.step_questions(pool, cfg, k)]
    used = {r["id"] for r in every}
    assert len(used) == len(every) == 40 * 32  # every question once
    # taking 7 gretel questions a step must not move the offset: run 1b took 10 a step
    gretel = sorted(idx(r) for r in every if r["id"].startswith("gretel-"))
    assert gretel == list(range(400, 400 + 40 * 7))
    assert sorted(idx(r) for r in every if r.get("kind") == "returned") == list(range(120))
    dev = {r["id"] for s in SETS for r in pool[s][-cfg["dev_per_set"] :]}
    dev |= {r["id"] for r in pool["gretel_returned"][-cfg["dev_returned"] :]}
    loop = rcfg.new_config({"pool_tail": True, "mix": dict.fromkeys(SETS, 2),
                            "returned_per_step": {"gretel": 1}})  # fmt: skip
    steps = [rcfg.step_questions(pool, loop, k) for k in (1, 2)]
    tail = {r["id"] for rows in steps for r in rows}
    assert not used & dev and not used & tail and not tail & dev and len(tail) == 16
    for rows in steps:  # a loop test holds every kind of question
        assert sum(r.get("kind") == "returned" for r in rows) == 1
        assert {r["origin"] for r in rows} == set(SETS)
    with pytest.raises(ValueError):  # the pool must reach the last step
        rcfg.step_questions(fake_pool(900), cfg, 40)
    with pytest.raises(ValueError):  # ... and so must the returned questions
        rcfg.step_questions(fake_pool(1000, back=200), cfg, 40)
    with pytest.raises(ValueError):  # a missing returned file is not a silent 7-question step
        rcfg.step_questions({s: pool[s] for s in SETS}, cfg, 1)
    with pytest.raises(ValueError):  # a loop test stays inside its reserve
        rcfg.step_questions(pool, rcfg.new_config({"pool_tail": True}), 12)
    none = rcfg.new_config({"returned_per_step": {}})  # without them: run 1b's arithmetic
    assert [idx(r) for r in rcfg.step_questions(pool, none, 2) if r["origin"] == "gretel"] == list(
        range(410, 420)
    )


def test_new_config_merges_nested_overrides():
    over = {"stop": {"kl_start_max": 0.1}, "obs": {"eval": {"rows": 5}}, "steps": 12, "lr": None}
    cfg = rcfg.new_config(over)
    assert cfg["stop"]["kl_start_max"] == 0.1 and cfg["stop"]["loops_up"] == 0.10
    assert cfg["obs"]["eval"] == {"rows": 5, "style": "table"}
    assert cfg["obs"]["rows"] == [5, 10, 20]
    assert cfg["quick_eval_steps"] == [5, 10] and cfg["eval_steps"] == [] and cfg["lr"] == 5e-6
    assert rcfg.lr_at(cfg, 1) == pytest.approx(1e-6) and rcfg.lr_at(cfg, 9) == 5e-6
    assert cfg["method_cfg"]["kl"]["turns"] == cfg["method_cfg"]["sdpo"]["turns"] == "first"


# ---- canaries --------------------------------------------------------------------------------
BASE = {
    "entropy_t1": 0.4,
    "distinct_final_mean": 4.0,
    "informative_frac": 0.5,
    "t1_tokens_median": 100.0,
    "kl_start": 0.0,
    "kl_step1": 0.010,
    "mismatch": 0.04,
    "loop_frac": 0.0,
    "too_long_frac": 0.0,
    "broken_frac": 0.02,
    "hardcoded_frac": 0.0,
    "look_leak_frac": 0.0,
    "select_star_frac": 0.0,
    "leak_frac": 0.0,
    "malformed_frac": 0.0,
    "calls_mean": 1.3,
    "limit_frac": 0.02,
    "first_solve_rate": 0.5,
    "copied_frac": 0.03,
    "distinct_frac": 0.2,
    "dup_differ_frac": 0.1,
}


def hist(n=8, **last):
    h = [dict(BASE, step=i + 1) for i in range(n)]
    h[-1].update(last)
    return h


def test_a_healthy_run_passes_every_rule():
    assert canaries.stop_reasons(hist(), CFG["stop"]) == []
    assert canaries.stop_reasons([], CFG["stop"]) == []


@pytest.mark.parametrize(
    "patch, word",
    [
        ({"entropy_t1": 0.09}, "entropy"),
        ({"t1_tokens_median": 20.0}, "length"),
        ({"t1_tokens_median": 400.0}, "length"),
        ({"kl_start": 0.06}, "KL to the start"),
        ({"kl_step1": 0.03}, "step 1"),
        ({"mismatch": 0.2}, "log-prob gap"),
        ({"too_long_frac": 0.1}, "too long"),
        ({"hardcoded_frac": 0.05}, "constants"),
        ({"look_leak_frac": 0.1}, "reference literal"),
        ({"select_star_frac": 0.1}, "SELECT *"),
        ({"leak_frac": 0.05}, "given solution"),
        ({"copied_frac": 0.2}, "read off a result"),
        ({"distinct_frac": 0.5}, "DISTINCT"),
    ],
)
def test_each_single_step_rule_fires(patch, word):
    why = canaries.stop_reasons(hist(**patch), CFG["stop"])
    assert len(why) == 1 and word in why[0], why


@pytest.mark.parametrize(
    "patch, word",
    [
        ({"distinct_final_mean": 1.5}, "different"),
        ({"malformed_frac": 0.1}, "no query"),
        ({"calls_mean": 2.4}, "calls an episode"),
        ({"limit_frac": 0.3}, "a call too many"),
        ({"loop_frac": 0.2}, "same query twice"),
        ({"broken_frac": 0.2}, "made wrong"),
        ({"first_solve_rate": 0.35}, "first query"),
        ({"dup_differ_frac": 0.4}, "repeat rows"),
    ],
)
def test_each_streak_rule_needs_its_streak(patch, word):
    h = hist(8)
    h[-1].update(patch)
    assert canaries.stop_reasons(h, CFG["stop"]) == []  # one step is not a streak
    for m in h[-3:]:
        m.update(patch)
    why = canaries.stop_reasons(h, CFG["stop"])
    assert len(why) == 1 and word in why[0], why


def test_informative_rule_and_the_start_of_a_run():
    h = hist(8)
    for m in h[-5:]:
        m["informative_frac"] = 0.1
    assert "informative" in canaries.stop_reasons(h, CFG["stop"])[0]
    # the relative rules wait for the run's own first five steps
    assert canaries.stop_reasons(hist(5, entropy_t1=0.01, copied_frac=0.9), CFG["stop"]) == []
    assert "step 1" in canaries.stop_reasons(hist(1, kl_step1=0.03), CFG["stop"])[0]


def test_text_canaries():
    e1 = {
        "final_query": "SELECT DISTINCT a FROM t",
        "gen_mask": [1] * 30,
        "prompt_ids": [0] * 10,
        "gen_ids": [0] * 30,
        "turns": [{"tokens": 30, "text": "the correct solution is given</think>x"}],
    }
    e2 = {
        "final_query": None,
        "gen_mask": [1] * 10 + [0] * 5 + [1] * 20,
        "prompt_ids": [0] * 10,
        "gen_ids": [0] * 35,
        "turns": [
            {"tokens": 10, "text": "a</think>b", "forced": True},
            # a later turn talking about its own retry is not the teacher leaking
            {"tokens": 20, "text": "let me write the correct query</think>d"},
        ],
    }
    c = canaries.text_canaries([e1, e2], max_seq=42)
    assert c["distinct_frac"] == 1.0 and c["leak_frac"] == 0.5 and c["think_forced_frac"] == 0.5
    assert c["t1_tokens_median"] == 20.0 and c["later_tokens_median"] == 20.0
    assert c["too_long_frac"] == 0.5 and c["episode_tokens_mean"] == 30.0


# ---- scoring's own fields and the summaries ---------------------------------------------------
def test_annotate_marks_hardcoded_copied_leaked_and_looped():
    r = row(gold="SELECT name FROM t WHERE name = 'Hidden Value'", question="Who is it?")
    seen = "name\nAlice Cooper\nHidden Value"
    e = {
        "final_query": "SELECT 'Alice Cooper'",
        "queries": [
            {"query": "SELECT name FROM t", "obs": seen, "tables": ["t"]},
            {"query": "select name   from t;", "obs": "name\nAlice Cooper", "tables": ["t"]},
            {"query": "SELECT 'Alice Cooper'", "obs": "x\nAlice Cooper", "tables": []},
        ],
    }
    scoring.annotate(e, r)
    assert e["hardcoded"] == "no_table" and e["copied"] == ["Alice Cooper"]
    assert e["look_leak"] == ["Hidden Value"] and e["loop"] is True
    one = {"query": "SELECT name FROM t", "obs": "name\nBob", "tables": ["t"]}
    clean = {"final_query": "SELECT name FROM t", "queries": [one]}
    scoring.annotate(clean, r)
    got = (clean["hardcoded"], clean["copied"], clean["look_leak"], clean["loop"])
    assert got == (None, [], [], False)
    none = {"final_query": None, "queries": []}
    scoring.annotate(none, r)
    assert none["hardcoded"] is None and none["loop"] is False
    # the answer run again after a look at the data is not a loop (the last query that ran is
    # the answer, so it has to be run again); the same query twice in a row is
    look = {"query": "SELECT * FROM t LIMIT 5", "obs": "0 rows", "tables": ["t"]}
    again = {"final_query": one["query"], "queries": [one, look, dict(one)]}
    scoring.annotate(again, {**r, "kind": "returned"})
    assert (again["loop"], again["rerun"], again["kind"]) == (False, True, "returned")
    assert e["rerun"] is True and clean["rerun"] is False and clean["kind"] is None


def test_returned_questions_are_reported_beside_the_other_gretel_questions():
    back = [
        sep(True, True, 1, kind="returned", exec_weak=True),
        sep(True, False, 2, kind="returned", exec_weak=False),  # rewrote a right answer
        sep(False, False, 1, kind="returned", exec_weak=True),  # empty, but not the question
        sep(False, None, 1, kind="returned", exec_weak=True),  # no filled copy could tell
    ]
    other = [sep(False, True, 2), sep(True, True, 1), sep(False, False, 3)]
    m = summary.returned_metrics(back + other + [sep(True, True, 1, "cypherbench")])
    assert m["returned_n"] == 4 and m["returned_solve"] == pytest.approx(1 / 3)
    assert (m["returned_broken"], m["returned_broken_frac"]) == (1, 0.5)
    assert (m["gretel_other_fixed"], m["gretel_other_broken"]) == (1, 0)
    assert m["gretel_other_solve"] == pytest.approx(2 / 3)
    assert m["returned_weak_only_frac"] == pytest.approx(2 / 3)
    assert summary.returned_metrics(other)["returned_weak_only_frac"] is None


def sep(first, final, calls_n=1, origin="gretel", **kw):
    return {
        "origin": origin,
        "first_ok": first,
        "exec_ok": final,
        "n_calls": calls_n,
        "final_query": "q" if calls_n else None,
        "end": "reply",
        "turns": [],
        **kw,
    }


def test_summary_counts_decided_and_strict_and_what_turns_changed():
    eps = [
        sep(True, True),
        sep(False, True, 2),
        sep(True, False, 2),
        sep(False, False, 3),
        sep(None, None),
        sep(None, None, 0),
        sep(False, None, 2),
    ]
    s = summary.summarise(eps)["overall"]
    f = s["final"]
    assert (f["right"], f["wrong"], f["undecided"], f["no_query"]) == (2, 2, 2, 1)
    assert f["decided"] == 0.5 and f["strict"] == pytest.approx(2 / 7)
    assert s["first"]["right"] == 2 and s["first"]["wrong"] == 3
    assert (s["changes"]["fixed"], s["changes"]["broken"], s["changes"]["net"]) == (1, 1, 0)
    assert s["changes"]["broken_frac"] == 0.5 and s["changes"]["fixed_frac"] == 0.5
    assert s["protocol"]["no_query_frac"] == pytest.approx(1 / 7)
    assert s["protocol"]["calls"] == {"0": 1, "1": 2, "2": 3, "3": 1}
    assert summary.headline(summary.summarise(eps))["final_decided"] == 0.5


def test_summary_retry_and_hacking_rates():
    leaky = sep(True, True, 2, "sqale", turns=[{"t": 1, "result": "empty"}])
    leaky.update(look_leak=["x"], copied=["y"], loop=True)
    eps = [
        sep(False, True, 2, turns=[{"t": 1, "result": "error"}]),
        sep(False, False, 1, turns=[{"t": 1, "result": "error"}]),
        sep(True, True, 1, turns=[{"t": 1, "result": "rows"}], hardcoded="constants"),
        leaky,
    ]
    s = summary.summarise(eps)
    o = s["overall"]
    assert o["protocol"]["retry_after"]["error"] == {"n": 2, "again": 0.5}
    assert o["protocol"]["retry_after"]["empty"]["again"] == 1.0
    assert o["hacking"]["hardcoded_frac"] == 0.25 and o["hacking"]["hardcoded_right"] == 1
    assert o["hacking"]["copied_frac"] == 0.25 and o["hacking"]["copied_multi_frac"] == 0.5
    assert o["hacking"]["look_leak_frac"] == 1.0 and o["hacking"]["loop_frac"] == 0.25
    assert set(s["by_set"]) == {"gretel", "sqale"}


def test_single_turn_episodes_count_in_the_score_and_not_in_the_turn_rates():
    """A dataset on run 1's protocol is a quarter of a step's episodes. They count toward the
    score; in the rates the stop rules read they would only make a problem look smaller."""
    turned = [
        sep(True, False, 2, loop=True, turns=[{"t": 1, "result": "rows"}]),
        sep(True, True, 3, end="limit", copied=["x"]),
        sep(False, True, 2, "text2cypher", look_leak=["y"]),
        sep(False, False, 1),
    ]
    single = [sep(True, True, 1, "sqale", end="single", hardcoded="constants"),
              sep(True, True, 1, "sqale", end="single"),
              sep(False, False, 1, "sqale", end="single"),
              sep(None, None, 0, "sqale", end="single")]  # fmt: skip
    alone, both = summary.block(turned), summary.block(turned + single)
    assert (both["n"], both["with_turns"]) == (8, 4)
    assert both["final"]["right"] == 4 and both["first"]["right"] == 4  # every episode scores
    assert both["final"]["no_query"] == 1
    for part in ("changes", "protocol"):
        assert both[part] == alone[part]
    assert both["protocol"]["calls_mean"] == 2.0 and both["protocol"]["limit_frac"] == 0.25
    assert both["changes"]["broken_frac"] == 0.5
    h = both["hacking"]
    assert (h["loop_frac"], h["copied_frac"], h["look_leak_frac"]) == (0.25, 0.25, 1.0)
    assert h["hardcoded_frac"] == pytest.approx(1 / 7)  # the constants rule reads every answer
    # a single-turn dataset's own block has no turn rates at all
    only = summary.block(single)
    assert only["protocol"]["calls_mean"] is None and only["changes"]["broken_frac"] is None
    assert only["hacking"]["look_leak_frac"] is None


def test_what_a_result_shows_and_what_the_turns_did_after_it():
    """The no-training test's key split: a right first answer by what its result showed."""
    assert summary.says_something([(None, 0), ("", "x")])
    assert not summary.says_something([]) and not summary.says_something([(None, 0, 0.0, "")])

    def ran(kind, obs, query="q1"):
        return {"kind": kind, "obs": obs, "query": query}

    assert summary.shown(ran("error", "Error: no such column")) == "error"
    assert summary.shown(ran("empty", "0 rows")) == "nothing"
    assert summary.shown(ran("rows", "n\nNULL")) == "nothing"  # an aggregate over no rows
    assert summary.shown(ran("rows", "a | n\nNULL | 0\n(1 of 9 rows)")) == "nothing"
    assert summary.shown(ran("rows", "name | n\nAlice | 0")) == "something"
    assert summary.shown(ran("rows", '{"n": null}\n{"n": 0}')) == "nothing"
    assert summary.shown(ran("rows", '{"name": "Alice", "n": null}')) == "something"

    def ep(first, final, first_obs, end="reply", last="q2", **kw):
        kind = "error" if first_obs.startswith("Error") else (
            "empty" if first_obs == "0 rows" else "rows")  # fmt: skip
        qs = [ran(kind, first_obs), ran("rows", "x\n1", last)]
        return {"first_ok": first, "exec_ok": final, "queries": qs, "end": end, **kw}

    eps = [
        ep(True, True, "name\nAlice"),
        ep(True, False, "name\nAlice"),  # broke a right answer that showed values: rewrote
        ep(True, False, "0 rows", end="limit"),  # explored after nothing came back: ran out
        ep(True, False, "n\nNULL"),  # rewrote until a number came back
        ep(True, True, "0 rows"),
        ep(False, True, "0 rows"),
        ep(False, False, "Error: no such table"),
        ep(True, False, "0 rows", one_query=True),  # no turns: not counted
        ep(None, True, "0 rows"),  # undecided first query: not counted
    ]
    s = summary.first_result(eps)
    assert s["something"]["kept_right"] == 1 and s["something"]["broken"] == 1
    assert s["something"]["broken_frac"] == 0.5 and s["something"]["rewrote"] == 1
    n = s["nothing"]
    assert (n["kept_right"], n["broken"], n["fixed"], n["kept_wrong"]) == (1, 2, 1, 0)
    assert (n["ran_out"], n["rewrote"]) == (1, 1)
    assert n["broken_frac"] == pytest.approx(2 / 3) and n["fixed_frac"] == 1.0
    assert s["error"]["kept_wrong"] == 1 and s["error"]["broken_frac"] is None


def test_an_evaluation_carries_the_returned_slices_numbers():
    from sqlrl.run2 import orchestrate

    eps = [sep(True, True, 1), sep(True, False, 2), sep(True, False, 3), sep(False, False, 1)]
    brief = orchestrate.returned_brief({"greedy": summary.summarise(eps)})
    assert (brief["n"], brief["first_right"], brief["right"]) == (4, 3, 1)
    assert (brief["fixed"], brief["broken"]) == (0, 2)
    assert brief["broken_frac"] == pytest.approx(2 / 3)
    assert orchestrate.returned_pubs(rcfg.new_config({"returned_per_step": {}})) == []


def test_eval_comparisons():
    assert eval_score.sign_test(0, 0) == 1.0 and eval_score.sign_test(10, 0) < 0.01
    mine = {
        "a": {"final": True, "right": 6, "scored": 8},
        "b": {"final": True, "right": 0, "scored": 8},
        "c": {"final": None, "right": 0, "scored": 0},
        "d": {"final": False},
    }
    ref = {
        "a": {"final": False, "right": 2, "scored": 8},
        "b": {"final": True, "right": 1, "scored": 8},
        "c": {"final": True, "right": 8, "scored": 8},
        "d": {"final": True},
    }
    c = eval_score.compare(mine, ref)
    assert (c["greedy"]["better"], c["greedy"]["worse"]) == (1, 1)  # undecided pairs left out
    assert (c["sampled"]["better"], c["sampled"]["worse"]) == (1, 1)
    assert (c["pass8"]["better"], c["pass8"]["worse"]) == (0, 1)
    base = {"qid": "a", "origin": "gretel"}
    eps = [
        {**base, "g": 0, "exec_ok": True, "first_ok": False, "n_calls": 2},
        {**base, "g": 1, "exec_ok": True},
        {**base, "g": 2, "exec_ok": None},
        {**base, "g": 3, "exec_ok": False},
    ]
    q = eval_score.questions(eps)["a"]
    assert (q["final"], q["first"], q["calls"], q["right"], q["scored"]) == (True, False, 2, 1, 2)


def _feed_run(tmp_path, monkeypatch, steps):
    """A run folder with `steps` trained steps, as the feed reads it."""
    monkeypatch.setattr(rpool, "data_dir", lambda: tmp_path)
    run = tmp_path / "r"
    run.mkdir()
    rcfg.save(rcfg.new_config({"run": "r"}), run)
    rows = [
        {"step": k, "solve_rate": 0.5 + k / 100, "first_solve_rate": 0.5, "step_s": 600.0,
         "solve_by_set": {"gretel": 0.4}, "reward_by_set": {"gretel": 0.3},
         "calls": {"1": 100, "2": 60}, "ends": {"reply": 150, "limit": 10},
         "retry_after": {"empty": {"n": 8, "again": 0.75}, "rows": {"n": 90, "again": 0.1}},
         "limit_frac": 0.1, "loop_frac": 0.0, "broken_frac": 0.2, "entropy_t1": 0.2,
         "t1_tokens_median": 90.0, "look_leak_frac": None, "template_match": True}
        for k in range(1, steps + 1)
    ]  # fmt: skip
    (run / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    (run / "events.jsonl").write_text(json.dumps(
        {"at": "2026-01-01T00:00:00Z", "kind": "info", "text": "step 1: rollouts",
         "phase": "rollouts", "step": 1}) + "\n", "utf-8")  # fmt: skip
    return run


def test_feed_never_creates_a_run_folder(tmp_path, monkeypatch):
    from sqlrl.run2 import feed

    monkeypatch.setattr(rpool, "data_dir", lambda: tmp_path)
    res = feed.write("run2")
    assert res["status"] == "not started" and res["files"] == []
    assert not (tmp_path / "run2").exists()  # the runbook reads "the folder exists" as a resume


def test_feed_rows_are_flat_and_chunked_and_bounds_follow_the_first_five_steps(
    tmp_path, monkeypatch
):
    from sqlrl.run2 import feed

    run = _feed_run(tmp_path, monkeypatch, steps=12)
    res = feed.write("r")
    assert res["files"] == ["state.json", "steps-1.json", "steps-2.json", "evals.json",
                            "refs.json"]  # fmt: skip
    assert res["changed"] == res["files"] and res["step_now"] == 12
    first = json.loads((run / "feed" / "steps-1.json").read_text("utf-8"))
    assert (first["from"], first["to"], len(first["steps"])) == (1, 10, 10)
    row = first["steps"][0]
    assert (row["solve_gretel"], row["reward_gretel"]) == (0.4, 0.3)
    assert (row["calls_1"], row["calls_2"], row["ends_limit"]) == (100, 60, 10)
    assert (row["retry_empty_n"], row["retry_empty_again"], row["retry_rows_again"]) == (
        8, 0.75, 0.1)  # fmt: skip
    assert row["look_leak_frac"] is None and row["template_match"] is True
    assert all(not isinstance(v, (dict, list)) for v in row.values())
    state = json.loads((run / "feed" / "state.json").read_text("utf-8"))
    stop = state["config"]["stop"]
    assert state["bounds"]["limit_frac"] == {
        "start": 0.1, "bound": round(0.1 + stop["limit_up"], 5), "side": "above"}  # fmt: skip
    assert state["bounds"]["entropy_t1"]["bound"] == round(0.2 * stop["entropy_min_frac"], 5)
    assert state["bounds"]["t1_tokens_median"]["high"] == 90.0 * stop["length_ratio"]
    assert state["shape"] == {"questions": 32, "with_turns": 20, "one_query": 12,
                              "returned": 3, "group_size": 8}  # fmt: skip
    # 28 steps of 600 s, and the evaluations still to come at their default lengths
    left = state["timing"]
    assert left["evals_left"] == [15, 20, 25, 30, 35, 40]
    assert left["hours_left"] == pytest.approx((28 * 600 + 4 * 330 + 2 * 1230) / 3600, rel=1e-3)
    assert feed.write("r")["changed"] in ([], ["state.json"])  # only the clock may differ


def test_feed_bounds_wait_for_step_six_and_a_dead_process_is_named(tmp_path, monkeypatch):
    import subprocess
    import sys

    from sqlrl.run2 import feed

    run = _feed_run(tmp_path, monkeypatch, steps=5)
    assert feed.build("r")["state"]["bounds"] == {}
    assert feed.build("r")["state"]["status"] == "running"  # no pid file: it can't be told
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (run / "run.pid").write_text(str(gone.pid), "utf-8")
    state = feed.build("r")["state"]
    assert state["status"] == "dead" and "resumes" in state["why"][0]
    (run / "run.pid").write_text(str(os.getpid()), "utf-8")
    assert feed.build("r")["state"]["status"] == "running"
    (run / "PAUSED").write_text(json.dumps({"step": 5, "why": ["a rule"]}), "utf-8")
    state = feed.build("r")["state"]
    assert (state["status"], state["why"]) == ("paused", ["a rule"])


def test_feed_eval_block_takes_a_quick_and_a_full_evaluation_and_a_bare_one():
    from sqlrl.run2 import feed

    block = {"n": 200, "with_turns": 100,
             "final": {"right": 98, "wrong": 89, "undecided": 10, "no_query": 3,
                       "decided": 0.524064, "strict": 0.49},
             "first": {"right": 96, "decided": 0.51},
             "changes": {"fixed": 7, "broken": 5, "net": 2, "broken_frac": 0.1},
             "protocol": {"calls_mean": 1.62, "calls": {"1": 62},
                          "retry_after": {"empty": {"n": 18, "again": 0.9}}},
             "hacking": {"loop_frac": 0.07}}  # fmt: skip
    quick = {"step": 5, "kind": "quick", "greedy": {"overall": block, "by_set": {"gretel": block}},
             "by_look": {"one_query": block}, "vs": {"best": {"step": 0}},
             "returned_dev": {"n": 50, "right": 11}}  # fmt: skip
    e = feed.eval_block(quick)
    g = e["greedy"]
    assert (g["right"], g["first_right"], g["fixed"], g["broken"], g["decided"]) == (
        98, 96, 7, 5, 0.52406)  # fmt: skip
    assert (g["retry_empty_n"], g["loop_frac"], g["limit_frac"]) == (18, 0.07, None)
    assert e["greedy_by_set"]["gretel"]["right"] == 98 and e["by_look"]["one_query"]["n"] == 200
    assert "sampled" not in e and e["returned_dev"]["right"] == 11
    sampled = {"questions": 195, "greedy_pass1": 0.52, "mean_at8": 0.476,
               "pass_at_k": [0.476, 0.718], "always": 40, "sometimes": 100, "never": 55,
               "hist": [1, 2]}  # fmt: skip
    full = {**quick, "kind": "full", "sampled": {"overall": sampled, "by_set": {"gretel": sampled}},
            "sampled_calls_mean": 1.636}  # fmt: skip
    e = feed.eval_block(full)
    assert e["sampled"]["pass_at_k"] == [0.476, 0.718] and "hist" not in e["sampled"]
    assert e["sampled_by_set"]["gretel"]["mean_at8"] == 0.476
    bare = feed.eval_block({"step": 5})  # nothing raises on a summary that lacks every block
    assert bare["greedy"] is None and bare["greedy_by_set"] == {} and bare["vs"] == {}
    assert feed.returned_brief({}) is None


def test_feed_watch_is_the_last_step_and_the_five_step_means(tmp_path, monkeypatch):
    from sqlrl.run2 import feed

    _feed_run(tmp_path, monkeypatch, steps=7)
    w = feed.write("r")["watch"]
    assert (w["step"], w["mean5_steps"]) == (7, 5)
    assert w["last"]["solve_rate"] == 0.57 and w["last"]["retry_empty_again"] == 0.75
    assert w["mean5"]["solve_rate"] == pytest.approx(0.55)  # steps 3 to 7
    assert w["mean5"]["returned_broken_frac"] is None  # not logged in this made-up run
    assert feed.watch([]) == {}
