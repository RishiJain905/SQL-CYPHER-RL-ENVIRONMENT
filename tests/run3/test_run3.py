"""Run 3's environment on the CPU: the gate, the protocol, rendering, porting to DuckDB, and
whole episodes played by a scripted policy on a small SQLite database (no server needed; the
PostgreSQL / MySQL / Neo4j parts are covered by `scripts/run3/check_env.py` on a live machine).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlrl.run3 import engines, env, episode, guards, porting, protocol, settings


@pytest.fixture()
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RUN3_DATA", str(tmp_path))
    d = tmp_path / "dbs" / "toy"
    d.mkdir(parents=True)
    db = sqlite3.connect(d / "shop.sqlite")
    db.executescript(
        """
        CREATE TABLE "Customer" (id INTEGER PRIMARY KEY, "Full Name" TEXT, city TEXT);
        CREATE TABLE orders (id INTEGER, customer_id INTEGER REFERENCES Customer(id),
                             amount REAL, day TEXT);
        INSERT INTO Customer VALUES (1, 'Ann Lee', 'Oslo'), (2, 'Bo Chan', 'Rome'),
                                    (3, 'Cy Diaz', 'Oslo');
        INSERT INTO orders VALUES (1, 1, 10.5, '2024-01-02'), (2, 1, 4, '2024-02-03'),
                                  (3, 2, 7.25, '2024-02-04'), (4, 3, NULL, '2024-03-01');
        """
    )
    db.commit()
    db.close()
    return tmp_path


def _checker_or_skip():
    try:
        return settings.checker()
    except Exception:
        pytest.skip("the trainer's checker isn't available on this machine")


# ---- the gate and the guards ------------------------------------------------------------------


def test_one_statement():
    assert guards.one_statement("SELECT 1;")
    assert guards.one_statement("SELECT ';' AS x")
    assert not guards.one_statement("SELECT 1; SELECT 2")
    assert not guards.one_statement("SELECT 1; DROP TABLE t;")


@pytest.mark.parametrize(
    "engine,q,ok",
    [
        ("sqlite", "PRAGMA table_info(t)", True),
        ("sqlite", "DELETE FROM t", False),
        ("duckdb", "SHOW TABLES", True),
        ("duckdb", "FROM t", True),
        ("duckdb", "COPY t TO 'x.csv'", False),
        ("postgres", "TABLE t", True),
        ("postgres", "SHOW search_path", False),
        ("mysql", "SHOW TABLES", True),
        ("mysql", "SET @a = 1", False),
        ("mysql", "SELECT 1; SELECT 2", False),
    ],
)
def test_gate(engine, q, ok):
    assert (engines.gate(engine, q) is None) == ok


def test_cypher_block():
    assert engines.cypher_block("MATCH (n) RETURN n.name") is None
    assert engines.cypher_block("MATCH (n) DETACH DELETE n") == engines.MSG_CYPHER_READ
    assert engines.cypher_block("CALL apoc.load.json('x')") == engines.MSG_CALL
    assert engines.cypher_block("CALL db.labels()") is None
    assert engines.cypher_block("MATCH (n {name: 'SET'}) RETURN n") is None


def test_sqlite_is_read_only(data):
    c = engines.open_sql("sqlite", "toy", "shop")
    assert c.run("SELECT count(*) FROM orders").rows == [(4,)]
    assert c.run("PRAGMA table_info(orders)").kind == "rows"
    assert c.run("PRAGMA writable_schema = 1").error == engines.MSG_READ
    assert c.run("WITH x AS (SELECT 1) SELECT * FROM x").kind == "rows"
    r = c.run("SELECT nope FROM orders")
    assert r.kind == "error" and "nope" in r.error
    assert c.run("SELECT 1 WHERE 0").kind == "empty"
    c.close()


# ---- porting ----------------------------------------------------------------------------------


def test_column_kinds_follow_the_values(data):
    src = sqlite3.connect(engines.sqlite_path("toy", "shop"))
    kinds = dict(porting.column_kinds(src, "orders"))
    src.close()
    assert kinds == {"id": "int", "customer_id": "int", "amount": "real", "day": "text"}


def test_target_names():
    assert porting.target_name("postgres", "Customer") == "customer"
    assert porting.target_name("postgres", "Full Name") == "Full Name"
    assert porting.target_name("mysql", "Customer") == "Customer"


def test_port_to_duckdb(data):
    pytest.importorskip("duckdb")
    out = porting.port("toy", "shop", ["duckdb"])
    assert out["engines"]["duckdb"]["ok"], out
    c = engines.open_sql("duckdb", "toy", "shop")
    assert c.run("SELECT count(*), sum(amount) FROM orders").rows == [(4, 21.75)]
    assert c.run('SELECT "Full Name" FROM Customer WHERE id = 2').rows == [("Bo Chan",)]
    assert c.run("SELECT 1; SELECT 2").error == engines.MSG_READ
    assert c.run("SELECT * FROM read_csv('x.csv')").kind == "error"
    c.close()


# ---- rendering --------------------------------------------------------------------------------


def test_render_styles():
    r = engines.Result.of(["a", "b"], [(1, "x"), (2, None), (3, "z")], 0.0)
    assert env.render(r, 2, "table", True) == "a | b\n1 | x\n2 | NULL\n(2 of 3 rows)"
    assert env.render(r, 5, "csv", True) == "a,b\n1,x\n2,NULL\n3,z"
    assert env.render(r, 1, "json", True).splitlines()[0] == '{"a": 1, "b": "x"}'
    e = engines.Result.fail("Binder Error: no column", "Did you mean x?")
    assert env.render(e, 5, "table", True) == "Error: Binder Error: no column\nDid you mean x?"
    assert env.render(e, 5, "table", False) == "Error: Binder Error: no column"
    assert env.render(engines.Result.of(["a"], [], 0.0), 5, "table", True) == "0 rows"


def test_schema_and_helpers(data):
    c = engines.open_sql("sqlite", "toy", "shop")
    sch = env.describe_sql(c, "toy", "shop")
    assert 'CREATE TABLE Customer (id INTEGER, "Full Name" TEXT, city TEXT);' in sch["full"]
    assert "FOREIGN KEY (customer_id) REFERENCES Customer (id)" in sch["full"]
    assert sch["names"] == "Tables: Customer, orders"
    h = env.SqlHelpers(c)
    assert h.describe_table("customer").rows[1] == ("Full Name", "TEXT")
    assert h.find_values("Customer.city", "osl").rows == [("Oslo",)]
    assert h.find_values("Customer.city", "x' OR '1'='1").kind == "empty"
    assert h.sample_rows("nope").kind == "error"
    c.close()


# ---- the protocol -----------------------------------------------------------------------------


def _setting(tools: str = "helpers") -> protocol.Setting:
    s = protocol.draw_setting([("sqlite", "SQLite 3")], None)
    s.tools = tools
    return s


def test_read_call():
    s = _setting()
    c = protocol.read_call("<think>hm</think>" + protocol.call_text(
        "run_query", {"query": "SELECT 1"}), s)  # fmt: skip
    assert (c.kind, c.name, c.args) == ("ok", "run_query", {"query": "SELECT 1"})
    assert protocol.read_call("</think>The answer is 3.", s).kind == "none"
    assert protocol.read_call("</think><tool_call>run_query<arg_key>query", s).why == "cut"
    two = protocol.call_text("list_tables", {}) * 2
    assert protocol.read_call(two, s).why == "several_calls"
    assert protocol.read_call(protocol.call_text("drop_db", {}), s).why == "unknown_tool"
    n = protocol.read_call(protocol.call_text("sample_rows", {"table": "t", "n": "3"}), s)
    assert n.args["n"] == 3


def test_settings_are_drawn_from_the_seed():
    a = protocol.draw_setting([("duckdb", "DuckDB 1")], 7)
    b = protocol.draw_setting([("duckdb", "DuckDB 1")], 7)
    assert a.record() == b.record()
    names = {t["function"]["name"] for t in protocol.tools(a)}
    assert {a.submit, a.decline, a.clarify, a.sources[0].tool} <= names
    many = protocol.draw_setting([("sqlite", "S"), ("neo4j", "N")], 3)
    assert many.sources[0].tool != many.sources[1].tool
    assert many.sources[1].dialect == "cypher"


def test_system_prompt_modes(data):
    s = _setting()
    sch = [{"full": "CREATE TABLE t (a INTEGER);", "names": "Tables: t"}]
    for mode, want in (("full", "CREATE TABLE t"), ("names", "Tables: t"), ("none", "isn't shown")):
        s.schema = mode
        assert want in protocol.system_prompt(s, sch)
    assert "`submit_answer`" in protocol.system_prompt(s, sch)


# ---- whole episodes ---------------------------------------------------------------------------


def _task(kind: str = "answer") -> dict:
    gold_sql = "SELECT c.\"Full Name\" FROM Customer c WHERE c.city = 'Oslo'"
    return {
        "q": {"id": "t1", "question": "Who lives in Oslo?", "evidence": ""},
        "kind": kind,
        "sources": [{"engine": "sqlite", "source": "toy", "db_id": "shop"}],
        "answer_source": 0,
        "gold": {"sql": gold_sql, "rows": [("Ann Lee",), ("Cy Diaz",)],
                 "columns": ["Full Name"], "full": None},
        "clarify_terms": ["home", "away"],
    }  # fmt: skip


def scripted(*turns: str):
    it = iter(turns)

    def policy(messages, tools):
        return next(it)

    return policy


GOOD = "SELECT \"Full Name\" FROM Customer WHERE city = 'Oslo'"


def test_episode_right_answer_after_looking(data):
    _checker_or_skip()
    s = _setting()
    pol = scripted(
        protocol.call_text("list_tables", {}),
        protocol.call_text("run_query", {"query": "SELECT * FROM Customer LIMIT 2"}),
        protocol.call_text("run_query", {"query": GOOD}),
        protocol.call_text("submit_answer", {"query": GOOD}),
    )
    rec = episode.run_episode(_task(), pol, s)
    assert rec["ending"] == "submit" and rec["n_calls"] == 3
    assert "Customer" in rec["turns"][0]["obs"]
    out = episode.grade(_task(), rec)
    assert out["outcome"] == "right" and out["reward"] == pytest.approx(0.98)
    assert out["last_query_ok"] is True


def test_episode_rewards(data):
    _checker_or_skip()
    s = _setting()
    sub = protocol.call_text("submit_answer", {"query": GOOD})
    wrong = protocol.call_text("submit_answer", {"query": "SELECT city FROM Customer"})
    const = protocol.call_text(
        "submit_answer", {"query": "SELECT 'Ann Lee' UNION SELECT 'Cy Diaz'"}
    )
    dec = protocol.call_text("decline", {"reason": "not stored"})
    cla = protocol.call_text("clarify", {"question": "Home or away score?"})
    cases = [
        ("answer", [sub], "right", 1.0),
        ("answer", [wrong], "wrong", 0.0),
        ("answer", [const], "wrong", 0.0),
        ("answer", [dec], "wrong", 0.0),
        ("answer", ["</think>Ann Lee and Cy Diaz."], "malformed", -0.1),
        ("write", [dec], "right", 1.0),
        ("write", [sub], "wrong", 0.0),
        ("unanswerable", [dec], "right", 1.0),
        ("ambiguous", [cla], "right", 1.0),
        ("answer", [cla], "wrong", 0.0),
    ]
    for kind, turns, outcome, reward in cases:
        t = _task(kind)
        rec = episode.run_episode(t, scripted(*turns), s)
        out = episode.grade(t, rec)
        assert (out["outcome"], out["reward"]) == (outcome, reward), (kind, turns, out)


def test_episode_call_limit(data):
    s = _setting()
    s.max_calls = 2
    look = protocol.call_text("list_tables", {})
    rec = episode.run_episode(_task(), scripted(look, look, look), s)
    assert rec["ending"] == "over_limit" and rec["n_calls"] == 2
    assert "No tool calls left" in rec["turns"][1]["obs"]
    assert episode.grade(_task(), rec)["reward"] == -0.1


def test_unreadable_call_costs_a_call(data):
    s = _setting()
    rec = episode.run_episode(_task(), scripted(
        "</think><tool_call>run_query<arg_key>query", protocol.call_text(
            "decline", {"reason": "x"})), s)  # fmt: skip
    assert rec["n_calls"] == 1 and rec["turns"][0]["obs"].startswith("Error: the call")
