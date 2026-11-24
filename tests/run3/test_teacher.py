import json

import pytest

from sqlrl.run3 import protocol, teacher


def _setting():
    return protocol.draw_setting([("duckdb", "DuckDB 1.3.2")], None)


def test_reply_with_a_call_reads_back_through_the_protocol():
    s = _setting()
    msg = {"reasoning": "look first", "tool_calls": [{"function": {
        "name": s.submit, "arguments": json.dumps({"query": "SELECT 1"})}}]}  # fmt: skip
    text = teacher.to_text(msg)
    call = protocol.read_call(text, s)
    assert call.kind == "ok" and call.name == s.submit and call.args == {"query": "SELECT 1"}
    assert text.startswith("look first" + protocol.THINK_END)


def test_reply_without_a_call_is_none_and_integer_args_survive():
    s = _setting()
    assert protocol.read_call(teacher.to_text({"content": "I think..."}), s).kind == "none"
    s2 = protocol.draw_setting([("duckdb", "x")], None, {"tools_w": [0, 1]})
    args = json.dumps({"table": "t", "n": 5})
    msg = {"tool_calls": [{"function": {"name": "sample_rows", "arguments": args}}]}
    call = protocol.read_call(teacher.to_text(msg), s2)
    assert call.kind == "ok" and call.args == {"table": "t", "n": 5}


def test_messages_get_call_ids_and_json_arguments():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [{"type": "function",
                             "function": {"name": "run_query", "arguments": {"sql": "SELECT 1"}}}]},
            {"role": "tool", "content": "1 row"}]  # fmt: skip
    out = teacher.to_openai(msgs)
    tc = out[2]["tool_calls"][0]
    assert tc["id"] == out[3]["tool_call_id"] and json.loads(tc["function"]["arguments"]) == {
        "sql": "SELECT 1"
    }
    assert "reasoning_content" not in out[2]


def test_budget_stops_the_batch():
    b = teacher.Budget(0.01)
    b.add(0.006)
    assert not b.over()
    b.add(0.006)
    assert b.over()
    pol = teacher.TeacherPolicy(b)
    with pytest.raises(teacher.OverBudget):
        pol([], [])


def test_pilot_refuses_without_approval(capsys):
    assert teacher.main(["pilot", "--out", "x"]) == 2
    assert "refused" in capsys.readouterr().out
