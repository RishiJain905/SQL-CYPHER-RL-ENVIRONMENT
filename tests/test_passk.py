from math import comb

import pytest

from sqlrl import passk


def test_pass_at_k_edges():
    assert passk.pass_at_k(8, 0, 1) == 0.0
    assert passk.pass_at_k(8, 8, 1) == 1.0
    assert passk.pass_at_k(8, 3, 1) == pytest.approx(3 / 8)
    assert passk.pass_at_k(8, 1, 8) == 1.0
    assert passk.pass_at_k(8, 3, 2) == pytest.approx(1 - comb(5, 2) / comb(8, 2))
    # fewer decided samples than k: any right answer counts
    assert passk.pass_at_k(5, 1, 8) == 1.0
    assert passk.pass_at_k(5, 0, 8) == 0.0
    with pytest.raises(ValueError):
        passk.pass_at_k(0, 0, 1)


def item(qid, origin, ok):
    return {"id": qid, "origin": origin, "exec_ok": ok}


def test_summarise_buckets_and_undecided():
    # q1 always right, q2 right 1 of 2, q3 never, q4 never decided
    s1 = [
        item("q1", "a", True),
        item("q2", "a", True),
        item("q3", "b", False),
        item("q4", "b", None),
    ]
    s2 = [
        item("q1", "a", True),
        item("q2", "a", False),
        item("q3", "b", False),
        item("q4", "b", None),
    ]
    s = passk.summarise([s1, s2], {"q1": True, "q2": False, "q3": None})
    o = s["overall"]
    assert o["questions"] == 3
    assert (o["always"], o["sometimes"], o["never"]) == (1, 1, 1)
    assert o["hist"] == [1, 1, 1]
    assert o["undecided_samples"] == 2
    assert o["greedy_pass1"] == pytest.approx(0.5)
    assert o["mean_at8"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert o["pass_at_k"] == pytest.approx([(1 + 0.5 + 0) / 3, 2 / 3])
    assert s["by_set"]["a"]["questions"] == 2 and s["by_set"]["b"]["questions"] == 1
    assert {q["id"]: q["right"] for q in s["questions"]} == {"q1": 2, "q2": 1, "q3": 0, "q4": 0}
