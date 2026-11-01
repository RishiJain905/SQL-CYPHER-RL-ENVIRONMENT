import pytest

from sqlrl.step2_pilots import config, rewards

CFG = config.pilot_config("p1-grpo")


def ro(qid, ok, json_ok=True, origin="gretel"):
    r = {"qid": qid, "origin": origin, "exec_ok": ok, "json_ok": json_ok}
    r["reward"] = rewards.reward(r, CFG)
    return r


def test_reward_values():
    assert ro("a", True)["reward"] == 1.0
    assert ro("a", False)["reward"] == 0.0
    assert ro("a", None, json_ok=False)["reward"] == -0.1  # no parsable call
    assert ro("a", None)["reward"] is None  # parsable but undecided: no reward


def test_advantages_center_std_and_signal():
    rs = [ro("q", True), ro("q", False), ro("q", False), ro("q", None),  # mixed group
          ro("s", True), ro("s", True)]  # fmt: skip # solved group: no signal
    rewards.advantages(rs, "center")
    assert [r["adv"] for r in rs[:3]] == pytest.approx([2 / 3, -1 / 3, -1 / 3])
    assert rs[3]["adv"] is None
    assert rs[4]["adv"] == 0.0 and rs[5]["adv"] == 0.0
    rewards.advantages(rs, "std")
    sd = (2 / 9) ** 0.5
    assert rs[0]["adv"] == pytest.approx((2 / 3) / sd, rel=1e-4)


def test_group_stats():
    rs = [ro("q", True), ro("q", False), ro("s", True, origin="sqale"), ro("s", None)]
    st = rewards.group_stats(rs)
    assert st["informative_frac"] == 0.5
    assert st["undecided_frac"] == 0.25
    assert st["solve_rate"] == pytest.approx(2 / 3)
    assert st["reward_by_set"] == {"gretel": 0.5, "sqale": 1.0}
