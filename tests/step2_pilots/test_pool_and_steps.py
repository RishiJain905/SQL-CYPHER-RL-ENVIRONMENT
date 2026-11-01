import json

import pytest

from sqlrl.step2_pilots import config, generate, pool
from sqlrl.step2_pilots.train_step import minibatches


def ex(i, origin, q, chars=100):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": q}]
    return {"id": f"{origin}-{i}", "origin": origin, "task": "call",
            "meta": {"schema_chars": chars}, "messages": msgs}  # fmt: skip


def test_pool_excludes_held_out_and_interleaves():
    train = [ex(i, s, f"{s} question {i}") for s in pool.SETS for i in range(6)]
    train.append(ex(99, "gretel", "Held Out  question", chars=100))
    train.append(ex(98, "gretel", "too long", chars=10**6))
    held = [{"id": "x", "messages": [{"role": "user", "content": "held out question"}]},
            {"id": "sqale-0", "messages": [{"role": "user", "content": "other"}]}]  # fmt: skip
    p = pool.build(train, held, per_set=3, seed=1)
    assert len(p) == 12
    assert [r["origin"] for r in p[:4]] == pool.SETS
    ids = {r["id"] for r in p}
    assert "sqale-0" not in ids and "gretel-99" not in ids and "gretel-98" not in ids
    assert all(r["rl_thinking"] == "high" for r in p)
    assert pool.build(train, held, per_set=3, seed=1) == p  # seeded


def test_step_questions_and_budget():
    cfg = config.pilot_config("p1-grpo")
    rows = [{"id": str(i)} for i in range(64)]
    assert [r["id"] for r in config.step_questions(rows, cfg, 2)] == [str(i) for i in range(32, 64)]
    with pytest.raises(ValueError):
        config.step_questions(rows, cfg, 3)
    for name in config.PILOTS:  # one budget: only the method block and smoke size differ
        c = config.pilot_config(name)
        if not c["smoke"]:
            assert (c["steps"], c["questions_per_step"], c["group_size"]) == (10, 32, 8)


def test_minibatches_keep_groups_whole():
    seqs = [{"qid": q, "g": g} for q in "abcde" for g in range(3)]
    bs = minibatches(seqs, 2, seed=7)
    assert sum(len(b) for b in bs) == 15
    for b in bs:
        for q in {s["qid"] for s in b}:
            assert sum(s["qid"] == q for s in b) == 3


def test_rollout_seeds_are_distinct():
    seeds = {generate.seed_for(s, q, g) + d for s in (1, 2) for q in range(32) for g in range(8)
             for d in (0, 1)}  # fmt: skip
    assert len(seeds) == 2 * 32 * 8 * 2


def test_read_done_skips_errors_and_partial_lines(tmp_path):
    f = tmp_path / "rollouts.jsonl"
    f.write_text(json.dumps({"qid": "a", "g": 0}) + "\n" + json.dumps({"qid": "a", "g": 1,
                 "error": "x"}) + "\n" + '{"qid": "b", "g"', encoding="utf-8")  # fmt: skip
    assert generate.read_done(f) == {("a", 0)}
