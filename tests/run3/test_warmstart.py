import json

from sqlrl.run3 import warmstart


def _rec(qid, outcome="right", final="SELECT 1", db="movie", turns=2, kind="answer"):
    return {"qid": qid, "engine": "duckdb", "stratum": "bird/duckdb", "kind": kind,
            "source": "bird", "db": db, "outcome": outcome, "ending": "submit",
            "final_query": final,
            "turns": [{"text": f"<think>t{i}</think>call{i}"} for i in range(turns)],
            "rendered": [f"PROMPT{i}" for i in range(turns)]}  # fmt: skip


def _write(tmp_path, recs):
    f = tmp_path / "episodes.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return f


def _train_qid():
    from sqlrl.run3 import splits

    return next(f"bird-{i}" for i in range(100) if splits.trainable("bird", "movie", f"bird-{i}"))


def test_collect_keeps_right_train_episodes_one_example_a_turn(tmp_path):
    q = _train_qid()
    recs = [_rec(q), _rec(q, outcome="wrong"), _rec("bird-x", db="airline")]  # held-out db
    exs, stats = warmstart.collect([_write(tmp_path, recs)])
    assert len(exs) == 2 and {e["turn"] for e in exs} == {0, 1}
    assert exs[0]["prompt"] == "PROMPT0" and exs[1]["target"] == "<think>t1</think>call1"
    assert stats["questions"] == 1


def test_collect_at_most_two_different_episodes_a_question(tmp_path):
    q = _train_qid()
    recs = [_rec(q, final="A"), _rec(q, final="A"), _rec(q, final="B"), _rec(q, final="C")]
    exs, _ = warmstart.collect([_write(tmp_path, recs)], per_question=2)
    assert len(exs) == 4  # two episodes (A and B) of two turns


def test_tokenize_closes_the_target_with_eos():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    ids, n_p = warmstart.tokenize(Tok(), {"prompt": "ab", "target": "c"})
    assert n_p == 2 and ids[:3] == [97, 98, 99] and len(ids) == 3 + len(warmstart.EOS)


def test_collect_max_calls_and_teacher_share(tmp_path):
    q = _train_qid()
    own = _rec(q, final="A")
    own["n_calls"] = 2
    long = _rec(q, final="B")
    long["n_calls"] = 8
    t1, t2 = _rec(q, final="C"), _rec(q, final="D")
    for t in (t1, t2):
        t["teacher"], t["n_calls"] = True, 3
    exs, stats = warmstart.collect([_write(tmp_path, [own, long, t1, t2])], per_question=4,
                                   max_calls=6)  # fmt: skip
    assert stats["dropped_over_max_calls"] == 1
    assert stats["episodes/own"] == 1 and stats["episodes/teacher"] == 1  # capped at own
