from sqlrl.run3 import rl


class Tok:
    """One id a character; `</think>` and the end-of-sentence token are one id each."""

    special = {rl.THINK_END: [9001], rl.EOS: [9002]}

    def __call__(self, text, add_special_tokens=False):
        if text in self.special:
            return {"input_ids": self.special[text]}
        return {"input_ids": [ord(c) for c in text]}


def test_advantages_floor_and_constant_groups():
    a = rl.advantages([1, 0, 0, 0, 0, 0, 0, 0])
    assert a[0] > 0 and all(x < 0 for x in a[1:])
    assert rl.advantages([1.0] * 8) == [0.0] * 8
    small = rl.advantages([0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(small[0]) <= 0.1 / 0.25 + 1e-9  # the std floor caps tiny differences


def test_examples_mask_forced_close_and_cut_answer():
    call = {"prompt": "P", "think": "ab", "closed": False, "answer": "cd", "eos": False,
            "tokens": 4}  # fmt: skip
    exs, over = rl.examples({"calls": [call]}, Tok(), 0.5, 100)
    e = exs[0]
    assert e["start"] == 1 and e["ids"] == [80, 97, 98, 9001, 99, 100]
    assert e["mask"] == [1, 1, 0, 1, 1]  # forced </think> not trained; no EOS when cut
    call2 = {**call, "closed": True, "eos": True}
    e2 = rl.examples({"calls": [call2]}, Tok(), 0.5, 100)[0][0]
    assert e2["mask"] == [1, 1, 1, 1, 1, 1] and e2["ids"][-1] == 9002
    assert rl.examples({"calls": [call2]}, Tok(), 0.5, 3) == ([], 1)  # too long: counted


def _row(**kw):
    base = {"malformed": 0.12, "false_decline": 0.02, "wrongdb_decline": 0.6, "kl": 0.001,
            "gen_tokens_mean": 800, "hardcoded": 0.0, "reward_mean": 0.4}  # fmt: skip
    return {**base, **kw}


def test_stop_rules_read_against_step_one():
    rows = [_row() for _ in range(5)]
    assert rl.stop_check(rows) == []  # too early
    ok = rows + [_row(malformed=0.15)]
    assert rl.stop_check(ok) == []
    bad = rows + [_row(malformed=0.9)] * 5
    assert any("malformed" in s for s in rl.stop_check(bad))
    kl = rows + [_row(kl=0.5)] * 5
    assert any("KL" in s for s in rl.stop_check(kl))
    fd = rows + [_row(false_decline=0.2)] * 5
    assert any("false declines" in s for s in rl.stop_check(fd))


def test_step_metrics():
    eps = [{"gid": 0, "kind": "answer", "stratum": "useful/sqlite", "outcome": "right",
            "reward": 1.0, "ending": "submit", "n_calls": 2, "calls": [{"tokens": 100}]},
           {"gid": 0, "kind": "answer", "stratum": "useful/sqlite", "outcome": "wrong",
            "reward": 0.0, "ending": "decline", "n_calls": 1, "calls": [{"tokens": 50}]},
           {"gid": 1, "kind": "unanswerable", "stratum": "nonanswer/x", "outcome": "right",
            "reward": 1.0, "ending": "decline", "n_calls": 0,
            "calls": [{"tokens": 30}]}]  # fmt: skip
    m = rl.step_metrics(eps)
    assert m["solve_rate"] == 0.5 and m["false_decline"] == 0.5 and m["wrongdb_decline"] == 1.0
    assert m["informative_groups"] == 1 and m["gen_tokens_mean"] == 60.0
