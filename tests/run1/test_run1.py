"""Run 1's pure parts (CPU): the scheduler, the settings, the batching, the canaries, the
rewards' floor, the copied losses against the pilots', the Q8_0 quantiser, the comparisons."""

from __future__ import annotations

import threading

import pytest
import torch

from sqlrl.run1 import canaries, eval_score, gguf_merge, pool, rewards
from sqlrl.run1 import config as rcfg
from sqlrl.run1 import losses as l1
from sqlrl.run1.rollouts import Plan, group_order, seed_for
from sqlrl.run1.train_step import minibatches, pack
from sqlrl.step2_pilots import losses as l0
from sqlrl.step2_pilots import rewards as rewards0


# ---- the rollout scheduler -------------------------------------------------------------------
def test_plan_keeps_a_worker_on_its_group_then_helps_the_largest():
    plan = Plan([(0, [0, 1, 2]), (1, [0, 1, 2, 3])])
    assert plan.take(None) == (0, 0, False)  # worker A starts group 0
    assert plan.take(None) == (1, 0, False)  # worker B starts group 1
    assert plan.take(0) == (0, 1, False)  # A stays on its group
    assert plan.take(0) == (0, 2, False)
    assert plan.take(0) == (1, 1, True)  # group 0 is done: A helps group 1
    assert plan.take(1) == (1, 2, False)
    assert plan.take(1) == (1, 3, False)
    assert plan.take(1) is None and plan.take(None) is None


def test_plan_hands_out_every_rollout_once_under_threads():
    groups = [(q, list(range(8))) for q in range(32)]
    plan, got, lock = Plan(groups), [], threading.Lock()

    def worker():
        cur = None
        while (job := plan.take(cur)) is not None:
            cur = job[0]
            with lock:
                got.append(job[:2])

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [(q, g) for q in range(32) for g in range(8)]


def test_plan_skips_finished_rollouts_and_empty_groups():
    plan = Plan([(0, []), (1, [5])])
    assert plan.take(None) == (1, 5, False)
    assert plan.take(1) is None


def test_group_order_puts_long_answer_sets_first():
    rows = [{"origin": "gretel", "prompt_tokens": 400}, {"origin": "cypherbench"},
            {"origin": "sqale", "prompt_tokens": 900}, {"origin": "sqale", "prompt_tokens": 1200},
            {"origin": "text2cypher"}]  # fmt: skip
    assert group_order(rows) == [1, 4, 3, 2, 0]


def test_seeds_are_distinct_and_even():
    seeds = {seed_for(s, q, g) for s in (1, 2) for q in range(32) for g in range(8)}
    assert len(seeds) == 2 * 32 * 8 and all(x % 2 == 0 for x in seeds)


# ---- the settings ----------------------------------------------------------------------------
def test_step_questions_follow_the_mix_and_never_repeat():
    p = {s: [{"id": f"{s}-{i}", "origin": s} for i in range(40)] for s in pool.SETS}
    cfg = rcfg.new_config()
    assert cfg["questions_per_step"] == 32
    s1, s2 = rcfg.step_questions(p, cfg, 1), rcfg.step_questions(p, cfg, 2)
    assert [sum(r["origin"] == s for r in s1) for s in cfg["mix"]] == list(cfg["mix"].values())
    assert not {r["id"] for r in s1} & {r["id"] for r in s2}
    with pytest.raises(ValueError):
        rcfg.step_questions(p, cfg, 5)  # 10 a step: 40 CypherBench questions end at step 4


def test_lr_warms_up_then_stays():
    cfg = rcfg.new_config({"lr": 1e-5, "warmup_steps": 5})
    assert [round(rcfg.lr_at(cfg, k) / 1e-5, 2) for k in (1, 3, 5, 6, 100)] == [0.2, 0.6, 1, 1, 1]
    assert rcfg.lr_at(rcfg.new_config({"warmup_steps": 0}), 1) == 1e-5


def test_new_config_adds_the_last_step_to_the_evaluations():
    cfg = rcfg.new_config({"steps": 60})
    assert cfg["eval_steps"] == [25, 50, 60]
    assert rcfg.new_config({"mix": rcfg.MIXES["even"]})["questions_per_step"] == 32


# ---- batching --------------------------------------------------------------------------------
def test_pack_stays_within_the_budget_and_keeps_everything():
    items = [900, 300, 1200, 3500, 800, 1000, 100]
    batches = pack(items, lambda x: x, 3072)
    assert sorted(x for b in batches for x in b) == sorted(items)
    assert all(len(b) == 1 or len(b) * b[0] <= 3072 for b in batches)
    assert batches[0] == [3500]  # longer than the budget: alone
    assert [len(b) for b in pack(items, lambda x: x, 0)] == [1] * len(items)


def test_minibatches_keep_groups_whole():
    seqs = [{"qid": f"q{q}", "g": g} for q in range(5) for g in range(3)]
    mbs = minibatches(seqs, 2, 7)
    assert len(mbs) == 2 and sum(len(b) for b in mbs) == 15
    for b in mbs:
        for q in {s["qid"] for s in b}:
            assert sum(s["qid"] == q for s in b) == 3


# ---- rewards and losses ----------------------------------------------------------------------
def _group(rewards_):
    return [{"qid": "q", "g": i, "reward": r} for i, r in enumerate(rewards_)]


def test_std_floor_keeps_a_malformed_only_group_small_and_mixed_groups_unchanged():
    a = _group([0.0] * 7 + [-0.1])
    rewards.advantages(a, "std", std_floor=0.25)
    assert max(abs(r["adv"]) for r in a) < 0.4  # 2.6 without the floor
    b, c = _group([1.0] + [0.0] * 7), _group([1.0] + [0.0] * 7)
    rewards.advantages(b, "std", std_floor=0.25)
    rewards0.advantages(c, "std")
    assert [r["adv"] for r in b] == pytest.approx([r["adv"] for r in c])
    d = _group([0.0] * 7 + [-0.1])
    rewards0.advantages(d, "std")
    assert max(abs(r["adv"]) for r in d) > 2  # what the pilots' version did


def test_copied_losses_equal_the_pilots():
    torch.manual_seed(0)
    lp = (torch.randn(40) * 0.3 - 1).requires_grad_()
    old, ref = lp.detach() + torch.randn(40) * 0.05, lp.detach() + torch.randn(40) * 0.05
    mask = torch.ones(40)
    mask[7] = 0
    for cfg in ({"adv": "std", "agg": "token_mean", "clip": [0.2, 0.28], "kl_beta": 0.0},
                {"adv": "center", "agg": "constant", "clip": [0.2, 0.2], "kl_beta": 0.01},
                {"adv": "std", "agg": "seq", "clip": [3e-4, 4e-4], "kl_beta": 0.0}):  # fmt: skip
        a, sa = l1.sequence_loss(cfg, lp, old, mask, 0.7, 39.0, ref)
        b, sb = l0.sequence_loss(cfg, lp, old, mask, 0.7, 39.0, ref)
        assert float(a.detach()) == pytest.approx(float(b.detach())) and sa == pytest.approx(sb)
    hs, ht, head = torch.randn(9, 16), torch.randn(9, 16), torch.randn(300, 16)
    assert torch.allclose(l1.topk_jsd(hs, ht, head, 20), l0.topk_jsd(hs, ht, head, 20))
    r1 = _group([1.0, 0.0, None, 0.0])
    r0 = _group([1.0, 0.0, None, 0.0])
    l1.route(r1)
    l0.route(r0)
    assert r1 == r0


# ---- the Q8_0 quantiser ----------------------------------------------------------------------
def test_q8_0_layout_and_roundtrip():
    torch.manual_seed(1)
    w = torch.randn(6, 64) * 0.05
    w[0, :32] = 0.0  # an all-zero block
    raw = gguf_merge.quantize_q8_0(w)
    assert raw.shape == (12, 34) and raw.dtype == torch.uint8
    back = gguf_merge.dequantize_q8_0(raw, (6, 64))
    step = w.reshape(-1, 32).abs().amax(dim=1) / 127
    err = (back - w).reshape(-1, 32).abs().amax(dim=1)
    assert bool((err <= step * 0.6 + 1e-7).all())  # half a step, plus the scale's float16
    assert float(back[0, :32].abs().max()) == 0.0
    q = raw[:, 2:].contiguous().view(torch.int8)
    assert int(q.abs().max()) == 127  # the largest value of a block maps to +-127
    with pytest.raises(ValueError):
        gguf_merge.quantize_q8_0(torch.randn(2, 33))


def test_gguf_names():
    f = gguf_merge.gguf_name
    assert f("base_model.model.model.layers.7.mlp.up_proj") == "blk.7.ffn_up.weight"
    assert f("model.layers.35.self_attn.q_k_v_proj") == "blk.35.attn_qkv.weight"
    assert f("model.layers.0.self_attn.out_proj") == "blk.0.attn_output.weight"
    assert f("model.layers.0.self_attn.g_proj") is None


# ---- canaries and stop rules -----------------------------------------------------------------
def _reply(thought: str, query: str, calls: int = 1) -> dict:
    call = f"<tool_call>run<arg_key>q</arg_key><arg_value>{query}</arg_value></tool_call>"
    return {"text": thought + "</think>" + call * calls, "gen_ids": [0] * 10}


def test_text_canaries_read_queries_and_thoughts():
    rs = [_reply("count rows", "SELECT DISTINCT a FROM t"), _reply("x", "select * from t"),
          _reply("The correct solution uses a join", "SELECT a FROM t", calls=2),
          {"text": "no call here</think>sorry", "gen_ids": [0] * 30}]  # fmt: skip
    c = canaries.text_canaries(rs)
    assert c["distinct_frac"] == pytest.approx(1 / 3)
    assert c["select_star_frac"] == pytest.approx(1 / 3)
    assert c["multi_call_frac"] == 0.25 and c["leak_frac"] == 0.25
    assert c["resp_tokens_median"] == 10


def test_stop_rules():
    rules = rcfg.DEFAULTS["stop"]
    ok = {
        "entropy": 0.45,
        "informative_frac": 0.5,
        "malformed_frac": 0.02,
        "mismatch": 0.03,
        "kl_sft": 0.004,
        "leak_frac": 0.0,
        "resp_tokens_median": 300,
        "distinct_frac": 0.15,
    }
    hist = [dict(ok) for _ in range(8)]
    assert canaries.stop_reasons(hist, rules) == []
    assert canaries.stop_reasons([], rules) == []
    assert "entropy" in canaries.stop_reasons(hist + [{**ok, "entropy": 0.2}], rules)[0]
    assert "KL" in canaries.stop_reasons(hist + [{**ok, "kl_sft": 0.06}], rules)[0]
    assert "DISTINCT" in canaries.stop_reasons(hist + [{**ok, "distinct_frac": 0.3}], rules)[0]
    assert "length" in canaries.stop_reasons(hist + [{**ok, "resp_tokens_median": 700}], rules)[0]
    low = [{**ok, "informative_frac": 0.2}] * 5
    assert "informative" in canaries.stop_reasons(hist + low, rules)[0]
    assert canaries.stop_reasons(hist + low[:4], rules) == []  # four steps: not yet
    bad = [{**ok, "malformed_frac": 0.08}] * 3
    assert "malformed" in canaries.stop_reasons(hist + bad, rules)[0]
    # the relative rules wait for a starting point (five steps)
    assert canaries.stop_reasons([dict(ok)] * 3 + [{**ok, "entropy": 0.1}], rules) == []


# ---- the evaluation's comparison and the pool ------------------------------------------------
def test_compare_counts_each_sign_test():
    base = {
        "questions": [
            {"id": "a", "scored": 8, "right": 0},
            {"id": "b", "scored": 8, "right": 4},
            {"id": "c", "scored": 8, "right": 2},
            {"id": "d", "scored": 0, "right": 0},
        ]
    }
    run = {
        "questions": [
            {"id": "a", "scored": 8, "right": 3},
            {"id": "b", "scored": 8, "right": 0},
            {"id": "c", "scored": 8, "right": 2},
            {"id": "d", "scored": 8, "right": 8},
        ]
    }
    r = eval_score.compare(run, base, {"a": True, "b": False, "c": None}, {"a": False, "b": False})
    assert (r["greedy"]["better"], r["greedy"]["worse"]) == (1, 0)
    assert (r["sampled"]["better"], r["sampled"]["worse"]) == (1, 1)
    assert (r["pass8"]["better"], r["pass8"]["worse"]) == (1, 1)
    assert eval_score.sign_test(0, 0) == 1.0 and eval_score.sign_test(9, 1) == pytest.approx(
        0.0215, abs=1e-3
    )


def _ex(i: str, origin: str, q: str, db: str = "") -> dict:
    return {
        "id": i,
        "origin": origin,
        "task": "call",
        "meta": {"db_id": db, "schema_chars": 10},
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": q}],
    }


def test_candidates_drop_held_out_questions_and_duplicates():
    train = [_ex("t1", "gretel", "How many?"), _ex("t2", "gretel", "how  many?"),
             _ex("t3", "gretel", "Held out?"), _ex("t4", "sqale", "A"), _ex("t5", "other", "B"),
             {**_ex("t6", "sqale", "C"), "meta": {"schema_chars": 99999}}]  # fmt: skip
    held = [_ex("h1", "gretel", "held out?")]
    c = pool.candidates(train, held)
    assert [e["id"] for e in c["gretel"]] in (["t1"], ["t2"])  # the duplicate text kept once
    assert [e["id"] for e in c["sqale"]] == ["t4"] and c["cypherbench"] == []
    assert pool.candidates(train, held) == c  # seeded


def test_reference_call_writes_the_tool_call_text():
    ex = {"meta": {"gold": "SELECT 1"}, "messages": [{}, {}, {"tool_calls": [
        {"function": {"name": "run_sql", "arguments": {"query": "SELECT 1"}}}]}]}  # fmt: skip
    gold, text = pool.reference_call(ex)
    assert gold == "SELECT 1"
    assert text == ("</think><tool_call>run_sql<arg_key>query</arg_key>"
                    "<arg_value>select 1</arg_value></tool_call>")  # fmt: skip


def test_probe_query_changes_the_text_not_the_query():
    assert pool.probe_query("MATCH (n) RETURN n") == "match (n) RETURN n"
    assert pool.probe_query("  select a FROM t") == "  SELECT a FROM t"
    assert pool.probe_query("(SELECT 1)") == "(select 1)"
