import pytest

torch = pytest.importorskip("torch")

from sqlrl.step2_pilots import config, losses  # noqa: E402
from sqlrl.step2_pilots.logprobs import answer_logprobs  # noqa: E402


def test_on_policy_token_gradient_is_policy_gradient():
    # at ratio 1 the clipped objective's gradient is -A * grad(lp) / norm, per kept token
    lp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    mask = torch.tensor([1, 0, 1])
    loss, st = losses.token_objective(lp, lp.detach(), mask, adv=0.5, clip=(0.2, 0.28), norm=4.0)
    loss.backward()
    assert lp.grad.tolist() == pytest.approx([-0.125, 0.0, -0.125])
    assert st["tokens"] == 2 and st["clipped"] == 0


def test_clip_higher_is_asymmetric():
    old = torch.zeros(2)
    lp = torch.log(torch.tensor([1.25, 1.35]))  # ratios 1.25 and 1.35
    mask = torch.ones(2)
    _, st = losses.token_objective(lp, old, mask, adv=1.0, clip=(0.2, 0.28), norm=1.0)
    assert st["clipped"] == 1  # only 1.35 is above 1 + 0.28


def test_kl_k3_is_zero_at_reference_and_positive_elsewhere():
    lp = torch.tensor([-1.0, -2.0])
    assert losses.k3(lp, lp).abs().max() < 1e-7
    assert (losses.k3(lp, lp + 0.3) > 0).all()


def test_sequence_ratio_uses_the_token_mean():
    lp = torch.tensor([-1.0, -1.0], requires_grad=True)
    old = torch.tensor([-1.1, -0.9])  # mean difference 0: s = 1
    loss, st = losses.sequence_objective(lp, old, torch.ones(2), adv=2.0, clip=(3e-4, 4e-4),
                                         norm=2.0)  # fmt: skip
    assert st["ratio_mean"] == pytest.approx(1.0)
    assert float(loss) == pytest.approx(-1.0)


def test_norms():
    cfg = config.pilot_config("p1-grpo")
    seqs = [{"qid": "a", "gen_mask": [1, 1, 0]}, {"qid": "a", "gen_mask": [1]},
            {"qid": "b", "gen_mask": [1, 1]}]  # fmt: skip
    assert losses.norm_for("constant", seqs, cfg) == 2 * 8 * 2048
    assert losses.norm_for("token_mean", seqs, cfg) == 5
    assert losses.norm_for("seq", seqs, cfg) == 3


def test_topk_jsd_zero_for_the_same_model_and_grad_only_to_the_student():
    torch.manual_seed(1)
    head = torch.randn(50, 8)
    hs = torch.randn(5, 8, requires_grad=True)
    same = losses.topk_jsd(hs, hs.detach(), head, k=10)
    assert same.abs().max() < 1e-6
    ht = torch.randn(5, 8)
    d = losses.topk_jsd(hs, ht, head, k=10)
    assert (d > 0).all() and (d <= torch.log(torch.tensor(2.0)) + 1e-6).all()
    d.sum().backward()
    assert hs.grad is not None and hs.grad.abs().sum() > 0


def test_srpo_routing():
    def r(g, reward, qid="q"):
        return {"qid": qid, "g": g, "reward": reward}

    rs = [r(0, 0.0), r(1, 1.0), r(2, -0.1), r(3, None), r(4, 1.0),  # mixed group
          r(0, 0.0, "w"), r(1, 0.0, "w")]  # fmt: skip # all wrong: GRPO fallback
    losses.route(rs)
    assert [x["route"] for x in rs] == ["sdpo", "grpo", "sdpo", None, "grpo", "grpo", "grpo"]
    assert rs[0]["teacher_g"] == 1 and rs[2]["teacher_g"] == 1


def test_answer_logprobs_match_a_full_softmax():
    torch.manual_seed(0)
    hidden, head = torch.randn(6, 4), torch.randn(10, 4)
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    lp, ent = answer_logprobs(hidden, head, ids, start=3, entropy=True)
    full = torch.log_softmax(hidden @ head.T, dim=-1)
    want = torch.stack([full[i - 1, ids[i]] for i in range(3, 6)])
    assert torch.allclose(lp, want, atol=1e-5)
    assert torch.allclose(ent, -(full.exp() * full).sum(-1)[2:5], atol=1e-5)
