"""The methods' policy losses, per sequence (torch; CPU-testable). Run 1's copy of the pilots'
module (tests prove the two equal).

Inputs for one rollout: `lp` the policy's log-probabilities of its sampled answer tokens (with
grad), `old` the same under the policy that sampled this batch (no grad), `ref` under the SFT
model (adapter off; only when `kl_beta` > 0), `mask` 1 for the model's own tokens (0 for the
forced `</think>`), `adv` its advantage. The functions return this sequence's contribution to the
minibatch loss and its statistics; `norm` turns the per-token sum into the method's average:

- GRPO in Dr. GRPO's form (Liu et al., 2025, "Understanding R1-Zero-Like Training"): the clipped
  token ratio objective with the advantage r - mean and the sum over tokens divided by a
  constant (G x max_tokens), so long and short answers aren't weighted differently; plus
  `kl_beta` x the k3 estimate of KL(policy || SFT) per token (Schulman's k3: e^(ref-lp) -
  (ref-lp) - 1).
- DAPO (Yu et al., 2025): the same token objective with an asymmetric clip ("clip-higher",
  [1 - 0.2, 1 + 0.28]), the sum divided by the number of tokens in the minibatch
  ("token-level"), no KL; groups with no signal are dropped before the loss (dynamic sampling).
- GSPO (Zheng et al., 2025): one ratio per sequence, exp(mean over tokens of lp - old), clipped
  to [1 - 3e-4, 1 + 4e-4], times the advantage; averaged over sequences.
"""

from __future__ import annotations

from typing import Any

import torch


def k3(lp: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    d = ref - lp
    return torch.exp(d) - d - 1


def token_objective(
    lp: torch.Tensor,
    old: torch.Tensor,
    mask: torch.Tensor,
    adv: float,
    clip: tuple[float, float],
    norm: float,
    ref: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO / DAPO: sum over tokens of -min(ratio * A, clip(ratio) * A) (+ beta KL), / norm."""
    m = mask.to(lp.dtype)
    ratio = torch.exp(lp - old)
    a = torch.as_tensor(adv, dtype=lp.dtype, device=lp.device)
    unclipped = ratio * a
    clipped = torch.clamp(ratio, 1 - clip[0], 1 + clip[1]) * a
    per_tok = -torch.minimum(unclipped, clipped)
    kl = None
    if ref is not None and kl_beta > 0:
        kl = k3(lp, ref)
        per_tok = per_tok + kl_beta * kl
    loss = (per_tok * m).sum() / norm
    n = m.sum().clamp(min=1)
    with torch.no_grad():
        was_clipped = ((clipped < unclipped) & (m > 0)).to(lp.dtype).sum()
        stats = {
            "tokens": float(m.sum()),
            "clipped": float(was_clipped),
            "kl_sum": float((kl * m).sum()) if kl is not None else 0.0,
            "ratio_mean": float((ratio * m).sum() / n),
        }
    return loss, stats


def sequence_objective(
    lp: torch.Tensor, old: torch.Tensor, mask: torch.Tensor, adv: float,
    clip: tuple[float, float], norm: float,
) -> tuple[torch.Tensor, dict[str, float]]:  # fmt: skip
    """GSPO: -min(s * A, clip(s) * A) / norm with s = exp(mean_t (lp - old))."""
    m = mask.to(lp.dtype)
    n = m.sum().clamp(min=1)
    s = torch.exp(((lp - old) * m).sum() / n)
    a = torch.as_tensor(adv, dtype=lp.dtype, device=lp.device)
    unclipped = s * a
    clipped = torch.clamp(s, 1 - clip[0], 1 + clip[1]) * a
    loss = -torch.minimum(unclipped, clipped) / norm
    with torch.no_grad():
        stats = {"tokens": float(m.sum()), "clipped": float(clipped < unclipped) * float(m.sum()),
                 "kl_sum": 0.0, "ratio_mean": float(s)}  # fmt: skip
    return loss, stats


def _jsd_chunk(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    """Per-token JSD between the student (from `hs`, with grad) and the teacher (from `ht`, no
    grad) over the student's top-k tokens plus one bucket for the rest of the vocabulary."""
    ls = torch.log_softmax(torch.nn.functional.linear(hs, w).float(), dim=-1)
    with torch.no_grad():
        lt = torch.log_softmax(torch.nn.functional.linear(ht, w).float(), dim=-1)
        idx = ls.detach().topk(k, dim=-1).indices
    ps = ls.gather(-1, idx).exp()
    pt = lt.gather(-1, idx).exp()
    ps = torch.cat([ps, (1 - ps.sum(-1, keepdim=True)).clamp(min=1e-12)], dim=-1)
    pt = torch.cat([pt, (1 - pt.sum(-1, keepdim=True)).clamp(min=1e-12)], dim=-1)
    m = 0.5 * (ps + pt)

    def kl(a, b):
        return (a * (a.clamp(min=1e-12).log() - b.clamp(min=1e-12).log())).sum(-1)

    return 0.5 * kl(ps, m) + 0.5 * kl(pt, m)


def topk_jsd(
    student_h: torch.Tensor, teacher_h: torch.Tensor, head: torch.Tensor, k: int = 100
) -> torch.Tensor:
    """SDPO's distillation term per answer token: the JSD between the student's next-token
    distribution and the self-teacher's on the same token prefix (Hübotter et al., 2026; SRPO's
    configuration: top-k 100, Jensen-Shannon). Chunked; under grad each chunk is checkpointed."""
    from torch.utils.checkpoint import checkpoint

    out = []
    for s in range(0, student_h.shape[0], 256):
        hs, ht = student_h[s : s + 256], teacher_h[s : s + 256]
        if torch.is_grad_enabled():
            out.append(checkpoint(_jsd_chunk, hs, ht, head, k, use_reentrant=False))
        else:
            out.append(_jsd_chunk(hs, ht, head, k))
    return torch.cat(out)


def _kl_chunk(hs: torch.Tensor, hr: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    """Per-token KL(student || reference) over the student's top-k tokens plus one bucket for
    the rest of the vocabulary (a lower bound of the full KL; exact when k is the vocabulary)."""
    ls = torch.log_softmax(torch.nn.functional.linear(hs, w).float(), dim=-1)
    with torch.no_grad():
        lr = torch.log_softmax(torch.nn.functional.linear(hr, w).float(), dim=-1)
        idx = ls.detach().topk(k, dim=-1).indices
    ps = ls.gather(-1, idx).exp()
    pr = lr.gather(-1, idx).exp()
    ps = torch.cat([ps, (1 - ps.sum(-1, keepdim=True)).clamp(min=1e-12)], dim=-1)
    pr = torch.cat([pr, (1 - pr.sum(-1, keepdim=True)).clamp(min=1e-12)], dim=-1)
    return (ps * (ps.clamp(min=1e-12).log() - pr.clamp(min=1e-12).log())).sum(-1)


def topk_kl(
    student_h: torch.Tensor, ref_h: torch.Tensor, head: torch.Tensor, k: int = 100
) -> torch.Tensor:
    """The KL penalty's term per answer token (run 1b, decisions R13): KL(policy || SFT model)
    between the two next-token distributions on the same token prefix. `ref_h` is the SFT
    model's states (adapter off, no grad). Unlike the k3 estimate in `token_objective`, which
    reads one sampled token, this compares the whole distributions, so it has no sampling noise
    and its gradient always points back toward the SFT model. Chunked like `topk_jsd`."""
    from torch.utils.checkpoint import checkpoint

    out = []
    for s in range(0, student_h.shape[0], 256):
        hs, hr = student_h[s : s + 256], ref_h[s : s + 256]
        if torch.is_grad_enabled():
            out.append(checkpoint(_kl_chunk, hs, hr, head, k, use_reentrant=False))
        else:
            out.append(_kl_chunk(hs, hr, head, k))
    return torch.cat(out)


def route(rollouts: list[dict[str, Any]]) -> None:
    """SRPO's routing (Sample-Routed Policy Optimization, arXiv 2604.02288), in place: a wrong
    rollout (reward < 0.5, decided) with at least one right sibling in its group goes to the
    self-distillation branch (`route` "sdpo", `teacher_qid_g` the sibling it learns from);
    everything else to GRPO ("grpo"). Undecided rollouts take no part (`route` None)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rollouts:
        groups.setdefault(r["qid"], []).append(r)
    for rs in groups.values():
        right = sorted((r for r in rs if r["reward"] is not None and r["reward"] >= 0.5),
                       key=lambda r: r["g"])  # fmt: skip
        for r in rs:
            if r["reward"] is None:
                r["route"] = None
            elif r["reward"] < 0.5 and right:
                r["route"] = "sdpo"
                r["teacher_g"] = right[0]["g"]
            else:
                r["route"] = "grpo"


def norm_for(agg: str, seqs: list[dict[str, Any]], cfg: dict[str, Any]) -> float:
    """The minibatch's divisor: "constant" G * max_tokens per group in it (Dr. GRPO);
    "token_mean" its trainable tokens (DAPO); "seq" its sequences (GSPO)."""
    if agg == "constant":
        groups = len({s["qid"] for s in seqs})
        return float(max(groups, 1) * cfg["group_size"] * cfg["max_tokens"])
    if agg == "token_mean":
        return float(max(sum(sum(s["gen_mask"]) for s in seqs), 1))
    if agg == "seq":
        return float(max(len(seqs), 1))
    raise ValueError(f"unknown aggregation {agg!r}")


def sequence_loss(
    method_cfg: dict[str, Any], lp, old, mask, adv: float, norm: float, ref=None
) -> tuple[torch.Tensor, dict[str, float]]:
    clip = tuple(method_cfg["clip"])
    if method_cfg["agg"] == "seq":
        return sequence_objective(lp, old, mask, adv, clip, norm)
    return token_objective(lp, old, mask, adv, clip, norm, ref, method_cfg.get("kl_beta", 0.0))
