"""SRPO's self-distillation branch (Sample-Routed Policy Optimization, arXiv 2604.02288, built
on SDPO, Hübotter et al., arXiv 2601.20802), inside `train_step` (torch, WSL).

- **Routing** (`losses.route`): a wrong rollout with a right sibling in its group learns from
  the self-teacher; every other rollout takes the GRPO branch.
- **The teacher** is the same model with an EMA of the LoRA (rate `ema`, 0.05 in SRPO), no
  grad, shown the question plus the right sibling's query as a correct solution (SRPO uses the
  sibling and no environment feedback). It doesn't generate: it re-scores the wrong rollout's
  own tokens, so the update stays on the rollout the policy sampled.
- **The loss** per answer token: the JSD over the student's top-100 tokens (+ a tail bucket)
  between the student and the teacher, weighted by exp(-beta * teacher entropy) normalised to
  mean 1 over the step's distilled tokens (SRPO's dynamic weighting: confident teacher tokens
  count more). It joins GRPO's token terms in one token-level average.
"""

from __future__ import annotations

import re
from typing import Any

import torch

ARG = re.compile(r"<arg_value>(.*?)</arg_value>", re.S)
TEMPLATE = "{question}\n\nCorrect solution:\n{solution}\n\nCorrectly solve the original question."


def solution_of(text: str) -> str:
    """The query a rollout called the tool with (its last argument value)."""
    found = ARG.findall(text.split("</think>", 1)[-1])
    return found[-1].strip() if found else ""


def render_ids(tok, messages: list[dict[str, Any]], tools, level: str = "high") -> list[int]:
    kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
    if level != "off":
        kwargs["reasoning_effort"] = level
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True, **kwargs)  # fmt: skip
    return list(tok(text, add_special_tokens=False)["input_ids"])


def teacher_prompt(tok, row: dict[str, Any], solution: str) -> list[int]:
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in row["messages"][:2]]
    msgs[-1]["content"] = TEMPLATE.format(question=msgs[-1]["content"], solution=solution)
    return render_ids(tok, msgs, row["tools"], row.get("rl_thinking") or "high")


def student_prompt_matches(tok, row: dict[str, Any], prompt_ids: list[int]) -> bool:
    """The trainer's rendering of the question equals the server's (the teacher's prompt is
    rendered here, the student's came from llama-server)."""
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in row["messages"][:2]]
    return render_ids(tok, msgs, row["tools"], row.get("rl_thinking") or "high") == prompt_ids


@torch.no_grad()
def prepare_teacher(
    model, backbone, head, sdpo: list[dict], by_qid_g: dict, rows: dict, tok,
    ema_state: dict, student_state: dict, beta: float,
) -> dict[str, Any]:  # fmt: skip
    """Teacher hidden states and entropies for every distilled rollout (kept on the CPU), with
    the EMA LoRA swapped in and the student's put back after."""
    from asktrainer.checkpoint import load_trainable

    from sqlrl.run1.logprobs import _chunk_entropy

    load_trainable(model, ema_state)
    try:
        weights_sum, n = 0.0, 0
        for r in sdpo:
            sib = by_qid_g[(r["qid"], r["teacher_g"])]
            ids = teacher_prompt(tok, rows[r["qid"]], solution_of(sib["text"])) + r["gen_ids"]
            start = len(ids) - len(r["gen_ids"])
            x = torch.tensor(ids, device=head.device)
            h = backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
            ht = h[start - 1 : len(ids) - 1]
            ent = torch.cat([_chunk_entropy(ht[s : s + 512], head)
                             for s in range(0, ht.shape[0], 512)])  # fmt: skip
            w = torch.exp(-beta * ent) * torch.tensor(r["gen_mask"], device=ent.device).float()
            r["_t_h"], r["_t_w"] = ht.cpu(), w.cpu()
            weights_sum += float(w.sum())
            n += int(sum(r["gen_mask"]))
    finally:
        load_trainable(model, student_state)
    mean_w = weights_sum / n if n else 1.0
    for r in sdpo:  # normalise to mean 1 over the step's distilled tokens
        r["_t_w"] = r["_t_w"] / max(mean_w, 1e-12)
    return {"sdpo_rollouts": len(sdpo), "sdpo_tokens": n, "teacher_weight_mean": mean_w}


def distill_loss(backbone, head, r: dict[str, Any], top_k: int, norm: float):
    """The distilled rollout's loss: weighted per-token JSD, summed, / the minibatch norm."""
    from sqlrl.run1.losses import topk_jsd

    ids = r["prompt_ids"] + r["gen_ids"]
    start = len(r["prompt_ids"])
    x = torch.tensor(ids, device=head.device)
    h = backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
    hs = h[start - 1 : len(ids) - 1]
    jsd = topk_jsd(hs, r.pop("_t_h").to(hs.device, hs.dtype), head, top_k)
    w = r.pop("_t_w").to(jsd.device)
    loss = (jsd * w).sum() / norm
    m = torch.tensor(r["gen_mask"], device=jsd.device).float()
    return loss, {"tokens": float(m.sum()), "jsd_sum": float((jsd.detach() * m).sum())}


def ema_update(ema: dict[str, torch.Tensor], student: dict[str, torch.Tensor], rate: float):
    return {k: (1 - rate) * ema[k] + rate * student[k] for k in ema}
