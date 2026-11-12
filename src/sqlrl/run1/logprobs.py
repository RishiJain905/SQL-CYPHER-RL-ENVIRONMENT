"""Per-token log-probabilities and entropies of a sampled answer under a causal LM (torch).

The logits are made only at the answer's positions and in chunks (fp32, the tied head), each
chunk checkpointed under grad so one chunk's [chunk, vocab] logits live at a time: the trainer's
`target_ce_sum` pattern, per token instead of summed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK = 512


def _chunk_lp(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, w).float()
    return logits.gather(-1, y[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


@torch.no_grad()
def _chunk_entropy(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(F.linear(h, w).float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def answer_logprobs(
    hidden: torch.Tensor, head: torch.Tensor, ids: torch.Tensor, start: int,
    entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # fmt: skip
    """`hidden` [T, H] the last layer's states for `ids` [T]; returns the log-probabilities [N]
    of `ids[start:]` (each predicted from the position before it) and, with `entropy`, the
    distributions' entropies [N] (no grad)."""
    h = hidden[start - 1 : ids.shape[0] - 1]
    y = ids[start:]
    lps, ents = [], []
    for s in range(0, h.shape[0], CHUNK):
        hs, ys = h[s : s + CHUNK], y[s : s + CHUNK]
        if torch.is_grad_enabled():
            lps.append(checkpoint(_chunk_lp, hs, head, ys, use_reentrant=False))
        else:
            lps.append(_chunk_lp(hs, head, ys))
        if entropy:
            ents.append(_chunk_entropy(hs.detach(), head.detach()))
    return torch.cat(lps), (torch.cat(ents) if entropy else None)


def sequence_logprobs(model, backbone, head, ids: list[int], start: int, entropy: bool = False):
    """One forward pass of `backbone` over `ids`, then `answer_logprobs`."""
    device = head.device
    x = torch.tensor(ids, dtype=torch.long, device=device)
    hidden = backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
    return answer_logprobs(hidden, head, x, start, entropy)
