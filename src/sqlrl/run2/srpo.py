"""SRPO's self-distillation branch in an episode with turns (inside `train_step`; torch, WSL).
Run 1's module, made turn-aware (decisions R15, D6).

- **Routing** (`losses.route`): a wrong episode with a right sibling in its group learns from
  the self-teacher; every other episode takes the reward update.
- **Only the first turn is distilled.** The teacher is shown the question plus the sibling's
  final query as a correct solution, and re-scores the wrong episode's own first-turn tokens.
  That target is clear for turn 1: write the right query the first time. On later turns the
  episode has seen a result its sibling may never have seen, so the teacher's view of those
  tokens would describe a different situation. The later turns of a distilled episode take the
  reward update with the episode's (negative) advantage.
- **The teacher** is the same model with an EMA of the LoRA (rate `ema`), no grad. It starts
  from the start adapter's own EMA state when there is one.
- **The loss** per first-turn token: the JSD over the student's top-100 tokens (+ a tail
  bucket) between student and teacher, weighted by exp(-beta x teacher entropy) normalised to
  mean 1 over the step's distilled tokens.
"""

from __future__ import annotations

from typing import Any

import torch

from sqlrl.run2 import prompt

TEMPLATE = "{question}\n\nCorrect solution:\n{solution}\n\nCorrectly solve the original question."


def render_ids(tok, messages: list[dict[str, Any]], tools, level: str = "high") -> list[int]:
    kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
    if level != "off":
        kwargs["reasoning_effort"] = level
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True, **kwargs)  # fmt: skip
    return list(tok(text, add_special_tokens=False)["input_ids"])


def teacher_prompt(tok, row: dict[str, Any], solution: str, cfg: dict[str, Any]) -> list[int]:
    """The episode's own prompt with the sibling's query given as the solution."""
    msgs = prompt.episode_messages(row, cfg)
    msgs[-1]["content"] = TEMPLATE.format(question=msgs[-1]["content"], solution=solution)
    return render_ids(tok, msgs, row["tools"], row.get("rl_thinking") or "high")


def student_prompt_matches(tok, row: dict[str, Any], prompt_ids: list[int],
                           cfg: dict[str, Any]) -> bool:  # fmt: skip
    """The trainer's rendering of the episode's prompt equals the server's (the teacher's
    prompt is rendered here, the student's came from llama-server)."""
    msgs = prompt.episode_messages(row, cfg)
    return render_ids(tok, msgs, row["tools"], row.get("rl_thinking") or "high") == prompt_ids


def ema_update(ema: dict[str, torch.Tensor], student: dict[str, torch.Tensor], rate: float):
    return {k: (1 - rate) * ema[k] + rate * student[k] for k in ema}
