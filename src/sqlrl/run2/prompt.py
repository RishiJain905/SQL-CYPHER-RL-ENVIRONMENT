"""The system prompt of a multi-turn episode: the SFT prompt with its "one query" rule changed
(standard library only; the rollouts, the trainer and the evaluation all render through here, so
the model reads one wording everywhere).

The SFT prompt (the trainer's `data/tool_prompt.py`) says "by calling the `tool` tool with ONE
read-only <language> query". Run 2 lets the model see the result and try again, so that line and
one rule change. Nothing else in the prompt is touched: the schema, the examples and the other
rules are the SFT model's own.

Variants (the wording is chosen on the dev slice of the pool, never on the evaluation set):

- `sft`: unchanged. The model is told "ONE query" and gets results anyway.
- `turns`: the first line says "one query at a time", and a rule says what a result is for.
- `check`: `turns`, plus a hint to compare the result with the question before stopping.

The scorer never reads this text: it gets the original example (the checker reads the schema
from the original system message).
"""

from __future__ import annotations

import re
from typing import Any

VARIANTS = ("sft", "turns", "check")

_ONE = re.compile(r"tool with ONE read-only (?P<lang>[^\n]+?) query\.")
_LAST_RULE = "- If the database can't answer the question, don't call the tool:"
TURN_RULE = (
    "- After each call you get its result: the rows, or an error. If the result shows the query "
    "is wrong, call the tool again with a corrected query; you have {n} calls in all. When the "
    "result answers the question, stop: reply with one short sentence and no tool call. The last "
    "query you ran is your answer."
)
CHECK_HINT = (
    " Before you stop, check the result against the question: the columns and the rows it asks "
    "for, no more and no fewer."
)


def system_prompt(system: str, variant: str, max_queries: int) -> str:
    """The episode's system message from the SFT one. Raises when the SFT wording isn't found:
    a prompt that silently stays single-turn would train the wrong protocol."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    if variant == "sft":
        return system
    if len(_ONE.findall(system)) != 1 or system.count(_LAST_RULE) != 1:
        raise ValueError("the system message isn't the SFT tool prompt: no rule to rewrite")
    out = _ONE.sub(lambda m: f"tool with one read-only {m['lang']} query at a time.", system)
    rule = TURN_RULE.format(n=max_queries) + (CHECK_HINT if variant == "check" else "")
    return out.replace(_LAST_RULE, rule + "\n" + _LAST_RULE)


def protocol(cfg: dict[str, Any], origin: str) -> tuple[str, int]:
    """(prompt variant, calls allowed) for a question of dataset `origin`. A dataset listed in
    `single_turn_sets` keeps run 1's protocol: the original prompt, one query, no result shown
    (decisions R5's other branch for a random-fill dataset, should turns hurt it)."""
    if origin in (cfg.get("single_turn_sets") or []):
        return "sft", 1
    return cfg.get("prompt_variant", "turns"), int(cfg["max_queries"])


def episode_messages(row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """The two messages an episode starts from (system, user), with the rule rewritten."""
    variant, max_queries = protocol(cfg, row["origin"])
    system, user = row["messages"][0], row["messages"][1]
    if system["role"] != "system" or user["role"] != "user":
        raise ValueError(f"{row.get('id')}: expected a system and a user message")
    return [
        {
            "role": "system",
            "content": system_prompt(system.get("content") or "", variant, max_queries),
        },  # fmt: skip
        {"role": "user", "content": user.get("content") or ""},
    ]
