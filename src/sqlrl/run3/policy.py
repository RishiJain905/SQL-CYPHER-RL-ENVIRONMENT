"""The model as a policy for `episode.run_episode` (standard library).

`LlamaPolicy` talks to a llama-server (the PC's sampler, as in runs 1 and 2): the chat template
renders the conversation so far (`/apply-template`), the thought is generated up to the budget
(and closed for the model if it runs over, run 2's rule), then the answer part.

This renders each turn's context as text, which is right for running and measuring episodes. The
trainer's rollouts (phase 2) build the context token by token instead (run 2's
`observation_ids`: a tool result is tokenised as plain text and can't forge a turn); this
module is not that path.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

THINK_END = "</think>"


def _post(url: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LlamaPolicy:
    def __init__(self, url: str, temperature: float = 0.0, think_budget: int = 1024,
                 max_tokens: int = 2048, seed: int = 0, level: str = "high"):  # fmt: skip
        self.url = url.rstrip("/")
        self.temperature, self.think_budget, self.max_tokens = temperature, think_budget, max_tokens
        self.seed, self.level = seed, level
        self.calls = 0

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {"enable_thinking": self.level != "off"}
        if self.level != "off":
            kwargs["reasoning_effort"] = self.level
        body = {"messages": messages, "tools": tools, "chat_template_kwargs": kwargs}
        return _post(self.url + "/apply-template", body)["prompt"]

    def _complete(self, prompt: str, n: int, stop: list[str] | None) -> dict[str, Any]:
        body = {"prompt": prompt, "n_predict": n, "temperature": self.temperature,
                "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": self.seed + self.calls,
                "cache_prompt": True}  # fmt: skip
        if stop:
            body["stop"] = stop
        return _post(self.url + "/completion", body)

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        self.calls += 1
        prompt = self.render(messages, tools)
        r1 = self._complete(prompt, self.think_budget, [THINK_END])
        think = r1.get("content") or ""
        r2 = self._complete(prompt + think + THINK_END, self.max_tokens, None)
        return think + THINK_END + (r2.get("content") or "")
