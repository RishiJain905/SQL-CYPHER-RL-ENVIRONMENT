"""Rollouts: each question answered `group_size` times through llama-server's raw completion
endpoint (Windows, standard library only, next to the server).

Why the raw endpoint: RL trains on the exact tokens the model sampled, so the server returns the
token ids it generated, and the trainer scores those same ids (no re-tokenizing). The prompt is
the server's own rendering of the chat template (`/apply-template`: the messages, the tools, the
thinking level), tokenized by the server (`/tokenize`).

The thinking budget is the step-1 yardstick's: the model thinks until it writes `</think>` or
`think_budget` tokens; at the budget, `</think>` is appended for it (that one token is not the
model's choice: its loss mask is 0) and the answer follows, up to `max_tokens` in all.

A rollout line: {"qid", "origin", "g", "seed", "prompt_ids", "gen_ids", "gen_mask", "text",
"finish", "think_tokens", "forced", "behavior_lp", "secs"}; `text` is the thought, `</think>`,
then the answer, the form the checker reads.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

THINK_END = "</think>"


def _post(url: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """A llama-server's endpoints, with the template / tokenizer answers cached per question."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._lock = threading.Lock()
        self._prompts: dict[str, list[int]] = {}
        self.think_end_id = self.tokenize(THINK_END)
        if len(self.think_end_id) != 1:
            raise RuntimeError(f"{THINK_END} is not one token: {self.think_end_id}")

    def tokenize(self, text: str) -> list[int]:
        r = _post(self.url + "/tokenize", {"content": text, "add_special": False,
                                           "parse_special": True})  # fmt: skip
        return list(r["tokens"])

    def prompt_ids(self, row: dict[str, Any]) -> list[int]:
        with self._lock:
            if row["id"] in self._prompts:
                return self._prompts[row["id"]]
        level = row.get("rl_thinking") or "high"
        kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
        if level != "off":
            kwargs["reasoning_effort"] = level
        msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in row["messages"][:2]]
        body = {"messages": msgs, "tools": row["tools"], "chat_template_kwargs": kwargs}
        text = _post(self.url + "/apply-template", body)["prompt"]
        ids = self.tokenize(text)
        with self._lock:
            self._prompts[row["id"]] = ids
        return ids

    def complete(self, ids: list[int], n: int, cfg: dict[str, Any], seed: int,
                 stop: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
        body = {"prompt": ids, "n_predict": n, "temperature": cfg["temperature"],
                "top_p": cfg["top_p"], "top_k": 0, "min_p": 0.0, "seed": seed,
                "return_tokens": True, "n_probs": 1, "cache_prompt": True}  # fmt: skip
        if stop:
            body["stop"] = stop
        return _post(self.url + "/completion", body)


def _chosen_logprobs(r: dict[str, Any]) -> list[float] | None:
    """The sampled tokens' log-probabilities as the server computed them (for the
    rollout-vs-trainer mismatch check); None when this build doesn't report them."""
    probs = r.get("completion_probabilities")
    if not probs:
        return None
    out = []
    for p in probs:
        if "logprob" in p:
            out.append(float(p["logprob"]))
        else:
            return None
    return out


def rollout(server: Server, row: dict[str, Any], g: int, seed: int, cfg: dict[str, Any]) -> dict:
    t0 = time.time()
    prompt = server.prompt_ids(row)
    end = server.think_end_id[0]
    r1 = server.complete(prompt, cfg["think_budget"], cfg, seed, stop=[THINK_END])
    think = list(r1.get("tokens") or [])
    lp1 = _chosen_logprobs(r1)
    if lp1 is not None and len(lp1) == len(think) - 1:  # the stop token comes without its
        lp1 = lp1 + [None]  # probability
    forced = not think or think[-1] != end
    mask = [1] * len(think)
    if forced:  # the budget ran out: close the thought for the model, as the yardstick does
        think.append(end)
        mask.append(0)
        lp1 = (lp1 or []) + [None] if lp1 is not None else None
    room = max(cfg["max_tokens"] - len(think), 1)
    r2 = server.complete(prompt + think, room, cfg, seed + 1)
    answer = list(r2.get("tokens") or [])
    lp2 = _chosen_logprobs(r2)
    behavior = lp1 + lp2 if lp1 is not None and lp2 is not None else None
    if behavior is not None and len(behavior) != len(think) + len(answer):
        behavior = None  # the server's lists don't line up with its tokens: don't guess
    return {
        "qid": row["id"],
        "origin": row["origin"],
        "g": g,
        "seed": seed,
        "prompt_ids": prompt,
        "gen_ids": think + answer,
        "gen_mask": mask + [1] * len(answer),
        "text": (r1.get("content") or "") + THINK_END + (r2.get("content") or ""),
        "finish": r2.get("stop_type"),
        "think_tokens": len(think),
        "forced": forced,
        "behavior_lp": behavior,
        "secs": round(time.time() - t0, 2),
    }


def read_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a half-written last line: answered again
                continue
            if "error" not in r:
                done.add((r["qid"], r["g"]))
    return done


def seed_for(step: int, q: int, g: int) -> int:
    """Distinct per step, question and rollout; even, so the answer phase's seed + 1 never
    collides with another rollout's thinking seed."""
    return 2 * (1_000_000 * step + 1_000 * q + g)


def generate(url: str, rows: list[dict[str, Any]], cfg: dict[str, Any], step: int,
             out: Path) -> dict[str, Any]:  # fmt: skip
    """Answer every question `group_size` times into `out` (append; a restart keeps what's
    done, and a failed rollout is retried)."""
    server = Server(url)
    done = read_done(out)
    jobs = [(q, row, g) for q, row in enumerate(rows) for g in range(cfg["group_size"])
            if (row["id"], g) not in done]  # fmt: skip
    lock = threading.Lock()
    errors = 0
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as f:

        def one(job: tuple[int, dict[str, Any], int]) -> None:
            nonlocal errors
            q, row, g = job
            seed = seed_for(step, q, g)
            try:
                res = rollout(server, row, g, seed, cfg)
            except Exception as e:  # retried on the next run
                with lock:
                    errors += 1
                res = {"qid": row["id"], "g": g, "seed": seed,
                       "error": f"{type(e).__name__}: {str(e)[:160]}"}  # fmt: skip
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()

        with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
            list(pool.map(one, jobs))
    return {"rollouts": len(jobs), "errors": errors, "seconds": round(time.time() - t0, 1)}
