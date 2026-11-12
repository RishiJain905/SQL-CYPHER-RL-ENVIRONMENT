"""Run 1's rollouts: each question answered `group_size` times through llama-server's raw
completion endpoint (Windows, standard library only, next to the server).

What the model reads and writes is the pilots': the server's own rendering of the chat template
(`/apply-template`), the exact token ids it sampled (`return_tokens`), the thinking budget (the
model thinks until `</think>` or `think_budget` tokens; at the budget `</think>` is appended for
it with loss mask 0, and the answer follows, up to `max_tokens` in all).

What changed is the scheduling (`documentation/run1/harness.md` has the measurements):

- **One worker per slot.** Worker i sends everything to slot i (`id_slot`), so a rollout's two
  requests (the thought, then the answer) always find their own cache, and no request ever
  waits for a slot another rollout took.
- **A question's rollouts run one after another on one slot.** The prompt is computed once and
  reused from the slot's cache for the other seven (the pilots computed it for every rollout).
- **Long questions first, and idle workers help.** Groups are started in the order of the
  datasets' answer lengths; a worker whose queue is empty takes a waiting rollout from the group
  with the most left, so every slot stays busy to the end.
- **Log-probabilities for some rollouts only** (`logprob_rollouts` per group): the server's own
  log-probability of each sampled token costs about a fifth of the decoding speed, and the
  rollout-vs-trainer check needs a sample, not all of them.

A rollout line: {"qid", "origin", "g", "seed", "slot", "prompt_ids", "gen_ids", "gen_mask",
"text", "finish", "think_tokens", "forced", "behavior_lp", "secs"}.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

THINK_END = "</think>"
LONG_FIRST = ["cypherbench", "text2cypher", "sqale", "gretel"]  # mean answer length, pilots


def _post(url: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """A llama-server's endpoints, with the rendered prompts cached per question."""

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

    def complete(
        self,
        ids: list[int],
        n: int,
        cfg: dict[str, Any],
        seed: int,
        slot: int,
        stop: list[str] | None = None,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        body = {"prompt": ids, "n_predict": n, "temperature": cfg["temperature"],
                "top_p": cfg["top_p"], "top_k": 0, "min_p": 0.0, "seed": seed,
                "return_tokens": True, "cache_prompt": True, "id_slot": slot}  # fmt: skip
        if logprobs:
            body["n_probs"] = 1
        if stop:
            body["stop"] = stop
        return _post(self.url + "/completion", body)


def _chosen_logprobs(r: dict[str, Any]) -> list[float] | None:
    """The sampled tokens' log-probabilities as the server computed them; None when the
    request didn't ask for them or the build doesn't report them."""
    probs = r.get("completion_probabilities")
    if not probs:
        return None
    out = []
    for p in probs:
        if "logprob" not in p:
            return None
        out.append(float(p["logprob"]))
    return out


def rollout(server: Server, row: dict[str, Any], g: int, seed: int, cfg: dict[str, Any],
            slot: int, logprobs: bool = False) -> dict[str, Any]:  # fmt: skip
    t0 = time.time()
    prompt = server.prompt_ids(row)
    end = server.think_end_id[0]
    r1 = server.complete(prompt, cfg["think_budget"], cfg, seed, slot, [THINK_END], logprobs)
    think = list(r1.get("tokens") or [])
    lp1 = _chosen_logprobs(r1)
    if lp1 is not None and len(lp1) == len(think) - 1:  # the stop token comes without its
        lp1 = lp1 + [None]  # probability
    forced = not think or think[-1] != end
    mask = [1] * len(think)
    if forced:  # the budget ran out: close the thought for the model, as the yardstick does
        think.append(end)
        mask.append(0)
        lp1 = lp1 + [None] if lp1 is not None else None
    room = max(cfg["max_tokens"] - len(think), 1)
    # the same slot: its cache holds the prompt and the thought (no second prefill)
    r2 = server.complete(prompt + think, room, cfg, seed + 1, slot, None, logprobs)
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
        "slot": slot,
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


class Plan:
    """Which rollout a worker does next. A worker stays on its group (the prompt is in its
    slot's cache); with the group done it starts the next unstarted group; with none left it
    takes a rollout from the group with the most still waiting."""

    def __init__(self, groups: list[tuple[int, list[int]]]):
        self._lock = threading.Lock()
        self._pending = {q: list(gs) for q, gs in groups if gs}
        self._unstarted = [q for q, gs in groups if gs]

    def take(self, current: int | None) -> tuple[int, int, bool] | None:
        """(question index, rollout index, whether it was taken from another worker's group)."""
        with self._lock:
            stolen = False
            if current is not None and current in self._pending:
                q = current
            elif self._unstarted:
                q = self._unstarted.pop(0)
            elif self._pending:
                q = max(self._pending, key=lambda k: len(self._pending[k]))
                stolen = True
            else:
                return None
            g = self._pending[q].pop(0)
            if not self._pending[q]:
                del self._pending[q]
            if q in self._unstarted:
                self._unstarted.remove(q)
            return q, g, stolen


def group_order(rows: list[dict[str, Any]]) -> list[int]:
    """Question indices, the datasets with the longest answers first (longer prompts first
    within a dataset), so the short groups fill the end of the step."""
    rank = {s: i for i, s in enumerate(LONG_FIRST)}
    return sorted(
        range(len(rows)),
        key=lambda q: (
            rank.get(rows[q]["origin"], len(rank)),
            -int(rows[q].get("prompt_tokens") or 0),
            q,
        ),
    )


def generate(url: str, rows: list[dict[str, Any]], cfg: dict[str, Any], step: int,
             out: Path) -> dict[str, Any]:  # fmt: skip
    """Answer every question `group_size` times into `out` (append; a restart keeps what's
    done, and a failed rollout is retried on the next run)."""
    server = Server(url)
    if out.exists():  # a crash mid-write leaves a last line without its newline: cut it
        data = out.read_bytes()
        if data and not data.endswith(b"\n"):
            out.write_bytes(data[: data.rfind(b"\n") + 1])
    done = read_done(out)
    groups = [(q, [g for g in range(cfg["group_size"]) if (rows[q]["id"], g) not in done])
              for q in group_order(rows)]  # fmt: skip
    plan = Plan(groups)
    n_jobs = sum(len(gs) for _, gs in groups)
    lock = threading.Lock()
    tot = {"errors": 0, "gen_tokens": 0, "stolen": 0, "done": 0}
    n_lp = int(cfg.get("logprob_rollouts", cfg["group_size"]))
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as f:

        def worker(slot: int) -> None:
            current: int | None = None
            fails = 0
            while fails < 3:  # a slot that keeps failing stops; the others finish its work
                job = plan.take(current)
                if job is None:
                    return
                q, g, stolen = job
                current = q
                row, seed = rows[q], seed_for(step, q, g)
                try:
                    res = rollout(server, row, g, seed, cfg, slot, logprobs=g < n_lp)
                    fails = 0
                except Exception as e:  # retried on the next run
                    fails += 1
                    res = {"qid": row["id"], "g": g, "seed": seed,
                           "error": f"{type(e).__name__}: {str(e)[:160]}"}  # fmt: skip
                with lock:
                    f.write(json.dumps(res) + "\n")
                    f.flush()
                    tot["done"] += 1
                    tot["stolen"] += stolen
                    if "error" in res:
                        tot["errors"] += 1
                    else:
                        tot["gen_tokens"] += len(res["gen_ids"])

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(cfg["slots"])]  # fmt: skip
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    secs = time.time() - t0
    return {"rollouts": n_jobs, "attempted": tot["done"], "errors": tot["errors"],
            "seconds": round(secs, 1), "gen_tokens": tot["gen_tokens"], "taken_over": tot["stolen"],
            "tok_s": round(tot["gen_tokens"] / secs, 1) if secs > 0 else None}  # fmt: skip
