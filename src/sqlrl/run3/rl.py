"""Run 3, phase 2: the RL loop's logic, no GPU (documentation/run3/phase2.md).

- `RecordingPolicy`: the sampler as a policy that records, for every call, the prompt the server
  rendered, the thought and whether the model closed it itself, the answer part and whether it
  ended on the end-of-sentence token, so the update trains exactly the tokens the model chose.
- `draw_step`: the groups a step plays, by phase2.md's shares, deterministic in (seed, step).
- `advantages`: run 2's SRPO advantage, (reward - group mean) / max(group std, 0.25).
- `examples`: one example a turn (prompt ids, target ids, a mask of the trained tokens).
- `stop_check`: the stop rules against step 1's reading.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlrl.run3 import splits
from sqlrl.run3.policy import THINK_END, LlamaPolicy

EOS = "<｜end▁of▁sentence｜>"
CFG = {"group": 8, "questions": 16, "temperature": 1.0, "think_budget": 1024,
       "max_tokens": 2048, "lr": 5e-6, "warmup": 5, "minibatches": 2, "grad_clip": 1.0,
       "kl_beta": 0.20, "adv_floor": 0.25, "max_seq": 3840, "max_hours": 4.0,
       "shares": {"useful": 0.55, "unmeasured": 0.20, "cypher": 0.15, "nonanswer": 0.10},
       "seed": 0}  # fmt: skip
STOP = {"from_step": 6, "window": 5, "malformed_up": 0.10, "malformed_max": 0.30,
        "false_decline_x": 3.0, "false_decline_min": 0.10, "wrongdb_down": 0.15,
        "kl_max": 0.10, "length_x": 2.0, "hardcoded_max": 0.05, "reward_down": 0.15}  # fmt: skip


def stable(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


# ---- the sampler ------------------------------------------------------------------------------


class RecordingPolicy(LlamaPolicy):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        prompt = self.render(messages, tools)
        r1 = self._complete(prompt, self.think_budget, [THINK_END])
        think = r1.get("content") or ""
        closed = r1.get("stop_type") == "word"  # the model wrote `</think>` itself
        r2 = self._complete(prompt + think + THINK_END, self.max_tokens, None)
        answer = r2.get("content") or ""
        self.calls.append({"prompt": prompt, "think": think, "closed": closed, "answer": answer,
                           "eos": r2.get("stop_type") == "eos",
                           "tokens": int(r1.get("tokens_predicted") or 0)
                           + int(r2.get("tokens_predicted") or 0)})  # fmt: skip
        return think + THINK_END + answer

    def _complete(self, prompt: str, n: int, stop: list[str] | None) -> dict[str, Any]:
        import urllib.request

        body = {"prompt": prompt, "n_predict": n, "temperature": self.temperature,
                "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": self.seed + len(self.calls),
                "cache_prompt": True}  # fmt: skip
        if stop:
            body["stop"] = stop
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        with urllib.request.urlopen(req, timeout=300) as r:  # was 900 s (phase 0)
            return json.loads(r.read())


# ---- the questions a step draws --------------------------------------------------------------


def load_pool(p0: Path, bundle_dbs: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Phase 0's groups by kind of draw, on the databases the bundle carries."""
    groups = {json.loads(x)["gid"]: json.loads(x)
              for x in (p0 / "groups.jsonl").read_text("utf-8").splitlines()}  # fmt: skip
    right: Counter = Counter()
    seen: Counter = Counter()
    for line in (p0 / "episodes.jsonl").read_text("utf-8").splitlines():
        r = json.loads(line)
        seen[r["gid"]] += 1
        right[r["gid"]] += r.get("outcome") == "right"
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gid, g in groups.items():
        useful = 0 < right[gid] < seen[gid]
        if g["source"] == "bird" and g["db"] not in bundle_dbs:
            continue
        if g["kind"] == "answer" and g["source"] == "bird" and useful:
            pool["useful"].append(g)
        elif g["kind"] == "answer" and g["source"] == "cypherbench":
            pool[f"cypher/{g['db']}/{'useful' if useful else 'other'}"].append(g)
        elif g["kind"] != "answer":
            pool["nonanswer"].append(g)
    return pool


def unmeasured(bundle_dbs: list[str], measured: set[str], step: int, n: int,
               seed: int = 0) -> list[dict[str, Any]]:  # fmt: skip
    """BIRD training questions phase 0 never measured, on the bundle's databases, a random
    engine whose gold passed (read lazily from the gold check)."""
    from sqlrl.run3 import engines, pools

    rnd = random.Random(f"run3-rl:unmeasured:{seed}:{step}")
    out: list[dict[str, Any]] = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        db = rnd.choice(bundle_dbs)
        e = rnd.choice(engines.SQL_ENGINES)
        ts = [t for t in pools.answer_tasks("bird", db, e)
              if splits.trainable("bird", db, t["q"]["id"])
              and t["q"]["id"] not in measured]  # fmt: skip
        if ts:
            t = rnd.choice(ts)
            out.append({"stratum": f"new/{e}", "kind": "answer", "source": "bird", "db": db,
                        "qid": t["q"]["id"], "engine": e})  # fmt: skip
    return out


def draw_step(pool: dict[str, list[dict[str, Any]]], step: int, q: int, bundle_dbs: list[str],
              cfg: dict[str, Any] = CFG) -> list[dict[str, Any]]:  # fmt: skip
    """The step's groups. Useful-band BIRD groups go round in a fixed shuffled order (each once
    before any repeats); one CypherBench graph a step in turn, its useful groups first."""
    sh = cfg["shares"]
    n_use = round(q * sh["useful"])
    n_cy = max(1, round(q * sh["cypher"]))
    n_na = max(1, round(q * sh["nonanswer"]))
    n_new = q - n_use - n_cy - n_na
    seed = cfg["seed"]
    use = list(pool["useful"])
    random.Random(f"run3-rl:useful:{seed}").shuffle(use)
    start = (step - 1) * n_use
    picked = [use[(start + i) % len(use)] for i in range(n_use)] if use else []
    graphs = list(splits.CYPHERBENCH_TRAIN_GRAPHS)
    gname = graphs[(step - 1) % len(graphs)]
    cy = list(pool.get(f"cypher/{gname}/useful", [])) + list(pool.get(f"cypher/{gname}/other", []))
    rnd = random.Random(f"run3-rl:step:{seed}:{step}")
    k = len(pool.get(f"cypher/{gname}/useful", []))
    head, tail = cy[:k], cy[k:]
    rnd.shuffle(head)
    rnd.shuffle(tail)
    picked += (head + tail)[:n_cy]
    na = list(pool["nonanswer"])
    rnd.shuffle(na)
    picked += na[:n_na]
    measured = {g.get("qid") for gs in pool.values() for g in gs}
    picked += unmeasured(bundle_dbs, measured, step, n_new, seed)
    for g in picked:
        if g["kind"] == "answer":
            assert splits.trainable(g["source"], g["db"], g["qid"]), g  # never held-out data
    # SQL first, then the graph (Neo4j holds one graph at a time)
    return sorted(picked, key=lambda g: g["engine"] == "neo4j")


# ---- the update's inputs ---------------------------------------------------------------------


def advantages(rewards: list[float], floor: float = 0.25) -> list[float]:
    m = sum(rewards) / len(rewards)
    sd = math.sqrt(sum((r - m) ** 2 for r in rewards) / len(rewards))
    return [(r - m) / max(sd, floor) for r in rewards]


def examples(ep: dict[str, Any], tok: Any, adv: float, max_seq: int) -> tuple[list[dict], int]:
    """One example a turn: ids = prompt + target, `mask` 1 on the tokens the model chose (a
    `</think>` closed for it at the budget and an answer cut at max_tokens without its end
    token are not). Returns (examples, how many were over `max_seq`)."""
    out, over = [], 0
    for c in ep.get("calls") or []:
        p = tok(c["prompt"], add_special_tokens=False)["input_ids"]
        th = tok(c["think"], add_special_tokens=False)["input_ids"] if c["think"] else []
        te = tok(THINK_END, add_special_tokens=False)["input_ids"]
        an = tok(c["answer"], add_special_tokens=False)["input_ids"] if c["answer"] else []
        eo = tok(EOS, add_special_tokens=False)["input_ids"] if c["eos"] else []
        target = th + te + an + eo
        mask = [1] * len(th) + [1 if c["closed"] else 0] * len(te) + [1] * len(an) + [1] * len(eo)
        if len(p) + len(target) > max_seq:
            over += 1
            continue
        if sum(mask):
            out.append({"ids": p + target, "start": len(p), "mask": mask, "adv": adv})
    return out, over


# ---- the step's numbers and the stop rules -----------------------------------------------------


def step_metrics(eps: list[dict[str, Any]]) -> dict[str, Any]:
    skipped = sum(1 for e in eps if e.get("outcome") == "skipped")
    eps = [e for e in eps if e.get("outcome") != "skipped"]  # no task on this VM (R30)
    ans = [e for e in eps if e.get("kind") == "answer"]
    wdb = [e for e in eps if e.get("kind") == "unanswerable"]
    right = [e for e in eps if e.get("outcome") == "right"]

    def share(rs: list[dict[str, Any]], f) -> float | None:
        return round(sum(1 for r in rs if f(r)) / len(rs), 4) if rs else None

    groups: dict[int, list[float]] = defaultdict(list)
    for e in eps:
        groups[e["gid"]].append(e.get("reward") or 0.0)
    informative = sum(1 for rs in groups.values() if len(set(rs)) > 1)
    by: dict[str, Any] = {}
    for key in sorted({e["stratum"].split("/")[0] for e in eps}):
        rs = [e for e in eps if e["stratum"].split("/")[0] == key]
        by[key] = share(rs, lambda r: r.get("outcome") == "right")
    return {
        "episodes": len(eps),
        "reward_mean": round(sum(e.get("reward") or 0.0 for e in eps) / max(len(eps), 1), 4),
        "solve_rate": share(ans, lambda r: r.get("outcome") == "right"),
        "malformed": share(eps, lambda r: r.get("outcome") in ("malformed", "crash")),
        "over_limit": share(eps, lambda r: r.get("ending") == "over_limit"),
        "false_decline": share(ans, lambda r: r.get("ending") == "decline"),
        "wrongdb_decline": share(wdb, lambda r: r.get("ending") == "decline"),
        "hardcoded": share(right, lambda r: str(r.get("why", "")).startswith("hardcoded")),
        "calls_mean": round(sum(e.get("n_calls", 0) for e in eps) / max(len(eps), 1), 3),
        "gen_tokens_mean": round(sum(sum(c["tokens"] for c in e.get("calls") or [])
                                     for e in eps) / max(len(eps), 1), 1),  # fmt: skip
        "informative_groups": informative, "groups": len(groups),
        "crashes": sum(1 for e in eps if e.get("outcome") == "crash"), "skipped": skipped,
        "solve_by_pool": by,
    }  # fmt: skip


def _roll(rows: list[dict[str, Any]], key: str, w: int) -> float | None:
    vals = [r[key] for r in rows[-w:] if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def stop_check(rows: list[dict[str, Any]], stop: dict[str, Any] = STOP) -> list[str]:
    """Reasons to stop, from step `from_step` on: 5-step means against step 1's reading."""
    if len(rows) < stop["from_step"]:
        return []
    s1, w = rows[0], stop["window"]
    out = []

    def m(key: str) -> float | None:
        return _roll(rows, key, w)

    if m("malformed") is not None and s1.get("malformed") is not None and (
            m("malformed") > s1["malformed"] + stop["malformed_up"]
            or m("malformed") > stop["malformed_max"]):  # fmt: skip
        out.append(f"malformed {m('malformed'):.3f} (step 1 {s1['malformed']})")
    fd_bound = max(stop["false_decline_x"] * (s1.get("false_decline") or 0.0),
                   stop["false_decline_min"])  # fmt: skip
    if m("false_decline") is not None and m("false_decline") > fd_bound:
        out.append(f"false declines {m('false_decline'):.3f}")
    if (m("wrongdb_decline") is not None and s1.get("wrongdb_decline") is not None
            and m("wrongdb_decline") < s1["wrongdb_decline"] - stop["wrongdb_down"]):  # fmt: skip
        out.append(f"wrong-database declines {m('wrongdb_decline'):.3f}")
    if m("kl") is not None and m("kl") > stop["kl_max"]:
        out.append(f"KL to the reference {m('kl'):.4f}")
    if (m("gen_tokens_mean") and s1.get("gen_tokens_mean")
            and m("gen_tokens_mean") > stop["length_x"] * s1["gen_tokens_mean"]):  # fmt: skip
        out.append(f"episode length {m('gen_tokens_mean'):.0f} tokens")
    if m("hardcoded") is not None and m("hardcoded") > stop["hardcoded_max"]:
        out.append(f"hard-coded constants {m('hardcoded'):.3f}")
    if m("reward_mean") is not None and m("reward_mean") < s1["reward_mean"] - stop["reward_down"]:
        out.append(f"mean reward {m('reward_mean'):.3f} (step 1 {s1['reward_mean']})")
    return out
