"""Answer every evalset question `samples` times, sampled, through llama-server (Windows,
standard library only, next to the server).

    python -m sqlrl.baseline.sample --run-dir <dir> [--url http://127.0.0.1:8092]

Sample k's answers go to `<run-dir>/preds_s<k>.jsonl`, in the trainer's prediction format
({"id", "output", "finish", "secs", "completion_tokens", "seed"}), so its checker scores each
file as it scores a greedy run. A restart keeps what was answered. The request is the greedy
evaluation's (the trainer's `llamaeval.request_body`: the same messages, tools and thinking
level) with sampling switched on: temperature, top-p, top-k and min-p as set, and a seed per
answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

SETTINGS = {
    "samples": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,  # off
    "min_p": 0.0,  # off (llama.cpp's default is 0.05)
    "max_tokens": 2048,
    "thinking_budget": 1024,  # the server's --reasoning-budget
    "seed_base": 1000,  # sample k, question i: seed_base * (k + 1) + i
}


def _llamaeval():
    """The trainer's request builder and reply parser (one parser for greedy and sampled)."""
    sys.path.insert(0, str(get_paths().trainer_repo / "src"))
    from asktrainer import llamaeval

    return llamaeval


def seed(k: int, i: int) -> int:
    return SETTINGS["seed_base"] * (k + 1) + i


def request_body(llamaeval, ex: dict[str, Any], k: int, i: int) -> dict[str, Any]:
    body = llamaeval.request_body(ex, SETTINGS["max_tokens"])
    body.update(
        temperature=SETTINGS["temperature"],
        top_p=SETTINGS["top_p"],
        top_k=SETTINGS["top_k"],
        min_p=SETTINGS["min_p"],
        seed=seed(k, i),
    )
    return body


def read_preds(path: Path) -> dict[str, dict[str, Any]]:
    """A sample file's answers by question id: a later line replaces an earlier one, and an
    answer the server failed on never replaces a real one (a re-run retries it)."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if "error" not in p or p["id"] not in out:
                    out[p["id"]] = p
    return out


def read_done(path: Path) -> set[str]:
    return {qid for qid, p in read_preds(path).items() if "error" not in p}


def sample(run_dir: Path, url: str, workers: int = 16) -> dict[str, Any]:
    llamaeval = _llamaeval()
    with get_paths().evalset.open(encoding="utf-8") as f:
        exs = [json.loads(line) for line in f if line.strip()]
    n = SETTINGS["samples"]
    files = [run_dir / f"preds_s{k}.jsonl" for k in range(n)]
    done = [read_done(p) for p in files]
    # sample-major: sample 0 finishes first, so the scorer can start on whole files
    todo = [(k, i, ex) for k in range(n) for i, ex in enumerate(exs) if ex["id"] not in done[k]]
    base = url.rstrip("/")
    locks = [threading.Lock() for _ in range(n)]
    handles = [p.open("a", encoding="utf-8", newline="\n") for p in files]
    errors = 0
    t0 = time.time()

    def one(job: tuple[int, int, dict[str, Any]]) -> None:
        nonlocal errors
        k, i, ex = job
        t = time.time()
        try:
            r = llamaeval._post(base + "/v1/chat/completions", request_body(llamaeval, ex, k, i))
            ch = r["choices"][0]
            res = {
                "id": ex["id"],
                "output": llamaeval.as_text(ch.get("message") or {}),
                "finish": ch.get("finish_reason"),
                "secs": round(time.time() - t, 2),
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                "seed": seed(k, i),
            }
        except Exception as e:  # the server: an empty answer scores as no call
            errors += 1
            res = {"id": ex["id"], "output": "", "seed": seed(k, i),
                   "error": f"{type(e).__name__}: {str(e)[:160]}"}  # fmt: skip
        with locks[k]:
            handles[k].write(json.dumps(res, ensure_ascii=False) + "\n")
            handles[k].flush()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
    finally:
        for h in handles:
            h.close()
    return {"answered": len(todo), "errors": errors, "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8092")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    res = sample(Path(a.run_dir), a.url, a.workers)
    print(json.dumps(res))
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
