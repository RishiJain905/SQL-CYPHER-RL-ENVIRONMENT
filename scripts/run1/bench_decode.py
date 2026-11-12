"""How fast does llama-server decode with every slot busy? (Windows, GPU, ~1 minute a server
setting.)

    C:\\Python314\\python.exe -I scripts\\run1\\bench_decode.py --server base --slots 16

Each slot gets a real prompt (the pilots' rollouts), cached first; then every slot generates
`--tokens` tokens at once (`ignore_eos`, so each makes exactly that many), once per request
variant. The number is generated tokens a second over all slots, by the wall clock.

Request variants (the same sampling distribution: temperature 1, nothing cut):
- `pilot`: the pilots' request (the default sampler chain, `n_probs` 1);
- `noprobs`: the same without `n_probs`;
- `post`: `n_probs` 1 with `post_sampling_probs` (the sampler's own probabilities);
- `lean`: only the temperature sampler in the chain, no `n_probs`;
- `lean+post`: `lean` with `post`;
- `backend`: sampling on the GPU (`backend_sampling`), no `n_probs`;
- `backend+post`: `backend` with `post`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.paths import get_paths  # noqa: E402
from sqlrl.run1 import server  # noqa: E402

VARIANTS = {
    "pilot": {"n_probs": 1},
    "noprobs": {},
    "post": {"n_probs": 1, "post_sampling_probs": True},
    "lean": {"samplers": ["temperature"]},
    "lean+post": {"samplers": ["temperature"], "n_probs": 1, "post_sampling_probs": True},
    "backend": {"backend_sampling": True},
    "backend+post": {
        "backend_sampling": True,
        "samplers": ["temperature"],
        "n_probs": 1,
        "post_sampling_probs": True,
    },  # fmt: skip
}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def prompts(n: int) -> list[list[int]]:
    path = get_paths().data / "pilots" / "p2-srpo" / "step-0010" / "rollouts.jsonl"
    seen, out = set(), []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] not in seen:
                seen.add(r["qid"])
                out.append(r["prompt_ids"])
    return [out[i % len(out)] for i in range(n)]


def run_variant(url: str, ids: list[list[int]], extra: dict, tokens: int, seed: int) -> dict:
    def one(i: int) -> dict:
        body = {"prompt": ids[i], "n_predict": tokens, "temperature": 1.0, "top_p": 1.0,
                "top_k": 0, "min_p": 0.0, "seed": seed + i, "return_tokens": True,
                "cache_prompt": True, "id_slot": i, "ignore_eos": True, **extra}  # fmt: skip
        return post(url + "/completion", body)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        res = list(pool.map(one, range(len(ids))))
    wall = time.time() - t0
    n = sum(len(r.get("tokens") or []) for r in res)
    probs = res[0].get("completion_probabilities") or []
    return {"tokens": n, "seconds": round(wall, 2), "tok_s": round(n / wall, 1),
            "prob_keys": sorted(probs[0].keys()) if probs else None,
            "first_tokens": (res[0].get("tokens") or [])[:6]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="base", help="a label for the server setting")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--model", default=None, help="a GGUF (default: the merged SFT Q8_0)")
    ap.add_argument("--lora", default=None, help="an adapter GGUF to load with --lora")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--extra", default="", help="more server flags, space separated")
    ap.add_argument("--ctx", type=int, default=65536, help="the KV size in tokens, all slots")
    ap.add_argument("--kv-unified", action="store_true", help="one shared KV pool (-kvu)")
    ap.add_argument("--out", default=None, help="append the results to this JSONL file")
    a = ap.parse_args(argv)
    p = get_paths()
    log_dir = p.data / "run1" / "bench" / a.server
    model = Path(a.model) if a.model else p.model_gguf
    info = server.start(model, log_dir, slots=a.slots, threads=a.threads, ctx=a.ctx,
                        kv_unified=a.kv_unified, lora=Path(a.lora) if a.lora else None,
                        extra=a.extra.split())  # fmt: skip
    url = f"http://127.0.0.1:{server.PORT}"
    rows = []
    try:
        ids = prompts(a.slots)
        run_variant(url, ids, {}, 8, 1)  # fill every slot's cache with its prompt
        for name in a.variants.split(","):
            try:
                r = run_variant(url, ids, VARIANTS[name], a.tokens, 1000)
            except Exception as e:  # a variant this build refuses
                r = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            row = {"server": a.server, "slots": a.slots, "threads": a.threads, "ctx": a.ctx,
                   "kv_unified": a.kv_unified, "lora": bool(a.lora), "variant": name,
                   "start_s": info["seconds"], **r}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        server.stop(log_dir)
    if a.out:
        with Path(a.out).open("a", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
