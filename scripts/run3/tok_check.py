"""Does the trainer tokenise a turn the way llama-server does? (phase2.md; advisor's check)

    python scripts/run3/tok_check.py dump --out <file> [--n 50]     # Windows: the server's ids
    python scripts/run3/tok_check.py compare --file <file>          # WSL: the trainer's ids

`dump` serves step 40, takes `--n` recorded turns of phase 0's right episodes (the prompt the
server rendered and the model's output), and stores the server's own tokenisation of the prompt
and of the output (special tokens parsed). `compare` tokenises the same texts with the trainer's
tokenizer as `rl.examples` does and reports how many match exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _tokenize(url: str, text: str) -> list[int]:
    body = {"content": text, "add_special": False, "parse_special": True}
    req = urllib.request.Request(url + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def dump(out: Path, n: int) -> int:
    from sqlrl.run2 import server
    from sqlrl.run2.model import models_dir
    from sqlrl.run3 import settings

    eps = settings.get().data / "phase0" / "pass8-step40" / "episodes.jsonl"
    log = settings.get().data / "render-server"
    model = models_dir() / "run2-step40" / "Spark-X2.6-4B-SQL-Cypher-Agent-RL2-Q8_0.gguf"
    server.start(model, log, port=8098, slots=2, ctx=2 * 12288)
    rows = []
    try:
        url = "http://127.0.0.1:8098"
        for line in eps.read_text("utf-8").splitlines():
            r = json.loads(line)
            if not r.get("rendered"):
                continue
            for prompt, t in zip(r["rendered"], r["turns"], strict=False):
                rows.append({"prompt": prompt, "target": t["text"],
                             "p_ids": _tokenize(url, prompt),
                             "t_ids": _tokenize(url, t["text"])})  # fmt: skip
                break  # one turn an episode, for variety
            if len(rows) >= n:
                break
    finally:
        server.stop(log)
    out.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    print(json.dumps({"dumped": len(rows)}))
    return 0


def compare(f: Path) -> int:
    from sqlrl.run2 import config as rcfg
    from sqlrl.run2.model import model_dir

    sys.path.insert(0, str(__import__("sqlrl.paths").paths.get_paths().trainer_repo / "src"))
    from asktrainer.modelio import load_tokenizer

    del rcfg
    tok = load_tokenizer(model_dir())
    same_p = same_t = n = 0
    first = None
    for line in f.read_text("utf-8").splitlines():
        x = json.loads(line)
        n += 1
        p = tok(x["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(x["target"], add_special_tokens=False)["input_ids"]
        same_p += p == x["p_ids"]
        same_t += t == x["t_ids"]
        if first is None and (p != x["p_ids"] or t != x["t_ids"]):
            first = {"prompt_len": [len(p), len(x["p_ids"])], "target_len": [len(t),
                     len(x["t_ids"])]}  # fmt: skip
    print(json.dumps({"turns": n, "prompt_identical": same_p, "target_identical": same_t,
                      "first_difference": first}))  # fmt: skip
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=50)
    c = sub.add_parser("compare")
    c.add_argument("--file", required=True)
    a = ap.parse_args()
    return dump(Path(a.out), a.n) if a.cmd == "dump" else compare(Path(a.file))


if __name__ == "__main__":
    sys.exit(main())
