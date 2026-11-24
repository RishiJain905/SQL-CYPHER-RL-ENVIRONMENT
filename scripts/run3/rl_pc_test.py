"""Run 3, phase 2's update tested on the PC before any Colab time (WSL, GPU):

    python scripts/run3/rl_pc_test.py --out <dir> --start <ckpt lora_state> --gguf <release Q8_0>

Stand-in rollouts: phase 0's recorded right episodes (their server-rendered prompts), each given
a made-up advantage, through the real path: `rl.examples` (masks), `Learner.update` (reference
pass, policy gradient, KL), `save`, the float32 merge into a copy of the released Q8_0 after its
template check, then a second `Learner` resumed from the saved state, which must read back the
same adapter. Prints one JSON line per stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run3 import rl  # noqa: E402
from sqlrl.run3.policy import THINK_END  # noqa: E402
from sqlrl.run3.rl_learner import Learner  # noqa: E402


def stand_in(p0_eps: Path, n: int) -> list[dict]:
    eps = []
    for line in p0_eps.read_text("utf-8").splitlines():
        r = json.loads(line)
        if not r.get("rendered") or len(r["rendered"]) != len(r["turns"]):
            continue
        calls = []
        for prompt, t in zip(r["rendered"], r["turns"], strict=True):
            think, _, answer = t["text"].partition(THINK_END)
            calls.append({"prompt": prompt, "think": think, "closed": True, "answer": answer,
                          "eos": True, "tokens": 0})  # fmt: skip
        eps.append({"calls": calls})
        if len(eps) >= n:
            break
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--episodes", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lr = Learner(Path(a.start))
    print(json.dumps({"stage": "load", "s": round(time.time() - t0, 1)}), flush=True)
    rnd = random.Random(0)
    exs, over = [], 0
    for ep in stand_in(Path(a.episodes), 24):
        e, o = rl.examples(ep, lr.tok, rnd.choice([-1.0, 1.0]), rl.CFG["max_seq"])
        exs += e
        over += o
    print(json.dumps({"stage": "examples", "n": len(exs), "over": over}), flush=True)
    m = lr.update(exs, 1)
    print(json.dumps({"stage": "update", **m}), flush=True)
    h = lr.save(out / "state")
    print(json.dumps({"stage": "save", **h}), flush=True)
    gguf = out / "rollout.gguf"
    shutil.copyfile(a.gguf, gguf)
    print(json.dumps({"stage": "template", **lr.check_template(gguf)}), flush=True)
    print(json.dumps({"stage": "merge", **lr.merge(gguf, 1)}), flush=True)
    del lr
    import torch

    torch.cuda.empty_cache()
    lr2 = Learner(Path(a.start), state_dir=out / "state")
    from asktrainer.checkpoint import trainable_state
    from safetensors.torch import save as st_bytes

    again = hashlib.sha256(st_bytes(trainable_state(lr2.model))).hexdigest()
    print(json.dumps({"stage": "resume", "same_adapter": again == h["lora_sha256"],
                      "optim_state": len(lr2.opt.state_dict()["state"])}), flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
