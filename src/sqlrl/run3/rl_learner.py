"""Run 3, phase 2: the update (GPU; documentation/run3/phase2.md).

`Learner` holds the policy (step 40's rank-16 adapter on the released model, R25) for the whole
run. A step's update: the reference's log-probabilities of every example's target tokens (the
start checkpoint's adapter swapped in, no gradient), then run 2's minibatches of the policy
gradient, every token of an episode's turns weighted by its advantage, token mean, plus
`kl_beta` x the k3 estimate of the KL to the reference on the same tokens. The adapter and the
optimiser are saved byte-checked (run 2's `save_verified`); the sampler's Q8_0 is rewritten from
the adapter in float32 (run 2's `gguf_merge`), never through bf16 (design 5.1).
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.run3.rl import CFG


class Learner:
    def __init__(self, start_lora: Path, cfg: dict[str, Any] = CFG, state_dir: Path | None = None):
        import torch
        from safetensors.torch import load_file

        from sqlrl.paths import get_paths
        from sqlrl.run2 import config as rcfg
        from sqlrl.run2.model import load_policy

        sys.path.insert(0, str(get_paths().trainer_repo / "src"))
        self.cfg = cfg
        self.torch = torch
        pcfg = rcfg.new_config({"init_from": "run2-step40", "seed": cfg["seed"]})
        resume = state_dir is not None and (state_dir / "lora_state.safetensors").exists()
        first = state_dir / "lora_state.safetensors" if resume else start_lora
        self.model, self.tok, self.backbone, self.head = load_policy(pcfg, first)
        self.ref = load_file(str(start_lora))  # the KL reference: the start checkpoint
        self.params = [q for q in self.model.parameters() if q.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], betas=(0.9, 0.999),
                                     weight_decay=0.0)  # fmt: skip
        if resume and (state_dir / "optim.pt").exists():
            self.opt.load_state_dict(torch.load(state_dir / "optim.pt", map_location="cuda"))

    # -- the update ---------------------------------------------------------------------------

    def _lps(self, ex: dict[str, Any]):
        from sqlrl.run2.logprobs import answer_logprobs

        x = self.torch.tensor(ex["ids"], dtype=self.torch.long, device=self.head.device)
        h = self.backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
        return answer_logprobs(h, self.head, x, ex["start"])[0]

    def update(self, exs: list[dict[str, Any]], step: int) -> dict[str, Any]:
        from asktrainer.checkpoint import load_trainable, trainable_state

        torch, cfg = self.torch, self.cfg
        t0 = time.time()
        lr = cfg["lr"] * min(1.0, step / max(cfg["warmup"], 1))
        for g in self.opt.param_groups:
            g["lr"] = lr
        exs = [e for e in exs if e["adv"] != 0.0]
        if not exs:
            return {"examples": 0, "lr": lr, "skipped": "no informative group"}
        # 1. the reference's log-probabilities (no gradient)
        mine = trainable_state(self.model)
        self.model.eval()
        load_trainable(self.model, self.ref)
        with torch.no_grad():
            for e in exs:
                e["_ref"] = self._lps(e).float()
        load_trainable(self.model, mine)
        del mine
        # 2. the policy gradient, run 2's minibatches
        self.model.train()
        random.Random(f"run3-rl:mb:{step}").shuffle(exs)
        m = max(1, int(cfg["minibatches"]))
        batches = [exs[i::m] for i in range(m)]
        tot = {"loss": 0.0, "pg": 0.0, "kl_sum": 0.0, "tokens": 0, "grad_norms": []}
        for batch in batches:
            norm = sum(sum(e["mask"]) for e in batch) or 1
            for e in batch:
                mask = torch.tensor(e["mask"], dtype=torch.float32, device=self.head.device)
                lp = self._lps(e).float()
                ref = e["_ref"]
                d = ref - lp
                k3 = torch.exp(d) - d - 1.0
                pg = -(e["adv"] * (lp * mask).sum()) / norm
                kl = (k3 * mask).sum() / norm
                loss = pg + cfg["kl_beta"] * kl
                loss.backward()
                tot["loss"] += float(loss.detach())
                tot["pg"] += float(pg.detach())
                tot["kl_sum"] += float((k3.detach() * mask).sum())
                tot["tokens"] += int(mask.sum())
            gn = float(torch.nn.utils.clip_grad_norm_(self.params, cfg["grad_clip"]))
            if not torch.isfinite(torch.tensor(gn)):
                self.opt.zero_grad(set_to_none=True)
                raise RuntimeError(f"step {step}: a non-finite gradient")
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            tot["grad_norms"].append(round(gn, 4))
        self.model.eval()
        for e in exs:
            e.pop("_ref", None)
        return {"examples": len(exs), "tokens": tot["tokens"], "lr": lr,
                "loss": round(tot["loss"], 6), "pg_loss": round(tot["pg"], 6),
                "kl": round(tot["kl_sum"] / max(tot["tokens"], 1), 6),
                "grad_norm": max(tot["grad_norms"]), "update_s": round(time.time() - t0, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}  # fmt: skip

    # -- saving, merging ----------------------------------------------------------------------

    def save(self, d: Path) -> dict[str, str]:
        from asktrainer.checkpoint import trainable_state
        from safetensors.torch import save as st_bytes

        from sqlrl.run2.integrity import save_verified

        d.mkdir(parents=True, exist_ok=True)
        h1 = save_verified(st_bytes(trainable_state(self.model)), d / "lora_state.safetensors")
        buf = io.BytesIO()
        self.torch.save(self.opt.state_dict(), buf)
        h2 = save_verified(buf.getvalue(), d / "optim.pt")
        return {"lora_sha256": h1, "optim_sha256": h2}

    def check_template(self, gguf: Path) -> dict[str, Any]:
        from sqlrl.run2 import gguf_merge

        return gguf_merge.check_template(self.model, gguf)

    def merge(self, gguf: Path, step: int) -> dict[str, Any]:
        from sqlrl.run2 import gguf_merge

        return gguf_merge.write_merged(self.model, gguf, step, with_lora=True)

    def tokenizer(self):
        return self.tok


def write_json(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    tmp.replace(p)
