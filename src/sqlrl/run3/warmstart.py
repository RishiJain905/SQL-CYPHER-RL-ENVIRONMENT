"""Run 3, phase 1: the warm start (documentation/run3/phase0.md "Phase 1", design.md 5.2).

    python -m sqlrl.run3.warmstart collect --episodes <pass8 dir>/episodes.jsonl --out <dir>
    python -m sqlrl.run3.warmstart train --dir <dir> [--until <optimizer step>]   # WSL, GPU
    python -m sqlrl.run3.warmstart memory [--tokens 3072,4096,5120,6144]         # WSL, GPU

- **collect** keeps right episodes of the training split only, at most two different ones a
  (question, engine) (by ending and submitted query), and turns each into one example a turn:
  the prompt text llama-server rendered for that turn (recorded by `measure`, so the warm start
  trains on exactly what the model saw) and the model's own output for it, closed by the end of
  sentence token. Tool results are in the prompt, so they are context, never targets. Writes
  `examples.jsonl` and `collect.json` (counts by stratum, kind and turn).
- **train** continues step 40's adapter (R25: rank 16, never a new one) with the token-mean
  cross-entropy on the targets only. AdamW, a short warmup, the gradient clipped at 1.0, an
  RL-scale learning rate (step 40's change is small: a large one would wash it out). Examples
  longer than `--max-seq` are counted and left out, never cut. Resumable: `state/` holds the
  adapter, the optimiser and the position; each `--until` stops at that optimiser step and saves
  `ckpt-<step>/lora_state.safetensors` (checked byte for byte), which the orchestrator merges
  and evaluates. `metrics.jsonl` gets one line an optimiser step; W&B logs live.
- **memory** measures the update's peak memory for one example of each length (as run 2's
  `tools memory`, with the warm start's lighter loss), which sets `--max-seq`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

EOS = "<｜end▁of▁sentence｜>"  # Spark's end of a turn (run 2's episodes close on it)
DEFAULTS = {"lr": 1e-5, "warmup": 10, "accum": 16, "epochs": 1, "max_seq": 3840,
            "grad_clip": 1.0, "weight_decay": 0.0, "betas": [0.9, 0.999], "seed": 0,
            "per_question": 2}  # fmt: skip


# ---- collect (no GPU) -------------------------------------------------------------------------


def _trainable(r: dict[str, Any]) -> bool:
    from sqlrl.run3 import splits

    if r.get("kind") == "answer":
        return splits.trainable(r["source"], r["db"], r["qid"])
    return r["db"] not in splits.HELDOUT_DBS.get(r["source"], frozenset())


def collect(episode_files: list[Path], per_question: int = 2) -> tuple[list[dict], dict]:
    kept: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: dict[tuple[str, str], set[str]] = {}
    for f in episode_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("outcome") != "right" or not r.get("rendered") or not _trainable(r):
                continue
            key = (r["qid"], r["engine"])
            sig = f"{r.get('ending')}|{(r.get('final_query') or '').strip()}"
            if sig in seen.setdefault(key, set()) or len(kept.get(key, [])) >= per_question:
                continue
            seen[key].add(sig)
            kept.setdefault(key, []).append(r)
    examples: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for (qid, engine), recs in kept.items():
        for r in recs:
            turns = r["turns"]
            if len(r["rendered"]) != len(turns):
                stats["skipped_render_mismatch"] += 1
                continue
            for t, (prompt, turn) in enumerate(zip(r["rendered"], turns, strict=True)):
                examples.append({"qid": qid, "engine": engine, "stratum": r["stratum"],
                                 "kind": r["kind"], "turn": t, "of": len(turns),
                                 "prompt": prompt, "target": turn["text"]})  # fmt: skip
            stats[f"episodes/{r['stratum'].split('/')[0]}"] += 1
            stats[f"ending/{r.get('ending')}"] += 1
    stats["questions"] = len(kept)
    stats["examples"] = len(examples)
    stats["turns_hist"] = dict(Counter(e["of"] for e in examples if e["turn"] == 0))
    return examples, dict(stats)


# ---- train (WSL, GPU) -------------------------------------------------------------------------


def _cfg(d: Path, over: dict[str, Any]) -> dict[str, Any]:
    f = d / "config.json"
    cfg = {**DEFAULTS, **(json.loads(f.read_text("utf-8")) if f.exists() else {})}
    cfg.update({k: v for k, v in over.items() if v is not None})
    f.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def _policy_cfg(seed: int) -> dict[str, Any]:
    from sqlrl.run2 import config as rcfg

    return rcfg.new_config({"init_from": "run2-step40", "seed": seed})


def tokenize(tok: Any, ex: dict[str, Any]) -> tuple[list[int], int]:
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    t = tok(ex["target"] + EOS, add_special_tokens=False)["input_ids"]
    return p + t, len(p)


def train(d: Path, until: int | None, over: dict[str, Any], launched_by: str = "agent") -> dict:
    import torch
    from safetensors.torch import load_file
    from safetensors.torch import save as st_bytes

    from sqlrl.paths import get_paths
    from sqlrl.run2.integrity import save_verified
    from sqlrl.run2.logprobs import answer_logprobs
    from sqlrl.run2.model import load_policy, start_state

    cfg = _cfg(d, over)
    sys.path.insert(0, str(get_paths().trainer_repo / "src"))
    from asktrainer.checkpoint import load_trainable, trainable_state

    state = d / "state"
    pos = json.loads((state / "position.json").read_text("utf-8")) if (
        state / "position.json").exists() else {"step": 0, "seen": 0}  # fmt: skip
    pcfg = _policy_cfg(cfg["seed"])
    start = state / "lora_state.safetensors" if pos["step"] else start_state(pcfg)
    model, tok, backbone, head = load_policy(pcfg, start)
    exs = [json.loads(x) for x in (d / "examples.jsonl").read_text("utf-8").splitlines()]
    order = list(range(len(exs))) * int(cfg["epochs"])
    random.Random(cfg["seed"]).shuffle(order)
    toks: dict[int, tuple[list[int], int]] = {}
    too_long = 0
    for i in set(order):
        ids, n_p = tokenize(tok, exs[i])
        if len(ids) > cfg["max_seq"]:
            too_long += 1
            continue
        toks[i] = (ids, n_p)
    order = [i for i in order if i in toks]
    total_steps = len(order) // cfg["accum"]
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])  # fmt: skip
    if pos["step"] and (state / "optim.pt").exists():
        opt.load_state_dict(torch.load(state / "optim.pt", map_location="cuda"))
    run = _wandb(d, cfg, launched_by, total_steps, too_long, len(exs))
    dev = head.device
    model.train()
    stop_at = min(until or total_steps, total_steps)
    t0 = time.time()
    while pos["step"] < stop_at:
        k = pos["step"] + 1
        lr = cfg["lr"] * min(1.0, k / max(cfg["warmup"], 1))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = order[pos["seen"] : pos["seen"] + cfg["accum"]]
        n_tok = sum(len(toks[i][0]) - toks[i][1] for i in batch)
        loss_sum, t1 = 0.0, time.time()
        for i in batch:
            ids, n_p = toks[i]
            x = torch.tensor(ids, dtype=torch.long, device=dev)
            h = backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
            lp, _ = answer_logprobs(h, head, x, n_p)
            loss = -lp.sum() / n_tok
            loss.backward()
            loss_sum += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]))
        if not torch.isfinite(torch.tensor(gn)):
            opt.zero_grad(set_to_none=True)
            raise RuntimeError(f"step {k}: non-finite gradient")
        opt.step()
        opt.zero_grad(set_to_none=True)
        pos = {"step": k, "seen": pos["seen"] + len(batch)}
        m = {"step": k, "of": total_steps, "loss": round(loss_sum, 5), "tokens": n_tok,
             "examples": len(batch), "grad_norm": round(gn, 4), "lr": lr,
             "seconds": round(time.time() - t1, 1),
             "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}  # fmt: skip
        with (d / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        if run:
            run.log({f"train/{key}": v for key, v in m.items() if key != "step"}, step=k)
        print(json.dumps(m), flush=True)
    model.eval()
    state.mkdir(exist_ok=True)
    blob = st_bytes(trainable_state(model))
    save_verified(blob, state / "lora_state.safetensors")
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    save_verified(buf.getvalue(), state / "optim.pt")
    (state / "position.json").write_text(json.dumps(pos), encoding="utf-8")
    ck = d / f"ckpt-{pos['step']:04d}"
    ck.mkdir(exist_ok=True)
    save_verified(blob, ck / "lora_state.safetensors")
    load_trainable(model, load_file(str(ck / "lora_state.safetensors")))  # it reads back
    out = {"step": pos["step"], "of": total_steps, "too_long": too_long, "examples": len(exs),
           "trained_examples": pos["seen"], "minutes": round((time.time() - t0) / 60, 1),
           "ckpt": ck.name, "sha256": hashlib.sha256(blob).hexdigest()}  # fmt: skip
    (ck / "train.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if run:
        run.summary.update({"too_long": too_long, "last_ckpt": ck.name})
        run.finish()
    return out


def _wandb(d: Path, cfg: dict[str, Any], who: str, steps: int, too_long: int, n: int):
    try:
        import wandb
    except ImportError:
        return None
    rid_f = d / "wandb_id.txt"
    rid = rid_f.read_text().strip() if rid_f.exists() else wandb.util.generate_id()
    rid_f.write_text(rid)
    safe = {k: v for k, v in cfg.items()
            if not any(w in k.lower() for w in ("token", "secret", "password", "key"))}  # fmt: skip
    return wandb.init(project="sql-cypher-rl", group="run3", job_type="warm-start",
                      name="run3-warmstart", id=rid, resume="allow", dir=str(d),
                      tags=["run3", "phase1", "warm-start", f"launched-by:{who}"],
                      config={**safe, "steps": steps, "too_long": too_long, "examples": n,
                              "start": "run2-step40"})  # fmt: skip


def memory(sizes: list[int]) -> dict[str, Any]:
    import torch

    from sqlrl.run2.logprobs import answer_logprobs
    from sqlrl.run2.model import load_policy, start_state

    pcfg = _policy_cfg(0)
    model, _tok, backbone, head = load_policy(pcfg, start_state(pcfg))
    model.train()
    dev = head.device
    params = [q for q in model.parameters() if q.requires_grad]
    out: dict[str, Any] = {"model_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                           "sizes": []}  # fmt: skip
    for n in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen = min(1500, n // 3)
        try:
            x = torch.randint(1000, 20000, (n,), device=dev)
            h = backbone(input_ids=x[None], use_cache=False).last_hidden_state[0]
            lp, _ = answer_logprobs(h, head, x, n - gen)
            (-lp.mean()).backward()
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            del x, h, lp
        except torch.OutOfMemoryError:
            peak = None
        for q in params:
            q.grad = None
        out["sizes"].append({"tokens": n, "peak_gb": peak})
        print(json.dumps(out["sizes"][-1]), flush=True)
        if peak is None:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--per-question", type=int, default=DEFAULTS["per_question"])
    t = sub.add_parser("train")
    t.add_argument("--dir", required=True)
    t.add_argument("--until", type=int, default=None)
    for k in ("lr", "accum", "epochs", "max_seq", "warmup"):
        t.add_argument(f"--{k.replace('_', '-')}", type=type(DEFAULTS[k]), default=None)
    t.add_argument("--launched-by", default="agent")
    m = sub.add_parser("memory")
    m.add_argument("--tokens", default="3072,4096,5120,6144,7168")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        exs, stats = collect([Path(x) for x in a.episodes], a.per_question)
        with (d / "examples.jsonl").open("w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        (d / "collect.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print(json.dumps(stats, indent=1))
    elif a.cmd == "train":
        over = {k: getattr(a, k) for k in ("lr", "accum", "epochs", "max_seq", "warmup")}
        print(json.dumps(train(Path(a.dir), a.until, over, a.launched_by)))
    else:
        print(json.dumps(memory([int(x) for x in a.tokens.split(",")])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
