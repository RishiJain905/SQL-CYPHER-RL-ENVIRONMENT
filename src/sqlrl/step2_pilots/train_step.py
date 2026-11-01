"""One RL step of a pilot (inside WSL, with the trainer's venv; one process per step, so the
GPU is free again for the rollout server when it exits):

    python -m sqlrl.step2_pilots.train_step --run-dir <dir> --step <k>

1. **Score** step k's rollouts with the trainer's checker (`publiceval.score`, checker 4) ->
   `scored.jsonl`; rewards and group advantages (`rewards.py`).
2. **Load** the merged SFT model (bf16) with the pilot's LoRA as it was after step k-1 (step 1:
   a new LoRA, seeded; its B matrices are zero, so it starts as the SFT model) and its AdamW
   state.
3. **Update:** the groups that carry a signal, split into `minibatches`, one optimizer step
   each, with the method's loss (`losses.py`). The first minibatch's "old" log-probabilities
   are the policy's own (on-policy); the later ones' are computed before any update. The KL
   reference (when `kl_beta` > 0) is the same model with the adapter switched off.
4. **Save** the LoRA (PEFT folder, its trainable state, the optimizer state) and convert it to
   GGUF for step k+1's rollout server; `metrics.json`, a `train_log.jsonl` row, the W&B step.

A finished step writes `TRAINED`; running it again does nothing. Scoring is kept when the update
fails, so a re-run starts at the update.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths
from sqlrl.step2_pilots import config as pcfg
from sqlrl.step2_pilots import rewards
from sqlrl.step2_pilots.pool import load_pool

WANDB_PROJECT = "sql-cypher-rl"


def step_dir(run_dir: Path, k: int) -> Path:
    return run_dir / f"step-{k:04d}"


def read_rollouts(path: Path, cfg: dict[str, Any], rows: list[dict]) -> list[dict[str, Any]]:
    """The step's rollouts, one per (question, g); a later line replaces an earlier one, a
    failed one never replaces a real one."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["qid"], r["g"])
            if "error" not in r or key not in out:
                out[key] = r
    want = {(row["id"], g) for row in rows for g in range(cfg["group_size"])}
    missing = want - {k for k, r in out.items() if "error" not in r}
    if missing:
        raise RuntimeError(f"{len(missing)} rollouts missing or failed (e.g. {sorted(missing)[0]})")
    return [out[k] for k in sorted(want)]


def score(sdir: Path, rollouts: list[dict], rows: list[dict], cfg: dict[str, Any]) -> float:
    """Checker verdicts into the rollouts (cached in `scored.jsonl`); returns seconds spent."""
    cache = sdir / "scored.jsonl"
    if cache.exists():
        got = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                got[(v["qid"], v["g"])] = v
        if all((r["qid"], r["g"]) in got for r in rollouts):
            for r in rollouts:
                r.update({k: got[(r["qid"], r["g"])][k] for k in ("json_ok", "exec_ok", "reason")})
            return 0.0
    from asktrainer import publiceval
    from asktrainer.data.schema import Example
    from asktrainer.settings import get_paths as trainer_paths

    data = publiceval.PublicData(trainer_paths().public)
    exs = {row["id"]: Example.model_validate(row) for row in rows}
    if any(e.dialect == "cypher" for e in exs.values()) and publiceval._graph() is None:
        # without it every Cypher answer would score "undecided" and get no reward, silently
        raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
    t = time.time()
    # one graph load per CypherBench graph: score grouped by graph
    order = sorted(range(len(rollouts)),
                   key=lambda i: (rollouts[i]["origin"],
                                  exs[rollouts[i]["qid"]].meta.get("db_id") or ""))  # fmt: skip
    for i in order:
        r = rollouts[i]
        item = publiceval.score(exs[r["qid"]], r["text"], data, checker=cfg["checker"])
        r.update(json_ok=bool(item["json_ok"]), exec_ok=item["exec_ok"], reason=item["reason"])
    tmp = cache.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rollouts:
            f.write(json.dumps({k: r[k] for k in ("qid", "g", "json_ok", "exec_ok", "reason")})
                    + "\n")  # fmt: skip
    tmp.replace(cache)
    return time.time() - t


def minibatches(seqs: list[dict[str, Any]], m: int, seed: int) -> list[list[dict[str, Any]]]:
    """Whole groups, shuffled with the step's seed, dealt round-robin into `m` minibatches."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in seqs:
        groups.setdefault(s["qid"], []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out: list[list[dict[str, Any]]] = [[] for _ in range(max(m, 1))]
    for i, k in enumerate(keys):
        out[i % len(out)].extend(groups[k])
    return [b for b in out if b]


def convert_gguf(adapter_dir: Path, out: Path) -> None:
    p = get_paths()
    src = p.llama_cpp_src
    cmd = [sys.executable, str(src / "convert_lora_to_gguf.py"), "--base", str(p.model_hf),
           "--outtype", "f16", "--outfile", str(out), str(adapter_dir)]  # fmt: skip
    env = {"PYTHONPATH": str(src / "gguf-py"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(src))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"LoRA -> GGUF failed: {r.stderr[-500:]}")


def train(run_dir: Path, k: int, launched_by: str = "agent") -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    cfg = pcfg.load(run_dir)
    sdir = step_dir(run_dir, k)
    if (sdir / "TRAINED").exists():
        return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer.checkpoint import load_trainable, trainable_state
    from asktrainer.config import LoraSpec
    from asktrainer.modelio import apply_lora, backbone_of, head_weight, load_base, save_adapter

    from sqlrl.step2_pilots import srpo
    from sqlrl.step2_pilots.logprobs import sequence_logprobs
    from sqlrl.step2_pilots.losses import norm_for, route, sequence_loss

    t_all = time.time()
    rows = pcfg.step_questions(load_pool(cfg["pool"]), cfg, k)
    rollouts = read_rollouts(sdir / "rollouts.jsonl", cfg, rows)
    score_s = score(sdir, rollouts, rows, cfg)
    for r in rollouts:
        r["reward"] = rewards.reward(r, cfg)
    mcfg = cfg["method_cfg"]
    rewards.advantages(rollouts, mcfg["adv"])
    stats = rewards.group_stats(rollouts)
    sd = mcfg.get("sdpo")  # SRPO: the self-distillation branch's settings
    if sd:
        route(rollouts)
        trainable = [r for r in rollouts if r["route"] == "sdpo"
                     or (r["route"] == "grpo" and r["adv"] not in (None, 0.0))]  # fmt: skip
    else:
        trainable = [r for r in rollouts if r["adv"] not in (None, 0.0)]
    too_long = [r for r in trainable if len(r["prompt_ids"]) + len(r["gen_ids"]) > cfg["max_seq"]]
    trainable = [r for r in trainable if r not in too_long]

    # the model and the LoRA after step k-1
    t_load = time.time()
    torch.manual_seed(cfg["seed"])
    model, tok, _info = load_base(p.model_hf, device="cuda", log=lambda *a: None)
    spec = LoraSpec(r=cfg["lora_r"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"],
                    targets=cfg["lora_targets"])  # fmt: skip
    model = apply_lora(model, spec, gradient_checkpointing=True)
    prev = step_dir(run_dir, k - 1)
    if k > 1:
        load_trainable(model, load_file(str(prev / "lora_state.safetensors")))
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])  # fmt: skip
    if k > 1 and (prev / "optim.pt").exists():
        opt.load_state_dict(torch.load(prev / "optim.pt", map_location="cuda"))
    load_s = time.time() - t_load
    backbone, head = backbone_of(model), head_weight(model)

    def lps(r, entropy=False):
        return sequence_logprobs(model, backbone, head, r["prompt_ids"] + r["gen_ids"],
                                 len(r["prompt_ids"]), entropy)  # fmt: skip

    t_train = time.time()
    sd_stats: dict[str, Any] = {}
    if sd:  # the EMA teacher's view of every distilled rollout, before any update
        student0 = trainable_state(model)
        ema_file = prev / "ema_state.safetensors"
        ema = load_file(str(ema_file)) if k > 1 and ema_file.exists() else student0
        by_qid_g = {(r["qid"], r["g"]): r for r in rollouts}
        rows_by_id = {row["id"]: row for row in rows}
        sd_stats = srpo.prepare_teacher(model, backbone, head,
                                        [r for r in trainable if r["route"] == "sdpo"],
                                        by_qid_g, rows_by_id, tok, ema, student0,
                                        sd["entropy_beta"])  # fmt: skip
        first = rollouts[0]
        sd_stats["template_match"] = srpo.student_prompt_matches(tok, rows_by_id[first["qid"]],
                                                                 first["prompt_ids"])  # fmt: skip
    model.train()
    batches = minibatches(trainable, cfg["minibatches"], cfg["seed"] * 100_000 + k)
    with torch.no_grad():  # "old" for the later minibatches, the KL reference for all
        for b in batches[1:]:
            for r in b:
                if r.get("route") != "sdpo":
                    r["_old"] = lps(r)[0]
        if mcfg.get("kl_beta", 0) > 0:
            with model.disable_adapter():
                for r in trainable:
                    r["_ref"] = lps(r)[0]
    tot = {"loss": 0.0, "tokens": 0.0, "clipped": 0.0, "kl_sum": 0.0, "ent_sum": 0.0,
           "ent_n": 0.0, "mis_abs": 0.0, "mis_n": 0, "jsd_sum": 0.0, "jsd_n": 0.0}  # fmt: skip
    grad_norms = []
    for bi, b in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        norm = norm_for(mcfg["agg"], b, cfg)
        for r in b:
            if r.get("route") == "sdpo":
                loss, st = srpo.distill_loss(backbone, head, r, sd["top_k"], norm)
                loss.backward()
                tot["loss"] += float(loss)
                tot["jsd_sum"] += st["jsd_sum"]
                tot["jsd_n"] += st["tokens"]
                continue
            lp, ent = lps(r, entropy=True)
            old = r.pop("_old", None)
            old = lp.detach() if old is None else old
            mask = torch.tensor(r["gen_mask"], device=lp.device)
            loss, st = sequence_loss(mcfg, lp, old, mask, r["adv"], norm, r.pop("_ref", None))
            loss.backward()
            m = mask.float()
            tot["loss"] += float(loss)
            for key in ("tokens", "clipped", "kl_sum"):
                tot[key] += st[key]
            tot["ent_sum"] += float((ent * m).sum())
            tot["ent_n"] += float(m.sum())
            if bi == 0 and r.get("behavior_lp"):  # the server's view of the same tokens
                known = torch.tensor([x is not None for x in r["behavior_lp"]], device=lp.device)
                beh = torch.tensor([x if x is not None else 0.0 for x in r["behavior_lp"]],
                                   device=lp.device)  # fmt: skip
                w = m * known.float()
                tot["mis_abs"] += float(((lp.detach() - beh).abs() * w).sum())
                tot["mis_n"] += float(w.sum())
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])))
        opt.step()
    model.eval()
    train_s = time.time() - t_train

    # save the LoRA for step k+1 (and the record)
    sdir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(sdir / "lora_state.safetensors"))
    torch.save(opt.state_dict(), sdir / "optim.pt")
    if sd:  # the teacher follows the student slowly
        save_file(srpo.ema_update(ema, trainable_state(model), sd["ema"]),
                  str(sdir / "ema_state.safetensors"))  # fmt: skip
    adapter = save_adapter(model, sdir / "adapter")
    t_conv = time.time()
    convert_gguf(adapter, sdir / "adapter.gguf")
    conv_s = time.time() - t_conv
    if k > 2:  # keep the optimizer state of the last two steps only (it's 250 MB)
        (step_dir(run_dir, k - 2) / "optim.pt").unlink(missing_ok=True)

    toks = max(tot["tokens"], 1.0)
    meta = sdir / "rollout_meta.json"
    rmeta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    metrics = {
        "step": k,
        **stats,
        "trained_seqs": len(trainable),
        "skipped_too_long": len(too_long),
        "loss": tot["loss"],
        "kl": tot["kl_sum"] / toks if mcfg.get("kl_beta", 0) > 0 else None,
        "entropy": tot["ent_sum"] / max(tot["ent_n"], 1.0),
        "distill_jsd": tot["jsd_sum"] / tot["jsd_n"] if tot["jsd_n"] else None,
        **sd_stats,
        "clip_frac": tot["clipped"] / toks,
        "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "lr": cfg["lr"],
        "resp_tokens": sum(len(r["gen_ids"]) for r in rollouts) / len(rollouts),
        "think_forced_frac": sum(bool(r.get("forced")) for r in rollouts) / len(rollouts),
        "mismatch": tot["mis_abs"] / tot["mis_n"] if tot["mis_n"] else None,
        "rollout_s": rmeta.get("seconds"),
        "score_s": round(score_s, 1),
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "convert_s": round(conv_s, 1),
        "step_s": round((rmeta.get("seconds") or 0) + time.time() - t_all, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (sdir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    with (run_dir / "train_log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metrics) + "\n")
    log_wandb(run_dir, cfg, metrics, launched_by)
    (sdir / "TRAINED").write_text("", encoding="utf-8")
    return metrics


def log_wandb(run_dir: Path, cfg: dict[str, Any], metrics: dict[str, Any], who: str) -> None:
    try:
        import wandb
    except ImportError:
        return
    id_file = run_dir / "wandb_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else None
    tags = ["pilot", f"method:{cfg['method']}", f"launched-by:{who}"] + (
        ["smoke"] if cfg.get("smoke") else [])  # fmt: skip
    run = wandb.init(project=WANDB_PROJECT, name=cfg["name"], id=run_id, resume="allow",
                     group="pilots", job_type="pilot", tags=tags, dir=str(run_dir),
                     config={k: v for k, v in cfg.items() if k != "name"})  # fmt: skip
    if run_id is None:
        id_file.write_text(run.id, encoding="utf-8")
    flat = {f"train/{k}": v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "step"}  # fmt: skip
    for s, v in metrics.get("reward_by_set", {}).items():
        flat[f"reward/{s}"] = v
    for s, v in metrics.get("solve_by_set", {}).items():
        flat[f"solve/{s}"] = v
    run.log(flat, step=metrics["step"])
    run.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    m = train(Path(a.run_dir), a.step, a.launched_by)
    print(json.dumps({k: m[k] for k in ("step", "reward_mean", "solve_rate", "informative_frac",
                                        "trained_seqs", "train_s", "peak_gb")}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
