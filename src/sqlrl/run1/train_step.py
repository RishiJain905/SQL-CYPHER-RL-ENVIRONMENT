"""One RL step of run 1 (inside WSL, with the trainer's venv; one process per step, so the GPU
is free again for the rollout server when it exits):

    python -m sqlrl.run1.train_step --run-dir <dir> --step <k>

1. **Score and load at the same time.** The checker (CPU, `publiceval.score`, checker 4)
   scores step k's rollouts into `scored.jsonl` in a process of its own (`--score-only`)
   while this one loads the model on the GPU.
2. **Rewards, advantages, routing** (`rewards.py`, `losses.route`): SRPO as piloted, with the
   floor under the group's standard deviation (decisions R8).
3. **The probe** (before any update): on the rollouts that carry the server's own
   log-probabilities, the gap between the server and the trainer (`mismatch`) and the drift
   from the SFT model measured from the server's side (`kl_sft`, decisions R7).
4. **The update:** the groups that carry a signal, split into `minibatches`, one optimizer step
   each. Sequences are padded into batches of at most `batch_tokens` tokens (right padding:
   with causal attention a sequence's own positions never see the padding).
5. **Save:** the LoRA, the optimizer state, the EMA teacher; then the LoRA is merged into the
   rollout model's Q8_0 file for step k+1's server (`gguf_merge.py`). At evaluation steps and
   the last step the adapter is also saved as a PEFT folder and a GGUF, for the record.

A finished step writes `TRAINED`; running it again does nothing. Scoring is kept when the update
fails, so a re-run starts at the update.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths
from sqlrl.run1 import canaries, rewards
from sqlrl.run1 import config as rcfg
from sqlrl.run1.pool import load_pool

WANDB_PROJECT = "sql-cypher-rl"


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


def score_only(run_dir: Path, k: int) -> int:
    """The scorer process: step k's verdicts into `scored.jsonl`, its time into
    `score_meta.json` (kept when the verdicts were already there)."""
    cfg = rcfg.load(run_dir)
    sdir = rcfg.step_dir(run_dir, k)
    sys.path.insert(0, str(get_paths().trainer_repo / "src"))
    rows = rcfg.step_questions(load_pool(cfg["pool"]), cfg, k)
    secs = score(sdir, read_rollouts(sdir / "rollouts.jsonl", cfg, rows), rows, cfg)
    meta = sdir / "score_meta.json"
    if secs > 0 or not meta.exists():
        meta.write_text(json.dumps({"seconds": round(secs, 1)}), encoding="utf-8")
    return 0


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


def pack(items: list[Any], length, budget: int) -> list[list[Any]]:
    """Batches whose padded size (count x the longest member) stays within `budget` tokens;
    longest first, so each batch pads to its first member. A member longer than the budget
    gets a batch of its own. `budget` <= 0: one member per batch."""
    order = sorted(items, key=length, reverse=True)
    out: list[list[Any]] = []
    for it in order:
        if out and budget > 0 and (len(out[-1]) + 1) * length(out[-1][0]) <= budget:
            out[-1].append(it)
        else:
            out.append([it])
    return out


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
    cfg = rcfg.load(run_dir)
    sdir = rcfg.step_dir(run_dir, k)
    if (sdir / "TRAINED").exists():
        return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from sqlrl.run1.model import load_policy, rollout_model_path

    t_all = time.time()
    rows = rcfg.step_questions(load_pool(cfg["pool"]), cfg, k)
    rollouts = read_rollouts(sdir / "rollouts.jsonl", cfg, rows)
    prev = rcfg.step_dir(run_dir, k - 1)
    state = prev / "lora_state.safetensors" if k > 1 else None
    # the checker scores in a process of its own (CPU) while this one loads the model
    scorer = subprocess.Popen([sys.executable, "-m", "sqlrl.run1.train_step", "--run-dir",
                               str(run_dir), "--step", str(k), "--score-only"])  # fmt: skip
    try:
        model, tok, backbone, head = load_policy(cfg, state)
        load_s = time.time() - t_all
        if scorer.wait() != 0:
            raise RuntimeError("scoring the rollouts failed (see above)")
    finally:
        if scorer.poll() is None:
            scorer.kill()
    score(sdir, rollouts, rows, cfg)  # reads the scorer's verdicts
    score_s = json.loads((sdir / "score_meta.json").read_text(encoding="utf-8"))["seconds"]

    import torch
    from asktrainer.checkpoint import load_trainable, trainable_state
    from asktrainer.modelio import save_adapter
    from safetensors.torch import load_file, save_file

    from sqlrl.run1 import gguf_merge, srpo
    from sqlrl.run1.logprobs import _chunk_entropy, answer_logprobs
    from sqlrl.run1.losses import norm_for, route, sequence_loss, topk_jsd

    for r in rollouts:
        r["reward"] = rewards.reward(r, cfg)
    mcfg = cfg["method_cfg"]
    rewards.advantages(rollouts, mcfg["adv"], std_floor=cfg.get("adv_std_floor", 0.0))
    stats = rewards.group_stats(rollouts)
    sd = mcfg.get("sdpo")  # SRPO: the self-distillation branch's settings
    if sd:
        route(rollouts)
        trainable = [r for r in rollouts if r["route"] == "sdpo"
                     or (r["route"] == "grpo" and r["adv"] not in (None, 0.0))]  # fmt: skip
    else:
        trainable = [r for r in rollouts if r["adv"] not in (None, 0.0)]

    def n_tokens(r: dict[str, Any]) -> int:
        return len(r["prompt_ids"]) + len(r["gen_ids"])

    too_long = [r for r in trainable if n_tokens(r) > cfg["max_seq"]]
    trainable = [r for r in trainable if n_tokens(r) <= cfg["max_seq"]]

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])  # fmt: skip
    if k > 1 and (prev / "optim.pt").exists():
        opt.load_state_dict(torch.load(prev / "optim.pt", map_location="cuda"))
    lr = rcfg.lr_at(cfg, k)
    for group in opt.param_groups:  # the saved state carries the step before's rate
        group["lr"] = lr
    dev = head.device
    budget = int(cfg.get("batch_tokens") or 0)

    def hidden(batch: list[dict[str, Any]], key: str | None = None) -> list[Any]:
        """The last layer's states of each member's tokens ([T_i, H]), one padded pass."""
        seqs = [r[key] if key else r["prompt_ids"] + r["gen_ids"] for r in batch]
        x = torch.zeros((len(seqs), max(len(s) for s in seqs)), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
        h = backbone(input_ids=x, use_cache=False).last_hidden_state
        return [h[i, : len(s)] for i, s in enumerate(seqs)]

    def lps_of(r: dict[str, Any], h, entropy: bool = False):
        x = torch.tensor(r["prompt_ids"] + r["gen_ids"], dtype=torch.long, device=dev)
        return answer_logprobs(h, head, x, len(r["prompt_ids"]), entropy)

    # 3. the probe: server against trainer, and the drift from the SFT model
    t_train = time.time()
    model.eval()
    probe = [r for r in rollouts if r.get("behavior_lp") and n_tokens(r) <= cfg["max_seq"]]
    acc = {"mis": 0.0, "kl_srv": 0.0, "kl_tr": 0.0, "n": 0.0}
    with torch.no_grad():
        for batch in pack(probe, n_tokens, budget):
            on = [lps_of(r, h)[0].float() for r, h in zip(batch, hidden(batch), strict=True)]
            with model.disable_adapter():
                off = [lps_of(r, h)[0].float() for r, h in zip(batch, hidden(batch), strict=True)]
            for r, a, b in zip(batch, on, off, strict=True):
                known = torch.tensor([v is not None for v in r["behavior_lp"]], device=dev)
                beh = torch.tensor([v if v is not None else 0.0 for v in r["behavior_lp"]],
                                   device=dev)  # fmt: skip
                m = known.float() * torch.tensor(r["gen_mask"], device=dev).float()
                acc["mis"] += float(((a - beh).abs() * m).sum())
                acc["kl_srv"] += float(((beh - b) * m).sum())
                acc["kl_tr"] += float(((a - b) * m).sum())
                acc["n"] += float(m.sum())

    # the EMA teacher's view of every distilled rollout, before any update
    sd_stats: dict[str, Any] = {}
    ema = None
    if sd:
        student0 = trainable_state(model)
        ema_file = prev / "ema_state.safetensors"
        ema = load_file(str(ema_file)) if k > 1 and ema_file.exists() else student0
        by_qid_g = {(r["qid"], r["g"]): r for r in rollouts}
        rows_by_id = {row["id"]: row for row in rows}
        sdpo = [r for r in trainable if r["route"] == "sdpo"]
        for r in sdpo:
            sib = by_qid_g[(r["qid"], r["teacher_g"])]
            r["_t_ids"] = (
                srpo.teacher_prompt(tok, rows_by_id[r["qid"]], srpo.solution_of(sib["text"]))
                + r["gen_ids"]
            )
        long_t = [r for r in sdpo if len(r["_t_ids"]) > cfg["max_seq"]]
        too_long += long_t
        trainable = [r for r in trainable if r not in long_t]
        sdpo = [r for r in sdpo if r not in long_t]
        load_trainable(model, ema)
        try:
            w_sum, n_tok = 0.0, 0
            with torch.no_grad():
                for batch in pack(sdpo, lambda r: len(r["_t_ids"]), budget):
                    for r, h in zip(batch, hidden(batch, "_t_ids"), strict=True):
                        start = len(r["_t_ids"]) - len(r["gen_ids"])
                        ht = h[start - 1 : len(r["_t_ids"]) - 1]
                        ent = torch.cat([_chunk_entropy(ht[s : s + 512], head)
                                         for s in range(0, ht.shape[0], 512)])  # fmt: skip
                        w = torch.exp(-sd["entropy_beta"] * ent) * torch.tensor(
                            r["gen_mask"], device=dev).float()  # fmt: skip
                        r["_t_h"], r["_t_w"] = ht.cpu(), w.cpu()
                        w_sum += float(w.sum())
                        n_tok += int(sum(r["gen_mask"]))
        finally:
            load_trainable(model, student0)
        mean_w = w_sum / n_tok if n_tok else 1.0
        for r in sdpo:  # normalise to mean 1 over the step's distilled tokens
            r["_t_w"] = r["_t_w"] / max(mean_w, 1e-12)
            del r["_t_ids"]
        first = rollouts[0]
        sd_stats = {"sdpo_rollouts": len(sdpo), "sdpo_tokens": n_tok,
                    "teacher_weight_mean": mean_w,
                    "template_match": srpo.student_prompt_matches(
                        tok, rows_by_id[first["qid"]], first["prompt_ids"])}  # fmt: skip

    # 4. the update
    batches = minibatches(trainable, cfg["minibatches"], cfg["seed"] * 100_000 + k)
    with torch.no_grad():  # "old" for the later minibatches: the policy before any update
        later = [r for b in batches[1:] for r in b if r.get("route") != "sdpo"]
        for batch in pack(later, n_tokens, budget):
            for r, h in zip(batch, hidden(batch), strict=True):
                r["_old"] = lps_of(r, h)[0]
    model.train()
    tot = {"loss": 0.0, "tokens": 0.0, "clipped": 0.0, "ent_sum": 0.0, "ent_n": 0.0,
           "jsd_sum": 0.0, "jsd_n": 0.0}  # fmt: skip
    grad_norms = []
    for mb in batches:
        opt.zero_grad(set_to_none=True)
        norm = norm_for(mcfg["agg"], mb, cfg)
        for batch in pack(mb, n_tokens, budget):
            loss_b = None
            for r, h in zip(batch, hidden(batch), strict=True):
                start, end = len(r["prompt_ids"]), n_tokens(r)
                mask = torch.tensor(r["gen_mask"], device=dev)
                if r.get("route") == "sdpo":
                    hs = h[start - 1 : end - 1]
                    jsd = topk_jsd(hs, r.pop("_t_h").to(dev, hs.dtype), head, sd["top_k"])
                    loss = (jsd * r.pop("_t_w").to(dev)).sum() / norm
                    tot["jsd_sum"] += float((jsd.detach() * mask.float()).sum())
                    tot["jsd_n"] += float(mask.sum())
                else:
                    lp, ent = lps_of(r, h, entropy=True)
                    old = r.pop("_old", None)
                    old = lp.detach() if old is None else old
                    loss, st = sequence_loss(mcfg, lp, old, mask, r["adv"], norm, None)
                    for key in ("tokens", "clipped"):
                        tot[key] += st[key]
                    tot["ent_sum"] += float((ent * mask.float()).sum())
                    tot["ent_n"] += float(mask.sum())
                tot["loss"] += float(loss.detach())
                loss_b = loss if loss_b is None else loss_b + loss
            loss_b.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])))
        opt.step()
    model.eval()
    train_s = time.time() - t_train

    # 5. save the LoRA and the next step's rollout model
    sdir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(sdir / "lora_state.safetensors"))
    torch.save(opt.state_dict(), sdir / "optim.pt")
    if sd:  # the teacher follows the student slowly
        save_file(srpo.ema_update(ema, trainable_state(model), sd["ema"]),
                  str(sdir / "ema_state.safetensors"))  # fmt: skip
    t_merge = time.time()
    merge: dict[str, Any] = {}
    if cfg.get("rollout_model") == "merged":
        out = rollout_model_path(cfg["run"])
        if not out.exists():  # normally copied by the orchestrator (faster from Windows)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.model_gguf, out)
        if k == 1:  # the file is still the released model: prove the map and the quantiser
            check = gguf_merge.check_template(model, p.model_gguf)
            merge["template_equal_share"] = check["equal_share"]
            if check["equal_share"] < 0.99 or check["max_diff"] > 1:
                raise RuntimeError(f"the trainer's weights don't reproduce the GGUF: {check}")
        merge["merge_tensors"] = gguf_merge.write_merged(model, out, k)["tensors"]
    keep = k in cfg["eval_steps"] or k == cfg["steps"] or cfg.get("rollout_model") != "merged"
    if keep:
        convert_gguf(save_adapter(model, sdir / "adapter"), sdir / "adapter.gguf")
    merge_s = time.time() - t_merge
    if k > 2:  # the optimizer state (250 MB) and the teacher of the last two steps only
        old_dir = rcfg.step_dir(run_dir, k - 2)
        (old_dir / "optim.pt").unlink(missing_ok=True)
        if (k - 2) not in cfg["eval_steps"]:
            (old_dir / "ema_state.safetensors").unlink(missing_ok=True)

    toks = max(tot["tokens"], 1.0)
    meta = sdir / "rollout_meta.json"
    rmeta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    n_probe = max(acc["n"], 1.0)
    metrics = {
        "step": k,
        **stats,
        "trained_seqs": len(trainable),
        "skipped_too_long": len(too_long),
        "loss": tot["loss"],
        "entropy": tot["ent_sum"] / max(tot["ent_n"], 1.0),
        "distill_jsd": tot["jsd_sum"] / tot["jsd_n"] if tot["jsd_n"] else None,
        **sd_stats,
        "clip_frac": tot["clipped"] / toks,
        "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "lr": lr,
        "resp_tokens": sum(len(r["gen_ids"]) for r in rollouts) / len(rollouts),
        "think_forced_frac": sum(bool(r.get("forced")) for r in rollouts) / len(rollouts),
        "mismatch": acc["mis"] / n_probe if acc["n"] else None,
        "kl_sft": acc["kl_srv"] / n_probe if acc["n"] else None,
        "kl_sft_trainer": acc["kl_tr"] / n_probe if acc["n"] else None,
        "probe_tokens": acc["n"],
        **canaries.text_canaries(rollouts),
        **merge,
        "rollout_s": rmeta.get("seconds"),
        "rollout_tok_s": rmeta.get("tok_s"),
        "serve_s": rmeta.get("serve_s"),
        "score_s": round(score_s, 1),
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "merge_s": round(merge_s, 1),
        "update_s": round(time.time() - t_all, 1),
        "step_s": round(
            (rmeta.get("seconds") or 0) + (rmeta.get("serve_s") or 0) + time.time() - t_all, 1
        ),  # fmt: skip
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
    try:
        id_file = run_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else None
        tags = ["run1", f"method:{cfg['method']}", f"launched-by:{who}"] + (
            ["smoke"] if cfg.get("smoke") else [])  # fmt: skip
        run = wandb.init(project=WANDB_PROJECT, name=cfg["run"], id=run_id, resume="allow",
                         group="run1", job_type="train", tags=tags, dir=str(run_dir),
                         config={k: v for k, v in cfg.items() if k != "run"})  # fmt: skip
        if run_id is None:
            id_file.write_text(run.id, encoding="utf-8")
        flat = {
            f"train/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "step"
        }
        for s, v in metrics.get("reward_by_set", {}).items():
            flat[f"reward/{s}"] = v
        for s, v in metrics.get("solve_by_set", {}).items():
            flat[f"solve/{s}"] = v
        run.log(flat, step=metrics["step"])
        run.finish()
    except Exception as e:  # W&B is a copy; train_log.jsonl is the record
        print(f"W&B logging skipped: {type(e).__name__}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--launched-by", default="agent")
    ap.add_argument("--score-only", action="store_true", help="only write scored.jsonl")
    a = ap.parse_args(argv)
    if a.score_only:
        return score_only(Path(a.run_dir), a.step)
    m = train(Path(a.run_dir), a.step, a.launched_by)
    print(
        json.dumps(
            {
                k: m.get(k)
                for k in (
                    "step",
                    "reward_mean",
                    "solve_rate",
                    "informative_frac",
                    "trained_seqs",
                    "mismatch",
                    "kl_sft",
                    "train_s",
                    "update_s",
                    "peak_gb",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
