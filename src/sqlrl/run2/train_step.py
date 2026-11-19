"""One RL step of run 2 (inside WSL, with the trainer's venv; one process per step, so the GPU
is free again for the rollout server when it exits):

    python -m sqlrl.run2.train_step --run-dir <dir> --step <k>

Run 1's step with an episode in place of a single answer (decisions R15):

1. **Score and load at the same time.** The checker (`scoring.score_episodes`) scores each
   episode's final and first query in a process of its own (`--score-only`) while this one loads
   the model.
2. **Rewards, advantages, routing** (`rewards.py`, `losses.route`).
3. **The probe** (before any update), on the episodes that carry the server's log-probabilities:
   the server against the trainer (`mismatch`); the drift from the **start adapter** on
   first-turn tokens as the server measures it (`kl_start`, less the reading at step 1, where
   the true drift is zero); the drift from the SFT model (`kl_sft`, for the record); and the
   token entropy per turn.
4. **The update.** One sequence per episode: the prompt, then every turn and every result in
   order. `gen_mask` is 0 on the result blocks, so they are read but never trained. Three terms
   join one token-level average:
   - the reward update (SRPO's GRPO branch) on the model's tokens;
   - for a wrong episode with a right sibling, self-distillation on its **first turn** (its
     later turns take the reward update);
   - the penalty: `beta` x KL(policy || start adapter) on **first-turn** tokens, a fixed weight.
     The first turn's context is the single-turn task, the skill to keep; later turns react to
     results the start model never saw, so they are left free.
5. **Save:** as run 1 (the LoRA, the optimizer state, the EMA teacher, each checked byte for
   byte; the LoRA merged into the rollout model for step k + 1).

A finished step writes `TRAINED`; running it again does nothing. Scoring is kept when the update
fails, so a re-run starts at the update.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths
from sqlrl.run2 import canaries, rewards, scoring, summary
from sqlrl.run2 import config as rcfg
from sqlrl.run2.integrity import mismatched, save_verified
from sqlrl.run2.pool import load_pool

WANDB_PROJECT = "sql-cypher-rl"


def read_rollouts(path: Path, cfg: dict[str, Any], rows: list[dict]) -> list[dict[str, Any]]:
    """The step's episodes, one per (question, g)."""
    out = {(r["qid"], r["g"]): r for r in scoring.read_episodes(path)}
    want = {(row["id"], g) for row in rows for g in range(cfg["group_size"])}
    missing = want - {k for k, r in out.items() if "error" not in r}
    if missing:
        raise RuntimeError(f"{len(missing)} episodes missing or failed (e.g. {sorted(missing)[0]})")
    return [out[k] for k in sorted(want)]


def score(sdir: Path, eps: list[dict], rows: list[dict], cfg: dict[str, Any]) -> float:
    """Checker verdicts into the episodes (cached in `scored.jsonl`); returns seconds spent."""
    cache = sdir / "scored.jsonl"
    if scoring.load(eps, cache):
        return 0.0
    secs = scoring.score_episodes(eps, {r["id"]: r for r in rows}, checker=cfg["checker"],
                                  dup_probe=bool(cfg.get("dup_probe")),
                                  log=lambda m: print(m, flush=True),
                                  guard=bool(cfg.get("returned_guard", True)))  # fmt: skip
    scoring.save(eps, cache)
    return secs


def score_only(run_dir: Path, k: int) -> int:
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
    """Batches whose padded size stays within `budget` tokens (<= 0: one member per batch)."""
    order = sorted(items, key=length, reverse=True)
    out: list[list[Any]] = []
    for it in order:
        if out and budget > 0 and (len(out[-1]) + 1) * length(out[-1][0]) <= budget:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def apply_minibatch(opt: Any, params: list[Any], grad_clip: float, loss: float) -> float | None:
    """One minibatch's optimizer step with its gradient clipped, and its gradient norm. A
    non-finite loss or gradient is dropped and the weights kept (None)."""
    import torch

    gn = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
    if not (math.isfinite(gn) and math.isfinite(loss)):
        opt.zero_grad(set_to_none=True)
        return None
    opt.step()
    return gn


def check_finite(params: list[Any]) -> None:
    import torch

    bad = sum(int((~torch.isfinite(q.detach())).sum()) for q in params)
    if bad:
        raise RuntimeError(f"{bad} weights aren't finite after the update: nothing was saved")


def convert_gguf(adapter_dir: Path, out: Path) -> None:
    p = get_paths()
    src = p.llama_cpp_src
    cmd = [sys.executable, str(src / "convert_lora_to_gguf.py"), "--base", str(p.model_hf),
           "--outtype", "f16", "--outfile", str(out), str(adapter_dir)]  # fmt: skip
    env = {"PYTHONPATH": str(src / "gguf-py"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(src))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"LoRA -> GGUF failed: {r.stderr[-500:]}")


def turn_masks(ep: dict[str, Any]) -> tuple[int, list[int], list[int]]:
    """(the first turn's token count, the mask of first-turn model tokens, the mask of later
    turns' model tokens), both as long as `gen_ids`. Result blocks are 0 in both."""
    n1 = canaries.turn1_tokens(ep)
    gm = ep["gen_mask"]
    return (
        n1,
        [m if i < n1 else 0 for i, m in enumerate(gm)],
        [m if i >= n1 else 0 for i, m in enumerate(gm)],
    )


def keep_steps(cfg: dict[str, Any]) -> set[int]:
    """Steps whose adapter is also saved as a folder and a GGUF, and whose teacher is kept."""
    return set(cfg["eval_steps"]) | set(cfg.get("quick_eval_steps") or []) | {cfg["steps"]}


def train(run_dir: Path, k: int, launched_by: str = "agent") -> dict[str, Any]:
    cfg = rcfg.load(run_dir)
    sdir = rcfg.step_dir(run_dir, k)
    if (sdir / "TRAINED").exists():
        return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from sqlrl.run2.model import init_dir, load_policy, rollout_model_path, start_state

    t_all = time.time()
    rows = rcfg.step_questions(load_pool(cfg["pool"]), cfg, k)
    eps = read_rollouts(sdir / "rollouts.jsonl", cfg, rows)
    prev = rcfg.step_dir(run_dir, k - 1)
    bad = mismatched(prev) if k > 1 else []
    if bad:  # never train on from a damaged state: step k-1 has to be trained again first
        raise RuntimeError(f"step {k - 1}'s {', '.join(bad)} differ from what was written")
    start_file = start_state(cfg)
    state = prev / "lora_state.safetensors" if k > 1 else start_file
    scorer = subprocess.Popen([sys.executable, "-m", "sqlrl.run2.train_step", "--run-dir",
                               str(run_dir), "--step", str(k), "--score-only"])  # fmt: skip
    try:
        model, tok, backbone, head = load_policy(cfg, state)
        load_s = time.time() - t_all
        if scorer.wait() != 0:
            raise RuntimeError("scoring the episodes failed (see above)")
    finally:
        if scorer.poll() is None:
            scorer.kill()
    score(sdir, eps, rows, cfg)  # reads the scorer's verdicts
    score_s = json.loads((sdir / "score_meta.json").read_text(encoding="utf-8"))["seconds"]

    import torch
    from asktrainer.checkpoint import load_trainable, trainable_state
    from asktrainer.modelio import save_adapter
    from safetensors.torch import load_file
    from safetensors.torch import save as st_bytes

    from sqlrl.run2 import gguf_merge, srpo
    from sqlrl.run2.logprobs import _chunk_entropy, answer_logprobs
    from sqlrl.run2.losses import norm_for, route, sequence_loss, topk_jsd, topk_kl

    for r in eps:
        r["reward"] = rewards.reward(r, cfg)
    mcfg = cfg["method_cfg"]
    rewards.advantages(eps, mcfg["adv"], std_floor=cfg.get("adv_std_floor", 0.0))
    stats = rewards.group_stats(eps, cfg)
    sd = mcfg.get("sdpo")  # SRPO: the self-distillation branch's settings
    if sd:
        route(eps)
        trainable = [r for r in eps if r["route"] == "sdpo"
                     or (r["route"] == "grpo" and r["adv"] not in (None, 0.0))]  # fmt: skip
    else:
        for r in eps:
            r["route"] = "grpo" if r["reward"] is not None else None
        trainable = [r for r in eps if r["adv"] not in (None, 0.0)]

    def n_tokens(r: dict[str, Any]) -> int:
        return len(r["prompt_ids"]) + len(r["gen_ids"])

    too_long = [r for r in trainable if n_tokens(r) > cfg["max_seq"]]
    trainable = [r for r in trainable if n_tokens(r) <= cfg["max_seq"]]
    for r in eps:
        r["_n1"], r["_m1"], r["_m2"] = turn_masks(r)

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])  # fmt: skip
    if k > 1 and (prev / "optim.pt").exists():
        opt.load_state_dict(torch.load(prev / "optim.pt", map_location="cuda"))
    lr = rcfg.lr_at(cfg, k)
    for group in opt.param_groups:
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

    student0 = trainable_state(model)
    start_sd = load_file(str(start_file)) if start_file is not None else None

    @contextlib.contextmanager
    def swapped(state_dict: dict[str, Any] | None):
        """The model with another LoRA state for a while (None: the adapter switched off, which
        is the SFT model), and the student's own state back afterwards."""
        if state_dict is None:
            with model.disable_adapter():
                yield
            return
        load_trainable(model, state_dict)
        try:
            yield
        finally:
            load_trainable(model, student0)

    # 3. the probe: server against trainer, the drift from the start and from SFT, the entropy
    t_train = time.time()
    model.eval()
    probe = [r for r in eps if r.get("behavior_lp") and n_tokens(r) <= cfg["max_seq"]]
    acc = dict.fromkeys(("mis", "n", "ks1", "n1", "ks2", "n2", "ksft", "ktr1", "e1", "en1",
                         "e2", "en2"), 0.0)  # fmt: skip
    with torch.no_grad():
        for batch in pack(probe, n_tokens, budget):
            for r, h in zip(batch, hidden(batch), strict=True):
                lp, ent = lps_of(r, h, entropy=True)
                r["_on"], r["_ent"] = lp.float().cpu(), ent.float().cpu()
        with swapped(start_sd):
            for batch in pack(probe, n_tokens, budget):
                for r, h in zip(batch, hidden(batch), strict=True):
                    r["_ref"] = lps_of(r, h)[0].float().cpu()
        if start_sd is not None:
            with swapped(None):
                for batch in pack(probe, n_tokens, budget):
                    for r, h in zip(batch, hidden(batch), strict=True):
                        r["_off"] = lps_of(r, h)[0].float().cpu()
    for r in probe:
        on, ref, ent = r.pop("_on"), r.pop("_ref"), r.pop("_ent")
        off = r.pop("_off", ref)
        known = torch.tensor([v is not None for v in r["behavior_lp"]]).float()
        beh = torch.tensor([v if v is not None else 0.0 for v in r["behavior_lp"]])
        m1, m2 = torch.tensor(r["_m1"]).float(), torch.tensor(r["_m2"]).float()
        k1, k2 = known * m1, known * m2
        acc["mis"] += float(((on - beh).abs() * (k1 + k2)).sum())
        acc["n"] += float((k1 + k2).sum())
        acc["ks1"] += float(((beh - ref) * k1).sum())
        acc["n1"] += float(k1.sum())
        acc["ks2"] += float(((beh - ref) * k2).sum())
        acc["n2"] += float(k2.sum())
        acc["ksft"] += float(((beh - off) * (k1 + k2)).sum())
        acc["ktr1"] += float(((on - ref) * k1).sum())
        acc["e1"] += float((ent * m1).sum())
        acc["en1"] += float(m1.sum())
        acc["e2"] += float((ent * m2).sum())
        acc["en2"] += float(m2.sum())
    kl_raw = acc["ks1"] / acc["n1"] if acc["n1"] else None
    # At step 1 the policy is the start adapter, so the true drift is zero and the reading is
    # the instrument's own offset (the server's Q8_0 merge against the trainer's bf16 adapter).
    # The offset that is subtracted is the one measured on many episodes before the run
    # (`kl.offset`; `phase0.py offset`); one step's reading swings by about +-0.0035 around it.
    # Step 1's own reading is kept as a check that the run started from the model it meant to
    off_file = run_dir / "kl_offset.json"
    if k == 1 and kl_raw is not None:
        off_file.write_text(json.dumps({"step": 1, "offset": kl_raw}), encoding="utf-8")
    kl_step1 = (json.loads(off_file.read_text(encoding="utf-8"))["offset"]
                if off_file.exists() else None)  # fmt: skip
    kl_cfg = mcfg.get("kl") or {}
    measured = kl_cfg.get("offset") if kl_cfg.get("offset_for") == cfg.get("init_from") else None
    kl_offset = measured if measured is not None else kl_step1  # another start: its own step 1

    # the EMA teacher's view of every distilled first turn, before any update
    sd_stats: dict[str, Any] = {}
    ema = None
    rows_by_id = {row["id"]: row for row in rows}
    if sd:
        ema_file = prev / "ema_state.safetensors"
        init_ema = init_dir(cfg) / "ema_state.safetensors" if init_dir(cfg) else None
        if k > 1 and ema_file.exists():
            ema = load_file(str(ema_file))
        elif k == 1 and init_ema is not None and init_ema.exists():
            ema = load_file(str(init_ema))  # the start adapter's own teacher
        else:
            ema = student0
        by_qid_g = {(r["qid"], r["g"]): r for r in eps}
        sdpo = [r for r in trainable if r["route"] == "sdpo"]
        for r in sdpo:
            sib = by_qid_g[(r["qid"], r["teacher_g"])]
            r["_t_ids"] = (srpo.teacher_prompt(tok, rows_by_id[r["qid"]], sib["final_query"], cfg)
                           + r["gen_ids"][: r["_n1"]])  # fmt: skip
        long_t = [r for r in sdpo if len(r["_t_ids"]) > cfg["max_seq"]]
        too_long += long_t
        trainable = [r for r in trainable if r not in long_t]
        sdpo = [r for r in sdpo if r not in long_t]
        w_sum, n_tok = 0.0, 0
        with torch.no_grad(), swapped(ema):
            for batch in pack(sdpo, lambda r: len(r["_t_ids"]), budget):
                for r, h in zip(batch, hidden(batch, "_t_ids"), strict=True):
                    n1 = r["_n1"]
                    start = len(r["_t_ids"]) - n1
                    ht = h[start - 1 : len(r["_t_ids"]) - 1]
                    ent = torch.cat([_chunk_entropy(ht[s : s + 512], head)
                                     for s in range(0, ht.shape[0], 512)])  # fmt: skip
                    w = torch.exp(-sd["entropy_beta"] * ent) * torch.tensor(
                        r["_m1"][:n1], device=dev).float()  # fmt: skip
                    r["_t_h"], r["_t_w"] = ht.cpu(), w.cpu()
                    w_sum += float(w.sum())
                    n_tok += int(sum(r["_m1"]))
        mean_w = w_sum / n_tok if n_tok else 1.0
        for r in sdpo:  # normalise to mean 1 over the step's distilled tokens
            r["_t_w"] = r["_t_w"] / max(mean_w, 1e-12)
            del r["_t_ids"]
        # the teacher's prompt is rendered here and the student's came from llama-server: they
        # must be the same text. One episode per dataset, because a dataset on run 1's protocol
        # reads another wording than the ones with turns (`prompt.protocol`)
        firsts: dict[str, dict[str, Any]] = {}
        for r in eps:
            firsts.setdefault(r["origin"], r)
        differ = [s for s, r in sorted(firsts.items()) if not srpo.student_prompt_matches(
            tok, rows_by_id[r["qid"]], r["prompt_ids"], cfg)]  # fmt: skip
        if differ:
            raise RuntimeError(
                "the trainer renders another prompt than the server for " + ", ".join(differ)
            )
        sd_stats = {"sdpo_rollouts": len(sdpo), "sdpo_tokens": n_tok,
                    "teacher_weight_mean": mean_w, "template_match": True}  # fmt: skip

    # the penalty: the start adapter's view of every trained first turn
    kcfg = mcfg.get("kl") or {}
    kl_beta = float(kcfg.get("beta") or 0.0)
    if kl_beta > 0:
        with torch.no_grad(), swapped(start_sd):
            for batch in pack(trainable, n_tokens, budget):
                for r, h in zip(batch, hidden(batch), strict=True):
                    a = len(r["prompt_ids"]) - 1
                    r["_ref_h"] = h[a : a + r["_n1"]].cpu()

    def has_reward_tokens(r: dict[str, Any]) -> bool:
        return r.get("route") != "sdpo" or sum(r["_m2"]) > 0

    # 4. the update
    batches = minibatches(trainable, cfg["minibatches"], cfg["seed"] * 100_000 + k)
    with torch.no_grad():  # "old" for the later minibatches: the policy before any update
        later = [r for b in batches[1:] for r in b if has_reward_tokens(r)]
        for batch in pack(later, n_tokens, budget):
            for r, h in zip(batch, hidden(batch), strict=True):
                r["_old"] = lps_of(r, h)[0]
    model.train()
    tot = dict.fromkeys(("loss", "tokens", "clipped", "ent_sum", "ent_n", "jsd_sum", "jsd_n",
                         "kl_sum", "kl_n"), 0.0)  # fmt: skip
    grad_norms, skipped = [], 0
    for mb in batches:
        opt.zero_grad(set_to_none=True)
        norm = norm_for(mcfg["agg"], mb, cfg)
        part = dict.fromkeys(tot, 0.0)  # this minibatch's sums: kept only if it's applied
        for batch in pack(mb, n_tokens, budget):
            loss_b = None
            for r, h in zip(batch, hidden(batch), strict=True):
                start, end, n1 = len(r["prompt_ids"]), n_tokens(r), r["_n1"]
                hs = h[start - 1 : end - 1]
                m1 = torch.tensor(r["_m1"], device=dev)
                m2 = torch.tensor(r["_m2"], device=dev)
                loss = None
                if r.get("route") == "sdpo":  # the first turn toward the teacher
                    jsd = topk_jsd(hs[:n1], r.pop("_t_h").to(dev, hs.dtype), head, sd["top_k"])
                    loss = (jsd * r.pop("_t_w").to(dev)).sum() / norm
                    part["jsd_sum"] += float((jsd.detach() * m1[:n1].float()).sum())
                    part["jsd_n"] += float(m1.sum())
                    mask = m2  # its later turns take the reward update
                else:
                    mask = m1 + m2
                if int(mask.sum()) > 0:
                    lp, ent = lps_of(r, h, entropy=True)
                    old = r.pop("_old", None)
                    old = lp.detach() if old is None else old
                    pg, st = sequence_loss(mcfg, lp, old, mask, r["adv"], norm, None)
                    loss = pg if loss is None else loss + pg
                    for key in ("tokens", "clipped"):
                        part[key] += st[key]
                    part["ent_sum"] += float((ent * mask.float()).sum())
                    part["ent_n"] += float(mask.sum())
                if kl_beta > 0:  # hold the first turn near the start adapter's distribution
                    kl = topk_kl(hs[:n1], r.pop("_ref_h").to(dev, hs.dtype), head,
                                 kcfg.get("top_k", 100))  # fmt: skip
                    pen = kl_beta * (kl * m1[:n1].float()).sum() / norm
                    loss = pen if loss is None else loss + pen
                    part["kl_sum"] += float((kl.detach() * m1[:n1].float()).sum())
                    part["kl_n"] += float(m1.sum())
                if loss is None:
                    continue
                part["loss"] += float(loss.detach())
                loss_b = loss if loss_b is None else loss_b + loss
            if loss_b is not None:
                loss_b.backward()
        gn = apply_minibatch(opt, params, cfg["grad_clip"], part["loss"])
        if gn is None:
            skipped += 1
            continue
        grad_norms.append(gn)
        for key, v in part.items():
            tot[key] += v
    model.eval()
    if batches and skipped == len(batches):
        raise RuntimeError(f"all {skipped} minibatches had a non-finite loss or gradient")
    check_finite(params)
    train_s = time.time() - t_train

    # 5. save the LoRA and the next step's rollout model, each state file checked byte for byte
    sdir.mkdir(parents=True, exist_ok=True)
    save_verified(st_bytes(trainable_state(model)), sdir / "lora_state.safetensors")
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    save_verified(buf.getvalue(), sdir / "optim.pt")
    del buf
    if sd:  # the teacher follows the student slowly
        save_verified(st_bytes(srpo.ema_update(ema, trainable_state(model), sd["ema"])),
                      sdir / "ema_state.safetensors")  # fmt: skip
    t_merge = time.time()
    merge: dict[str, Any] = {}
    if cfg.get("rollout_model") == "merged":
        out = rollout_model_path(cfg["run"])
        if not out.exists():  # normally made by the orchestrator before step 1
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.model_gguf, out)
        if k == 1:  # prove the name map and the quantiser on the released file
            check = gguf_merge.check_template(model, p.model_gguf)
            merge["template_equal_share"] = check["equal_share"]
            if check["equal_share"] < 0.99 or check["max_diff"] > 1:
                raise RuntimeError(f"the trainer's weights don't reproduce the GGUF: {check}")
        merge["merge_tensors"] = gguf_merge.write_merged(model, out, k)["tensors"]
    keeps = keep_steps(cfg)
    if k in keeps or cfg.get("rollout_model") != "merged":
        convert_gguf(save_adapter(model, sdir / "adapter"), sdir / "adapter.gguf")
    merge_s = time.time() - t_merge
    if k > 2:  # the optimizer state (250 MB) and the teacher of the last two steps only
        old_dir = rcfg.step_dir(run_dir, k - 2)
        (old_dir / "optim.pt").unlink(missing_ok=True)
        if (k - 2) not in keeps:
            (old_dir / "ema_state.safetensors").unlink(missing_ok=True)

    toks = max(tot["tokens"], 1.0)
    meta = sdir / "rollout_meta.json"
    rmeta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    blk = summary.block(eps)
    metrics = {
        "step": k,
        **stats,
        "trained_seqs": len(trainable),
        "skipped_too_long": len(too_long),
        "loss": tot["loss"],
        "entropy": tot["ent_sum"] / max(tot["ent_n"], 1.0),  # the reward-update tokens only
        "entropy_t1": acc["e1"] / acc["en1"] if acc["en1"] else None,  # the probe: unbiased
        "entropy_later": acc["e2"] / acc["en2"] if acc["en2"] else None,
        "distill_jsd": tot["jsd_sum"] / tot["jsd_n"] if tot["jsd_n"] else None,
        **sd_stats,
        "clip_frac": tot["clipped"] / toks,
        "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "skipped_minibatches": skipped,
        "lr": lr,
        "mismatch": acc["mis"] / acc["n"] if acc["n"] else None,
        # the drift from the start adapter, first-turn tokens, the server's reading
        "kl_start_raw": kl_raw,
        "kl_offset": kl_offset,
        "kl_step1": kl_step1,
        "kl_start": kl_raw - kl_offset if kl_raw is not None and kl_offset is not None else None,
        "kl_start_later": acc["ks2"] / acc["n2"] if acc["n2"] else None,
        "kl_start_trainer": acc["ktr1"] / acc["n1"] if acc["n1"] else None,
        "kl_sft": acc["ksft"] / acc["n"] if acc["n"] else None,
        "kl_beta": kl_beta if kcfg else None,
        "kl_pen": tot["kl_sum"] / tot["kl_n"] if tot["kl_n"] else None,
        "probe_tokens": acc["n"],
        # how the episodes used their turns, and the signs of hacking
        "calls_mean": blk["protocol"]["calls_mean"],
        "calls": blk["protocol"]["calls"],
        "ends": blk["protocol"]["ends"],
        "limit_frac": blk["protocol"]["limit_frac"],
        "retry_after": blk["protocol"]["retry_after"],
        "look_error_frac": blk["protocol"]["look_error_frac"],
        "fixed": blk["changes"]["fixed"],
        "broken": blk["changes"]["broken"],
        "fixed_frac": blk["changes"]["fixed_frac"],
        "broken_frac": blk["changes"]["broken_frac"],
        **blk["hacking"],
        **summary.returned_metrics(eps),
        **canaries.text_canaries(eps, cfg["max_seq"]),
        **merge,
        "rollout_s": rmeta.get("seconds"),
        "rollout_tok_s": rmeta.get("tok_s"),
        "serve_s": rmeta.get("serve_s"),
        "graph_load_s": rmeta.get("graph_load_s"),
        "score_s": round(score_s, 1),
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "merge_s": round(merge_s, 1),
        "update_s": round(time.time() - t_all, 1),
        "step_s": round(
            (rmeta.get("seconds") or 0) + (rmeta.get("serve_s") or 0) + time.time() - t_all, 1
        ),
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
        tags = ["run2", f"method:{cfg['method']}", f"launched-by:{who}"] + (
            ["smoke"] if cfg.get("smoke") else [])  # fmt: skip
        run = wandb.init(project=WANDB_PROJECT, name=cfg["run"], id=run_id, resume="allow",
                         group="run2", job_type="train", tags=tags, dir=str(run_dir),
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
        for kind, v in (metrics.get("retry_after") or {}).items():
            flat[f"retry_after/{kind}"] = v["again"]
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
    keys = ("step", "reward_mean", "solve_rate", "first_solve_rate", "informative_frac",
            "trained_seqs", "mismatch", "kl_start", "calls_mean", "train_s", "update_s",
            "peak_gb")  # fmt: skip
    print(json.dumps({k: m.get(k) for k in keys}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
