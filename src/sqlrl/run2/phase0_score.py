"""Scoring for phase 0 (inside WSL, CPU):

    python -m sqlrl.run2.phase0_score anchor --dir <dir> --ref <scores_greedy.json>
    python -m sqlrl.run2.phase0_score episodes --file <episodes.jsonl> --rows evalset|dev
    python -m sqlrl.run2.phase0_score lookcheck --rows dev|evalset     # no model: references
    python -m sqlrl.run2.phase0_score firstresult --dir <scored evaluation folder>

`anchor` answers two questions before any number with turns is trusted:

1. **Is the start model the model that scored 54.5?** The evaluation set answered greedily
   through the old path (the chat endpoint, as every yardstick so far) on the freshly merged
   start model, against the kept evaluation's verdicts, question by question.
2. **Does the new engine measure the same thing?** The engine's first query (the original
   prompt, greedy) against the same verdicts.

`episodes` scores an episode file (`scoring.score_episodes`) and writes `<file>.scored.jsonl`
and `<file>.summary.json` (`summary.summarise`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths
from sqlrl.run2 import pool, scoring, summary


def _rows(kind: str) -> dict[str, dict[str, Any]]:
    if kind == "evalset":
        return {r["id"]: r for r in pool.read_jsonl(get_paths().evalset)}
    cfg_pool = "pool-v2"
    return {r["id"]: r for rs in pool.load_pool(cfg_pool).values() for r in rs}


def agreement(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two sets of verdicts ({id: True / False / None}) question by question."""
    ids = sorted(set(a) & set(b))
    both = [i for i in ids if a[i] is not None and b[i] is not None]
    return {
        "questions": len(ids),
        "both_decided": len(both),
        "agree": sum(a[i] == b[i] for i in both),
        "a_right_b_wrong": sum(a[i] is True and b[i] is False for i in both),
        "a_wrong_b_right": sum(a[i] is False and b[i] is True for i in both),
        "decided_only_in_a": sum(a[i] is not None and b[i] is None for i in ids),
        "decided_only_in_b": sum(a[i] is None and b[i] is not None for i in ids),
        "a": _acc(a, ids),
        "b": _acc(b, ids),
    }


def _acc(v: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    right = sum(v[i] is True for i in ids)
    wrong = sum(v[i] is False for i in ids)
    return {"right": right, "wrong": wrong, "undecided": len(ids) - right - wrong,
            "decided": right / (right + wrong) if right + wrong else None}  # fmt: skip


def anchor(d: Path, ref: Path) -> dict[str, Any]:
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import publiceval
    from asktrainer.data.schema import read_jsonl
    from asktrainer.settings import get_paths as trainer_paths

    kept = {i["id"]: i.get("exec_ok") for i in json.loads(ref.read_text("utf-8"))["items"]}
    out: dict[str, Any] = {"ref": ref.parent.name}
    chat = d / "preds_chat.jsonl"
    if chat.exists():
        preds = [json.loads(x) for x in chat.read_text("utf-8").splitlines() if x.strip()]
        res = publiceval.verify(
            list(read_jsonl(p.evalset)),
            preds,
            publiceval.PublicData(trainer_paths().public),
            checker=4,
        )
        (d / "scores_chat.json").write_text(json.dumps(res, indent=1, default=str), "utf-8")
        out["old_path_vs_kept"] = agreement({i["id"]: i.get("exec_ok") for i in res["items"]}, kept)
    epf = d / "episodes.jsonl"
    if epf.exists():
        eps = [e for e in scoring.read_episodes(epf) if "error" not in e]
        scoring.score_episodes(eps, _rows("evalset"), dup_probe=False)
        scoring.save(eps, d / "episodes.scored.jsonl")
        # a first query that never came (no call) is undecided in the old count too
        first = {ep["qid"]: ep.get("first_ok") for ep in eps}
        out["engine_first_vs_kept"] = agreement(first, kept)
        if chat.exists():
            out["engine_first_vs_old_path"] = agreement(
                first, {i["id"]: i.get("exec_ok") for i in res["items"]})  # fmt: skip
        out["engine"] = summary.headline(summary.summarise(eps))
    (d / "anchor.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def score_file(path: Path, rows_kind: str, dup_probe: bool = True) -> dict[str, Any]:
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    every = scoring.read_episodes(path)
    eps = [e for e in every if "error" not in e]
    scored = path.with_suffix(".scored.jsonl")
    if not scoring.load(eps, scored):
        scoring.score_episodes(eps, _rows(rows_kind), dup_probe=dup_probe)
        scoring.save(eps, scored)
    s = summary.summarise(eps)
    s["episodes"] = len(eps)
    s["failed"] = len(every) - len(eps)
    path.with_suffix(".summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def lookcheck(rows_kind: str = "dev", per_set: int = 50) -> dict[str, Any]:
    """Does a correct query's result say something on the look-around copy? The reference
    queries of the dev slice (`rows_kind` "dev") or of the evaluation set ("evalset") are run on
    the copies the model would look at. This is a diagnostic that only the scoring side may
    run: the builder itself never sees a reference. Where the reference's result says nothing
    (`0 rows`, or only NULL and zero: `summary.says_something`), a model that wrote the right
    query is shown nothing and is nudged to change it.

    **Run it on both.** The pool keeps a question only when its reference's result says
    something to the checker, and the dev slice is cut from the pool, so the dev slice passes
    almost by construction; the evaluation set has no such filter (results/run2-phase0.md,
    section 8). Nothing else may use the Neo4j while this runs: it loads graphs."""
    from collections import Counter

    from sqlrl.run2 import env, guards

    if rows_kind == "evalset":
        rows = pool.read_jsonl(get_paths().evalset)
        scripts = pool.load_scripts("evalset")
    else:
        rows = pool.dev_slice(pool.load_pool("pool-v2"), per_set)
        scripts = pool.load_scripts("pool-v2")
    graphs = env.GraphHost()
    out: dict[str, Any] = {}
    detail = []
    for row in sorted(rows, key=lambda r: (r["origin"], (r.get("meta") or {}).get("db_id") or "")):
        pub = env.public_row(row, scripts.get(row["id"]))
        gold = row["meta"]["gold"]
        if pub["dialect"] == "cypher":
            graphs.ensure(pub)
        look = env.Look(pub, graphs)
        try:
            res = look.run(gold)
        finally:
            look.close()
        lits = guards.query_literals(gold, row["dialect"])
        asked = scoring.prompt_text(row).lower()
        says = summary.says_something(res.rows)
        c = out.setdefault(row["origin"], Counter())
        c[res.kind] += 1
        c["questions"] += 1
        c["says_something"] += says
        c["ref_literals"] += len(lits)
        c["ref_literals_in_prompt"] += sum(s.lower() in asked for s in lits)
        detail.append({"id": row["id"], "origin": row["origin"], "kind": res.kind,
                       "rows": res.total, "says_something": says, "error": res.error})  # fmt: skip
    rep = {s: dict(c) for s, c in sorted(out.items())}
    d = pool.data_dir() / "phase0"
    d.mkdir(parents=True, exist_ok=True)
    name = "lookcheck.json" if rows_kind == "dev" else f"lookcheck-{rows_kind}.json"
    (d / name).write_text(json.dumps({"by_set": rep, "questions": detail}, indent=1),
                          encoding="utf-8")  # fmt: skip
    return rep


def first_result(d: Path) -> dict[str, Any]:
    """An evaluation folder's episodes split by what the first query's result showed
    (`summary.first_result`): greedy, the sampled episodes, and greedy per dataset. Needs the
    folder's `scored.jsonl` (`eval_score`)."""
    eps = [e for e in scoring.read_episodes(d / "episodes.jsonl") if "error" not in e]
    if not scoring.load(eps, d / "scored.jsonl"):
        raise RuntimeError(f"{d.name} isn't scored yet: run eval_score on it first")
    greedy = [e for e in eps if e["g"] == 0]
    out = {"greedy": summary.first_result(greedy),
           "by_set": {s: summary.first_result([e for e in greedy if e["origin"] == s])
                      for s in sorted({e["origin"] for e in greedy})}}  # fmt: skip
    sampled = [e for e in eps if e["g"] > 0]
    if sampled:
        out["sampled"] = summary.first_result(sampled)
    (d / "first_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def offset(path: Path, init: str, batch: int = 32) -> dict[str, Any]:
    """How far apart the server and the trainer read the *same* policy (GPU). The episodes were
    sampled from the start adapter merged into the rollout model; here the trainer loads that
    same adapter and scores the same tokens. The true drift is zero, so the mean of (server
    log-prob - trainer log-prob) on first-turn tokens is the offset the run's drift reading
    carries. Given overall, per dataset, and per batch of `batch` episodes (a step's probe is 32):
    a stable offset can be subtracted; a noisy one can't."""
    import random
    from statistics import mean, pstdev

    import torch

    from sqlrl.run2 import config as rcfg
    from sqlrl.run2.logprobs import sequence_logprobs
    from sqlrl.run2.model import load_policy, start_state
    from sqlrl.run2.train_step import turn_masks

    cfg = rcfg.new_config({"init_from": init})
    eps = [e for e in scoring.read_episodes(path) if "error" not in e and e.get("behavior_lp")
           and len(e["prompt_ids"]) + len(e["gen_ids"]) <= cfg["max_seq"]]  # fmt: skip
    model, _tok, backbone, head = load_policy(cfg, start_state(cfg), grad_checkpointing=False)
    model.eval()
    rows = []
    with torch.no_grad():
        for e in eps:
            ids, start = e["prompt_ids"] + e["gen_ids"], len(e["prompt_ids"])
            on = sequence_logprobs(model, backbone, head, ids, start)[0].float().cpu()
            with model.disable_adapter():
                off = sequence_logprobs(model, backbone, head, ids, start)[0].float().cpu()
            _, m1, m2 = turn_masks(e)
            known = torch.tensor([v is not None for v in e["behavior_lp"]]).float()
            beh = torch.tensor([v if v is not None else 0.0 for v in e["behavior_lp"]])
            k1, k2 = known * torch.tensor(m1).float(), known * torch.tensor(m2).float()
            rows.append({"origin": e["origin"],
                         "d1": float(((beh - on) * k1).sum()), "n1": float(k1.sum()),
                         "d2": float(((beh - on) * k2).sum()), "n2": float(k2.sum()),
                         "abs": float(((beh - on).abs() * (k1 + k2)).sum()),
                         "sft1": float(((beh - off) * k1).sum())})  # fmt: skip

    def rate(rs: list[dict[str, Any]], num: str, den: str) -> float | None:
        d = sum(r[den] for r in rs)
        return sum(r[num] for r in rs) / d if d else None

    random.Random(0).shuffle(rows)
    batches = [rows[i : i + batch] for i in range(0, len(rows) - batch + 1, batch)]
    per_batch = [rate(b, "d1", "n1") for b in batches]
    out = {
        "episodes": len(rows), "first_turn_tokens": sum(r["n1"] for r in rows),
        "offset_first_turn": rate(rows, "d1", "n1"),
        "offset_later_turns": rate(rows, "d2", "n2"),
        "mismatch": sum(r["abs"] for r in rows) / max(sum(r["n1"] + r["n2"] for r in rows), 1),
        "kl_sft_first_turn": rate(rows, "sft1", "n1"),
        "by_set": {s: rate([r for r in rows if r["origin"] == s], "d1", "n1")
                   for s in sorted({r["origin"] for r in rows})},
        "batches": per_batch, "batch_mean": mean(per_batch) if per_batch else None,
        "batch_sd": pstdev(per_batch) if len(per_batch) > 1 else None,
    }  # fmt: skip
    path.with_suffix(".offset.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offset")
    o.add_argument("--file", required=True)
    o.add_argument("--init", default="run1b-step30")
    a = sub.add_parser("anchor")
    a.add_argument("--dir", required=True)
    a.add_argument("--ref", required=True)
    e = sub.add_parser("episodes")
    e.add_argument("--file", required=True)
    e.add_argument("--rows", choices=["evalset", "dev"], required=True)
    lc = sub.add_parser("lookcheck")
    lc.add_argument("--rows", choices=["evalset", "dev"], default="dev")
    fr = sub.add_parser("firstresult")
    fr.add_argument("--dir", required=True)
    x = ap.parse_args(argv)
    if x.cmd == "offset":
        sys.path.insert(0, str(get_paths().trainer_repo / "src"))
        print(json.dumps(offset(Path(x.file), x.init)))
    elif x.cmd == "anchor":
        print(json.dumps(anchor(Path(x.dir), Path(x.ref))))
    elif x.cmd == "lookcheck":
        sys.path.insert(0, str(get_paths().trainer_repo / "src"))
        print(json.dumps(lookcheck(x.rows)))
    elif x.cmd == "firstresult":
        print(json.dumps(first_result(Path(x.dir))))
    else:
        print(json.dumps(summary.headline(score_file(Path(x.file), x.rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
