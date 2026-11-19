"""Score an evaluation with turns and compare it with its references (inside WSL):

    python -m sqlrl.run2.eval_score --dir <eval dir> [--run-dir <run dir> --step <k>]
        [--ref name=<dir> ...] [--rows evalset|dev]

An evaluation is an episode file (`episodes.jsonl`): episode 0 of each question is greedy,
episodes 1-8 (a full evaluation) are sampled. `summary.json` holds:

- `greedy`: `summary.summarise` of the greedy episodes. **`final.decided` is the headline:
  greedy with turns**, counted as every earlier number was (undecided and no-query answers left
  out); `final.strict` counts them wrong; `first` is the forced single-query number; `changes`
  says what the turns fixed and broke.
- `sampled` (full only): `passk.summarise` over the sampled episodes: mean@8 and pass@8 with
  turns, the headroom left.
- `vs`: paired sign tests against each reference, question by question on the questions both
  decided: `greedy` (the final greedy answer right for one, wrong for the other), and, when both
  have samples, `sampled` and `pass8` (as `run1.eval_score`).
- `questions`: the per-question verdicts, so a later evaluation can be paired with this one.

A reference is either an evaluation in this format (its `summary.json`) or one of run 1's
(`scores_greedy.json` and `summary.json`, single turn). With `--run-dir` the run's `BEST.json`
is rewritten: the step to keep is the highest greedy with turns (decisions R14: greedy picks).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Any

from sqlrl import passk
from sqlrl.paths import get_paths
from sqlrl.run2 import pool, scoring, summary

SAMPLES = 8


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial p for `better` vs `worse` (ties dropped), p = 1/2."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def _test(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    better = sum(a > b for a, b in pairs)
    worse = sum(a < b for a, b in pairs)
    return {"better": better, "worse": worse, "p": sign_test(better, worse)}


def compare(mine: dict[str, dict[str, Any]], ref: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sign tests between two evaluations' per-question records ({id: {"final", "right",
    "scored"}}; `final` is the greedy verdict)."""
    ids = sorted(set(mine) & set(ref))
    greedy = [(float(mine[i]["final"]), float(ref[i]["final"])) for i in ids
              if mine[i].get("final") is not None and ref[i].get("final") is not None]  # fmt: skip
    out = {"greedy": _test(greedy)}
    both = [i for i in ids if mine[i].get("scored") and ref[i].get("scored")]
    if both:
        out["sampled"] = _test([(mine[i]["right"] / mine[i]["scored"],
                                 ref[i]["right"] / ref[i]["scored"]) for i in both])  # fmt: skip
        out["pass8"] = _test([(float(mine[i]["right"] > 0), float(ref[i]["right"] > 0))
                              for i in both])  # fmt: skip
    return out


def load_ref(d: Path) -> dict[str, dict[str, Any]]:
    """A reference evaluation's per-question records, from either format."""
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    qs = {q["id"]: dict(q) for q in s.get("questions") or []}
    old = d / "scores_greedy.json"
    if old.exists():  # run 1's format: the greedy verdicts are in their own file
        for it in json.loads(old.read_text(encoding="utf-8"))["items"]:
            qs.setdefault(it["id"], {"id": it["id"]})["final"] = it.get("exec_ok")
    return qs


def questions(eps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ep in eps:
        q = out.setdefault(
            ep["qid"],
            {
                "id": ep["qid"],
                "origin": ep["origin"],
                "final": None,
                "first": None,
                "calls": None,
                "right": 0,
                "scored": 0,
            },
        )
        if ep["g"] == 0:
            q.update(final=ep.get("exec_ok"), first=ep.get("first_ok"), calls=ep.get("n_calls"))
        elif ep.get("exec_ok") is not None:
            q["scored"] += 1
            q["right"] += bool(ep["exec_ok"])
    return out


def score_dir(d: Path, rows_kind: str = "evalset", refs: dict[str, Path] | None = None,
              step: int | None = None) -> dict[str, Any]:  # fmt: skip
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    eps = [e for e in scoring.read_episodes(d / "episodes.jsonl") if "error" not in e]
    if not scoring.load(eps, d / "scored.jsonl"):
        if rows_kind == "evalset":
            rows = {r["id"]: r for r in pool.read_jsonl(p.evalset)}
        else:
            rows = {r["id"]: r for rs in pool.load_pool("pool-v2").values() for r in rs}
        # the yardstick is the checker alone: no duplicate probe (as run 1b's evaluations)
        scoring.score_episodes(eps, rows, checker=4, dup_probe=False)
        scoring.save(eps, d / "scored.jsonl")
    greedy = [e for e in eps if e["g"] == 0]
    sampled = [e for e in eps if e["g"] > 0]
    qs = questions(eps)
    s: dict[str, Any] = {"step": step, "kind": "full" if sampled else "quick",
                         "episodes": len(eps), "greedy": summary.summarise(greedy)}  # fmt: skip
    if sampled:
        files = [[{"id": e["qid"], "origin": e["origin"], "exec_ok": e.get("exec_ok")}
                  for e in sampled if e["g"] == g] for g in range(1, SAMPLES + 1)]  # fmt: skip
        ps = passk.summarise(files, {q["id"]: q["final"] for q in qs.values()})
        s["sampled"] = {"overall": ps["overall"], "by_set": ps["by_set"]}
        s["sampled_calls_mean"] = summary.protocol(sampled)["calls_mean"]
        s["sampled_hacking"] = summary.hacking(sampled)
    s["vs"] = {}
    for name, ref_dir in (refs or {}).items():
        if (ref_dir / "summary.json").exists():
            s["vs"][name] = compare(qs, load_ref(ref_dir))
    s["questions"] = list(qs.values())
    (d / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def headline(s: dict[str, Any]) -> dict[str, Any]:
    h = summary.headline(s["greedy"])
    if "sampled" in s:
        o = s["sampled"]["overall"]
        h.update(mean_at8=o["mean_at8"], pass_at8=o["pass_at_k"][-1])
    return h


def update_best(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """`BEST.json`: the evaluated step with the highest greedy with turns (the earlier one on a
    tie), with what the keep rule (design.md section 6) says about it."""
    evals = []
    for d in sorted(run_dir.glob("eval-*")):
        f = d / "summary.json"
        if f.exists() and d.name[5:].isdigit():
            evals.append((int(d.name[5:]), json.loads(f.read_text(encoding="utf-8"))))
    best = None
    for k, s in evals:
        v = s["greedy"]["overall"]["final"]["decided"]
        if v is not None and (best is None or v > best[2]):
            best = (k, s, v)
    if best is None:
        return None
    k, s, v = best
    first = s["greedy"]["overall"]["first"]["decided"]
    floor = cfg.get("first_floor")
    out = {"step": k, "greedy_with_turns": v, "strict": s["greedy"]["overall"]["final"]["strict"],
           "first_query": first, "kind": s["kind"],
           "first_query_ok": None if floor is None or first is None else first >= floor,
           "vs": s.get("vs"), "evaluated": [e[0] for e in evals],
           "lora": f"step-{k:04d}/lora_state.safetensors"}  # fmt: skip
    (run_dir / "BEST.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--rows", choices=["evalset", "dev"], default="evalset")
    ap.add_argument("--ref", action="append", default=[], help="name=<evaluation dir>")
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    refs = {x.split("=", 1)[0]: Path(x.split("=", 1)[1]) for x in a.ref}
    d = Path(a.dir)
    if a.run_dir:  # the run's best earlier evaluation is a reference too
        run_dir = Path(a.run_dir)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        best = run_dir / "BEST.json"
        if best.exists():
            b = json.loads(best.read_text(encoding="utf-8"))
            if b["step"] != a.step:
                refs["best"] = run_dir / f"eval-{b['step']:04d}"
    s = score_dir(d, a.rows, refs, a.step)
    if a.run_dir:
        if "best" in s["vs"]:
            s["vs"]["best"]["step"] = b["step"]
            (d / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        update_best(run_dir, cfg)
        log_wandb(run_dir, cfg, s, a.step)
    print(json.dumps({**headline(s), "vs": s["vs"]}))
    return 0


def log_wandb(run_dir: Path, cfg: dict[str, Any], s: dict[str, Any], step: int) -> None:
    try:
        import wandb

        run_id = (run_dir / "wandb_id.txt").read_text().strip()
        run = wandb.init(project="sql-cypher-rl", id=run_id, resume="allow", name=cfg["run"],
                         dir=str(run_dir))  # fmt: skip
        run.define_metric("rl_step")
        run.define_metric("eval/*", step_metric="rl_step")
        flat: dict[str, Any] = {"rl_step": step}
        for name, blk in [("overall", s["greedy"]["overall"]), *s["greedy"]["by_set"].items()]:
            for part in ("final", "first"):
                for key in ("decided", "strict"):
                    if blk[part][key] is not None:
                        flat[f"eval/{name}/{part}_{key}"] = blk[part][key]
            flat[f"eval/{name}/fixed"] = blk["changes"]["fixed"]
            flat[f"eval/{name}/broken"] = blk["changes"]["broken"]
            flat[f"eval/{name}/calls_mean"] = blk["protocol"]["calls_mean"]
        if "sampled" in s:
            o = s["sampled"]["overall"]
            flat["eval/overall/mean_at8"] = o["mean_at8"]
            flat["eval/overall/pass@8"] = o["pass_at_k"][-1]
        for ref, tests in s["vs"].items():
            for test, res in tests.items():
                if isinstance(res, dict):
                    for key, v in res.items():
                        flat[f"eval/vs_{ref}/{test}_{key}"] = v
        run.log(flat)
        run.finish()
    except Exception as e:  # W&B is a copy; the summary file is the record
        print(f"W&B logging skipped: {type(e).__name__}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
