"""Score one of run 1's evaluations and compare it with the baseline (inside WSL):

    python -m sqlrl.run1.eval_score --run-dir <run dir> --step <k>

`eval-<k>/summary.json`: step 1's summary for the policy after step k (`passk.summarise`, with
its own greedy verdicts as pass@1) and `vs_baseline`, three paired sign tests on the questions
both sides decided: `greedy` (right for one, wrong for the other), `sampled` (more of the 8
answers right for one than the other) and `pass8` (at least one of the 8 right for one, none for
the other). `pass8` is the gate. Logged into the run's W&B run as `eval/*` at `rl_step` k.
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
from sqlrl.step1_baseline.score import score_all

BASELINE = "baseline-sft1500-t1.0"


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial p for `better` vs `worse` (ties dropped), p = 1/2."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def compare(run: dict[str, Any], base: dict[str, Any], run_greedy: dict, base_greedy: dict):
    b = {q["id"]: q for q in base["questions"]}
    c = {"greedy": [0, 0], "sampled": [0, 0], "pass8": [0, 0]}
    for q in run["questions"]:
        rg, bg = run_greedy.get(q["id"]), base_greedy.get(q["id"])
        if rg is not None and bg is not None and rg != bg:
            c["greedy"][0] += rg is True
            c["greedy"][1] += bg is True
        o = b.get(q["id"])
        if o and q["scored"] and o["scored"]:
            d = q["right"] / q["scored"] - o["right"] / o["scored"]
            c["sampled"][0] += d > 0
            c["sampled"][1] += d < 0
            c["pass8"][0] += q["right"] > 0 and o["right"] == 0
            c["pass8"][1] += q["right"] == 0 and o["right"] > 0
    return {k: {"better": v[0], "worse": v[1], "p": sign_test(v[0], v[1])} for k, v in c.items()}


def earlier_evals(run_dir: Path, step: int) -> list[dict[str, Any]]:
    """The run's finished evaluations before `step`, in step order."""
    out = []
    for d in sorted(run_dir.glob("eval-*")):
        f = d / "summary.json"
        if f.exists() and d.name[5:].isdigit() and int(d.name[5:]) < step:
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def best_of(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The evaluation to keep: the highest mean@8 (the measure with the most samples behind
    it); the earlier step on a tie."""
    best = None
    for s in summaries:
        if best is None or s["overall"]["mean_at8"] > best["overall"]["mean_at8"]:
            best = s
    return best


def greedy_of(run_dir: Path, step: int) -> dict[str, Any]:
    f = run_dir / f"eval-{step:04d}" / "scores_greedy.json"
    return {i["id"]: i.get("exec_ok") for i in json.loads(f.read_text(encoding="utf-8"))["items"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    out = run_dir / f"eval-{a.step:04d}"
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import publiceval
    from asktrainer.data.schema import read_jsonl
    from asktrainer.settings import get_paths as trainer_paths

    exs = list(read_jsonl(p.evalset))
    greedy_preds = [json.loads(x) for x in (out / "preds_greedy.jsonl").read_text("utf-8")
                    .splitlines() if x.strip()]  # fmt: skip
    gres = publiceval.verify(exs, greedy_preds, publiceval.PublicData(trainer_paths().public),
                             checker=4)  # fmt: skip
    (out / "scores_greedy.json").write_text(json.dumps(gres, indent=1, default=str), "utf-8")
    run_greedy = {i["id"]: i.get("exec_ok") for i in gres["items"]}
    items = score_all(out, 4, 8)
    summary = passk.summarise(items, run_greedy)
    base = json.loads((p.data / "baseline" / BASELINE / "summary.json").read_text("utf-8"))
    base_greedy = {i["id"]: i.get("exec_ok") for i in
                   json.loads(p.greedy_scores.read_text(encoding="utf-8"))["items"]}  # fmt: skip
    summary["vs_baseline"] = compare(summary, base, run_greedy, base_greedy)
    summary["baseline"], summary["step"] = BASELINE, a.step
    # against the run's best earlier evaluation: has the run turned? (run 1's did after step 19)
    before = earlier_evals(run_dir, a.step)
    prev_best = best_of(before)
    if prev_best is not None:
        summary["vs_best"] = {"step": prev_best["step"], **compare(
            summary, prev_best, run_greedy, greedy_of(run_dir, prev_best["step"]))}  # fmt: skip
    best = best_of([*before, summary])
    (run_dir / "BEST.json").write_text(json.dumps({
        "step": best["step"], "mean_at8": best["overall"]["mean_at8"],
        "greedy_pass1": best["overall"]["greedy_pass1"],
        "pass_at8": best["overall"]["pass_at_k"][-1],
        "evaluated": [s["step"] for s in [*before, summary]],
        "lora": f"step-{best['step']:04d}/lora_state.safetensors",
    }, indent=1), encoding="utf-8")  # fmt: skip
    try:
        import wandb

        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run_id = (run_dir / "wandb_id.txt").read_text().strip()
        run = wandb.init(project="sql-cypher-rl", id=run_id, resume="allow", name=cfg["run"],
                         dir=str(run_dir))  # fmt: skip
        run.define_metric("rl_step")
        run.define_metric("eval/*", step_metric="rl_step")
        flat: dict[str, Any] = {"rl_step": a.step}
        for name, blk in [("overall", summary["overall"]), *summary["by_set"].items()]:
            for key in ("greedy_pass1", "mean_at8", "always", "sometimes", "never"):
                if blk[key] is not None:
                    flat[f"eval/{name}/{key}"] = blk[key]
            flat[f"eval/{name}/pass@8"] = blk["pass_at_k"][-1]
        for test, res in summary["vs_baseline"].items():
            for key, v in res.items():
                flat[f"eval/vs_baseline/{test}_{key}"] = v
        for test, res in summary.get("vs_best", {}).items():
            if isinstance(res, dict):
                for key, v in res.items():
                    flat[f"eval/vs_best/{test}_{key}"] = v
        run.log(flat)
        run.finish()
    except Exception as e:  # W&B is a copy; the summary file is the record
        print(f"W&B logging skipped: {type(e).__name__}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    o = summary["overall"]
    head = {"greedy": o["greedy_pass1"], "mean@8": o["mean_at8"], "pass@8": o["pass_at_k"][-1]}
    print(json.dumps({**head, "vs_baseline": summary["vs_baseline"],
                      "vs_best": summary.get("vs_best")}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
