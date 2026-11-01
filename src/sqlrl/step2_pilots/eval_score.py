"""Score a pilot's evaluation answers and compare them with the baseline (inside WSL):

    python -m sqlrl.step2_pilots.eval_score --run-dir <pilot dir>

`eval/summary.json`: the step-1 summary for the pilot (`passk.summarise`, with the pilot's own
greedy verdicts as pass@1) and `vs_baseline`, two paired sign tests on the questions both sides
decided: greedy (right for one and wrong for the other) and sampled (more of the 8 answers
right for one than the other). Logged into the pilot's W&B run as `eval/*`.
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
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(pilot: dict[str, Any], base: dict[str, Any], pilot_greedy: dict, base_greedy: dict):
    b = {q["id"]: q for q in base["questions"]}
    g_better = g_worse = s_better = s_worse = 0
    for q in pilot["questions"]:
        pg, bg = pilot_greedy.get(q["id"]), base_greedy.get(q["id"])
        if pg is not None and bg is not None and pg != bg:
            g_better += pg is True
            g_worse += bg is True
        o = b.get(q["id"])
        if o and q["scored"] and o["scored"]:
            d = q["right"] / q["scored"] - o["right"] / o["scored"]
            s_better += d > 0
            s_worse += d < 0
    return {
        "greedy": {"better": g_better, "worse": g_worse, "p": sign_test(g_better, g_worse)},
        "sampled": {"better": s_better, "worse": s_worse, "p": sign_test(s_better, s_worse)},
        # the tracker page's headline: the greedy test
        "better": g_better,
        "worse": g_worse,
        "p": sign_test(g_better, g_worse),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    out = run_dir / "eval"
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
    pilot_greedy = {i["id"]: i.get("exec_ok") for i in gres["items"]}
    items = score_all(out, 4, 8)
    summary = passk.summarise(items, pilot_greedy)
    base_dir = p.data / "baseline" / BASELINE
    base = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    base_greedy = {i["id"]: i.get("exec_ok") for i in
                   json.loads(p.greedy_scores.read_text(encoding="utf-8"))["items"]}  # fmt: skip
    summary["vs_baseline"] = compare(summary, base, pilot_greedy, base_greedy)
    summary["baseline"] = BASELINE
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    try:
        import wandb

        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run_id = (run_dir / "wandb_id.txt").read_text().strip()
        run = wandb.init(project="sql-cypher-rl", id=run_id, resume="allow", name=cfg["name"],
                         dir=str(run_dir))  # fmt: skip
        for name, blk in [("overall", summary["overall"]), *summary["by_set"].items()]:
            for key in ("greedy_pass1", "mean_at8", "always", "sometimes", "never"):
                if blk[key] is not None:
                    run.summary[f"eval/{name}/{key}"] = blk[key]
            for k, v in enumerate(blk["pass_at_k"], start=1):
                run.summary[f"eval/{name}/pass@{k}"] = v
        for test in ("greedy", "sampled"):
            for key, v in summary["vs_baseline"][test].items():
                run.summary[f"eval/vs_baseline/{test}_{key}"] = v
        run.finish()
    except Exception as e:  # W&B is a copy; the summary file is the record
        print(f"W&B logging skipped: {type(e).__name__}", flush=True)
    o = summary["overall"]
    head = {"greedy": o["greedy_pass1"], "mean@8": o["mean_at8"], "pass@8": o["pass_at_k"][-1]}
    print(json.dumps({**head, "vs_baseline": summary["vs_baseline"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
