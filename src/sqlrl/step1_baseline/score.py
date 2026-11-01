"""Score the baseline's sample files with the trainer's checker, then summarise (inside WSL, with
the trainer's venv: the checker needs its SQLite fills and the trainer's Neo4j).

    python -m sqlrl.step1_baseline.score --run-dir <dir> [--checker 4] [--wandb]

Each `preds_s<k>.jsonl` gets `scores_s<k>.json` (the trainer's `publiceval.verify` result:
summary and one item per question), written as soon as it's scored, so the dashboard shows
partial results. With every file scored: `summary.json` (`passk.summarise`) and, with --wandb,
a W&B run of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl import passk
from sqlrl.paths import get_paths
from sqlrl.step1_baseline.sample import SETTINGS, read_preds

WANDB_PROJECT = "sql-cypher-rl"


def greedy_exec(path: Path) -> dict[str, bool | None]:
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return {i["id"]: i.get("exec_ok") for i in items}


def score_all(run_dir: Path, checker: int, samples: int) -> list[list[dict[str, Any]]]:
    sys.path.insert(0, str(get_paths().trainer_repo / "src"))
    from asktrainer import publiceval
    from asktrainer.data.schema import read_jsonl
    from asktrainer.settings import get_paths as trainer_paths

    exs = list(read_jsonl(get_paths().evalset))
    data = publiceval.PublicData(trainer_paths().public)
    out = []
    for k in range(samples):
        preds_path, scores_path = run_dir / f"preds_s{k}.jsonl", run_dir / f"scores_s{k}.json"
        if scores_path.exists():
            out.append(json.loads(scores_path.read_text(encoding="utf-8"))["items"])
            continue
        preds = list(read_preds(preds_path).values())
        if len(preds) != len(exs):
            raise RuntimeError(f"{preds_path.name}: {len(preds)} answers for {len(exs)} questions")
        t = time.time()
        res = publiceval.verify(exs, preds, data, checker=checker)
        res["sample"], res["seconds"] = k, round(time.time() - t, 1)
        tmp = scores_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        tmp.replace(scores_path)
        print(json.dumps({"sample": k, "exec_acc": res["summary"]["all"]["exec_acc"],
                          "seconds": res["seconds"]}), flush=True)  # fmt: skip
        out.append(res["items"])
    return out


def log_wandb(run_dir: Path, summary: dict[str, Any], checker: int, launched_by: str) -> str:
    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_dir.name,
        job_type="baseline",
        tags=["baseline", f"launched-by:{launched_by}", f"checker:{checker}"],
        config={
            **SETTINGS,
            "checker": checker,
            "model": "Spark-X2.6-4B-SQL-Cypher-Agent Q8_0",
            "questions": summary["overall"]["questions"],
        },  # fmt: skip
        dir=str(run_dir),
    )
    for name, b in [("overall", summary["overall"]), *summary["by_set"].items()]:
        for key in ("greedy_pass1", "mean_at8", "always", "sometimes", "never", "questions"):
            if b[key] is not None:
                run.summary[f"{name}/{key}"] = b[key]
        for k, v in enumerate(b["pass_at_k"], start=1):
            run.summary[f"{name}/pass@{k}"] = v
    curve = wandb.Table(columns=["set", "k", "pass@k"])
    for name, b in [("overall", summary["overall"]), *summary["by_set"].items()]:
        for k, v in enumerate(b["pass_at_k"], start=1):
            curve.add_data(name, k, v)
    qs = wandb.Table(columns=["id", "set", "greedy", "right", "scored"])
    for q in summary["questions"]:
        qs.add_data(q["id"], q["origin"], q["greedy"], q["right"], q["scored"])
    run.log({"pass_at_k": curve, "questions": qs})
    url = run.url
    run.finish()
    return url


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checker", type=int, default=4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--launched-by", default="agent")
    ap.add_argument("--samples", type=int, default=SETTINGS["samples"], help="fewer: a smoke test")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    items = score_all(run_dir, a.checker, a.samples)
    summary = passk.summarise(items, greedy_exec(get_paths().greedy_scores))
    summary["checker"], summary["settings"] = a.checker, SETTINGS
    if a.wandb:
        summary["wandb"] = log_wandb(run_dir, summary, a.checker, a.launched_by)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    o = summary["overall"]
    print(json.dumps({"questions": o["questions"], "greedy": o["greedy_pass1"],
                      "mean@8": o["mean_at8"], "pass@8": o["pass_at_k"][-1],
                      "wandb": summary.get("wandb")}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
