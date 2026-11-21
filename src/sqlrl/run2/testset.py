"""The test set, once, on the adapter a run keeps (Windows; RUNBOOK section 9, step 2):

    python scripts/run2/testset.py build-db              # the SQL questions' database scripts (WSL)
    python scripts/run2/testset.py run --run run2 --step 40

The 1,000 questions (`pool.testset_path()`, 250 a dataset) answered greedily three times, each
under one llama-server, one GPU job at a time:

- `step<k>-turns`: the kept adapter (the run's rollout GGUF, which must hold step k merged in)
  under the run's own protocol (`config.json`: the prompt, the call limit, the single-turn
  datasets, the result format);
- `start-turns`: the start adapter (run 1b's step 30, `<models>/run2/phase0/step30.gguf`) under
  the same protocol: what the training added;
- `start-one`: the start adapter on one query with the released prompt (the anchor's setting,
  `phase0.anchor`): the single-turn number the multi-turn protocol is compared with.

Scored inside WSL by `eval_score --rows testset`, the same yardstick as every evaluation, with
paired sign tests: the kept adapter against both references, and the start with turns against
the start on one query. Everything lands in `<data>/run2/testset/`. A folder that holds a
summary is never generated again (the test set is used once); a run that was cut keeps what
it made and answers the rest.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import to_wsl
from sqlrl.run2 import config as rcfg
from sqlrl.run2 import env, episodes, pool, server
from sqlrl.run2.model import merged_step, models_dir, rollout_model_path
from sqlrl.run2.orchestrate import public_rows, run_dir_of, wsl

OUT = "testset"


def out_dir() -> Path:
    d = pool.data_dir() / OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_pubs() -> list[dict[str, Any]]:
    """The test set as public rows: the reference stops here, as for the evaluation set."""
    return public_rows(pool.read_jsonl(pool.testset_path()), pool.load_scripts("testset"))


def one_query(cfg: dict[str, Any]) -> dict[str, Any]:
    """The run's settings on one query with the released prompt (`phase0.anchor`'s setting)."""
    c = copy.deepcopy(cfg)
    c.update(prompt_variant="sft", max_queries=1)
    return c


def jobs(run: str, step: int) -> list[dict[str, Any]]:
    """The three answer sets, in the order they are made: the references first."""
    cfg = rcfg.load(run_dir_of(run))
    kept = rollout_model_path(run)
    if merged_step(kept) != step:
        raise RuntimeError(f"{kept.name} holds step {merged_step(kept)}, not step {step}: merge "
                           f"it again (tools merge --step {step})")  # fmt: skip
    start = models_dir() / "run2" / "phase0" / "step30.gguf"
    if not start.exists():
        raise FileNotFoundError(f"{start} is missing (tools merge, phase 0)")
    return [
        {"name": "start-one", "model": start, "cfg": one_query(cfg)},
        {"name": "start-turns", "model": start, "cfg": cfg},
        {"name": f"step{step}-turns", "model": kept, "cfg": cfg},
    ]


def generate(job: dict[str, Any]) -> dict[str, Any]:
    d = out_dir() / job["name"]
    d.mkdir(parents=True, exist_ok=True)
    if (d / "summary.json").exists():
        return {"name": job["name"], "skipped": "already answered and scored (used once)"}
    cfg = job["cfg"]
    proto = episodes.claim(d / "episodes.jsonl", cfg)
    server.start(job["model"], d, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"])
    try:
        url = f"http://127.0.0.1:{server.PORT}"
        res: dict[str, Any] = {}
        for _ in range(3):  # a failed episode is retried
            res = episodes.generate(url, test_pubs(), cfg, d / "episodes.jsonl",
                                    episodes.eval_spec(cfg), 1, env.GraphHost())  # fmt: skip
            if not res["errors"]:
                break
    finally:
        if not server.stop(d):
            raise RuntimeError("the test-set server did not stop")
    meta = {**res, "model": str(job["model"].name), **proto}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if res.get("errors"):
        raise RuntimeError(f"{job['name']}: {res['errors']} episodes failed after 3 tries: "
                           "run it again to retry them")  # fmt: skip
    return {"name": job["name"], **res}


def score(name: str, refs: dict[str, str]) -> dict[str, Any]:
    d = out_dir() / name
    ref = " ".join(f"--ref {k}={to_wsl(out_dir() / v)}" for k, v in refs.items())
    (d / "scored.jsonl").unlink(missing_ok=True)
    code = wsl("sqlrl.run2.eval_score", f"--dir {to_wsl(d)} --rows testset {ref}",
               d / "score.log")  # fmt: skip
    if code != 0 or not (d / "summary.json").exists():
        raise RuntimeError(f"scoring {name} failed (see testset/{name}/score.log)")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    o = s["greedy"]["overall"]
    return {"name": name, "final": o["final"], "first": o["first"], "changes": o["changes"],
            "by_set": {k: {"final": v["final"]["decided"], "strict": v["final"]["strict"],
                           "first": v["first"]["decided"]}
                       for k, v in s["greedy"]["by_set"].items()},
            "vs": s.get("vs")}  # fmt: skip


def run(run_name: str, step: int) -> dict[str, Any]:
    js = jobs(run_name, step)
    made = [generate(j) for j in js]
    kept = js[-1]["name"]
    scored = [
        score("start-one", {}),
        score("start-turns", {"start_one": "start-one"}),
        score(kept, {"start_with_turns": "start-turns", "start_one": "start-one"}),
    ]
    out = {"run": run_name, "step": step, "generated": made, "scored": scored}
    (out_dir() / "testset.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-db")
    r = sub.add_parser("run")
    r.add_argument("--run", default="run2")
    r.add_argument("--step", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build-db":
        log = out_dir() / "build-db.log"
        code = wsl("sqlrl.run2.pool", "build-db --testset", log)
        print(log.read_text(encoding="utf-8")[-2000:])
        return code
    if server.running():
        print("another llama-server is running: one GPU job at a time")
        return 1
    print(json.dumps(run(a.run, a.step)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
