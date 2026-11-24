"""Run 3, phase 2's W&B run, logged from the PC (WSL's stored login; no key goes to the VM).

    python scripts/run3/rl_wandb.py --local <synced dir>

Logs every row of the synced `metrics.jsonl` not yet logged (step = the RL step) to the run
`run3-rl` (project `sql-cypher-rl`, group `run3`, `launched-by:agent`), the PC's checkpoint
evaluations (`evals.jsonl`, written by the watcher) as `eval/*`, and the devcheck as summary.
Resumes the same run every time (its id is kept in the folder). Prints the run's URL.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    a = ap.parse_args()
    import wandb

    loc = Path(a.local)
    rid_f = loc / "wandb_id.txt"
    rid = rid_f.read_text().strip() if rid_f.exists() else uuid.uuid4().hex[:8]
    rid_f.write_text(rid)
    done_f = loc / "wandb_logged.json"
    done = json.loads(done_f.read_text()) if done_f.exists() else {"step": 0, "evals": []}
    cfg = json.loads((loc / "config.json").read_text()) if (loc / "config.json").exists() else {}
    safe = {k: v for k, v in cfg.items()
            if not any(w in k.lower() for w in ("token", "secret", "password", "key"))}  # fmt: skip
    run = wandb.init(project="sql-cypher-rl", group="run3", job_type="rl", name="run3-rl",
                     id=rid, resume="allow", dir=str(loc),
                     tags=["run3", "phase2", "rl", "colab", "launched-by:agent"],
                     config={**safe, "start": "phase1/ckpt-0129",
                             "kl_reference": "phase1/ckpt-0129"})  # fmt: skip
    rows = [json.loads(x) for x in (loc / "metrics.jsonl").read_text().splitlines() if x] if (
        loc / "metrics.jsonl").exists() else []  # fmt: skip
    n = 0
    for r in rows:
        if r["step"] <= done["step"]:
            continue
        flat = {}
        for k, v in r.items():
            if k in ("step", "groups", "parts"):
                continue
            if isinstance(v, dict):
                flat.update(
                    {f"{k}/{kk}": vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                )
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[k] = v
        run.log({f"rl/{k}": v for k, v in flat.items()}, step=r["step"])
        done["step"] = r["step"]
        n += 1
    evf = loc / "evals.jsonl"
    for line in evf.read_text().splitlines() if evf.exists() else []:
        ev = json.loads(line)
        if ev["ckpt"] in done["evals"]:
            continue
        run.log({f"eval/{k}": v for k, v in ev.items() if isinstance(v, (int, float))},
                step=max(done["step"], ev["step"]))  # fmt: skip
        done["evals"].append(ev["ckpt"])
    dc = loc / "devcheck.json"
    if dc.exists():
        d = json.loads(dc.read_text())
        run.summary["devcheck/right_rate"] = d.get("right_rate")
    done_f.write_text(json.dumps(done))
    print(json.dumps({"logged_steps": n, "last": done["step"], "url": run.url}))
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
