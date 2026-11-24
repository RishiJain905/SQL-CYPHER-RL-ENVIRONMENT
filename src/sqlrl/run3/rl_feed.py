"""The "Run 3 RL" dashboard's feed (phase 2), from the PC's synced copy of the VM's run.

    python -m sqlrl.run3.rl_feed --local <synced dir>   # writes <dir>/feed/{state,steps,evals}.json

The page reads collection `rl` of its Artifact store, one document a file, pushed with
`ArtifactData` as runs 1 and 2's were. Every field may be missing: the page draws a gap.

- `state`: the VM's status and step, hours used of `max_hours`, the devcheck, the start's
  references (phase 1's checkpoint 129 on the dev set and run 2's evaluation set), the stop
  rules' bounds, the events (the VM's and the watcher's), the W&B run's state.
- `steps`: one row a step (rollout numbers, the update, timing), without the long lists.
- `evals`: the PC's evaluations of downloaded checkpoints (`evals.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.run3 import rl

START = {"ckpt": "phase1/ckpt-0129", "dev_right": 0.464, "evalset": 64.2,
         "malformed": 0.113, "wrongdb_decline": 0.55, "false_decline": 0.018}  # fmt: skip


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lines(p: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def build(loc: Path) -> dict[str, Any]:
    rows = _lines(loc / "metrics.jsonl")
    steps = [{k: v for k, v in r.items() if k not in ("groups", "parts")} for r in rows]
    events = _lines(loc / "events.jsonl") + _lines(loc / "watch-events.jsonl")
    events.sort(key=lambda e: e.get("t", ""))
    vm = _json(loc / "state.json") or {}
    cfg = _json(loc / "config.json") or {}
    stop = rl.stop_check(rows) if rows else []
    state = {
        "vm": vm, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": rows[-1]["step"] if rows else 0, "hours": rows[-1]["hours"] if rows else 0.0,
        "max_hours": cfg.get("max_hours", rl.CFG["max_hours"]), "questions": cfg.get("questions"),
        "devcheck": _json(loc / "devcheck.json"), "start": START, "stop_bounds": rl.STOP,
        "stop_now": stop, "step1": steps[0] if steps else None, "events": events[-40:],
        "watch": _json(loc / "watch.json") or {},
    }  # fmt: skip
    return {
        "state": state,
        "steps": {"steps": steps},
        "evals": {"evals": _lines(loc / "evals.jsonl")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    a = ap.parse_args()
    loc = Path(a.local)
    out = loc / "feed"
    out.mkdir(parents=True, exist_ok=True)
    changed = []
    for name, doc in build(loc).items():
        f = out / f"{name}.json"
        text = json.dumps(doc, default=str)
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            changed.append(name)
        f.write_text(text, encoding="utf-8")
    print(json.dumps({"dir": str(out), "changed": changed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
