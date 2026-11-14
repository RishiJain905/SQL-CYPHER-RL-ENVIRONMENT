"""The dashboard's data: everything a page needs to show a run, as small JSON files (Windows or
WSL, standard library only, read-only on the run):

    C:\\Python314\\python.exe -I scripts\\run1\\feed.py --run run1

Writes `<run dir>/feed/`:

- `state.json`: the run's status and phase, its settings, the baseline, every evaluation with
  its sign tests, the last events, the step in progress (rollouts done), the mean step time and
  the hours left, and the stop rules that would fire now.
- `steps-<n>.json` (n = 1, 2, ...; 25 steps each): one flat row of numbers per trained step,
  everything `train_step` logged (rewards and solve rates overall and per dataset, the
  informative share, entropy, KL, the server/trainer gap, the distilled share, the canaries,
  the gradient norm, the learning rate, the seconds of each phase, peak GPU memory).

The files are rewritten whole each time, so pushing them to a page's store is idempotent.
`documentation/run1/dashboard-brief.md` describes every field.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths
from sqlrl.run1 import canaries
from sqlrl.run1 import config as rcfg

BASELINE = "baseline-sft1500-t1.0"
CHUNK = 25


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lines(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _num(v: Any) -> Any:
    return float(f"{v:.5g}") if isinstance(v, float) else v


def step_row(m: dict[str, Any]) -> dict[str, Any]:
    """One step's metrics as a flat row of numbers (per-dataset values become
    `reward_<set>` and `solve_<set>`)."""
    row: dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, bool):
            row[k] = v
        elif isinstance(v, (int, float)) or v is None:
            row[k] = _num(v)
    for name, key in (("reward", "reward_by_set"), ("solve", "solve_by_set")):
        for s, v in (m.get(key) or {}).items():
            row[f"{name}_{s}"] = _num(v)
    return row


def history(run_dir: Path) -> list[dict[str, Any]]:
    by: dict[int, dict[str, Any]] = {}
    for m in _lines(run_dir / "train_log.jsonl"):
        if "step" in m:
            by[m["step"]] = m
    return [by[k] for k in sorted(by)]


def eval_block(summary: dict[str, Any]) -> dict[str, Any]:
    def block(b: dict[str, Any]) -> dict[str, Any]:
        return {"greedy_pass1": _num(b.get("greedy_pass1")), "mean_at8": _num(b.get("mean_at8")),
                "pass_at_k": [_num(x) for x in b.get("pass_at_k") or []],
                "always": b.get("always"), "sometimes": b.get("sometimes"),
                "never": b.get("never"), "questions": b.get("questions")}  # fmt: skip

    return {"step": summary.get("step"), "overall": block(summary["overall"]),
            "by_set": {s: block(b) for s, b in (summary.get("by_set") or {}).items()},
            "vs_baseline": summary.get("vs_baseline"),
            "vs_best": summary.get("vs_best")}  # fmt: skip


COMPARE_KEYS = ("step", "solve_rate", "reward_mean", "kl_sft", "entropy", "informative_frac",
                "resp_tokens_median", "distinct_frac", "lr")  # fmt: skip


def compare_block(name: str) -> dict[str, Any] | None:
    """An earlier run to draw behind this one (run 1b against run 1: the same questions at
    every step, so the solve rates pair up): its evaluations and a few numbers per step."""
    run_dir = get_paths().data / "run1" / name
    hist = history(run_dir)
    if not hist:
        return None
    evals = []
    for d in sorted(run_dir.glob("eval-*")):
        s = _read_json(d / "summary.json")
        if s:
            evals.append(eval_block(s))
    steps = [{k: _num(h.get(k)) for k in COMPARE_KEYS} for h in hist]
    for row, h in zip(steps, hist, strict=True):
        for s, v in (h.get("solve_by_set") or {}).items():
            row[f"solve_{s}"] = _num(v)
    return {"run": name, "evals": evals, "steps": steps}


def status_of(run_dir: Path, events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    paused = _read_json(run_dir / "PAUSED") if (run_dir / "PAUSED").exists() else None
    if (run_dir / "DONE").exists():
        return "done", []
    if (run_dir / "FAILED").exists():
        last = next((e["text"] for e in reversed(events) if e.get("kind") == "error"), "")
        return "failed", [last]
    if (run_dir / "PAUSED").exists():
        return "paused", list((paused or {}).get("why") or [])
    if (run_dir / "STOPPED").exists():
        return "stopped", ["stopped after a step (STOP file)"]
    return ("running" if events else "not started"), []


def build(name: str) -> dict[str, Any]:
    p = get_paths()
    run_dir = p.data / "run1" / name
    cfg = _read_json(run_dir / "config.json") or {}
    events = _lines(run_dir / "events.jsonl")
    hist = history(run_dir)
    status, why = status_of(run_dir, events)
    last_phase = next((e for e in reversed(events) if e.get("phase")), {})
    step_now = hist[-1]["step"] if hist else 0
    # the step in progress: how many of its rollouts are written
    cur = rcfg.step_dir(run_dir, step_now + 1) / "rollouts.jsonl"
    done = sum(1 for _ in cur.open(encoding="utf-8")) if cur.exists() else 0
    total = cfg.get("questions_per_step", 0) * cfg.get("group_size", 0)
    secs = [h["step_s"] for h in hist[-10:] if isinstance(h.get("step_s"), (int, float))]
    mean_s = sum(secs) / len(secs) if secs else None
    left = max(cfg.get("steps", 0) - step_now, 0)
    evals = []
    for d in sorted(run_dir.glob("eval-*")):
        s = _read_json(d / "summary.json")
        if s:
            evals.append(eval_block(s))
    base = _read_json(p.data / "baseline" / BASELINE / "summary.json")
    # the same SFT model measured through run 1's own evaluation path (`run.py --evalcheck`)
    same = _read_json(p.data / "run1" / "evalcheck" / "eval-0000" / "summary.json")
    meta = _read_json(run_dir / "run_meta.json") or {}
    state = {
        "run": name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "why": why,
        "phase": last_phase.get("phase"),
        "phase_text": last_phase.get("text"),
        "phase_at": last_phase.get("at"),
        "step_now": step_now,
        "steps_total": cfg.get("steps"),
        "started_at": meta.get("started_at"),
        "launched_by": meta.get("launched_by"),
        "in_progress": {"step": step_now + 1, "rollouts_done": done, "rollouts_total": total},
        "timing": {
            "mean_step_s": _num(mean_s) if mean_s else None,
            "hours_left": _num(left * mean_s / 3600) if mean_s else None,
        },  # fmt: skip
        "stop_rules_now": canaries.stop_reasons(hist, cfg["stop"]) if cfg.get("stop") else [],
        "config": {
            k: cfg.get(k)
            for k in (
                "method",
                "mix",
                "questions_per_step",
                "group_size",
                "lr",
                "warmup_steps",
                "minibatches",
                "eval_steps",
                "temperature",
                "think_budget",
                "max_tokens",
                "slots",
                "rollout_model",
                "adv_std_floor",
                "pool",
                "stop",
                "gate_p",
                "lora_r",
                "lora_alpha",
                "preset",
                "method_cfg",
                "dup_probe",
                "turn_p",
                "compare_run",
                "steps",
            )
        },  # fmt: skip
        "best": _read_json(run_dir / "BEST.json"),  # the evaluated step to keep (run 1b)
        "baseline": {"name": BASELINE, **eval_block(base)} if base else None,
        "baseline_run1_path": eval_block(same) if same else None,
        "evals": evals,
        "events": events[-40:],
        "step_files": (len(hist) + CHUNK - 1) // CHUNK,
    }
    rows = [step_row(h) for h in hist]
    compare = compare_block(cfg["compare_run"]) if cfg.get("compare_run") else None
    return {"state": state, "steps": rows, "compare": compare}


def write(name: str) -> dict[str, Any]:
    data = build(name)
    out = get_paths().data / "run1" / name / "feed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps(data["state"]), encoding="utf-8")
    files = ["state.json"]
    rows = data["steps"]
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        f = f"steps-{i // CHUNK + 1}.json"
        (out / f).write_text(json.dumps({"from": chunk[0]["step"], "to": chunk[-1]["step"],
                                         "steps": chunk}), encoding="utf-8")  # fmt: skip
        files.append(f)
    if data["compare"]:  # the earlier run doesn't change: push this document once
        (out / "compare.json").write_text(json.dumps(data["compare"]), encoding="utf-8")
        files.append("compare.json")
    s = data["state"]
    return {"feed": str(out), "files": files, "status": s["status"], "step_now": s["step_now"],
            "phase": s["phase"], "stop_rules_now": s["stop_rules_now"]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="run1")
    a = ap.parse_args(argv)
    print(json.dumps(write(a.run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
