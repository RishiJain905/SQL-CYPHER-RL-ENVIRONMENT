"""The pilot tracker's feed (Windows, standard library): a pilot's files -> a full snapshot in
`<pilot>/dashboard/tick-NNNNN.json` when something changed, plus the pilot's document once and
whenever its status changes. Claude pushes them to the Spark RL Pilots page
(documentation/pilots/README.md has the contract).

    python -I scripts\\step2_pilots\\feed.py --pilot p1-grpo
    python -I scripts\\step2_pilots\\feed.py --baseline     (the `reference/baseline` document)

Prints one JSON line: {"tick_file", "seq", "status", "pilot_doc"} (null when nothing is new).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

STEP_KEYS = ["step", "reward_mean", "reward_by_set", "solve_rate", "solve_by_set",
             "informative_frac", "undecided_frac", "malformed_frac", "loss", "kl", "entropy",
             "resp_tokens", "clip_frac", "grad_norm", "lr", "mismatch", "think_forced_frac",
             "trained_seqs", "rollout_s", "score_s", "load_s", "train_s", "convert_s", "step_s",
             "peak_gb"]  # fmt: skip
EVAL_KEYS = ["greedy_pass1", "mean_at8", "pass_at_k", "always", "sometimes", "never"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                       capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # fmt: skip
    return str(pid) in r.stdout


def eval_block(summary: dict[str, Any]) -> dict[str, Any]:
    def blk(b):
        return {k: b.get(k) for k in EVAL_KEYS}

    return {**blk(summary["overall"]),
            "by_set": {s: blk(b) for s, b in summary["by_set"].items()},
            "vs_baseline": summary.get("vs_baseline")}  # fmt: skip


def status_of(run_dir: Path, events: list[dict[str, Any]]) -> str:
    if (run_dir / "EVALUATED").exists():
        return "done"
    if (run_dir / "FAILED").exists() or (events and events[-1].get("phase") == "failed"):
        return "failed"
    if (run_dir / "STOPPED").exists() or (run_dir / "PAUSED").exists():
        return "stopped"  # the page's word; the note (the last event) says "paused"
    phase = events[-1].get("phase") if events else None
    pid_file = run_dir / "run.pid"
    running = pid_file.exists() and alive(int(pid_file.read_text()))
    if phase == "evaluating":
        return "evaluating" if running else "failed"
    if (run_dir / "TRAINED").exists():
        return "evaluating" if running else "trained"
    if not events:
        return "planned"
    return "running" if running else "stopped"


def snapshot(run_dir: Path) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    events = read_jsonl(run_dir / "events.jsonl")
    steps = [{k: m.get(k) for k in STEP_KEYS} for m in read_jsonl(run_dir / "train_log.jsonl")]
    status = status_of(run_dir, events)
    done = len(steps)
    eta = None
    if status == "running" and done:
        per = sum(s["step_s"] or 0 for s in steps[-5:]) / len(steps[-5:])
        eta = int(per * (cfg["steps"] - done))
    summary = run_dir / "eval" / "summary.json"
    ev = eval_block(json.loads(summary.read_text(encoding="utf-8"))) if summary.exists() else None
    last = events[-1]["text"] if events else "Not started."
    return {"status": status, "note": last, "step": done, "steps_total": cfg["steps"],
            "eta_s": eta, "steps": steps, "eval": ev,
            "events": [{"at": e["at"], "kind": e["kind"], "text": e["text"]}
                       for e in events[-20:]]}  # fmt: skip


def pilot_doc(run_dir: Path, status: str, last_tick_at: str | None) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    meta_p = run_dir / "run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    keep = ["steps", "questions_per_step", "group_size", "temperature", "think_budget",
            "max_tokens", "lr", "lora_r", "lora_alpha", "minibatches", "reward_malformed",
            "pool", "seed"]  # fmt: skip
    m = cfg["method_cfg"]
    title = f"{cfg['name']}: {cfg['method']}"
    return {"name": cfg["name"], "method": cfg["method"], "title": title, "status": status,
            "smoke": bool(cfg.get("smoke")),
            "config": {**{k: cfg.get(k) for k in keep}, "adv": m["adv"], "agg": m["agg"],
                       "clip": m["clip"], "kl_beta": m.get("kl_beta", 0.0)},
            "started_at": meta.get("started_at"), "last_tick_at": last_tick_at,
            "notes": ""}  # fmt: skip


def feed(run_dir: Path) -> dict[str, Any]:
    out = run_dir / "dashboard"
    out.mkdir(exist_ok=True)
    sp = out / "state.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    snap = snapshot(run_dir)
    key = json.dumps({k: v for k, v in snap.items() if k != "eta_s"}, sort_keys=True)
    tick_file = None
    if key != state.get("last_key"):
        seq = state.get("seq", 0) + 1
        snap = {"seq": seq, "made_at": now(), **snap}
        tick_file = out / f"tick-{seq:05d}.json"
        tick_file.write_text(json.dumps(snap), encoding="utf-8")
        state.update(seq=seq, last_key=key, last_made_at=snap["made_at"])
    doc_file = None
    if tick_file or state.get("doc_status") != snap["status"]:
        doc_file = out / "pilot.json"
        doc_file.write_text(json.dumps(pilot_doc(run_dir, snap["status"],
                                                 state.get("last_made_at"))), "utf-8")  # fmt: skip
        state["doc_status"] = snap["status"]
    sp.write_text(json.dumps(state), encoding="utf-8")
    return {"tick_file": str(tick_file) if tick_file else None, "seq": state.get("seq"),
            "status": snap["status"], "step": snap["step"],
            "pilot_doc": str(doc_file) if doc_file else None}  # fmt: skip


def baseline_doc() -> Path:
    p = get_paths()
    s = json.loads((p.data / "baseline" / "baseline-sft1500-t1.0" / "summary.json")
                   .read_text(encoding="utf-8"))  # fmt: skip
    doc = {
        "name": "baseline-sft1500-t1.0",
        **{k: s["overall"].get(k) for k in EVAL_KEYS},
        "by_set": {n: {k: b.get(k) for k in EVAL_KEYS} for n, b in s["by_set"].items()},
    }
    out = p.data / "pilots" / "reference-baseline.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args(argv)
    if a.baseline:
        print(json.dumps({"reference_doc": str(baseline_doc())}))
        return 0
    print(json.dumps(feed(get_paths().data / "pilots" / a.pilot)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
