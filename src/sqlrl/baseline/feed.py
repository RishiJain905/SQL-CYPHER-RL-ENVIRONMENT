"""The dashboard feed for a baseline run (Windows, standard library): reads the run's files and
writes a full snapshot as `<run-dir>/dashboard/tick-NNNNN.json` when something changed, plus
`run.json` once. Claude pushes them to the page's store (documentation/dashboard/README.md).

    python -I scripts\\baseline_feed.py --run-dir <dir>

Prints one JSON line: {"tick_file", "seq", "status", "run_doc"} (`tick_file` / `run_doc` null
when there's nothing new to push).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl import passk
from sqlrl.baseline.sample import SETTINGS, read_preds
from sqlrl.paths import get_paths

STALL_S = 15 * 60
MODEL = "Spark-X2.6-4B-SQL-Cypher-Agent Q8_0"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                       capture_output=True, text=True)  # fmt: skip
    return str(pid) in r.stdout


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def snapshot(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    n, total_q = SETTINGS["samples"], 200
    preds = [read_preds(run_dir / f"preds_s{k}.jsonl") for k in range(n)]
    good = [p for ps in preds for p in ps.values() if "error" not in p]
    answered = len(good)
    events = read_jsonl(run_dir / "events.jsonl")
    phase = events[-1].get("phase", "starting") if events else "starting"

    # the answer rate over the last ~10 minutes of feed history, else since sampling began
    hist = [h for h in state.get("history", []) if time.time() - h[0] < 600]
    hist.append([time.time(), answered])
    state["history"] = hist
    rate = None
    if len(hist) >= 2 and hist[-1][0] - hist[0][0] > 60:
        rate = (hist[-1][1] - hist[0][1]) / ((hist[-1][0] - hist[0][0]) / 60)
    total = n * total_q
    eta = int((total - answered) / rate * 60) if rate and answered < total else None
    if answered > state.get("last_answered", -1):
        state["last_answered"], state["last_progress"] = answered, time.time()

    scored = []
    for k in range(n):
        f = run_dir / f"scores_s{k}.json"
        if f.exists():
            scored.append(json.loads(f.read_text(encoding="utf-8"))["items"])
    results, questions = None, []
    if scored:
        greedy = {
            i["id"]: i.get("exec_ok")
            for i in json.loads(get_paths().greedy_scores.read_text(encoding="utf-8"))["items"]
        }
        s = passk.summarise(scored, greedy)  # fmt: skip
        results = {"samples_scored": len(scored), "overall": s["overall"], "by_set": s["by_set"]}
        questions = s["questions"]

    pid_file = run_dir / "run.pid"
    running = pid_file.exists() and alive(int(pid_file.read_text()))
    # scoring re-run by hand: alive while files keep landing (a log appended to from WSL keeps
    # its old Windows modified time, so the newest scores file is the signal)
    marks = [p.stat().st_mtime for p in [run_dir / "score.log", *run_dir.glob("scores_s*.json")]
             if p.exists()]  # fmt: skip
    if not running and marks:
        running = time.time() - max(marks) < 300
    if (run_dir / "DONE").exists():
        status = "done"
    elif (run_dir / "FAILED").exists():
        status = "failed"
    elif not running:
        status = "stopped"
    elif phase == "sampling" and time.time() - state.get("last_progress", time.time()) > STALL_S:
        status = "stalled"
    else:
        status = {"sampled": "scoring"}.get(phase, phase)
    stopped = status == "stopped"
    if stopped:  # the page knows "failed"; the note says what happened
        status = "failed"
    note = {
        "starting": "Loading the model into llama-server.",
        "sampling": f"Answering: {answered} of {total} answers so far.",
        "scoring": f"Checking the answers: {len(scored)} of {n} sample files scored.",
        "done": "Finished: every answer sampled and checked.",
        "stalled": "No new answer for 15 minutes while sampling.",
        "failed": events[-1]["text"] if events else "The run stopped.",
    }[status]
    if stopped:
        note = "The run's process stopped without finishing; it resumes from the answers kept."
    toks = [p["completion_tokens"] for p in good if p.get("completion_tokens")]
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    return {
        "status": status,
        "note": note,
        "settings": {
            "model": MODEL,
            "questions": total_q,
            "samples": n,
            "temperature": SETTINGS["temperature"],
            "top_p": SETTINGS["top_p"],
            "thinking_budget": SETTINGS["thinking_budget"],
            "max_tokens": SETTINGS["max_tokens"],
            "checker": meta.get("checker", 4),
            "seeds": [SETTINGS["seed_base"] * (k + 1) for k in range(n)],
        },  # fmt: skip
        "sampling": {
            "answered": answered,
            "total": total,
            "per_sample": [sum(1 for p in ps.values() if "error" not in p) for ps in preds],
            "answers_per_min": round(rate, 1) if rate is not None else None,
            "eta_s": eta,
            "errors": sum(1 for ps in preds for p in ps.values() if "error" in p),
            "cut_by_length": sum(1 for p in good if p.get("finish") == "length"),
            "mean_completion_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        },
        "scoring": {"scored_samples": len(scored), "total_samples": n},
        "results": results,
        "questions": questions,
        "events": [{"at": e["at"], "kind": e["kind"], "text": e["text"]} for e in events[-20:]],
    }


def feed(run_dir: Path) -> dict[str, Any]:
    out_dir = run_dir / "dashboard"
    out_dir.mkdir(exist_ok=True)
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    snap = snapshot(run_dir, state)
    key = json.dumps({k: v for k, v in snap.items() if k != "sampling"} | {
        "answered": snap["sampling"]["answered"]}, sort_keys=True)  # fmt: skip
    tick_file = None
    if key != state.get("last_key"):
        seq = state.get("seq", 0) + 1
        snap = {"seq": seq, "made_at": now(), **snap}
        tick_file = out_dir / f"tick-{seq:05d}.json"
        tick_file.write_text(json.dumps(snap), encoding="utf-8")
        state["seq"], state["last_key"], state["last_made_at"] = seq, key, snap["made_at"]
    run_doc = None
    if not state.get("run_sent"):
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        run_doc = out_dir / "run.json"
        run_doc.write_text(json.dumps({
            "name": run_dir.name, "kind": "baseline",
            "title": "Sampling baseline: the released SFT model",
            "started_at": meta["started_at"], "last_tick_at": state.get("last_made_at", now()),
            "model": MODEL,
            "notes": "200 evalset questions x 8 sampled answers, temperature 1.0, checker 4.",
        }), encoding="utf-8")  # fmt: skip
        state["run_sent"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {"tick_file": str(tick_file) if tick_file else None, "seq": state.get("seq"),
            "status": snap["status"], "answered": snap["sampling"]["answered"],
            "last_made_at": state.get("last_made_at"),
            "run_doc": str(run_doc) if run_doc else None}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(feed(Path(a.run_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
