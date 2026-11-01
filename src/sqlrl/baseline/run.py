"""The whole baseline, start to finish (Windows; start it detached, it runs ~1 hour):

    C:\\Python314\\python.exe -I scripts\\baseline.py --name baseline-sft1500-t1.0

Start it as a detached process with no console (`Win32_Process.Create` with DETACHED_PROCESS),
never through a `cmd /c` wrapper: a console's Ctrl+C killed the first run's driver mid-sampling.

1. llama-server with the released Q8_0 (the trainer's serve.ps1, port 8092, thinking budget
   1,024): the same build and settings as the greedy evalset scores;
2. every question answered `samples` times (`sample.py`);
3. the server stopped, then every sample file scored inside WSL (`score.py`, W&B);
4. `DONE`, or `FAILED` with the step that failed.

`<run-dir>/events.jsonl` records each step (`feed.py` reads it for the dashboard); `run.pid`
holds this process's id. A restart resumes: answered questions and scored files are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sqlrl.baseline import sample
from sqlrl.paths import REPO, get_paths, to_wsl

PORT = 8092
TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def event(run_dir: Path, kind: str, text: str, **extra) -> None:
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "text": text,
           **extra}  # fmt: skip
    with (run_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def llama_servers() -> list[str]:
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                       capture_output=True, **TEXT)  # fmt: skip
    return [line for line in r.stdout.splitlines() if "llama-server" in line]


def serve(start: bool, log: Path) -> None:
    p = get_paths()
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(p.serve_script),
            "-Port", str(PORT)]  # fmt: skip
    if start:
        args += ["-Model", str(p.model), "-Extra",
                 f"--reasoning-budget {sample.SETTINGS['thinking_budget']}"]  # fmt: skip
    else:
        args += ["-Stop"]
    with log.open("w", encoding="utf-8") as f:  # a file, not a pipe: the server inherits it
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if start and "ready" not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("llama-server did not start (see serve.log)")


def score(run_dir: Path, launched_by: str) -> None:
    p = get_paths()
    cmd = (f"cd {to_wsl(REPO)} && PYTHONPATH=src {p.wsl_venv}/bin/python -m sqlrl.baseline.score "
           f"--run-dir {to_wsl(run_dir)} --wandb --launched-by {launched_by}")  # fmt: skip
    with (run_dir / "score.log").open("a", encoding="utf-8") as f:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)  # fmt: skip
    if r.returncode != 0:
        raise RuntimeError(f"scoring failed (exit {r.returncode}, see score.log)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    run_dir = get_paths().data / "baseline" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # started detached (no console): anything printed or raised goes to the run's own log
    log = (run_dir / "driver.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    for marker in ("DONE", "FAILED"):
        (run_dir / marker).unlink(missing_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    meta = run_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"name": a.name, "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "launched_by": a.launched_by,
            "settings": sample.SETTINGS, "checker": 4, "port": PORT}, indent=1),
            encoding="utf-8")  # fmt: skip
    step = "start"
    try:
        step = "serve"
        if llama_servers():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        event(run_dir, "info", "starting llama-server with the released Q8_0", phase="starting")
        serve(True, run_dir / "serve.log")
        step = "sample"
        kept = sum(len(sample.read_done(run_dir / f"preds_s{k}.jsonl"))
                   for k in range(sample.SETTINGS["samples"]))  # fmt: skip
        event(run_dir, "info", "sampling: 200 questions x 8 answers"
              + (f" (resumed: {kept} answers kept)" if kept else ""), phase="sampling")  # fmt: skip
        try:
            res = sample.sample(run_dir, f"http://127.0.0.1:{PORT}")
        finally:
            serve(False, run_dir / "serve-stop.log")
        event(run_dir, "warn" if res["errors"] else "info",
              f"sampling done: {res['answered']} answers in {res['seconds'] / 60:.1f} min, "
              f"{res['errors']} errors; server stopped", phase="sampled")  # fmt: skip
        if res["errors"]:
            raise RuntimeError(f"{res['errors']} answers failed; re-run to retry them")
        step = "score"
        event(run_dir, "info", "scoring the 8 sample files with checker 4", phase="scoring")
        score(run_dir, a.launched_by)
        event(run_dir, "info", "done: summary.json written, W&B run logged", phase="done")
        (run_dir / "DONE").write_text("", encoding="utf-8")
        return 0
    except Exception as e:
        event(run_dir, "error", f"failed at {step}: {type(e).__name__}: {str(e)[:200]}",
              phase="failed")  # fmt: skip
        (run_dir / "FAILED").write_text(step, encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
