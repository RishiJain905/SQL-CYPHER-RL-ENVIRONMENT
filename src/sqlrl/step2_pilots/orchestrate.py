"""Run a pilot, step by step (Windows; start it hidden and detached, it runs for hours):

    C:\\Python314\\python.exe -I scripts\\step2_pilots\\run.py --pilot p1-grpo

Each step: llama-server with the merged Q8_0 and the LoRA after the step before (none at step 1:
the new LoRA starts as the SFT model), the step's rollouts (`generate.py`), the server stopped,
then the update inside WSL (`train_step.py`, which scores, trains, saves and converts the LoRA).
The GPU holds one of them at a time.

Every child process gets CREATE_NO_WINDOW: a console window that pops up and is closed ends its
process with a Ctrl+C exit (0xC000013A), the likely cause of the baseline's two interruptions
(decisions R3). A restart resumes at the first step without `TRAINED` (its kept rollouts are
reused).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import REPO, get_paths, to_wsl
from sqlrl.step2_pilots import config as pcfg
from sqlrl.step2_pilots.generate import generate, read_done

PORT = 8095
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def event(run_dir: Path, kind: str, text: str, **extra: Any) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"at": now(), "kind": kind, "text": text, **extra}) + "\n")


def llama_servers() -> list[str]:
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                       capture_output=True, creationflags=NO_WINDOW, **TEXT)  # fmt: skip
    return [line for line in r.stdout.splitlines() if "llama-server" in line]


def serve(start: bool, log: Path, lora: Path | None = None, parallel: int = 16) -> None:
    p = get_paths()
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(p.serve_script),
            "-Port", str(PORT)]  # fmt: skip
    if start:
        extra = "--reasoning-budget -1" + (f" --lora {lora}" if lora else "")
        args += ["-Model", str(p.model_gguf), "-Parallel", str(parallel), "-Extra", extra]
    else:
        args += ["-Stop"]
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                       creationflags=NO_WINDOW)  # fmt: skip
    if start and "ready" not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"llama-server did not start (see {log.name})")


def wsl_train(run_dir: Path, k: int, launched_by: str) -> dict[str, Any]:
    p = get_paths()
    cmd = (f"cd {to_wsl(REPO)} && PYTHONPATH=src {p.wsl_venv}/bin/python "
           f"-m sqlrl.step2_pilots.train_step --run-dir {to_wsl(run_dir)} --step {k} "
           f"--launched-by {launched_by}")  # fmt: skip
    log = run_dir / f"step-{k:04d}" / "train.log"
    with log.open("a", encoding="utf-8") as f:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)  # fmt: skip
    if r.returncode != 0:
        raise RuntimeError(f"step {k}: the update failed (exit {r.returncode}, see train.log)")
    return json.loads((log.parent / "metrics.json").read_text(encoding="utf-8"))


def run(name: str, launched_by: str = "agent", steps: int | None = None) -> int:
    p = get_paths()
    run_dir = p.data / "pilots" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "driver.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    if not (run_dir / "config.json").exists():
        pcfg.save(pcfg.pilot_config(name, {"steps": steps} if steps else None), run_dir)
    cfg = pcfg.load(run_dir)
    resumed = (run_dir / "PAUSED").exists()
    for marker in ("FAILED", "STOPPED", "PAUSED", "STOP"):
        (run_dir / marker).unlink(missing_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    meta = run_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"name": name, "started_at": now(), "launched_by": launched_by},
                                   indent=1), encoding="utf-8")  # fmt: skip
    from sqlrl.step2_pilots.pool import load_pool

    pool = load_pool(cfg["pool"])
    k = 0
    try:
        if llama_servers():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        event(run_dir, "info", f"pilot {name} ({cfg['method']}): {cfg['steps']} steps"
              + (" (resumed after a pause)" if resumed else ""), phase="running")  # fmt: skip
        for k in range(1, cfg["steps"] + 1):
            sdir = run_dir / f"step-{k:04d}"
            if (sdir / "TRAINED").exists():
                continue
            if (run_dir / "STOP").exists():
                event(run_dir, "warn", f"stopped before step {k} (STOP file)", phase="stopped")
                (run_dir / "STOPPED").write_text(str(k), encoding="utf-8")
                return 0
            sdir.mkdir(parents=True, exist_ok=True)
            rows = pcfg.step_questions(pool, cfg, k)
            out = sdir / "rollouts.jsonl"
            want = len(rows) * cfg["group_size"]
            if len(read_done(out)) < want:
                lora = run_dir / f"step-{k - 1:04d}" / "adapter.gguf" if k > 1 else None
                event(run_dir, "info", f"step {k}: rollouts", phase="rollouts", step=k)
                serve(True, sdir / "serve.log", lora, cfg["parallel"])
                try:
                    res = generate(f"http://127.0.0.1:{PORT}", rows, cfg, k, out)
                finally:
                    serve(False, sdir / "serve-stop.log")
                rm = sdir / "rollout_meta.json"
                prev = json.loads(rm.read_text(encoding="utf-8")) if rm.exists() else {}
                res["seconds"] = round(res["seconds"] + prev.get("seconds", 0.0), 1)
                rm.write_text(json.dumps(res), encoding="utf-8")
                if len(read_done(out)) < want:
                    raise RuntimeError(f"step {k}: {res['errors']} rollouts failed; re-run")
            event(run_dir, "info", f"step {k}: update", phase="training", step=k)
            m = wsl_train(run_dir, k, launched_by)
            event(run_dir, "info", f"step {k} done: reward {m['reward_mean']:.3f}, solve "
                  f"{m['solve_rate']:.3f}, {m['trained_seqs']} sequences trained, "
                  f"{m['step_s'] / 60:.1f} min", phase="running", step=k)  # fmt: skip
        if not (run_dir / "TRAINED").exists():
            event(run_dir, "info", "training done; next: the evaluation", phase="trained")
            (run_dir / "TRAINED").write_text("", encoding="utf-8")
        if cfg.get("smoke") or (run_dir / "EVALUATED").exists():
            return 0
        from sqlrl.step2_pilots.evaluate import evaluate  # one command runs (or resumes) it all

        return evaluate(name, launched_by)
    except Exception as e:
        event(run_dir, "error", f"failed at step {k}: {type(e).__name__}: {str(e)[:300]}",
              phase="failed", step=k)  # fmt: skip
        (run_dir / "FAILED").write_text(str(k), encoding="utf-8")
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--launched-by", default="agent")
    ap.add_argument("--steps", type=int, default=None, help="override the step count (new runs)")
    a = ap.parse_args(argv)
    return run(a.pilot, a.launched_by, a.steps)


if __name__ == "__main__":
    sys.exit(main())
