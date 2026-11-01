"""A finished pilot measured with step 1's yardstick (Windows; ~50 minutes):

    C:\\Python314\\python.exe -I scripts\\step2_pilots\\evaluate.py --pilot p1-grpo

The merged Q8_0 with the pilot's last LoRA, served exactly as the baseline was (the trainer's
serve.ps1, thinking budget 1,024, the chat endpoint):

1. the 200 evalset questions answered greedily (the trainer's `llamaeval.generate`, as the
   release was scored);
2. the same questions answered 8 times sampled with the baseline's settings and seeds
   (`step1_baseline.sample`), so each answer pairs with the baseline's;
3. the server stopped, then everything scored inside WSL (`eval_score.py`): pass@1 greedy,
   mean@8, pass@1-8, the buckets, and the sign tests against the baseline.

Writes `<pilot>/eval/` and `EVALUATED`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sqlrl.paths import REPO, get_paths, to_wsl
from sqlrl.step1_baseline import sample
from sqlrl.step2_pilots import config as pcfg
from sqlrl.step2_pilots.orchestrate import NO_WINDOW, PORT, event, llama_servers

EVAL_BUDGET = 1024


def serve_eval(start: bool, log: Path, lora: Path | None = None) -> None:
    p = get_paths()
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(p.serve_script),
            "-Port", str(PORT)]  # fmt: skip
    if start:
        extra = f"--reasoning-budget {EVAL_BUDGET}" + (f" --lora {lora}" if lora else "")
        args += ["-Model", str(p.model_gguf), "-Extra", extra]
    else:
        args += ["-Stop"]
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                       creationflags=NO_WINDOW)  # fmt: skip
    if start and "ready" not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"llama-server did not start (see {log.name})")


def evaluate(name: str, launched_by: str = "agent") -> int:
    p = get_paths()
    run_dir = p.data / "pilots" / name
    cfg = pcfg.load(run_dir)
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    lora = run_dir / f"step-{cfg['steps']:04d}" / "adapter.gguf"
    if not lora.exists():
        raise FileNotFoundError(f"{lora} is missing: has the pilot finished training?")
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import llamaeval

    try:
        if llama_servers():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        event(run_dir, "info", "evaluation: greedy + 8 sampled answers per evalset question",
              phase="evaluating")  # fmt: skip
        serve_eval(True, out / "serve.log", lora)
        try:
            url = f"http://127.0.0.1:{PORT}"
            g = llamaeval.generate(p.evalset, out / "preds_greedy.jsonl", url)
            s = sample.sample(out, url)
        finally:
            serve_eval(False, out / "serve-stop.log")
        if g["errors"] or s["errors"]:
            raise RuntimeError(f"{g['errors']} greedy and {s['errors']} sampled answers failed")
        cmd = (f"cd {to_wsl(REPO)} && PYTHONPATH=src {p.wsl_venv}/bin/python "
               f"-m sqlrl.step2_pilots.eval_score --run-dir {to_wsl(run_dir)} "
               f"--launched-by {launched_by}")  # fmt: skip
        with (out / "score.log").open("a", encoding="utf-8") as f:
            r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               creationflags=NO_WINDOW)  # fmt: skip
        if r.returncode != 0:
            raise RuntimeError(f"scoring failed (exit {r.returncode}, see eval/score.log)")
        event(run_dir, "info", "evaluation done (eval/summary.json)", phase="done")
        (run_dir / "EVALUATED").write_text("", encoding="utf-8")
        return 0
    except Exception as e:
        event(run_dir, "error", f"evaluation failed: {type(e).__name__}: {str(e)[:300]}",
              phase="failed")  # fmt: skip
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    run_dir = get_paths().data / "pilots" / a.pilot
    log = (run_dir / "driver.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    (run_dir / "PAUSED").unlink(missing_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")  # pause.py stops it
    return evaluate(a.pilot, a.launched_by)


if __name__ == "__main__":
    sys.exit(main())
