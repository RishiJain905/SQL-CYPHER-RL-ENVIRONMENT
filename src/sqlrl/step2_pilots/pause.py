"""Free the GPU now: stop a pilot (or its evaluation) wherever it is (Windows):

    C:\\Python314\\python.exe -I scripts\\step2_pilots\\pause.py --pilot p1-grpo
    C:\\Python314\\python.exe -I scripts\\step2_pilots\\pause.py --pilot p1-grpo --after-step

Without `--after-step`, at once: the orchestrator and its children (the process tree), the
pilots' llama-server, and any pilot process inside WSL (`train_step`, `eval_score`). Then it
checks that nothing of the pilot is left on the GPU and writes `PAUSED`. With `--after-step`,
the current step finishes and the run stops before the next one (the `STOP` file).

Resume with the same command that started it (`run.py --pilot <id>`, or `evaluate.py`): a step
without `TRAINED` starts its update again from the LoRA after the step before, keeping the
rollouts and scores already made; nothing that finished is redone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from sqlrl.paths import get_paths
from sqlrl.step2_pilots.orchestrate import NO_WINDOW, PORT, TEXT, event, llama_servers, serve


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, creationflags=NO_WINDOW, **TEXT)


def pause(name: str, after_step: bool = False) -> dict:
    run_dir = get_paths().data / "pilots" / name
    if after_step:
        (run_dir / "STOP").write_text("", encoding="utf-8")
        event(run_dir, "warn", "pause requested: stopping after the current step")
        return {"pilot": name, "stop_after_step": True}
    pid_file = run_dir / "run.pid"
    killed = None
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        r = _run(["taskkill", "/PID", pid, "/T", "/F"])  # the orchestrator and its children
        killed = r.returncode == 0
    serve(False, run_dir / "serve-stop.log")  # the pilots' server (port PORT), if up
    _run(["wsl", "-d", "Ubuntu", "--", "pkill", "-f", "sqlrl.step2_pilots."])
    left_server = [s for s in llama_servers() if str(PORT) in s] or llama_servers()
    w = _run(["wsl", "-d", "Ubuntu", "--", "pgrep", "-af", "sqlrl.step2_pilots."])
    left_wsl = [x for x in w.stdout.splitlines() if "pgrep" not in x]
    (run_dir / "PAUSED").write_text("", encoding="utf-8")
    event(run_dir, "warn", "paused: GPU freed (resume with the same command)", phase="paused")
    return {"pilot": name, "orchestrator_killed": killed, "llama_servers_left": left_server,
            "wsl_processes_left": left_wsl}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--after-step", action="store_true")
    a = ap.parse_args(argv)
    res = pause(a.pilot, a.after_step)
    print(json.dumps(res))
    return 0 if not res.get("llama_servers_left") and not res.get("wsl_processes_left") else 1


if __name__ == "__main__":
    sys.exit(main())
