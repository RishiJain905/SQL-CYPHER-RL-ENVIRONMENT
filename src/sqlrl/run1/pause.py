"""Free the GPU now: stop run 1 wherever it is (Windows):

    C:\\Python314\\python.exe -I scripts\\run1\\pause.py --run run1
    C:\\Python314\\python.exe -I scripts\\run1\\pause.py --run run1 --after-step

Without `--after-step`, at once: the orchestrator and its children (the process tree), the
run's llama-server (rollouts or evaluation), and any run-1 process inside WSL. Then it checks
that nothing of the run is left on the GPU and writes `PAUSED`. With `--after-step`, the
current step finishes and the run stops before the next one (the `STOP` file).

Resume with the command that started it (`run.py --run <name>`): a step without `TRAINED`
starts again, keeping the rollouts and scores already made; nothing that finished is redone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from sqlrl.run1 import server
from sqlrl.run1.orchestrate import event, run_dir_of

NO_WINDOW = server.NO_WINDOW


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, creationflags=NO_WINDOW, **server.TEXT)


def pause(name: str, after_step: bool = False) -> dict:
    run_dir = run_dir_of(name)
    if not run_dir.exists():
        raise FileNotFoundError(f"no run named {name}")
    if after_step:
        (run_dir / "STOP").write_text("", encoding="utf-8")
        event(run_dir, "warn", "pause requested: stopping after the current step")
        return {"run": name, "stop_after_step": True}
    killed = None
    pid_file = run_dir / "run.pid"
    if pid_file.exists():
        r = _run(["taskkill", "/PID", pid_file.read_text().strip(), "/T", "/F"])
        killed = r.returncode == 0  # the orchestrator and its children
    for pid in run_dir.glob("*/server.pid"):  # the run's own server (a step's or an eval's)
        server.stop(pid.parent)
    _run(["wsl", "-d", "Ubuntu", "--", "pkill", "-f", "sqlrl.run1."])
    left_server: list[str] = []
    for _ in range(15):  # a stopped server takes a moment to exit
        left_server = server.running()
        if not left_server:
            break
        time.sleep(1)
    w = _run(["wsl", "-d", "Ubuntu", "--", "pgrep", "-af", "sqlrl.run1."])
    left_wsl = [x for x in w.stdout.splitlines() if "pgrep" not in x]
    (run_dir / "PAUSED").write_text(json.dumps({"why": ["paused by hand"]}), encoding="utf-8")
    event(run_dir, "warn", "paused: GPU freed (resume with the same command)", phase="paused")
    return {"run": name, "orchestrator_killed": killed, "llama_servers_left": left_server,
            "wsl_processes_left": left_wsl}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="run1")
    ap.add_argument("--after-step", action="store_true")
    a = ap.parse_args(argv)
    res = pause(a.run, a.after_step)
    print(json.dumps(res))
    return 0 if not res.get("llama_servers_left") and not res.get("wsl_processes_left") else 1


if __name__ == "__main__":
    sys.exit(main())
