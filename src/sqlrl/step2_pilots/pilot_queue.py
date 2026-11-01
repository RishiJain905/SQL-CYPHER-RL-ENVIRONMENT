"""Run pilots one after another (Windows; start it hidden and detached):

    C:\\Python314\\python.exe -I scripts\\step2_pilots\\pilot_queue.py \\
        --after p1-grpo p0-srpo-smoke p2-srpo

It waits for `--after` (a pilot run by another process) to finish, then runs each pilot in turn
(`orchestrate.run`: training, then the evaluation). It stops, starting nothing more, as soon as
any pilot it watches or runs is paused, fails or stops. `pause.py` also stops the queue (its
pid is in `<data>/pilots/queue.pid`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from sqlrl.paths import get_paths


def finished(run_dir) -> str | None:
    for m in ("EVALUATED", "FAILED", "PAUSED", "STOPPED"):
        if (run_dir / m).exists():
            return m
    cfg = run_dir / "config.json"
    if (run_dir / "TRAINED").exists() and cfg.exists():
        if json.loads(cfg.read_text(encoding="utf-8")).get("smoke"):
            return "EVALUATED"  # smoke pilots aren't evaluated
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pilots", nargs="+")
    ap.add_argument("--after", default=None)
    ap.add_argument("--launched-by", default="agent")
    a = ap.parse_args(argv)
    root = get_paths().data / "pilots"
    log = (root / "queue.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    (root / "queue.pid").write_text(str(os.getpid()), encoding="utf-8")
    from sqlrl.step2_pilots.orchestrate import run

    def say(text: str) -> None:
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), text, flush=True)

    if a.after:
        say(f"waiting for {a.after}")
        while (state := finished(root / a.after)) is None:
            time.sleep(30)
        if state != "EVALUATED":
            say(f"{a.after} ended {state}: the queue stops")
            return 1
    for name in a.pilots:
        say(f"starting {name}")
        rc = run(name, a.launched_by)
        state = finished(root / name)
        if rc != 0 or state != "EVALUATED":
            say(f"{name} ended {state or rc}: the queue stops")
            return 1
    say("queue done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
