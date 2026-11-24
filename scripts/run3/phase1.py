"""Run 3, phase 1 on the PC: the warm start, trained in chunks, each checkpoint evaluated.

    python scripts/run3/phase1.py collect                 # examples from phase 0's right episodes
    python scripts/run3/phase1.py memory                  # the update's peak memory (WSL, GPU)
    python scripts/run3/phase1.py run --chunks 3 --ref <run 2 eval-0040>  # train, evaluate

Windows (it starts llama-server for the evaluations; training and the merge run inside WSL). For
each chunk: `warmstart train --until <step>` (WSL), the checkpoint merged in float32 into a copy
of the released Q8_0 (`run2.tools merge`, WSL), run 2's evaluation set (`anchor.py`), the run-3
dev set greedy (`pass8.py --greedy`, the same groups as step 40's baseline), the feed rebuilt. A
checkpoint below the evaluation-set gate (57.0, or worse than step 40 by the sign test) stops the
run for a look (documentation/run3/phase0.md). One GPU job at a time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.paths import REPO, get_paths, to_wsl  # noqa: E402
from sqlrl.run2.model import models_dir  # noqa: E402
from sqlrl.run3 import feed, settings  # noqa: E402

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def p1() -> Path:
    return settings.get().data / "phase1"


def p0() -> Path:
    return settings.get().data / "phase0"


def event(level: str, text: str, **status) -> None:
    f = p1() / "status.json"
    st = json.loads(f.read_text("utf-8")) if f.exists() else {"events": []}
    st.update(status)
    st.setdefault("events", []).append(
        {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level, "text": text}
    )
    f.write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"[{level}] {text}", flush=True)
    subprocess.run([sys.executable, "-m", "sqlrl.run3.feed"], cwd=REPO,
                   env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
                   capture_output=True)  # fmt: skip


def wsl(module: str, args: str, log: Path) -> int:
    venv = get_paths().wsl_venv
    cmd = f"cd {to_wsl(REPO)} && PYTHONPATH=src {venv}/bin/python -m {module} {args}"
    with log.open("a", encoding="utf-8") as f:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)  # fmt: skip
    return r.returncode


def py(script: str, args: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as f:
        return subprocess.run([sys.executable, str(REPO / "scripts" / "run3" / script), *args],
                              stdout=f, stderr=subprocess.STDOUT).returncode  # fmt: skip


def total_steps() -> int:
    """The plan's step count before examples over the length limit are left out (the trainer
    knows the exact one; the last chunk trains to the end whatever it is)."""
    from sqlrl.run3.warmstart import DEFAULTS

    f = p1() / "config.json"
    cfg = {**DEFAULTS, **(json.loads(f.read_text("utf-8")) if f.exists() else {})}
    n = sum(1 for _ in (p1() / "examples.jsonl").open(encoding="utf-8"))
    return n * int(cfg["epochs"]) // int(cfg["accum"])


def trained_to() -> int:
    f = p1() / "state" / "position.json"
    return json.loads(f.read_text("utf-8"))["step"] if f.exists() else 0


def evaluate(ck: Path) -> dict:
    log = p1() / "phase1.log"
    busy = settings.get().data / "neo4j.busy"  # another job (the teacher pilot) holds Neo4j
    if busy.exists():
        event("info", f"{ck.name}: waiting for the run-3 Neo4j ({busy.read_text().strip()})")
        while busy.exists():
            time.sleep(30)
    gguf = models_dir() / "run3" / f"ws-{ck.name}.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(get_paths().model_gguf, gguf)  # the merge rewrites a copy of the release
    if wsl("sqlrl.run2.tools", f"merge --lora-state {to_wsl(ck / 'lora_state.safetensors')} "
           f"--out {to_wsl(gguf)}", log) != 0:  # fmt: skip
        raise RuntimeError(f"the merge of {ck.name} failed (phase1.log)")
    event("info", f"{ck.name}: merged in float32 into a Q8_0 copy; evaluating", status="evaluating")
    if py("anchor.py", ["--model", str(gguf), "--name", f"ws-{ck.name}",
                        "--ref", str(run2_eval40())], log) != 0:  # fmt: skip
        raise RuntimeError(f"the evaluation set on {ck.name} failed (phase1.log)")
    dev = p1() / f"eval-{ck.name}" / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p0() / "dev-step40" / "groups.jsonl", dev / "groups.jsonl")
    if py("pass8.py", ["--dir", str(dev), "--model", str(gguf), "--greedy", "--samples", "1"],
          log) != 0:  # fmt: skip
        raise RuntimeError(f"the dev set on {ck.name} failed (phase1.log)")
    docs = feed.build()
    ev = next(e for e in docs["evals"]["evals"] if e["ckpt"] == ck.name)
    gguf.unlink(missing_ok=True)  # 4.4 GB each; the adapter is kept and merges again in minutes
    return ev


REF: list[Path] = []  # run 2's eval-0040 folder, from --ref


def run2_eval40() -> Path:
    """Run 2's evaluation after step 40 (moved with the finished run to the history drive)."""
    return REF[0]


def cmd_run(chunks: int, until: int | None) -> int:
    total = total_steps()
    marks = [round(total * (i + 1) / chunks) for i in range(chunks)]
    if until:
        marks = [m for m in marks if m <= until] or [until]
    event("info", f"phase 1: {total} optimiser steps, checkpoints after {marks}",
          status="training", phase="1")  # fmt: skip
    for i, m in enumerate(marks):
        last = i == len(marks) - 1
        if trained_to() < m:  # the last chunk trains to the trainer's own end
            event("info", f"training to step {'the end' if last else m} of about {total}",
                  status="training")  # fmt: skip
            arg = "" if last else f" --until {m}"
            if wsl("sqlrl.run3.warmstart", f"train --dir {to_wsl(p1())}{arg}",
                   p1() / "train.log") != 0:  # fmt: skip
                event("error", f"training to step {m} failed (train.log)", status="dead",
                      why="training failed")  # fmt: skip
                return 1
        ck = p1() / f"ckpt-{trained_to():04d}"
        if (p1() / f"eval-{ck.name}" / "dev" / "summary.json").exists():
            continue  # evaluated already (a resumed run)
        ev = evaluate(ck)
        g = ev["gates"]
        es = (ev.get("evalset") or {}).get("decided")
        passed = sum(1 for v in g.values() if v["ok"])
        dev_rate = (ev.get("dev") or {}).get("right_rate")
        event("info", f"{ck.name}: evaluation set {es}%, dev right {dev_rate}, gates "
              f"{passed} of {len(g)} pass", status="training")  # fmt: skip
        if g["evalset"]["ok"] is False:
            event("warn", f"{ck.name} is below the evaluation-set gate: stopped for a look",
                  status="paused", why="evaluation-set gate")  # fmt: skip
            return 3
    event("info", "phase 1 finished: every checkpoint evaluated", status="done")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sub.add_parser("memory")
    r = sub.add_parser("run")
    r.add_argument("--chunks", type=int, default=3)
    r.add_argument("--until", type=int, default=None)
    r.add_argument("--ref", required=True, help="run 2's eval-0040 folder")
    a = ap.parse_args(argv)
    p1().mkdir(parents=True, exist_ok=True)
    if a.cmd == "collect":
        return subprocess.run([sys.executable, "-m", "sqlrl.run3.warmstart", "collect",
                               "--episodes", str(p0() / "pass8-step40" / "episodes.jsonl"),
                               "--out", str(p1())], cwd=REPO,
                              env={**__import__("os").environ,
                                   "PYTHONPATH": str(REPO / "src")}).returncode  # fmt: skip
    if a.cmd == "memory":
        return wsl("sqlrl.run3.warmstart", "memory", p1() / "memory.log")
    REF.append(Path(a.ref))
    return cmd_run(a.chunks, a.until)


if __name__ == "__main__":
    sys.exit(main())
