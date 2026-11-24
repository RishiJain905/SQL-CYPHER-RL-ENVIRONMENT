"""Run 3, phase 2: evaluate a downloaded RL checkpoint on the PC while Colab trains.

    python scripts/run3/rl_eval.py --local <synced dir> --step <k> --ref <phase0/anchor-ws-ckpt-0129>

The step's adapter (synced to `<local>/step-<k>/lora_state.safetensors`) is merged in float32 into
a copy of the released Q8_0 (`run2.tools merge`, WSL), then run 2's evaluation set (`anchor.py`)
and the run-3 dev set greedy (`pass8.py`, step 40's 300 triples) are run on it, and one row is
appended to `<local>/evals.jsonl`: the dev set's right rate and behaviour, both question by
question against the start checkpoint (phase 1's 129). One GPU job at a time on the PC.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.paths import REPO, get_paths, to_wsl  # noqa: E402
from sqlrl.run2.model import models_dir  # noqa: E402
from sqlrl.run3 import feed, settings  # noqa: E402

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def wsl(module: str, args: str, log: Path) -> int:
    venv = get_paths().wsl_venv
    cmd = f"cd {to_wsl(REPO)} && PYTHONPATH=src {venv}/bin/python -m {module} {args}"
    with log.open("a", encoding="utf-8") as f:
        return subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              creationflags=NO_WINDOW).returncode  # fmt: skip


def py(script: str, args: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as f:
        return subprocess.run([sys.executable, str(REPO / "scripts" / "run3" / script), *args],
                              stdout=f, stderr=subprocess.STDOUT).returncode  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    loc = Path(a.local)
    name = f"rl-{a.step:04d}"
    lora = loc / f"step-{a.step:04d}" / "lora_state.safetensors"
    if not lora.exists():
        print(json.dumps({"error": f"{lora} not synced yet"}))
        return 2
    s = settings.get()
    log = loc / "eval.log"
    gguf = models_dir() / "run3" / f"{name}.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(get_paths().model_gguf, gguf)
    if wsl("sqlrl.run2.tools", f"merge --lora-state {to_wsl(lora)} --out {to_wsl(gguf)}",
           log) != 0:  # fmt: skip
        raise RuntimeError("the merge failed (eval.log)")
    if py("anchor.py", ["--model", str(gguf), "--name", name, "--ref", a.ref], log) != 0:
        raise RuntimeError("the evaluation set failed (eval.log)")
    dev = loc / "evals" / name / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s.data / "phase0" / "dev-step40" / "groups.jsonl", dev / "groups.jsonl")
    if py("pass8.py", ["--dir", str(dev), "--model", str(gguf), "--greedy", "--samples", "1"],
          log) != 0:  # fmt: skip
        raise RuntimeError("the dev set failed (eval.log)")
    gguf.unlink(missing_ok=True)
    eps = [json.loads(x) for x in (dev / "episodes.jsonl").read_text("utf-8").splitlines()]
    start = [json.loads(x) for x in (s.data / "phase1" / "eval-ckpt-0129" / "dev" /
                                     "episodes.jsonl").read_text("utf-8").splitlines()]  # fmt: skip
    b = feed.behaviour(eps)
    es = feed._evalset(s.data / "phase0" / f"anchor-{name}")
    start_es = s.data / "phase0" / "anchor-ws-ckpt-0129" / "scored.jsonl"
    row = {"ckpt": name, "step": a.step, "dev_right": b["right_rate"],
           "dev_vs": feed._paired(start, eps), "malformed": b["malformed"],
           "wrongdb_decline": b["wrongdb_decline"], "false_decline": b["false_decline"],
           "write_decline": b["write_decline"], "look_when_none": b["look_when_none"],
           "evalset": es["decided"] if es else None,
           # anchor.py's comparison is against --ref: the start's own evaluation-set run
           "evalset_vs": es.get("vs_step40") if es else None,
           "start_scored": str(start_es.exists())}  # fmt: skip
    with (loc / "evals.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
