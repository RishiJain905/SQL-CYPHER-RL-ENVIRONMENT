"""Run 3, phase 2: bring the VM's run to the PC (WSL, standard library; the Colab CLI's files
API, which works while the kernel is busy).

    python3 scripts/run3/rl_sync.py --local <dir> [--session rl3]

Pulls `state.json`, `metrics.jsonl`, `events.jsonl`, `config.json`, `devcheck.json`; then, for
each step in `metrics.jsonl` not yet on the PC, its adapter (the 90 MB parts, joined, checked
against the SHA-256 the VM wrote in `TRAINED`), and the newest step's optimiser (so a lost VM
resumes from the PC). Every CLI line is filtered for tokens. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

COLAB = str(Path.home() / ".local" / "bin" / "colab")
RED = [(re.compile(r"(token=)[^&\" ]+"), r"\1REDACTED"), (re.compile(r"eyJ[A-Za-z0-9._-]{20,}"),
                                                         "REDACTED")]  # fmt: skip
REMOTE = "/content/rlrun"


def red(s: str) -> str:
    for rx, rep in RED:
        s = rx.sub(rep, s)
    return s


def pull(session: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([COLAB, "download", "-s", session, remote, str(local)],
                       capture_output=True, text=True, timeout=600)  # fmt: skip
    if r.returncode != 0:
        print(red((r.stdout + r.stderr)[-300:]), file=sys.stderr)
    return r.returncode == 0 and local.exists()


def joined(session: str, step_dir: str, name: str, parts: list[str], local: Path) -> bool:
    tmp = local.parent / "parts"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in parts:
        if not pull(session, f"{step_dir}/parts/{p}", tmp / p):
            return False
    with (local.parent / (name + ".tmp")).open("wb") as out:
        for p in parts:
            out.write((tmp / p).read_bytes())
    (local.parent / (name + ".tmp")).replace(local)
    for p in parts:
        (tmp / p).unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--session", default="rl3")
    a = ap.parse_args()
    loc = Path(a.local)
    loc.mkdir(parents=True, exist_ok=True)
    got = {}
    for f in ("state.json", "metrics.jsonl", "events.jsonl", "config.json", "devcheck.json"):
        got[f] = pull(a.session, f"{REMOTE}/{f}", loc / f)
    new_steps, bad = [], []
    rows = []
    if (loc / "metrics.jsonl").exists():
        rows = [json.loads(x) for x in (loc / "metrics.jsonl").read_text().splitlines() if x]
    for row in rows:
        k = row["step"]
        sd = loc / f"step-{k:04d}"
        if (sd / "lora_state.safetensors").exists():
            continue
        sd.mkdir(exist_ok=True)
        rdir = f"{REMOTE}/step-{k:04d}"
        if not pull(a.session, f"{rdir}/TRAINED", sd / "TRAINED"):
            bad.append(k)
            continue
        want = json.loads((sd / "TRAINED").read_text())["lora_sha256"]
        parts = row["parts"]["lora_state.safetensors"]
        if not joined(a.session, rdir, "lora_state.safetensors", parts,
                      sd / "lora_state.safetensors"):  # fmt: skip
            bad.append(k)
            continue
        have = hashlib.sha256((sd / "lora_state.safetensors").read_bytes()).hexdigest()
        if have != want:
            (sd / "lora_state.safetensors").unlink()
            bad.append(k)
            continue
        pull(a.session, f"{rdir}/episodes.jsonl", sd / "episodes.jsonl")
        new_steps.append(k)
    if rows and new_steps:  # the newest step's optimiser, for a resume from the PC
        k = rows[-1]["step"]
        sd = loc / f"step-{k:04d}"
        if joined(a.session, f"{REMOTE}/step-{k:04d}", "optim.pt",
                  rows[-1]["parts"]["optim.pt"], sd / "optim.pt"):  # fmt: skip
            want = json.loads((sd / "TRAINED").read_text())["optim_sha256"]
            if hashlib.sha256((sd / "optim.pt").read_bytes()).hexdigest() != want:
                (sd / "optim.pt").unlink()
                bad.append(f"optim-{k}")
    print(json.dumps({"files": got, "steps": len(rows), "new_adapters": new_steps,
                      "failed": bad}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
