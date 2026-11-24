"""Run 3, phase 2: the data the Colab VM needs for the RL run, packed on the PC.

    python scripts/run3/rl_bundle.py --out <dir> [--mb 500]

- the BIRD training databases richest in phase 0's useful groups per MB, up to `--mb` MB, plus
  every database the dev set uses (the served-model check plays the same 300 triples), never a
  held-out one; their gold checks; the four CypherBench training graphs, their questions and
  gold checks; BIRD train's question file;
- `rl/pool.json`: phase 0's groups by kind of draw (`rl.load_pool`), so the VM needs no phase-0
  episode file; `rl/dev-groups.jsonl`; `rl/bundle-dbs.json`;
- the start checkpoint's adapter (`rl/start/lora_state.safetensors`).

Written as `run3-rl.tar.gz` split into 90 MB parts (`colab upload` fails near 110 MB), with a
SHA-256 of the whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run3 import graphs, rl, settings, splits  # noqa: E402

PART = 90 * 1024 * 1024


def choose_dbs(p0: Path, mb_cap: float) -> list[str]:
    s = settings.get()
    lines = (p0 / "pass8-step40" / "groups.jsonl").read_text("utf-8").splitlines()
    groups = {json.loads(x)["gid"]: json.loads(x) for x in lines}
    right, seen = Counter(), Counter()
    for line in (p0 / "pass8-step40" / "episodes.jsonl").read_text("utf-8").splitlines():
        r = json.loads(line)
        seen[r["gid"]] += 1
        right[r["gid"]] += r.get("outcome") == "right"
    useful = Counter(g["db"] for gid, g in groups.items()
                     if g["source"] == "bird" and 0 < right[gid] < seen[gid])  # fmt: skip
    size = {db: (s.dbs / "bird" / f"{db}.sqlite").stat().st_size / 1e6 for db in useful}
    order = sorted(useful, key=lambda d: useful[d] / max(size[d], 0.01), reverse=True)
    out, total = [], 0.0
    for db in order:
        if total + size[db] > mb_cap:
            continue
        out.append(db)
        total += size[db]
    dev = {json.loads(x)["db"] for x in (p0 / "dev-step40" / "groups.jsonl").read_text("utf-8")
           .splitlines() if json.loads(x)["source"] == "bird"}  # fmt: skip
    out += sorted(dev - set(out))
    assert not set(out) & splits.HELDOUT_DBS["bird"], "a held-out database in the bundle"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=500)
    ap.add_argument("--start", required=True, help="the start checkpoint's lora_state")
    a = ap.parse_args(argv)
    s = settings.get()
    p0 = s.data / "phase0"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dbs = choose_dbs(p0, a.mb)
    pool = rl.load_pool(p0 / "pass8-step40", set(dbs))
    (out / "pool.json").write_text(json.dumps(pool), encoding="utf-8")
    (out / "bundle-dbs.json").write_text(json.dumps(dbs), encoding="utf-8")
    files: list[tuple[Path, str]] = []
    for db in dbs:
        files.append((s.dbs / "bird" / f"{db}.sqlite", f"dbs/bird/{db}.sqlite"))
        files.append((s.pools / "goldcheck" / "bird" / f"{db}.jsonl",
                      f"pools/goldcheck/bird/{db}.jsonl"))  # fmt: skip
    for g in splits.CYPHERBENCH_TRAIN_GRAPHS:
        files.append((s.pools / "goldcheck" / "cypherbench" / f"{g}.jsonl",
                      f"pools/goldcheck/cypherbench/{g}.jsonl"))  # fmt: skip
    cb = graphs.cypherbench_root()
    files.append((cb / "train.json", "raw/cypherbench/train.json"))
    for p in sorted((cb / "graphs" / "simplekg_sampled").glob("*.json")):
        if p.name.split("_sampled")[0] in splits.CYPHERBENCH_TRAIN_GRAPHS:
            files.append((p, f"raw/cypherbench/graphs/simplekg_sampled/{p.name}"))
    tj = sorted((s.raw / "bird-train").rglob("train.json"))[0]
    files.append((tj, f"raw/{tj.relative_to(s.raw).as_posix()}"))
    files.append((out / "pool.json", "rl/pool.json"))
    files.append((out / "bundle-dbs.json", "rl/bundle-dbs.json"))
    files.append((p0 / "dev-step40" / "groups.jsonl", "rl/dev-groups.jsonl"))
    files.append((Path(a.start), "rl/start/lora_state.safetensors"))
    tar = out / "run3-rl.tar.gz"
    with tarfile.open(tar, "w:gz", compresslevel=6) as t:
        for p, arc in files:
            t.add(p, arcname=arc)
    h = hashlib.sha256(tar.read_bytes()).hexdigest()
    (out / "run3-rl.sha256").write_text(h, encoding="utf-8")
    parts = []
    with tar.open("rb") as f:
        i = 0
        while chunk := f.read(PART):
            pp = out / f"run3-rl.part-{i:02d}"
            pp.write_bytes(chunk)
            parts.append(pp.name)
            i += 1
    print(json.dumps({"dbs": len(dbs), "files": len(files), "pool": {k: len(v) for k, v in
                      pool.items()}, "tar_mb": round(os.path.getsize(tar) / 1e6, 1),
                      "parts": parts, "sha256": h}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
