"""The data a Colab VM needs, packed on the PC (design.md 7.2).

    python -m sqlrl.run3.bundle --out <file.tar.gz> [--source bird --source birddev ...]

What goes in (paths relative to the run's data folder, so it unpacks into `RUN3_DATA` as is):
- `dbs/<source>/*.sqlite`: the SQLite sources. The VM ports them into DuckDB, PostgreSQL and
  MySQL itself (`porting.py`, deterministic, minutes): smaller to carry than dumps of three
  engines, and the same code builds both machines;
- `pools/`: the gold checks (the VM checks again on its own engine versions);
- `raw/cypherbench/`: `train.json`, `test.json` and the sampled graphs.

Nothing else: no models, no runs, no secrets. The VM's model comes from Hugging Face, the
adapter separately (`scripts/run3/colab/README.md`).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

from sqlrl.run3 import graphs, settings


def files(sources: list[str]) -> list[tuple[Path, str]]:
    s = settings.get()
    out: list[tuple[Path, str]] = []
    for src in sources:
        for p in sorted((s.dbs / src).glob("*.sqlite")):
            out.append((p, f"dbs/{src}/{p.name}"))
    for p in sorted(s.pools.rglob("*")):
        if p.is_file():
            out.append((p, "pools/" + p.relative_to(s.pools).as_posix()))
    cb = graphs.cypherbench_root()
    for name in ("train.json", "test.json"):
        if (cb / name).exists():
            out.append((cb / name, f"raw/cypherbench/{name}"))
    for p in sorted((cb / "graphs" / "simplekg_sampled").glob("*.json")):
        out.append((p, f"raw/cypherbench/graphs/simplekg_sampled/{p.name}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack run 3's data for a Colab VM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    fs = files(a.source or ["birddev"])
    size = sum(p.stat().st_size for p, _ in fs)
    print(json.dumps({"files": len(fs), "gb_uncompressed": round(size / 1e9, 2)}), flush=True)
    if a.dry_run:
        return 0
    with tarfile.open(a.out, "w:gz", compresslevel=6) as tar:
        for p, arc in fs:
            tar.add(p, arcname=arc)
    print(json.dumps({"out": a.out, "gb": round(Path(a.out).stat().st_size / 1e9, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
