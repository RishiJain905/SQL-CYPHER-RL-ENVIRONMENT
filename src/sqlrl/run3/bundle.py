"""The data a Colab VM needs, packed on the PC (design.md 7.2).

    python -m sqlrl.run3.bundle --out <file.tar.gz> [--source bird --source birddev ...]

What goes in (paths relative to the run's data folder, so it unpacks into `RUN3_DATA` as is):
- `dbs/<source>/*.sqlite`: the SQLite sources. The VM ports them into DuckDB, PostgreSQL and
  MySQL itself (`porting.py`, deterministic, minutes): smaller to carry than dumps of three
  engines, and the same code builds both machines;
- `pools/`: the gold checks (the VM checks again on its own engine versions);
- `raw/cypherbench/`: `train.json`, `test.json` and the sampled graphs;
- `raw/bird-train/.../train.json`, `raw/bird-dev/.../dev.json`: the BIRD question files, for the
  sources packed (the environment check's task check reads them; the first Colab smoke failed
  without them).

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

# {source: (folder under raw/, question file)}: as `sources.py` finds them
QUESTIONS = {"bird": ("bird-train", "train.json"), "birddev": ("bird-dev", "dev.json")}


def files(sources: list[str], dbs: list[str] | None = None,
          graphs_: list[str] | None = None) -> list[tuple[Path, str]]:  # fmt: skip
    """`dbs` / `graphs_`: only these databases and CypherBench graphs (None: all)."""
    s = settings.get()
    out: list[tuple[Path, str]] = []
    keep_db = (lambda name: name in dbs) if dbs else (lambda name: True)
    for src in sources:
        if src in QUESTIONS:  # the source's question file: the task checks read it
            folder, name = QUESTIONS[src]
            q = sorted((s.raw / folder).rglob(name))
            if q:
                out.append((q[0], f"raw/{q[0].relative_to(s.raw).as_posix()}"))
        for p in sorted((s.dbs / src).glob("*.sqlite")):
            if keep_db(p.stem):
                out.append((p, f"dbs/{src}/{p.name}"))
        for p in sorted((s.pools / "goldcheck" / src).glob("*.jsonl")):
            if keep_db(p.stem):
                out.append((p, f"pools/goldcheck/{src}/{p.name}"))
    for p in sorted((s.pools / "goldcheck" / "cypherbench").glob("*.jsonl")):
        if not graphs_ or p.stem in graphs_:
            out.append((p, f"pools/goldcheck/cypherbench/{p.name}"))
    cb = graphs.cypherbench_root()
    for name in ("train.json", "test.json"):
        if (cb / name).exists():
            out.append((cb / name, f"raw/cypherbench/{name}"))
    for p in sorted((cb / "graphs" / "simplekg_sampled").glob("*.json")):
        if not graphs_ or p.name.startswith(tuple(f"{g}_" for g in graphs_)):
            out.append((p, f"raw/cypherbench/graphs/simplekg_sampled/{p.name}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack run 3's data for a Colab VM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--db", action="append", default=[], help="only these databases")
    ap.add_argument("--graph", action="append", default=[], help="only these CypherBench graphs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    fs = files(a.source or ["birddev"], a.db or None, a.graph or None)
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
