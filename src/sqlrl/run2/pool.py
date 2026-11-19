"""Run 2's questions: run 1's pool (`pool-v2`, unchanged: the same ids in the same order), a
40-step offset so no question is one run 1b trained on, a dev slice held back for wording tests,
and each SQL question's database script joined on from a sidecar file.

    python -m sqlrl.run2.pool build-db [--pool pool-v2]     (inside WSL: reads the parquet files)
    python -m sqlrl.run2.pool build-db --evalset

- **The pool file is run 1's** (`<data>/run1/pool-v2.jsonl`). Rebuilding it could change the
  order, and then the offset would skip the wrong questions. So nothing is rebuilt: the scripts
  go into `<data>/run2/<pool>.db.jsonl` ({"id", "origin", "script"}), keyed by id.
- **gretel**'s script is the dataset's own `sql_context` (tables and rows); **SQaLe**'s is its
  DDL. The Cypher sets need none: CypherBench's graph is named in the row, text2cypher's schema
  is in the prompt.
- **The dev slice** is the last `dev_per_set` questions of each dataset in pool order. No step
  may reach them (`config.step_questions` checks). Wording and environment choices are tested
  there, so the evaluation set is never tuned on.
- **The evaluation set** gets the same sidecar (`evalset.db.jsonl`): an evaluation with turns
  needs the databases too. It is read only by the evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

SETS = ["cypherbench", "gretel", "sqale", "text2cypher"]
SQL_SETS = ("gretel", "sqale")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def data_dir() -> Path:
    return get_paths().data / "run2"


def pool_path(name: str) -> Path:
    return get_paths().data / "run1" / f"{name}.jsonl"


def db_path(name: str) -> Path:
    return data_dir() / f"{name}.db.jsonl"


def load_pool(name: str) -> dict[str, list[dict[str, Any]]]:
    """{dataset: its questions in pool order} (run 1's file, as run 1 read it)."""
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    for row in read_jsonl(pool_path(name)):
        by_set[row["origin"]].append(row)
    for rows in by_set.values():
        rows.sort(key=lambda r: r["set_index"])
    return by_set


def load_scripts(name: str) -> dict[str, str]:
    """{question id: its database script} from the sidecar (`evalset` for the evaluation set)."""
    path = db_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: python -m sqlrl.run2.pool build-db")
    return {r["id"]: r["script"] for r in read_jsonl(path)}


def dev_slice(pool: dict[str, list[dict[str, Any]]], per_set: int) -> list[dict[str, Any]]:
    """The held-back questions: the last `per_set` of each dataset."""
    return [r for s in SETS for r in pool.get(s, [])[-per_set:]] if per_set > 0 else []


def build_db(rows: list[dict[str, Any]], out: Path, log=print) -> dict[str, Any]:
    """Write the SQL rows' database scripts to `out` (inside WSL)."""
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import publiceval
    from asktrainer.data.schema import Example
    from asktrainer.settings import get_paths as trainer_paths

    data = publiceval.PublicData(trainer_paths().public)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {s: {"rows": 0, "missing": 0, "chars": 0} for s in SQL_SETS}
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            if row["origin"] not in SQL_SETS:
                continue
            script = data.schema_script(Example.model_validate(row))
            st = stats[row["origin"]]
            st["rows"] += 1
            if not script:
                st["missing"] += 1
                continue
            st["chars"] += len(script)
            f.write(json.dumps({"id": row["id"], "origin": row["origin"], "script": script},
                               ensure_ascii=False) + "\n")  # fmt: skip
    tmp.replace(out)
    log(json.dumps({"file": out.name, **stats}))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-db")
    b.add_argument("--pool", default="pool-v2")
    b.add_argument("--evalset", action="store_true")
    a = ap.parse_args(argv)
    if a.evalset:
        build_db(read_jsonl(get_paths().evalset), db_path("evalset"))
    else:
        build_db(read_jsonl(pool_path(a.pool)), db_path(a.pool))
    return 0


if __name__ == "__main__":
    sys.exit(main())
