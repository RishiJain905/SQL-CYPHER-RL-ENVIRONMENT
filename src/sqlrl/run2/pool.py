"""Run 2's questions: run 1's pool (`pool-v2`, unchanged: the same ids in the same order), a
40-step offset so no question is one run 1b trained on, a dev slice held back for wording tests,
and each SQL question's database script joined on from a sidecar file.

    python -m sqlrl.run2.pool build-db [--pool pool-v2]     (inside WSL: reads the parquet files)
    python -m sqlrl.run2.pool build-db --evalset
    python -m sqlrl.run2.pool build-returned                (inside WSL, once: decisions R17)

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
RETURNED = "gretel_returned"  # the pool's key for the returned gretel questions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def data_dir() -> Path:
    return get_paths().data / "run2"


def pool_path(name: str) -> Path:
    return get_paths().data / "run1" / f"{name}.jsonl"


def testset_path() -> Path:
    """The 1,000-question test set: beside the evaluation set in the trainer's public data
    (`public/testset/examples_val.jsonl`). Read only by `testset.py`, once, on the adapter a
    run keeps; no rollout, reward or filter ever sees it."""
    return get_paths().evalset.parent.parent / "testset" / "examples_val.jsonl"


def db_path(name: str) -> Path:
    return data_dir() / f"{name}.db.jsonl"


def returned_path(name: str) -> Path:
    return data_dir() / f"{name}.returned.jsonl"


def load_pool(name: str) -> dict[str, list[dict[str, Any]]]:
    """{dataset: its questions in pool order} (run 1's file, as run 1 read it), plus, when the
    file exists, `gretel_returned`: the gretel questions run 1's pool rejected because the
    right result says nothing (`build_returned`; decisions R17). Those rows are gretel rows
    (`origin` "gretel") tagged `kind: "returned"`; the tag never leaves the scoring side."""
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    for row in read_jsonl(pool_path(name)):
        by_set[row["origin"]].append(row)
    if returned_path(name).exists():
        by_set[RETURNED] = read_jsonl(returned_path(name))
    for rows in by_set.values():
        rows.sort(key=lambda r: r["set_index"])
    return by_set


def load_scripts(name: str) -> dict[str, str]:
    """{question id: its database script} from the sidecars (`evalset` for the evaluation
    set; a pool's returned questions have their own file)."""
    path = db_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: python -m sqlrl.run2.pool build-db")
    out = {r["id"]: r["script"] for r in read_jsonl(path)}
    extra = db_path(name + ".returned")
    if extra.exists():
        out.update({r["id"]: r["script"] for r in read_jsonl(extra)})
    return out


def dev_slice(pool: dict[str, list[dict[str, Any]]], per_set: int) -> list[dict[str, Any]]:
    """The held-back questions: the last `per_set` of each dataset."""
    return [r for s in SETS for r in pool.get(s, [])[-per_set:]] if per_set > 0 else []


def dev_returned(pool: dict[str, list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    """The second dev slice: the last `n` returned gretel questions. On the first dev slice a
    result that says nothing is always a wrong query; here it is the right result."""
    return list(pool.get(RETURNED, [])[-n:]) if n > 0 else []


def kind_of_nothing(db: Any, rows: list[tuple]) -> str:
    """Why a result says nothing: `no_rows`, `null` (only NULLs) or `zero` (zeros and empties);
    `+empty_tables` when the database holds no row at all."""
    if not rows:
        kind = "no_rows"
    elif all(v is None for r in rows for v in r):
        kind = "null"
    else:
        kind = "zero"
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    held = sum(db.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables)
    return kind if held else kind + "+empty_tables"


def strict_check(query: str, gold: str, script: str | None, checker: int = 4) -> str:
    """pass | fail | unverified: the checker's full SQL check (`querycheck.check_sql`) on the
    dataset's own rows **and** on filled copies of the schema with the queries' literals
    planted. `publiceval.score` checks gretel on the dataset's own rows only, and there any
    query that returns nothing equals a reference that returns nothing. The filled copies are
    where a right query and a merely empty one come apart."""
    from asktrainer.querycheck import check_sql

    return check_sql(query, gold, script, real_rows=True, version=checker)["verdict"]


def build_returned(name: str = "pool-v2", seed: int = 42, force: bool = False,
                   log=print) -> dict[str, Any]:  # fmt: skip
    """Write `<data>/run2/<name>.returned.jsonl` (inside WSL, CPU): the gretel questions run
    1's pool looked at and rejected with `reference_says_nothing`, in a seeded order.

    A question is taken back when (1) run 1's own self-check still says exactly that (the
    checker passes the reference; its result on the dataset's own rows says nothing), and (2)
    the strict check can decide the reference on filled copies, so an episode can earn the
    reward under the guard (`config.returned_guard`). Written once: the order is what
    `config.step_questions` counts in, so the file is never rebuilt (`force` is for a build
    that was cut)."""
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    import random
    from collections import Counter

    from asktrainer import publiceval
    from asktrainer.data.schema import Example
    from asktrainer.querycheck import _connect, _run
    from asktrainer.settings import get_paths as trainer_paths

    # run 1's own functions, read only: the rule that rejected these questions is applied again
    from sqlrl.run1 import pool as pool1

    out = returned_path(name)
    if out.exists() and not force:
        raise FileExistsError(f"{out.name} exists: its order is in use, it is never rebuilt")
    checked = read_jsonl(pool_path(name).with_suffix(".checked.jsonl"))
    tokens = {r["id"]: r.get("prompt_tokens") for r in checked}
    want = {r["id"] for r in checked
            if r["origin"] == "gretel" and r.get("reason") == "reference_says_nothing"}  # fmt: skip
    public = trainer_paths().public
    data = publiceval.PublicData(public)
    rows = []
    with (public / "gold-full" / "examples_train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if '"gretel"' in line:
                ex = json.loads(line)
                if ex["id"] in want and ex.get("task") == "call":
                    rows.append(ex)
    rows = list({r["id"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["id"])
    random.Random(f"{seed}-gretel-returned").shuffle(rows)
    why, kinds, kept = Counter(), Counter(), []
    for ex in rows:
        ok, reason = pool1.self_check(ex, data)
        if ok or reason != "reference_says_nothing":
            why[f"self_check:{reason}"] += 1
            continue
        e = Example.model_validate(ex)
        gold, _ = pool1.reference_call(ex)
        script = data.schema_script(e)
        verdict = strict_check(pool1.probe_query(gold), gold, script)
        if verdict != "pass":
            why[f"strict:{verdict}"] += 1
            continue
        db = _connect(script)
        try:
            kind = kind_of_nothing(db, _run(db, gold))
        finally:
            db.close()
        kinds[kind] += 1
        row = dict(ex)
        row.update(rl_thinking="high", kind="returned", nothing=kind, set_index=len(kept),
                   prompt_tokens=tokens.get(ex["id"]))  # fmt: skip
        kept.append(row)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    build_db(kept, db_path(name + ".returned"), log=log)
    # the evaluation set's own gretel questions of this kind, for comparison (counts only)
    ev = Counter()
    for ex in read_jsonl(p.evalset):
        if ex["origin"] != "gretel":
            continue
        e = Example.model_validate(ex)
        gold, script = ex["meta"]["gold"], data.schema_script(e)
        db = _connect(script)
        try:
            res = _run(db, gold)
            if not _informative_rows(res):
                ev[kind_of_nothing(db, res)] += 1
                ev["strict:" + strict_check(pool1.probe_query(gold), gold, script)] += 1
        finally:
            db.close()
    stats = {"pool": name, "seed": seed, "rejected_by_run1": len(want), "found": len(rows),
             "kept": len(kept), "left_out": dict(why), "kinds": dict(kinds),
             "evalset_gretel_says_nothing": dict(ev)}  # fmt: skip
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(json.dumps(stats))
    return stats


def _informative_rows(rows: list[tuple]) -> bool:
    from asktrainer.querycheck import _informative

    return _informative(rows)


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
    b.add_argument("--testset", action="store_true")
    r = sub.add_parser("build-returned")
    r.add_argument("--pool", default="pool-v2")
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build-returned":
        build_returned(a.pool, force=a.force)
    elif a.evalset:
        build_db(read_jsonl(get_paths().evalset), db_path("evalset"))
    elif a.testset:
        build_db(read_jsonl(testset_path()), db_path("testset"))
    else:
        build_db(read_jsonl(pool_path(a.pool)), db_path(a.pool))
    return 0


if __name__ == "__main__":
    sys.exit(main())
