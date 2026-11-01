"""The pilots' question pool (inside WSL, with the trainer's venv):

    python -m sqlrl.step2_pilots.pool [--name pool-v1] [--per-set 320]

Training-split questions only: the SFT run's `gold-full` set (the release's own training data,
tool-call format, each with its gold query), never the evalset or the test set (their questions
are excluded by id and by question text). Equal per dataset, interleaved so every step of
`questions_per_step` has the same mix (8 per set at 32), seeded. Prompts longer than
`--max-schema-chars` are left out so a rollout fits in training memory.

Each row is the training example itself (for the checker) plus `rl_thinking`, the thinking level
the rollouts use (the evalset's: high).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

SETS = ["cypherbench", "gretel", "sqale", "text2cypher"]


def question_text(ex: dict[str, Any]) -> str:
    users = [m for m in ex["messages"] if m["role"] == "user"]
    return " ".join((users[0].get("content") or "").split()).lower() if users else ""


def build(
    train: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    per_set: int,
    seed: int = 42,
    max_schema_chars: int = 12000,
) -> list[dict[str, Any]]:
    banned_ids = {e["id"] for e in held_out}
    banned_q = {question_text(e) for e in held_out}
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    for ex in train:
        if ex["origin"] not in by_set or ex.get("task") != "call":
            continue
        if ex["id"] in banned_ids or question_text(ex) in banned_q:
            continue
        if (ex.get("meta") or {}).get("schema_chars", 0) > max_schema_chars:
            continue
        by_set[ex["origin"]].append(ex)
    rng = random.Random(seed)
    picked = {}
    for s in SETS:
        if len(by_set[s]) < per_set:
            raise ValueError(f"{s}: {len(by_set[s])} questions, {per_set} wanted")
        picked[s] = rng.sample(by_set[s], per_set)
    pool = []
    for i in range(per_set):  # interleave: one of each set in turn
        for s in SETS:
            row = dict(picked[s][i])
            row["rl_thinking"] = "high"
            pool.append(row)
    return pool


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool_path(name: str) -> Path:
    return get_paths().data / "pilots" / f"{name}.jsonl"


def load_pool(name: str) -> list[dict[str, Any]]:
    return read_jsonl(pool_path(name))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="pool-v1")
    ap.add_argument("--per-set", type=int, default=320)  # 40 steps x 8 per set
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer.settings import get_paths as trainer_paths

    public = trainer_paths().public
    train = read_jsonl(public / "gold-full" / "examples_train.jsonl")
    held_out = read_jsonl(p.evalset) + read_jsonl(public / "testset" / "examples_val.jsonl")
    pool = build(train, held_out, a.per_set, a.seed)
    out = pool_path(a.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"pool": str(out), "questions": len(pool), "held_out": len(held_out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
