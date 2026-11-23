"""The question sources of run 3 as one record shape (design.md 3.1).

    {"id", "source", "split", "db_id", "question", "evidence", "gold", "difficulty"}

`source` names the database folder (`<data>/dbs/<source>/<db_id>.sqlite`); `split` is the
source's own split. What is held out (BIRD dev, CypherBench test, whole held-out databases) is
decided in `pools.py`, never here.

| Source | Files (under `<data>/raw`) | Licence |
|---|---|---|
| `bird` | `bird-train/train/train.json`, `train_databases/` | CC BY-SA 4.0 |
| `birddev` | `bird-dev/dev_20240627/dev.json`, `dev_databases/` | CC BY-SA 4.0 (held out) |
| `spider` | `spider/train_spider.json`, `train_others.json`, `database/` | CC BY-SA 4.0 |
| `cypherbench` | `train.json` and the sampled graphs (`graphs.cypherbench_root`) | Apache 2.0 |
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlrl.run3 import settings


def _find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} under {root}")
    return hits[0]


def bird(split: str = "train") -> list[dict[str, Any]]:
    raw = settings.get().raw
    if split == "train":
        path, source = _find(raw / "bird-train", "train.json"), "bird"
    else:
        path, source = _find(raw / "bird-dev", "dev.json"), "birddev"
    out = []
    for i, r in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        qid = r.get("question_id", i)
        out.append({
            "id": f"{source}-{qid}",
            "source": source,
            "split": split,
            "db_id": r["db_id"],
            "question": r["question"].strip(),
            "evidence": (r.get("evidence") or "").strip(),
            "gold": r["SQL"].strip(),
            "difficulty": r.get("difficulty"),
        })  # fmt: skip
    return out


def spider() -> list[dict[str, Any]]:
    root = settings.get().raw / "spider"
    out = []
    for name in ("train_spider.json", "train_others.json"):
        try:
            path = _find(root, name)
        except FileNotFoundError:
            continue
        for i, r in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            out.append({
                "id": f"spider-{name.split('_')[1].split('.')[0]}-{i}",
                "source": "spider",
                "split": "train",
                "db_id": r["db_id"],
                "question": r["question"].strip(),
                "evidence": "",
                "gold": r["query"].strip(),
                "difficulty": None,
            })  # fmt: skip
    return out


def load(source: str) -> list[dict[str, Any]]:
    if source == "bird":
        return bird("train")
    if source == "birddev":
        return bird("dev")
    if source == "spider":
        return spider()
    if source == "cypherbench":
        from sqlrl.run3 import graphs

        return graphs.cypherbench_questions("train")
    raise ValueError(f"unknown source {source!r}")
