"""Run 3's splits (design.md 3.6), fixed before any measurement (decision R27).

    python -m sqlrl.run3.splits            # the counts per split

- **Never trained on, ever:** BIRD dev, CypherBench's test split (7 graphs no training graph
  shares), run 2's evalset and test set, and the real deployment (design 1). Run 3's pools never
  load them for training.
- **Held-out databases:** six of BIRD train's 58 gold-checked databases, picked by a fixed rule
  (the SHA-256 of `run3-holdout:<name>`, lowest first, among databases with at least 50
  questions; `retail_world` excluded: only 43 of its 373 questions pass the gold check), 793
  questions. No rollout, reward, filter or curriculum choice may read them: they measure "a
  database it has never seen" (design 6). CypherBench's whole-graph held-out set is its test
  split, so no training graph is held out (there are only four).
- **Dev slices:** 1 question in 20 of every training database (by the SHA-256 of
  `run3-dev:<question id>`), used to read the warm start and each RL phase, never trained on.
- **Train:** everything else in the training databases and graphs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

HELDOUT_DBS: dict[str, frozenset[str]] = {
    "bird": frozenset(
        {"superstore", "address", "video_games", "university", "airline", "ice_hockey_draft"}
    ),
    "cypherbench": frozenset(),
}
NEVER_SOURCES = frozenset({"birddev", "cypherbench_test"})  # benchmarks: evaluation only
CYPHERBENCH_TRAIN_GRAPHS = ("art", "biology", "soccer", "terrorist_attack")
DEV_ONE_IN = 20


def _bucket(text: str, mod: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % mod


def split_of(source: str, db_id: str, qid: str) -> str:
    """ "never", "heldout", "dev" or "train" for one question."""
    if source in NEVER_SOURCES:
        return "never"
    if source == "cypherbench" and db_id not in CYPHERBENCH_TRAIN_GRAPHS:
        return "never"
    if db_id in HELDOUT_DBS.get(source, frozenset()):
        return "heldout"
    if _bucket(f"run3-dev:{qid}", DEV_ONE_IN) == 0:
        return "dev"
    return "train"


def trainable(source: str, db_id: str, qid: str) -> bool:
    return split_of(source, db_id, qid) == "train"


def main() -> int:
    from sqlrl.run3 import pools, settings

    out = {}
    for src in ("bird", "cypherbench"):
        c: Counter = Counter()
        for f in sorted((settings.get().pools / "goldcheck" / src).glob("*.jsonl")):
            for r in pools.load_checks(src, f.stem):
                c[split_of(src, f.stem, r["id"])] += 1
        out[src] = dict(c)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
