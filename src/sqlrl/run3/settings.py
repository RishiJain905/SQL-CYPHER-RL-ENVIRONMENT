r"""Where run 3's data lives and how its engines are reached (standard library only).

Run 3 runs on two machines: the home PC (Windows, engines in Docker, bound to 127.0.0.1) and a
Colab VM (Linux, engines installed from packages on their default ports). So nothing here needs
the older `configs/paths.local.toml`: a machine either has `configs/run3.local.toml` (git-ignored,
the PC) or sets `RUN3_DATA` and runs with the defaults (Colab, `scripts/run3/colab/`).

```toml
[run3]
data = 'F:\<folder>\rl\run3'        # pools, engine dumps, episodes, runs
trainer_repo = 'F:\<folder>\nfl-ask-trainer'
[run3.engines]                       # optional: only what differs from the defaults
postgres = "host=127.0.0.1 port=55432"
mysql = "127.0.0.1:53306"
neo4j = "http://127.0.0.1:7477"
```

The engines carry no passwords: each binds to the machine's loopback only, the loader connects
as the engine's administrator without a password, and every model query runs as `run3_reader`,
a role that can only read (`engines.py`). There is nothing secret to store.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO / "configs" / "run3.local.toml"

# The engines' addresses when nothing says otherwise (a Colab VM: the packages' own ports)
DEFAULT_ENGINES = {
    "postgres": "host=127.0.0.1 port=5432",
    "mysql": "127.0.0.1:3306",
    "neo4j": "http://127.0.0.1:7474",
}
READER = "run3_reader"  # the read-only role every model query runs as (PostgreSQL and MySQL)


@dataclass(frozen=True)
class Settings:
    data: Path
    trainer_repo: Path | None
    engines: dict[str, str] = field(default_factory=dict)

    def engine(self, name: str) -> str:
        return self.engines.get(name) or DEFAULT_ENGINES[name]

    # where things go under `data`
    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def dbs(self) -> Path:
        """The SQLite source of every database (`<dbs>/<source>/<db_id>.sqlite`)."""
        return self.data / "dbs"

    @property
    def duckdb_dir(self) -> Path:
        return self.data / "duckdb"

    @property
    def pools(self) -> Path:
        return self.data / "pools"


def _local() -> dict:
    if not LOCAL.exists():
        return {}
    # utf-8-sig: Windows PowerShell writes a byte-order mark that TOML doesn't allow
    return tomllib.loads(LOCAL.read_text(encoding="utf-8-sig")).get("run3", {})


def get() -> Settings:
    raw = _local()
    data = os.environ.get("RUN3_DATA") or raw.get("data")
    if not data:
        raise FileNotFoundError(
            "run 3 has no data folder: write configs/run3.local.toml or set RUN3_DATA"
        )
    trainer = os.environ.get("RUN3_TRAINER_REPO") or raw.get("trainer_repo")
    return Settings(
        data=Path(data),
        trainer_repo=Path(trainer) if trainer else None,
        engines=dict(raw.get("engines") or {}),
    )


def checker():
    """The trainer's `querycheck` and `cyphercheck` (its public repo; cloned on Colab). Run 3
    uses their result comparison (one yardstick with runs 1 and 2) and their graph loader."""
    s = get()
    if s.trainer_repo is None:
        raise FileNotFoundError("run 3 needs the trainer repo: trainer_repo or RUN3_TRAINER_REPO")
    src = str(s.trainer_repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from asktrainer import cyphercheck, querycheck

    return querycheck, cyphercheck
