# Run 3: operating the environment

**Status (2026-11-23): the environment is built and working on the PC; the run has not
started.** Nothing here trains. The design is `design.md`; why the environment is built the way
it is, `../guides/run3-environment.md`; Colab, `../guides/colab.md`; decisions R21-R23.

## Where things are

| What | Where |
|---|---|
| Code | `src/sqlrl/run3/` (nothing imported from runs 1-2) |
| Tests (CPU) | `tests/run3/` |
| Colab scripts | `scripts/run3/colab/` (`bootstrap.sh`, `gpu_smoke.py`) |
| This machine's settings | `configs/run3.local.toml` (git-ignored; template in `settings.py`) |
| Data | `<data>/run3/`: `raw/` (downloads), `dbs/<source>/*.sqlite`, `duckdb/`, `pools/goldcheck/`, `smoke/` |

The modules, in the order an episode uses them:

| Module | Does |
|---|---|
| `settings.py` | the data folder, the engines' addresses, the trainer's checker |
| `sources.py` | BIRD, BIRD dev, Spider, CypherBench as one question record |
| `engines.py` | the five engines behind one read-only `run(query)` |
| `porting.py` | a SQLite database into DuckDB, PostgreSQL, MySQL |
| `gold.py` | the gold in each dialect (sqlglot), the gold check, the comparison |
| `graphs.py` | graph copies and their round trip, CypherBench graphs, the one-graph host |
| `pools.py` | the gold check per database, tasks, non-answers |
| `protocol.py` | the setting drawn per episode, the tools, the prompt, reading a reply |
| `env.py` | the schema text, the helper tools, what a result looks like |
| `episode.py` | one episode and its reward |
| `policy.py` | the model behind llama-server as a policy |
| `smoke.py`, `check_env.py`, `bundle.py` | the smoke, the machine check, the Colab data bundle |

## The PC

The engines are three Docker containers, started once and restarted with Docker:

```
run3-postgres  postgres:16           127.0.0.1:55432
run3-mysql     mysql:8.4             127.0.0.1:53306   (--lower-case-table-names=1)
run3-neo4j     neo4j:5.26-community  127.0.0.1:7477 (HTTP), 7690 (Bolt)
```

Python: the system Python with `duckdb`, `psycopg[binary]`, `pymysql`, `sqlglot`. Commands run
from the repo with `PYTHONPATH=src`:

```
python -m sqlrl.run3.porting setup                          # the read-only role, once
python -m sqlrl.run3.porting port --source birddev --db superhero
python -m sqlrl.run3.pools check --source birddev --db superhero   # or --all
python -m sqlrl.run3.pools summary --source birddev
python -m sqlrl.run3.check_env                              # is everything working?
python -m sqlrl.run3.smoke --db superhero --per-engine 4 --graph soccer --two-source 4 \
    --out <data>/run3/smoke/<name>                          # needs a llama-server
pytest tests/run3 -q
```

`check_env` on 2026-11-23: every line `ok` (five engines, each refuses a write on its own, the
same rows on every SQL engine, a scripted episode right on each, the graph copy's round trip).

## The smokes so far (step 40, greedy, untrained, the evaluation's fixed setting)

| Smoke | Right | Notes |
|---|---|---|
| `smoke1`: 3 BIRD-dev databases, 8 a SQL engine, 12 non-answers | SQLite 6/8, PostgreSQL 4/8, MySQL 4/8, DuckDB 3/8; write requests declined 6/6; wrong-database declined 5/6 | 1.4 minutes |
| `smoke2`: 4 an engine + CypherBench soccer + 4 two-source | SQLite 3/4, PostgreSQL 3/4, DuckDB 2/4, MySQL 2/4, Neo4j 3/4, two-source 3/4 | two episodes called a tool after the last allowed call |

Small samples: they prove the pieces work together, not how good the model is. Phase 0's
pass@8 is the measurement.

## Colab (not used yet)

1. 🧑 **Rishi signs the CLI in, once:** in WSL, `colab sessions` prints a Google link; open it,
   allow, paste the code back. Then `colab usage` shows the balance and the A100's rate.
2. **The data bundle** (PC): `python -m sqlrl.run3.bundle --out <file>.tar.gz --source bird
   --source birddev`.
3. ✋ **The paid smoke (about 0.5 h, Rishi's OK first):** `colab new -s run3 --gpu A100`,
   `colab upload` the bundle and step 40's PEFT adapter, run `bootstrap.sh`, then
   `gpu_smoke.py`; read the heartbeat file a while later; `colab stop -s run3`. It answers: which
   A100 (40 or 80 GB), whether a detached process survives, whether the model loads on NVIDIA
   with the adapter unmerged, whether vLLM loads it.

## What phase 0 still has to build

- BIRD train (downloaded, 69 databases): port the ones under the size limit, gold-check them;
  Spider (download); CypherBench's other graphs.
- Held-out databases (design 3.6) chosen and written down before any measurement.
- The anchor check on the VM (design 5.1) and the trainer on NVIDIA.
- Phase 0's measurements (pass@8 of step 40 per pool, engine and harness), the useful band.
- Ambiguous questions and graph-copy golds: both need the teacher (paid, D1).
