# SQL-CYPHER-RL-ENVIRONMENT

Reinforcement learning for a small database agent: SQL and Cypher, on a single 16 GB consumer GPU
(AMD RX 7800 XT).

The starting point is
[trjxter/Spark-X2.6-4B-SQL-Cypher-Agent](https://huggingface.co/trjxter/Spark-X2.6-4B-SQL-Cypher-Agent),
a 4B tool-calling agent that answers questions about a database with one read-only SQLite SQL or
Neo4j Cypher query (43.7% execution accuracy on its test set, against 32.5% for the base model).
It was fine-tuned in [nfl-ask-trainer](https://github.com/RishiJain905/nfl-ask-trainer). This repo
holds the RL environments, the training loop and the measurements.

## The plan

[`documentation/rl.md`](documentation/rl.md):

1. **The sampling baseline:** each evalset question answered 8 times, sampled; pass@1 against
   pass@8 shows how much room RL has.
2. **The method pilots:** short runs of GRPO-family methods, SDPO and SRPO, scored the same way.
3. **The real RL run:** single-turn with an execution reward, then a multi-turn agent in real
   databases.

## Where it stands

**Step 1 is done** (2026-11-01): greedy 43.2%, sampled mean 36.9%, **pass@8 62.6%** on the
200-question evalset; 94 of 198 questions are right sometimes, the pool RL learns from
([`documentation/results/step1-baseline.md`](documentation/results/step1-baseline.md)).
**Step 2 is running:** two pilots of 10 RL steps, GRPO and SRPO
([`documentation/pilots/TRACKER.md`](documentation/pilots/TRACKER.md)). Progress:
[`documentation/PROGRESS.md`](documentation/PROGRESS.md).

## Running it

Windows Python runs next to llama-server; the checker runs inside WSL. Paths come from
`configs/paths.local.toml` (copy `configs/paths.example.toml`; it's git-ignored).

| Command | What it does |
|---|---|
| `C:\Python314\python.exe -I scripts\step1_baseline\run.py --name <run>` | the sampling baseline, start to finish (~45 min; start it detached with no console) |
| `C:\Python314\python.exe -I scripts\step1_baseline\feed.py --run-dir <dir>` | one tick for the live page |
| `PYTHONPATH=src <venv>/bin/python -m sqlrl.step1_baseline.score --run-dir <dir> --wandb` (WSL) | score a baseline's sample files, write `summary.json`, log to W&B |
| `PYTHONPATH=src <venv>/bin/python -m sqlrl.step2_pilots.pool` (WSL) | step 2's question pool (`pool-v1`) |
| `C:\Python314\python.exe -I scripts\step2_pilots\run.py --pilot <id>` | a pilot: its RL steps, then its evaluation (start it hidden and detached); also resumes |
| `C:\Python314\python.exe -I scripts\step2_pilots\pilot_queue.py --after <id> <id> ...` | pilots one after another |
| `C:\Python314\python.exe -I scripts\step2_pilots\pause.py --pilot <id> [--after-step]` | free the GPU now (or after the current step) |
| `C:\Python314\python.exe -I scripts\step2_pilots\feed.py --pilot <id>` | one tick for the pilots' live page |
| `PYTHONPATH=src <venv>/bin/python -m pytest -q` / `ruff check .` (WSL) | the tests (CPU) and lint |

## Documentation

- [`documentation/rl.md`](documentation/rl.md): the plan
- [`documentation/rl-runs.md`](documentation/rl-runs.md): the two RL runs, the methods, the RL facts that matter here, the open questions
- [`documentation/run1-design.md`](documentation/run1-design.md): run 1's design (draft): environment, questions, method, length, stop rules
- [`documentation/pilots/`](documentation/pilots/): step 2's tracker and live page
- [`documentation/guides/sampling-baseline.md`](documentation/guides/sampling-baseline.md): pass@k, how the baseline works, how to run it
- [`documentation/results/`](documentation/results/): each step's numbers
- [`documentation/decisions.md`](documentation/decisions.md): every choice made along the way
- [`documentation/dashboard/`](documentation/dashboard/): the live page
