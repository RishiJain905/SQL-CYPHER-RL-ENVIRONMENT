# The sampling baseline (pass@k)

How much room reinforcement learning has, measured before any RL, and the yardstick every later
step is scored with.

## What it is

Every earlier score of this model is **greedy**: one answer a question, always taking the most
likely next token, so the answer never changes. RL doesn't learn from one fixed answer. Methods
like GRPO ask the same question several times with some randomness (**sampling**), compare the
answers, and push the model toward the ones that checked out. So the useful question before RL
is: *if the model answers 8 times, how often is it right, and on which questions?*

- **pass@1:** right on the first try (greedy, or the average over samples).
- **pass@k:** right in at least one of k tries. pass@8 is roughly the ceiling a GRPO-style method
  can lift pass@1 toward: it can turn "sometimes right" into "right first time", but it can't
  invent answers the model never produces.
- **The buckets:** per question, how many of its 8 answers were right. *Always* (8/8) teaches
  nothing; *sometimes* (1-7) is GRPO's learning pool; *never* (0/8) gives GRPO nothing, and only
  methods that learn from the error message (SDPO, SRPO) might reach it.

pass@k here is the unbiased estimate from n samples with c right: 1 − C(n−c, k) / C(n, k) (Chen
et al., 2021, the HumanEval paper). It's better than "any right in the first k" because it uses
all 8 samples for every k.

## How it's wired

```
evalset (200 questions)
   │
   ▼
sample.py (Windows) ──► llama-server, released Q8_0, port 8092, thinking budget 1,024
   │  8 answers a question, temperature 1.0, a seed per answer
   ▼
preds_s0..7.jsonl ──► score.py (WSL) ──► the trainer's checker 4 (SQLite fills, Neo4j)
                           │
                           ├─► scores_s0..7.json (one item per question per sample)
                           ├─► summary.json (passk.summarise)
                           └─► W&B: sql-cypher-rl / <run>
feed.py (Windows) reads all of it ──► dashboard/tick-NNNNN.json ──► the live page
```

- `src/sqlrl/baseline/sample.py` builds each request exactly like the greedy evaluation (the
  trainer's `llamaeval.request_body`: same messages, tools, thinking level), then switches
  sampling on: temperature 1.0, top-p 1.0, top-k off, **min-p 0** (llama.cpp's default 0.05 would
  quietly cut the tail), seed `1000 * (k + 1) + i`. The same seed gives the same answer.
- `score.py` scores each sample file with the trainer's `publiceval.verify` (checker 4), the
  same code that scored the release. Smoke test: re-scoring the release's own greedy answers gives
  43.17% on 183 questions, exactly the release number.
- An answer the checker can't decide (`exec_ok` None: no database tells the two queries apart)
  counts neither way. A question's pass@k uses its decided answers.
- `run.py` does it all: server up, sample, server down, score, `DONE`. `feed.py` turns the run's
  files into snapshots for the page.

## Running it

```
C:\Python314\python.exe -I scripts\baseline.py --name <run>          (Windows, ~45 min)
C:\Python314\python.exe -I scripts\baseline_feed.py --run-dir <dir>  (a dashboard tick)
```

Start `baseline.py` as a detached process with **no console** (`Win32_Process.Create` with
`CreateFlags = 0x208`, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP), never through `cmd /c`.
Needs the trainer's Neo4j (`asktrainer-neo4j`) and nothing else on the GPU. A restart resumes:
answered questions and scored files are kept. Scoring alone, inside WSL:

```
cd <repo> && PYTHONPATH=src <trainer venv>/bin/python -m sqlrl.baseline.score --run-dir <wsl dir> --wandb
```

## What it produces

Under `<data>/rl/baseline/<run>/`: `preds_s<k>.jsonl` (answer, finish reason, seconds, tokens,
seed), `scores_s<k>.json` (the checker's items), `summary.json` (overall and per-dataset blocks:
greedy pass@1, mean@8, pass@1-8, always / sometimes / never, the 0-8 histogram, undecided
answers; and every question's right / scored count), `events.jsonl`, `dashboard/`.

**W&B** (project `sql-cypher-rl`, job type `baseline`): the summary holds `<set>/pass@k`,
`<set>/mean_at8`, `<set>/greedy_pass1`, the bucket counts; two tables, `pass_at_k` (set, k,
pass@k: plot k against pass@k, one line per set) and `questions` (id, set, greedy, right,
scored).

**The live page** ([Spark RL Baseline](https://claude.ai/artifact/JcR5uM76GHeJUCf1UkHooJ),
`documentation/dashboard/`): progress while sampling and scoring, then the per-dataset table,
the pass@k curves, the buckets, the histogram and a grid of all 200 questions (ringed: greedy
wrong but some sample right).

## The first run's numbers (2026-11-01)

Greedy 43.2%, sampled mean 36.9%, **pass@8 62.6%**; 94 of 198 questions "sometimes", 74
"never", 30 "always". CypherBench has the most room (44.9% to 80.0%), text2cypher is mostly
"never" (33 of 49). Full table and reading: `documentation/results/step1-baseline.md`.

## Limits

- 200 questions: about ±3.5 points of noise on any single rate, so compare runs question by
  question (a sign test on the per-question counts), not by the headline alone.
- text2cypher and SQaLe are checked on random databases built from the schema; many answers
  there can't be decided, which makes those two the noisiest.
- 8 samples: pass@8 is the ceiling *at 8 tries*. More samples would show a higher ceiling.

## Changing it safely

The settings live in `sample.SETTINGS`. Changing any of them (temperature, samples, budget) makes
a new yardstick: give the run a new name, and re-measure the baseline with the new settings
before comparing an RL model against it. Every later step must use the same settings as the
baseline it's compared with.

## Lessons from running it

- A `cmd /c` wrapper's console passed a Ctrl+C to the driver and killed it mid-sampling, and
  later a stray Ctrl+C ended `wsl.exe` 6 s into scoring even with the driver detached. The run
  resumed both times with nothing lost. For multi-day RL runs, start long jobs from inside WSL
  (`setsid nohup`) or find the source first.
- A file appended to from WSL keeps its old Windows modified time; the feed uses the newest
  scores file as the "still working" signal instead.
