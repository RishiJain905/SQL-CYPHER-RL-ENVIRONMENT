# The multi-turn environment (run 2)

This guide explains the environment the agent trains in during run 2: what an "environment" is
here, how one episode goes, what it produces, how to look at it, and how to change it safely.
The design choices and their reasons are in `run2/design.md`.

## What it is

In run 1 the model wrote one query and never saw it run. The "environment" was only a reward
function: the checker ran the query after the fact.

In run 2 the model **acts and observes**. It calls a query tool, the query really runs on a
database, the result comes back into the conversation, and the model decides what to do next.
That loop, plus the rule for what is rewarded, is the environment. Three terms:

- **Episode:** one attempt at one question, from the prompt to the model's last word.
- **Turn:** one thing the model writes: a thought and a tool call, or a closing reply.
- **Observation:** what the environment sends back after a call: rows, `0 rows`, or an error.

## One episode, step by step

A real shape (the texts are shortened):

```
system   You answer questions about the database below by calling the `run_query` tool
         with one read-only SQLite SQL query at a time. ... (schema, examples, rules)
user     Which customers spent the most on paid invoices?
model    <think>join cust to invoices ...</think>
         run_query(query = SELECT c.nam, SUM(i.total) ...)
tool     Error: no such column: c.nam
model    <think>the column is cusstrname</think>
         run_query(query = SELECT c.cusstrname, SUM(i.total) ...)
tool     cusstrname | SUM(i.total)
         Alldridge | 912.5
         (1 of 14 rows)
model    <think>one row was asked for; add ORDER BY and LIMIT 1 ...</think>
         run_query(query = SELECT c.cusstrname, SUM(i.total) ... ORDER BY 2 DESC LIMIT 1)
tool     cusstrname | SUM(i.total)
         Marlowe | 2140.0
model    Marlowe spent the most.
```

The episode's **answer is the last query that ran** (the third one). It is scored by the same
checker as a single-turn answer. The first query is scored as well, so we can count how often
turns fixed a wrong first answer and how often they broke a right one.

Limits: 3 calls. A fourth is not run. No call at all is a malformed answer.

**One dataset keeps the old protocol.** SQaLe questions get the original prompt and one query,
and nothing is shown (decisions R16; why is under "Limits" below). Such an episode ends at its
first call (`end` is `single`). The setting is `single_turn_sets`.

## The pieces

| File (`src/sqlrl/run2/`) | What it does | Runs on |
|---|---|---|
| `prompt.py` | The SFT system prompt with the "one query" rule changed | both |
| `calls.py` | Reads a turn's tool call; writes a query back as a call for the checker | both |
| `env.py` | The look-around databases, read-only execution, the result text | Windows |
| `guards.py` | The constants rule and the copied-literal canary | both |
| `episodes.py` | The episode loop against llama-server, and the scheduler | Windows |
| `scoring.py` | Scores an episode's final and first query with the checker | WSL |
| `summary.py` | Decided and strict scores, fixed / broken, how turns were used | both |
| `rewards.py`, `canaries.py` | The reward, the group statistics, the stop rules | both |
| `train_step.py` | The update on one sequence per episode | WSL (GPU) |
| `eval_score.py` | Scores an evaluation with turns, compares it with references | WSL |
| `orchestrate.py`, `pause.py` | The run's loop; freeing the GPU | Windows |
| `phase0.py`, `phase0_score.py` | The no-training tests | Windows / WSL |
| `pool.py` | The questions, the dev slice, the database scripts | both |

"Windows" modules use only the standard library (plus the trainer repo's checker modules, which
are standard library too), because they run next to llama-server.

## The databases the model looks at

| Dataset | Engine | The model looks at |
|---|---|---|
| gretel | SQLite, in memory | the dataset's own tables and rows |
| SQaLe | SQLite, in memory | its schema, filled with 60 random rows a table. Built, but not shown in run 2: SQaLe keeps one query |
| CypherBench | Neo4j (the checker's container) | the benchmark's sampled graph |
| text2cypher | Neo4j | a random graph built from the schema in the prompt |

Two rules shape this:

1. **The reward is never decided on a random look-around copy.** The checker has its own
   databases and keeps using them.
2. **A look-around copy is built without the reference answer.** The builder gets the schema
   and the question. It plants the values the question names (so `WHERE county = 'King'` can
   find rows) and never sees the reference query. `env.public_row` is the one place where the
   reference is dropped; nothing downstream of it has access.

The SQL scripts live in `<data>/run2/pool-v2.db.jsonl` and `evalset.db.jsonl`, built once
inside WSL (`python -m sqlrl.run2.pool build-db`), so the Windows side needs no parquet reader.

### What the model can and can't do there

- It can read: `SELECT`, `WITH ... SELECT`, `MATCH ... RETURN`, schema listings.
- It can't write, attach files, change settings or call procedures. SQLite enforces this with
  an authorizer (a callback that approves each operation; only reads are approved) and
  `query_only`; Neo4j with a read transaction. Both were tested with the guard switched off on
  our side: the engines refuse by themselves.
- A query is stopped after 2 seconds (SQLite) or 5 seconds (Neo4j). Results are capped.
- The errors it reads are the engines' own (`no such column: x`, Neo4j's syntax message), cut
  to one line.

## What comes back

```
name | n            a table: column names, then rows
Alice | 1
(2 of 218 rows)     only when rows were cut

{"name": "Alice", "n": 1}      or one JSON object a row

0 rows
Error: no such column: nam
```

Cells are cut at 40 characters. In training each episode draws a row cap (5, 10 or 20) and one
of the two forms; evaluation always uses 10 rows as a table. The variation is deliberate: other
people's harnesses will format results differently, and the model shouldn't depend on one form.

**A result is data, never markup.** The result text is tokenised as plain text and placed
between the chat template's own wrapper tokens. A cell containing the template's control text
can't end the tool message or start a turn.

## What a run produces

Per step, in `<data>/run2/<run>/step-000k/`:

- `rollouts.jsonl`: one line per episode. The useful fields for reading by hand:
  `turns` (each turn's text, the call's kind, the observation), `queries` (each query that ran,
  what kind of result, how many rows, which tables it read), `first_query`, `final_query`,
  `n_calls`, `end` (`reply`, `limit`, `no_query`, `cut`, `context`, `single`), `obs` (the result
  format that episode drew).
- `scored.jsonl`: per episode, the final query's verdict (`exec_ok`, `reason`), the first
  query's (`first_ok`), and the flags `hardcoded`, `copied`, `look_leak`, `loop`, `dup`.
- `metrics.json`: the step's numbers (also a line of `train_log.jsonl`).

Per evaluation, in `eval-000k/`: `episodes.jsonl`, `scored.jsonl`, `summary.json`, and
`protocol.json`: the prompt variant, the call limit, the one-query datasets, the result format
and whether thoughts stayed in the history. An episode file is only ever continued under the
protocol it was started with, and the run checks its start reference against its own.

## How to look at it

Read one episode:

```python
import json
eps = [json.loads(x) for x in open("step-0001/rollouts.jsonl", encoding="utf-8")]
ep = eps[0]
for t in ep["turns"]:
    print(t["t"], t["call"], "\n", t["text"][-300:], "\n ->", t.get("obs"))
print(ep["end"], ep["n_calls"], ep["final_query"])
```

The no-training tests (`scripts/run2/phase0.py`) are the quickest way to see the environment
work without training anything:

| Command | What it shows | GPU |
|---|---|---|
| `timing` | how long each look-around database takes to build; Neo4j's time limit and read-only mode | no |
| `roundtrip` | the context the engine builds equals the chat template's rendering | yes |
| `anchor` | the start model and the engine reproduce the kept evaluation | yes |
| `wording` | the prompt variants on the dev slice | yes |
| `heldout` | the evaluation set with turns on an untrained model | yes |

And one check that only the scoring side may run: `python -m sqlrl.run2.phase0_score lookcheck`
runs each dev-slice reference on its look-around copy and counts how often it returns rows.

## Changing it safely

- **The prompt wording:** add a variant to `prompt.py`, test it with `phase0.py wording` on the
  dev slice. Never tune wording on the evaluation set.
- **The result format:** `env.render` and the `obs` setting. If you add a form, keep the
  evaluation's fixed, or old evaluations stop being comparable.
- **Which datasets get turns:** `single_turn_sets` in the run's settings (a new run only;
  `["sqale"]` by default). Changing it changes the task the evaluations measure, so the start
  reference (`phase0.py heldout --samples 8`) has to be made again; the run refuses to start
  against one made under another protocol.
- **The look-around fill:** `env.question_literals` and the `FILL_*` / `PLANT_P` constants.
  Rerun `lookcheck` after any change and keep `tests/run2`'s test that the fill never contains a
  reference-only value.
- **Never:** pass a row to the rollout side without `env.public_row`; score on a look-around
  copy; let a result's text be tokenised with special tokens.

## Limits

- On SQaLe about a quarter of correct queries return nothing on the look-around copy, because
  they filter on a key or a constant the question doesn't spell out. The model would be shown
  `0 rows` for a right answer. On the dev slice turns fixed 1 SQaLe answer and broke 6, so
  SQaLe keeps one query. This is a limit of the environment, not of the model: against a real
  database a right query returns its rows. It also means run 2 can't teach or measure turns on
  SQaLe-like schemas (large, many tables).
- The one-query rule goes by dataset, and real use has no dataset label. A deployment picks one
  prompt: with turns (what three datasets train) or the original (what SQaLe trains).
- **The pool keeps only questions whose reference's result says something** (rows with a value
  that is not NULL, zero or empty; for gretel on the dataset's own rows). So in training, and
  on the dev slice, an empty result always means a wrong query. In real use, and on the
  evaluation set, it is often the right answer: 13 of 50 gretel and 32 of 50 text2cypher
  evaluation references say nothing on the copy the model looks at. The untrained model
  rewrites a right answer after such a result about half the time, and loses 4 points on the
  evaluation set with turns (`results/run2-phase0.md`, sections 7 and 8). The 627 gretel
  questions the pool rejected for this are still in the training split (ids in
  `pool-v2.checked.jsonl`). This is the open problem of run 2.
- The constants rule catches the plain form of answer-copying, not every form. The canaries
  watch the rest.
- One Neo4j holds one graph at a time. text2cypher questions run one after another; an
  evaluation spends most of its time there.
- Checker strictness (an extra column, an order) is part of what the model sees as "wrong".
