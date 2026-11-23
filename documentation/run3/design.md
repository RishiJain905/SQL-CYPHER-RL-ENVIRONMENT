# Run 3 design: the database agent

**Status: approved 2026-11-23 (R21); the environment is built and working on the PC (R22, `README.md`); the run has not started.** This file says what run 3 is for, what its environment looks like,
where its data comes from, how it is rewarded and trained, where it runs, and what has to be
decided before any code. It follows `rl.md`'s stage C in spirit, but the stage is general: the
model never trains on the NFL data.

## 0. The goal in one paragraph

A small model that sits between a person and a database: it is told which database it is connected
to (SQLite, DuckDB, PostgreSQL, MySQL or Neo4j), it decides for itself whether one query is enough
or whether it should first look around (list tables, read a few rows, look up how a value is
spelled), it recovers when the engine says no, and it ends by **submitting** a query as its answer,
or by declining a question the database can't answer, or by asking which of two things was meant.
Runs 1 and 2 taught it to write a better first query on small, fully described databases. Run 3 puts
it in front of real ones.

## 1. Why: what a real deployment showed

After run 2, step 40 was connected to a real deployment it had never seen: one DuckDB file and two
Neo4j graphs (a sports-statistics project, kept outside this repository), behind a read-only guard,
with 165 questions whose answers were known. The same harness and grader ran a large hosted model
for comparison.

| | Step 40 | Large hosted model |
|---|---|---|
| Right answers (strict) | 26.7% | 65.5% |
| First query alone | 23.6% | 74.5% |
| SQL / graph answers | 31.7% / 12.5% | 73.0% / 53.6% |
| Declined the trick questions | 7 of 14 | 13 of 14 |

What went wrong, and what run 2's environment had or didn't have:

| Seen in the deployment | What run 2 had | What run 3 changes |
|---|---|---|
| SQLite habits fail on DuckDB (a bare column beside `GROUP BY`) | SQL checked on SQLite only | Every question checked on the engines it is valid on: SQLite, DuckDB, PostgreSQL, MySQL |
| Lost in a large schema (6,500 prompt tokens, three databases) | 1-3 small tables, all in the prompt | Real databases with many tables, wide columns and dirty values |
| The wrong database chosen (63% right) | One tool, one database a question | Episodes with two or more sources to choose between |
| Never looks anything up | Everything it needed was in the prompt | A prompt that may hold the full schema, only the table names, or nothing, and tools to explore |
| The same failing query sent again | Errors were rare and fixing them was rarely rewarded | A real environment produces real errors; recovering is what earns the reward |
| "The last query is the answer" | Built into the protocol | An explicit `submit` action; exploring is free of that risk |
| Half the trick questions answered anyway | No question where declining was right | A small share of unanswerable, ambiguous and write-request questions |

The same table, read the other way: none of these are about sports. They are what any deployment
asks of the model. The deployment stays as a **test the model never trains on**; run 3 succeeds if
its 26.7% rises without one question from it in training.

## 2. The environment

### 2.1 Engines

| Engine | How it runs (no Docker needed) | Read-only by |
|---|---|---|
| SQLite | the Python module | `query_only`, an authorizer (run 2's) |
| DuckDB | the Python package | `read_only=True` connection, no file or network functions |
| PostgreSQL | the system package | a role with `SELECT` only, `default_transaction_read_only`, `statement_timeout` |
| MySQL | the system package | a user with `SELECT` only, `max_execution_time` |
| Neo4j | the server tarball and a Java runtime | read transactions, a server-side time limit, no procedures but the schema listings (run 2's) |

All engines: a time limit a query, a cap on rows and on the size of a value, the engine's own error
text. Others (SQL Server, MariaDB, Memgraph) are left for later.

### 2.2 An episode

1. **The prompt** names the engine and its version, says what the tools are, and shows the schema
   in one of three ways, drawn per episode: the full schema, the table (or label) names only, or
   nothing. A real harness may do any of the three.
2. The model thinks and calls a tool. The environment runs it and returns the result as a tool
   message (run 2's token-by-token context: results are read, never trained, and can't forge a
   turn).
3. It may call again, up to a limit (8 calls is the starting value).
4. It ends with exactly one of:
   - `submit(query)`: the answer. The grader runs this query on the scoring copy.
   - `decline(reason)`: the database can't answer this.
   - `clarify(options)`: the question could mean two things; it names them.
   
   Ending without one of these is malformed.

### 2.3 Tools, and how the harness varies

Real harnesses differ, so the tool set is drawn per episode:

- **Query only:** one `run_query` tool a database. Exploring means writing the engine's own
  catalog queries (`information_schema`, `SHOW TABLES`, `PRAGMA table_info`, `CALL db.labels()`).
- **With helpers:** `run_query` plus `list_tables`, `describe(table)`, `sample_rows(table, n)` and
  `find_values(column, text)` (rows whose value looks like the text: how "LA Rams" is stored).
- **Several sources:** two or more `run_query` tools, each tied to its own database (a SQL database
  and a graph copy of the same data, or two different databases), and the model picks.

Tool names are drawn from a few spellings (`run_query`, `execute_sql`, `query_graph`, ...), as run 2's
data already had, so the model learns the role, not the name.

### 2.4 What comes back

- Rows as a table, JSON lines or CSV, with 5, 10, 20 or 50 rows shown and `(10 of 2,318 rows)` when
  cut (run 2 drew the format the same way).
- `0 rows`.
- The engine's error: its first line, or its first line plus the hint lines (PostgreSQL's `HINT:`,
  DuckDB's "Either add it to the GROUP BY list ..."), drawn per episode, because harnesses differ.
- A timeout, a permission refusal, a "result too large" notice.

### 2.5 The scenarios it must meet

Each of these is produced on purpose somewhere in the pools, so the model meets it in training:

- dialect differences: `LIMIT` against `TOP`, identifier quoting and case, `||` against `CONCAT`,
  date functions, integer division, NULL ordering, `ILIKE`, schemas (`schema.table`) in PostgreSQL;
- messy databases: names with spaces or capitals, JSON columns, enums, views, codes the question
  doesn't spell out, foreign keys that aren't declared;
- an empty result that is the right answer (run 2's open problem: on real data it is common and
  needs no special questions);
- results too large to show, slow queries, permission refusals;
- unanswerable questions (decline), ambiguous ones (clarify), write requests (decline);
- a question that needs two sources (the graph for the path, SQL for the totals) is out of scope for
  run 3: each episode's answer comes from one source.

## 3. Data

### 3.1 Pools

| Pool | What it is | Used for | Licence |
|---|---|---|---|
| BIRD train | 9,428 questions on 69 real databases (Kaggle-style, dirty values, long schemas); the full benchmark is 95 databases, 33.4 GB | SQL, the main new pool | CC BY-SA 4.0 |
| Spider 1.0 train | multi-table questions on many small-to-medium databases | SQL, breadth | CC BY-SA 4.0 (to confirm) |
| SynSQL-2.5M | synthetic questions on thousands of generated multi-table databases | SQL, scale if BIRD and Spider are not enough | Apache 2.0 (to confirm) |
| CypherBench | over 10,000 questions on 11 large property graphs built from Wikidata | Cypher, real graphs (already served by the checker's Neo4j) | Apache 2.0 |
| text2cypher | the pool runs 1 and 2 used | Cypher, kept | as in run 1 |
| gretel, SQaLe | the pools runs 1 and 2 used | a small share, so the run-2 evalset stays comparable | as in run 1 |
| Graph copies (3.3) | BIRD and Spider databases converted to Neo4j | Cypher volume and source choice | follows the source |
| Non-answers (3.4) | generated | decline and clarify | ours |

Every licence is checked from the dataset's own page before any of its data is committed, and data
files never go into this repository (as now: they live in the data folder).

### 3.2 One question, several engines

Each SQLite database is loaded into DuckDB, PostgreSQL and MySQL. The gold query is translated with
`sqlglot` into each dialect and run there; a (question, engine) pair is kept only when the
translated gold's result equals the SQLite gold's result under the checker's rules. A question is
valid on the engines it passed, and the pool records which. This gives dialects without new
questions, and every gold answer is checked on the engine it is used with.

### 3.3 Graph copies of the SQL databases

The reward compares results, not query text, so a graph question needs only the right result, not
a gold Cypher query. A relational database becomes a property graph by fixed rules:

- a row of a table becomes a node labelled with the table's name, its columns as properties;
- a foreign key becomes a relationship from the row to the row it points at;
- a table that only joins two others (two foreign keys and nothing else, or a few columns) becomes
  relationships, its other columns as their properties.

The conversion is checked by a round trip (the graph read back into tables equals the source).
A question goes into the graph pool only when its SQL gold result can be reproduced on the graph:
by a teacher's Cypher query in phase 0 (3.5), which is then also a warm-start example. Questions
whose answer depends on what the conversion changed (a join table's own key) are dropped. This is
new work; it starts on a handful of databases and grows only if the round trip and the teacher
check hold.

### 3.4 Non-answers

- **Unanswerable:** a question rewritten to ask for something the database doesn't hold (a column
  or table taken out of its copy, or an entity swapped for one with no rows), checked by running
  the original gold: it must fail or change meaning.
- **Ambiguous:** a question rewritten so two columns fit (a database with `home_score` and
  `away_score` asked for "the score"); the right reply is `clarify` naming both.
- **Write requests:** "delete", "update", "add" rewrites of real questions.

About 10% of a step's questions. Declining an answerable question earns nothing and is counted
(3.6), so declining is never the safe bet.

### 3.5 The teacher

A large hosted model (the one that scored 65.5% above) runs the run-3 environment on training
questions: exploring, submitting, declining. Its episodes are graded like the model's; only the
right ones are kept. They serve twice: as warm-start examples (5.2) and as the Cypher proof for
graph copies (3.3). This costs money: a 200-episode pilot measures the price an episode before any
larger batch, and the batch size is a decision for the owner (section 9).

### 3.6 Splits

- BIRD's dev set, CypherBench's test split, run 2's evalset and test set, and the real deployment
  in section 1 are never trained on.
- **Whole databases are held out**, not only questions: a share of BIRD's and CypherBench's
  training databases is set aside, so "a database it has never seen" is measured directly.
- Dev slices (no step trains on them) for the warm start and for each RL phase.

## 4. The reward

An episode gets one reward, from how it ended.

| Ending | Reward |
|---|---|
| `submit`, right | 1, less 0.02 for each call after the second (at least 0.9) |
| `submit`, wrong | 0 |
| `decline` on an unanswerable question or a write request | 1 |
| `decline` on an answerable question | 0 (and counted) |
| `clarify` naming the meanings the question has | 1 |
| `clarify` on a clear question | 0 (and counted) |
| No ending, or a malformed one | -0.1 |
| Right rows that were not computed (constants) | 0 |

"Right" is the checker's comparison of the submitted query's result on the scoring copy with the
gold result (run 2's checker 4: column order free, row order free unless the question asks for an
order, numbers within a tolerance), with each engine's types normalised (decimals, dates,
booleans). The call cost is small on purpose: looking around is supposed to pay; the cost only
breaks ties between equally right episodes. Its value is checked on the pilot (8.1).

**Ways to cheat it, and the guards:**

| Cheat | Guard |
|---|---|
| Read the answer while exploring, then submit it as constants | A submitted query must read a table (SQL) or match a pattern (Cypher), run 2's `guards.hardcoded` |
| Explore without end | The call limit, the call cost, a stop rule on calls an episode |
| Stall the engine | Time limits, row and size caps |
| Decline everything hard | Declining an answerable question earns 0 while an attempt can earn 1; a stop rule on the false-decline rate |
| Clarify everything | The same, for clarify |
| Submit everything (`SELECT *`) | The result must match the gold result's rows and columns; run 2's `SELECT *` watch |
| Write to the database | Read-only roles in every engine |

## 5. Training

### 5.1 Where it starts

From run 2's step 40. The lesson of its export (`results/run2.md`, `run2/export.py`): the RL changes
are smaller than bf16's step (at most 0.00017 a weight). Merged into bf16 weights, only 7.8% of them
survived and the model fell back to about the SFT model's score (46.7% against 60.2%). So:

- **The change is never rounded into bf16.** The base weights sit exactly on bf16's grid, so a change
  smaller than half a grid step rounds straight back to the base value: that is how 92% of step 40's
  change vanished. Q8_0's grid is not aligned with the base values, so a merge done in float32 and
  quantised straight to Q8_0 keeps the change on average (run 2's evaluations all ran on such a file:
  60.2% on the evalset, 60.8% on the test set).
- **The trainer never merges.** Step 40 stays an adapter on the released SFT model, as run 2
  continued run 1b's adapter: the base loads in its own bf16 (which loses nothing, because the base
  is what bf16 holds), and the adapter's change is added at run time from its own float32 matrices.
  The warm start and RL train that adapter further (or a new adapter stacked on a frozen step 40;
  phase 0 picks one), and the KL reference is step 40 itself.
- **The sampler gets the new policy every step** in one of two safe ways. With vLLM, no merge: the
  adapter is applied at run time and each step loads the new adapter file (about 120 MB). With
  llama.cpp, a merge every step, done in float32 and quantised straight to Q8_0, as in runs 1 and 2
  (`gguf_merge.py`).
- **Every change of form is checked against step 40's own score** before anything continues from it:
  the model as loaded for training, as served for rollouts (vLLM or llama.cpp), and the warm start's
  output. The check is run 2's anchor: greedy on the run-2 evalset must agree with step 40's
  evaluation question by question (run 2's anchor agreed on 183 of 186) and score within noise of
  60.2. A form that falls toward 46.7 stops the run. Phase 0 runs it first; it is the gate's first
  line.
- **The release** is merged in float32 (`run2/export.py`'s default) and every quant is measured the
  same way.

### 5.2 Phases of the one run

The Colab budget for run 3 is **5 to 6 hours at most** (section 7), so only RL runs there; every
other phase runs on the home PC, free.

| Phase | Where | What | Gate to the next |
|---|---|---|---|
| 0. Build and measure (no training) | PC, plus a 30-minute Colab smoke | The engines, the pools (3.2-3.4), the harness variants, the grader; the round trip for graph copies; the teacher pilot; **pass@8 of step 40** on every pool, engine and harness variant; the smoke measures steps an hour on the GPU Colab gives (a G4 or an A100) | The pools have enough questions in the useful band (8.1); the step count that fits the budget is known; the owner reads the numbers |
| 1. Warm start | PC (Colab for at most 1 hour only if its episodes don't fit the 16 GB card) | Fine-tuning on the teacher's right episodes and the model's own right episodes, mixed so its own style stays | No loss on run 2's evalset; better on the run-3 dev slices; it now uses `submit`, explores when the prompt hides the schema, declines where it should |
| 2. RL | Colab, about 4.5 hours | All engines, all harness variants, the non-answers, one database an episode; **several-source episodes join at a mid-run check** (a share of each step, the graph copies offered beside their SQL source) once the one-database episodes improve on the dev slice | The run-3 dev slices and held-out databases improve; no stop rule fires |
| 3. Evaluation | PC (llama.cpp, as runs 1 and 2) | Everything in 3.6, once, on the kept step | — |

Each phase saves a checkpoint that the next continues from; a gate that fails stops the run for a
look, as in run 2. Phase 2 also stops itself, and saves, when its hours are used (`max_hours` in the
run's config), whatever step it is on.

### 5.3 The method

SRPO as in run 2, because it worked there: 8 episodes a question, the advantage
(reward - group mean) / max(group std, 0.25), every token of an episode sharing it, a wrong episode
with a right sibling distilled toward the EMA teacher shown that sibling's submitted query (first
turn only). The KL penalty to the warm-start model starts at 0.20 (run 2 needed that from step 12 on).
The card's memory allows episodes of up to about 16,000 tokens (run 2: 3,840 on 16 GB), which
exploring needs. Questions are drawn mostly from the useful band (8.1), so that most groups hold a
right and a wrong episode.

## 6. Measuring it

| Number | What it is |
|---|---|
| **Greedy, right answers, on held-out databases**, by engine | The headline: a database and an engine pair it has never trained on |
| The same on BIRD dev and CypherBench test | Public numbers others report |
| pass@8 | Headroom: how much RL can still draw out |
| Run 2's evalset | No loss on what runs 1 and 2 taught |
| False declines, false clarifies, calls an episode, explore share | Behaviour |
| **The real deployment** (section 1) | The transfer test, after the warm start and at the end |

## 7. Where it runs

### 7.1 The split

- **The home PC (one 16 GB AMD card):** building the environment, phase 0, the warm start (if its
  episodes fit) and the evaluation, free, at any pace.
- **Google Colab, a G4 (RTX PRO 6000, 96 GB) or an A100:** RL only, **5 to 6 hours at most** for
  run 3 (lowered from 5 to 7 on 2026-11-23, R24), not the whole account. The G4 is asked for
  first (more memory, in the account's tier; about 9-10 units an hour by users' measurements),
  the A100 second (about 7 an hour, 80 GB). 6 hours is about 54-60 units on a G4, 42 on an A100,
  of the account's 202.5. Rates change; `colab usage` shows the real one once a VM runs.

| Use | Colab hours |
|---|---|
| Smoke in phase 0 (which GPU, setup works, throughput) | 0.5 |
| RL (phase 2) | about 4.5 |
| Setup and restarts | 0.5 |
| Warm start, only if it can't run on the PC | up to 0.5 |
| **Cap** | **6** |

The smoke's steps-an-hour sets the run's length: the number of steps is chosen to fill about 4.5
hours, and `max_hours` stops the run, saved, if it runs long. With a fixed budget the choice that
matters is signal a step: questions come from the useful band (8.1), so few groups are wasted on
all-right or all-wrong.

### 7.2 A Colab session

- A setup cell mounts Google Drive and copies the prepared data to the machine's own disk (reading
  from Drive directly is slow): database files, PostgreSQL and MySQL dumps, Neo4j database dumps,
  all built once at home in phase 0.
- PostgreSQL and MySQL install from the system's packages; Neo4j from its tarball with a Java
  runtime; DuckDB and SQLite from pip. No Docker.
- Sessions end without warning, so every step saves to Drive (the adapter, the optimiser, the run's
  state) and a new session resumes by redoing any step without its `TRAINED` mark, as runs 1 and 2
  already do.

### 7.3 Serving the model on NVIDIA

Runs 1 and 2 sampled with llama.cpp (a fork with the model's architecture ported) from a merged
Q8_0 file rewritten every step. The model's architecture is its own (remote code), so:

1. Try vLLM's Transformers backend in phase 0 (fast batching if it loads the model), with the
   adapter applied at run time, never merged into bf16 weights.
2. If not, build the same llama.cpp fork with CUDA: the same sampling path as runs 1 and 2, with
   the merge made from float32 every step.

Either way, the served model passes the anchor check (5.1) before the first rollout.

## 8. Phase 0's measurements

### 8.1 The useful band

RL learns from a group only when it holds a right and a wrong episode. Before training, step 40 (and
the SFT model, for reference) answers every candidate question 8 times on each engine and harness
variant. Questions it always gets right teach nothing; questions it never gets right teach nothing
to RL and go to the warm start instead. The run's pools are drawn mostly from the questions in
between, and phase 0 reports how many there are a pool and engine. If a pool has almost none in
the band, the warm start has to move it first, or it is dropped.

### 8.2 The rest

- The dialect check (3.2): how many questions survive on each engine.
- The graph round trip and the teacher check (3.3).
- The teacher pilot: the price an episode and how often it is right (3.5).
- The call cost (4): a short comparison of two values on the pilot.
- Throughput on the Colab GPU: episodes an hour, steps an hour, how long a step's update takes.

## 9. Decided, and still to decide

Decided (in conversation before this file): one run with phases, not separate runs; general data
only, the real deployment never trained on; the engines in 2.1; Colab for RL, **5 to 6 hours at
most**, the rest of the account's units kept for other work.

To decide before building:

1. **The teacher's budget** for warm-start examples and graph proofs (after the pilot's price).
2. **Batch against steps:** how phase 2's 5 hours are spent (more questions a step or more steps),
   chosen from the smoke's numbers.
3. **The start point:** step 40 (this file's proposal) or the released SFT model; with step 40,
   whether to train its adapter further or stack a new one on it (phase 0 measures both against the
   anchor).
4. **SynSQL:** in from the start, or only if BIRD and Spider leave the useful band thin.
5. **The engines:** the five in 2.1, or more.

## 10. Risks

| Risk | What is done about it |
|---|---|
| 5 hours of RL is too few steps to move the model | Signal a step first (the useful band); the smoke's numbers set batch and steps; if the dev slices are still rising at the cap, a second short Colab block is a later decision, not part of this run |
| The model's architecture doesn't load in vLLM | The llama.cpp CUDA build (7.3) |
| BIRD's databases are large (tens of GB) | Only the training databases phase 0 keeps are prepared; stored compressed on Drive |
| A Colab session ends mid-step | Every step saves; a resume redoes the step |
| The graph conversion changes what a question means | The round trip and the teacher check; it starts small |
| Few questions in the useful band | The warm start, or a different pool mix, decided in phase 0 |
| Longer episodes make credit fuzzier | Outcome-level credit as in run 2; SRPO's distillation stays on the first turn |
| Step 40's change lost in a bf16 merge or a bf16 load (it happened once) | Step 40 stays an adapter; no bf16 merge anywhere; the anchor check (5.1) on every form the model takes, before anything continues from it |
