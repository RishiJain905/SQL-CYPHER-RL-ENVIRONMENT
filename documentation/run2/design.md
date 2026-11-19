# Run 2 design: the multi-turn agent

Run 2 is stage B of `rl.md`. The model still answers a question with a query, but now it sees
what the query returned and may fix it. This file says what was built and why. The choices are
the recommendations of the design page Rishi reviewed (D1 to D14), taken as a whole (decisions
R15). What the no-training tests measured is in `results/run2-phase0.md`; how to run it is in
`RUNBOOK.md`; the environment has its own guide, `guides/run2-environment.md`.

**Status: built and tested, not started. Rishi starts the run.** The first no-training test on
the evaluation set (2026-11-19) landed in the "do not start RL" row of the design page's table:
turns cost the untrained start model 4 points there, because a right query whose result says
nothing gets rewritten, and the training pool held no such question. Rishi chose to fix the
training questions (decisions R17; section 3 below). Under that setup the untrained start
model scores the same with turns as on its first query (52.4%), gains 3 to 8 points where a
look can inform (two runs: net +3 and +7 questions of 87, at most one broken), and still
breaks 7 of 10 right answers whose right result is nothing: the part training now has examples
for (`results/run2-phase0.md`, section 10).

## 1. What changes from run 1b

| | Run 1b | Run 2 |
|---|---|---|
| An answer | one query, never run in front of the model | up to 3 queries; the model reads each result; the last query that ran is the answer. SQaLe and text2cypher questions keep one query (section 3; decisions R16, R17) |
| Starts from | a new LoRA on the SFT model | run 1b's step-30 adapter, continued |
| Drift is measured against | the SFT model | the start adapter, on first-turn tokens |
| The penalty's weight | follows the drift (doubled up to 6.4) | fixed at 0.05 |
| Reward | 1 / 0 / -0.1 | 1 - 0.03 x (calls - 1) when right, else 0 / -0.1 |
| Self-distillation | the whole answer | the first turn only |
| Headline score | greedy, one query | greedy with turns (the first query is reported beside it) |
| Questions | pool steps 1-40 | the same pool from step 41 on, 200 held back as a dev slice; 3 of a step's 10 gretel questions are ones the pool had rejected because the right result is nothing (section 3), 50 of those held back as a second dev slice |
| Code | `src/sqlrl/run1/` with a preset | `src/sqlrl/run2/`, self-contained |

Unchanged: the datasets and the mix (10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher a
step, 8 episodes a question), checker 4 and the rule that rewards come from execution only, the
200-question evaluation set, the rollout server and its flags, the LoRA's shape, the learning
rate (5e-6) and warm-up, the save-and-verify path, the pause tool.

## 2. An episode

1. The prompt is the SFT prompt with one rule changed (`prompt.py`). The first line says "one
   query at a time", and a rule says what a result is for: fix the query, at most 3 calls, stop
   with one short sentence, the last query you ran is your answer. Everything else (schema,
   examples, the other rules) is the SFT model's own. The wording was chosen on the dev slice.
2. The model thinks (up to 1,024 tokens) and calls the query tool.
3. The environment runs the query on the question's look-around database (section 3) and the
   result comes back as a tool message.
4. The model may call again or stop with a plain reply. A fourth call is not run.

**The answer is the last query that ran.** No query at all is malformed (-0.1), as in run 1. A
call the environment can't read (two calls at once, the wrong tool, no query) comes back as an
error and uses a call.

**What a result looks like.** Column names, the first rows, cells cut at 40 characters, and
`(10 of 218 rows)` when rows were cut; `0 rows`; or `Error: ` and the engine's own message
(first line, 200 characters). Training draws the row cap (5, 10 or 20) and the form (a table or
one JSON object a row) per episode, because other harnesses will differ (decisions R5).
Evaluation always uses 10 rows as a table.

**The context is built token by token and only grows.** After a call, the result is appended
exactly as the chat template renders a tool message, followed by the next turn's opening. The
no-training test compared this with the template's own rendering on 18 conversation prefixes:
all equal. Two consequences:

- The server's slot cache holds the whole episode, and the trainer reads **one sequence per
  episode**. A mask is 1 on the tokens the model sampled and 0 on result blocks, so results are
  read but never trained.
- **A result can't forge the conversation.** The result's own text is tokenised as plain text.
  A cell that holds `</tool_response>` or the turn markers stays data. (Tested: the same text
  parsed as markup would produce an end-of-turn token; here it doesn't.)

## 3. What the model looks at, and what decides the reward

| Dataset | The model looks at | The reward is decided on |
|---|---|---|
| gretel | the dataset's own rows | the same rows |
| CypherBench | the benchmark's sampled graph | the same graph |
| SQaLe | the schema filled with random rows | the checker's own fills |
| text2cypher | a random graph from the prompt's schema | the checker's own graphs |

The checker's random fills plant the reference query's literals into the data, so its copies
would show the model values lifted from the answer. The look-around copies for SQaLe and
text2cypher are therefore built separately, **from the schema and the question only**, with
seeds the checker never uses. One function (`env.public_row`) drops the reference query and the
reference reply before a question reaches the rollout side at all.

The fills plant the values the question itself names (quoted phrases, proper names, codes,
dates, numbers), so a filter on such a value finds rows. Measured on the dev slice by running
each reference on its look-around copy (a check only the scoring side may run):

| Dataset | A correct query returns rows |
|---|---|
| gretel | 50 of 50 |
| CypherBench | 50 of 50 |
| text2cypher | 46 of 50 |
| SQaLe | 36 of 50 |

**This table is the dev slice, and the dev slice flatters gretel and text2cypher.** The pool
keeps a question only when its reference's result says something (for gretel on the dataset's
own rows), so every dev question passes by construction. The evaluation set is unfiltered. On
it a correct query's result says something (a value that is not NULL, zero or empty) for 37 of
50 gretel questions and 18 of 50 text2cypher questions (CypherBench 50, SQaLe 39). There the
untrained model rewrites a right answer after an empty result about half the time
(`results/run2-phase0.md`, section 8). This is open.

So on SQaLe about a quarter of correct first queries would be shown `0 rows`. The remaining
cases filter on a key value or on a constant the question doesn't spell out, which a fill built
without the reference can't know.

**SQaLe therefore keeps run 1's protocol** (decisions R16): the original prompt, one query,
nothing shown. The no-training test on the dev slice showed the cost of turns there: on SQaLe
they fixed 1 answer and broke 6 (three after `0 rows` for a right query, three after rows the
model second-guessed), while on the other three datasets they fixed 8 and broke 1. Rishi's
call: SQaLe must not regress.

**text2cypher keeps one query too** (decisions R17; `single_turn_sets: ["sqale",
"text2cypher"]`). 31 of the evaluation set's 50 text2cypher prompts describe a graph with no
relationships and ask about paths; the look-around graph follows the schema, so the right
answer is "no" or 0 and a look can only mislead. What the two mean:

- A one-query episode is run 1b's episode: one turn, reward 1 / 0 / -0.1, distilled as a whole
  when it is wrong and a sibling is right. The first-turn penalty covers all of it. It gets no
  look-around database and doesn't wait for the graph.
- **Turns are for gretel and CypherBench**, where the model looks at real data: 20 of a step's
  32 questions, 160 of its 256 episodes. The headline, greedy with turns, is those two
  datasets with turns and the other two with one query.
- This is a limit of the environment, not of the model. Against a real database a right query
  returns its rows. Run 2 can't teach or measure turns on the two random-fill datasets.
- The rule goes by dataset, and real use has no dataset label: a deployment picks one prompt.

**When nothing is the right result** (decisions R17). Run 1's pool keeps a question only when
its reference's result says something, so on the pool an empty result always means a wrong
query. On the evaluation set, and in real use, it is often the right result (13 of 50 gretel
evaluation questions). Trained on the pool alone, the model can only learn "nothing came back:
change the query until something does". So:

- **The returned questions.** The 627 gretel questions the pool rejected for this are taken
  back (`pool-v2.returned.jsonl`; 604 after a second check). Three of a step's ten gretel
  questions come from that file, in its own order; its last 50 are a second dev slice. The
  pool file is untouched, and the offset still counts run 1b's ten gretel questions a step.
- **Their reward is stricter.** On the dataset's own rows every query that returns nothing
  equals a reference that returns nothing, so the checker alone would pay "it showed nothing,
  stop" whatever the query. A returned question's answer is right only when the checker's full
  SQL check also passes on filled copies of the schema (`pool.strict_check`). On the second dev
  slice the filled copies reject 17 of the 42 first answers the checker passes.
- **The model never learns which questions these are.** They are gretel questions with gretel
  ids; the mark stays on the scoring side (tested).
- **Three quarters of them have a database with no rows at all** (452 of 604; 10 of the
  evaluation set's 13): the dataset gave tables and no data.

An evaluation records the protocol it was made under (`protocol.json`: the prompt variant, the
call limit, the one-query datasets, the result format, and the version of the look-around
data, `env.LOOK_VERSION`, which must be raised with any change to what the model is shown).
Its episode file is never continued under another one, and the run refuses to start when the start adapter's reference evaluation
was made under another one.

**Read only, enforced by the engines.** SQLite: `query_only`, plus an authorizer that allows
reading and nothing else (no write, PRAGMA, ATTACH or temp table), a 2-second limit and caps on
rows and value size. Neo4j: a read transaction, a 5-second server-side limit per query, a cap on
the response's size, and no procedure calls except schema listings. All of these are tested
against the real engines.

**Neo4j holds one graph at a time**, so Cypher questions take turns: episodes that need the
loaded graph run side by side (up to 8), SQL questions fill the other slots, and the graph
changes only when no Cypher episode is running. Loading takes 0.1 to 5 seconds a graph.

## 4. The reward, and the ways to cheat it

An episode gets one reward, from its final query, scored by checker 4 exactly as a single-turn
answer is (`scoring.py` wraps the query as a one-call reply).

| Outcome | Reward |
|---|---|
| Right on the first query | 1.00 |
| Right on the second / third | 0.97 / 0.94 |
| Wrong, whatever it took | 0 |
| No query at all | -0.1 |
| Undecided by the checker, or withheld by the duplicate probe | none (left out) |
| Right rows that were not computed (below) | 0 |

The first query is scored too. That gives "fixed" (wrong, then right) and "broken" (right, then
wrong), and the first-try skill can be watched for free.

**Cheats this design closes or watches:**

| Cheat | Guard |
|---|---|
| Reading the answer off real rows and returning it as constants (`SELECT 'Alice', 42`) | `guards.hardcoded`: a final query that read no table (the SQLite authorizer reports this) or no graph pattern, or whose result columns are all literals, earns 0. Stop rule at 2% |
| The same, hidden better (`SELECT 'Alice', (SELECT 42) FROM t`) | The canary `copied_frac`: final queries with a literal that appeared in an earlier result and nowhere in the prompt. Reading a value's spelling is legitimate, so it is watched against the run's own start (+5 points pauses) |
| Seeing the reference's values in the look-around data | The fills are built without the reference. `look_leak_frac` counts random-fill episodes whose results show a reference literal that isn't in the prompt; above 5% pauses. (With both random-fill datasets on one query nothing from a fill is shown, so this reads nothing; it matters again if one of them gets turns) |
| "It showed nothing, so stop" on a question whose right result is nothing | The stricter reward for returned questions (section 3): the query must also match the reference on filled copies. `returned_weak_only_frac` counts the answers only the checker passes |
| Free information from many queries | 3 calls, the 0.03 cost, stop rules on calls an episode and on repeated queries |
| Ending without an answer to dodge a wrong one | No query is -0.1, and the strict score counts it wrong |
| Forging a turn through a result | Results are tokenised as plain text |
| Writing to the database, or stalling it | Read-only engines, time limits, size caps |
| DISTINCT dropped where the sample graph can't tell | Run 1b's duplicate probe, kept |
| Copying the teacher's situation ("the correct solution is...") | Run 1's `leak_frac`, kept |

## 5. Learning

**Outcome-level credit.** Every token the model wrote in an episode shares the episode's
advantage, (reward - group mean) / max(group std, 0.25). A group is informative when it holds a
right and a not-right outcome; the turn cost alone doesn't count.

**SRPO, first turn only for the teacher.** A wrong episode with a right sibling distills its
first turn toward the EMA teacher, which is shown the question plus the sibling's final query.
The sibling that needed the fewest calls is used. The episode's later turns take the reward
update with its (negative) advantage: after a result the sibling may never have seen, the
teacher's view would describe another situation.

**Staying near the start.** The reference for drift is the start adapter, frozen, swapped in for
one extra forward pass (as the EMA teacher is). The penalty is a fixed 0.05 x KL(policy || start)
on first-turn tokens. The first turn's context is the single-turn task, the skill to keep; later
turns react to results the start model never saw and are left free. Nothing pulls the model
back automatically: when first-turn drift passes 0.05 the run pauses for a person.

**The drift reading has an offset, and it was measured.** The server samples from a Q8_0 file
with the adapter merged in; the trainer reads the same adapter in bf16 and under-reads a small
adapter (decisions R7). So the two disagree a little even about the same policy. On 200
episodes sampled from the start adapter (22,500 first-turn tokens), where the true drift is
zero, the reading is **0.0099**; over batches of 32 episodes (one step's probe) it ranges from
0.005 to 0.016, a standard deviation of 0.0035. Measured again under the final protocol
(text2cypher on one query): **0.0084**, 0.005 to 0.012, standard deviation 0.0021. The run
subtracts 0.0084 (`kl.offset`). The
0.05 bound is about 14 standard deviations above that, so it can be read.

The design page's rule was "switch to merging the start adapter into a new base model if the
step-1 offset reads above 0.01, because then the bound could not be told from noise". The
measured offset is at that line, and its noise is a fifteenth of the bound, so the adapter is
continued as planned. Step 1's own raw reading is still checked: above 0.025 means the run did
not start from the model the offset was measured on, and it pauses.

**How long an episode may be.** The update reads the prompt and every turn as one sequence, and
its memory grows with the length. Measured on the 16 GB card (`tools memory`): 12.2 GB at 3,072
tokens, 13.5 at 3,650, 14.5 at 4,096, and 4,608 doesn't fit. So `max_seq` is 3,840 (run 1's
6,144 was never reached there, and would not fit). A longer episode is left out of the loss
and counted (`too_long_frac`, a stop rule at 5%); none of 328 sampled episodes was longer than
3,650.

## 6. Measuring it

Every evaluation runs the 200-question set through the same episode engine as the rollouts.

| Number | What it is |
|---|---|
| **Greedy with turns, decided** | The headline. Right / (right + wrong), undecided and no-query answers left out, as every earlier number was counted |
| The same, strict | Undecided and no-query answers count as wrong, over all 200 |
| First query | The greedy episode's first query: the forced single-query score, at no cost |
| Fixed / broken | What the later turns did to the first answer |
| mean@8 and pass@8 with turns | Full evaluations only: the headroom left |

**The yardstick is anchored.** Before any number with turns, the start model was rebuilt from
the kept adapter and measured two ways against the kept step-30 evaluation (102 of 187, 54.5%):
through the old path it agrees on 183 of 186 questions (52.9%), and the new engine's first query
agrees on 184 of 186 (54.3%). Reruns of one model differ by about this much.

Greedy with turns runs after steps 5, 10, ... 40; the full evaluation after 20 and 40. `BEST.json`
names the step with the highest greedy with turns (decisions R14: greedy picks), and says
whether its first-query score is still within 3 points of 54.5.

The run pauses after an evaluation when pass@8 with turns is below the start adapter's with
turns (sign test, p < 0.05), or when greedy with turns is worse than the run's best earlier
evaluation (p < 0.10).

## 7. Stop rules

Any one pauses the run for a look. "The start" is the mean of run 2's own first five steps.

| Guards against | Rule |
|---|---|
| Collapse | First-turn token entropy below 25% of the start. It is measured before the update on every episode that carries the server's log-probabilities, so it doesn't follow the routing |
| | A group holds fewer than half as many different final queries as at the start, 3 steps |
| | Informative groups below 25%, 5 steps |
| | First-turn length 3x the start, either way |
| Drift | First-turn KL to the start (less the measured offset) above 0.05; step 1's raw reading above 0.025 |
| | Server against trainer above 0.10 |
| The protocol | Episodes with no query above 5%, 3 steps |
| | More than 2.0 calls an episode, 3 steps |
| | Episodes trying a call too many: 15 points above the start, 3 steps |
| | The same query twice in a row: 10 points above the start, 3 steps (the answer run again after a look at the data is not counted: the answer rule asks for it) |
| | Episodes too long to train on above 5% |
| The first-try skill | Right first answers a retry made wrong: 8 points above the start, 3 steps |
| | The first query's solve rate 0.10 below the start, 3 steps |
| Reward hacking | Constants above 2%; copied literals 5 points above the start; look-around leak above 5% |
| | DISTINCT 10 points above the start; `SELECT *` above 5%; a first-turn thought citing a given solution above 2%; the duplicate probe as in run 1b |

Three rules are measured against the start because the untrained start model already does
these things (the no-training test on the dev slice: 13% of episodes try a fourth call, 8% run
a query twice, 6% of right first answers get broken). A fixed bound would pause the run at step
1 for what it started with. The teacher-leak rule reads first-turn thoughts only: in later
turns "the correct query should be ..." is the model talking about its own retry.

The protocol rules, the broken-answer rule, copied literals and the look-around leak are
counted over the episodes that had turns (160 of a step's 256). The one-query
episodes can't retry, loop, copy or see a leak; counted in, they would make every such problem
look a quarter smaller than the bound it is read against. The scores, the collapse rules, the
drift and the constants rule count every episode.

Nothing here adds an entropy bonus or pulls the model back by itself. Those would be method
changes, and they are Rishi's to make.

## 8. Where the files go

Everything a run writes is on the internal drive: the run's folder (`<data>/run2/<run>/`) and
its rollout model (`<models>/run2/<run>/`, a 4 GB file rewritten every step). `models` is its
own setting in `configs/paths.local.toml`, so that file never follows the released model onto
another drive. The checker's Neo4j (Docker's disk) is on the internal drive too. The USB data
drive keeps what a run only reads: the released model's master copy, the evaluation set, the
CypherBench graphs. A finished run moves there as history; the kept adapter stays.

## 9. Order of work

1. **Phase 0, no training** (`phase0.py`): the template round trip, look-around timings, the
   anchor, the wording test on the dev slice, and the evaluation set with turns on the start
   model, on step 26 and on the SFT model. Results: `results/run2-phase0.md`.
2. **A loop test** (`run.py --smoke`): 2 steps of 8 questions from a reserve at the pool's tail.
3. **Rishi reads Phase 0** against the design page's table (D10). Done 2026-11-19 for SQaLe
   (one query: R16). The evaluation set then read "do not start RL"; he chose to fix the
   training questions (R17), and the tests were run again under that setup
   (`results/run2-phase0.md`, section 10).
4. **The run**: 40 steps, about 10 hours with evaluations. When Rishi says go.
