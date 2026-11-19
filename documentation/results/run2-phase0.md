# Run 2, phase 0: what the untrained model does with turns

No step is trained here. These are the checks and measurements `run2/design.md` asks for before
run 2 starts (`scripts/run2/phase0.py`). Finished 2026-11-19. The model is run 1b's step-30
adapter merged into the SFT model, unless a row says otherwise.

**The result in three lines:**

- On the pool's dev slice turns add 4 points (57.8% to 61.8%; fixed 9, broke 1).
- **On the evaluation set turns cost 4 points** (52.4% to 48.1%; fixed 4, broke 10). By the
  design page's table (D10) that reads "results derail it: do not start RL". Section 7.
- The cause is one case: a right query whose result says nothing (`0 rows`, or only NULL or
  zero). The model breaks 9 of 25 such answers and 1 of 53 others. That case is a third of the
  evaluation set's right first answers and is absent from the pool, by the pool's own filter.
  Section 8. What to do about it is Rishi's choice (section 9).
- **Rishi chose to fix the training questions** (decisions R17). Under that setup step 30
  scores 52.4% with turns, the same as its first query; 60.5% against 52.3% where a look can
  inform, with nothing broken; and it still breaks 7 of 10 right answers whose right result is
  nothing, which training now has examples for. Section 10.

## 1. Checks

| Check | Result |
|---|---|
| The context the engine builds token by token against the chat template's own rendering | equal on 18 of 18 conversation prefixes (12 episodes) |
| A result that carries the template's markup | stays plain text: 32 tokens, no end-of-turn token (parsed as markup it would be 8 tokens with one) |
| The chat endpoint with a tool message in the conversation | accepted; the model replied in a sentence |
| Look-around build times | CypherBench graphs 0.6 to 4.9 s; text2cypher fills 0.1 to 0.9 s; SQL under 0.1 s |
| Neo4j's limits | a runaway query stopped after 6 s; a write refused by the server itself; an oversized result refused |

## 2. The anchor: is this the model, and the measurement, behind 54.5?

The kept evaluation of step 30 is 102 right of 187 decided (54.5%), one greedy query.

| | Right / decided | Agrees with the kept evaluation |
|---|---|---|
| The start model rebuilt from the kept adapter, the old path (chat endpoint) | 99 / 187 (52.9%) | 183 of 186 questions |
| The new engine's first query, the original prompt | 101 / 186 (54.3%) | 184 of 186 questions |

Reruns of one model differ on about 3 questions in 186. So a difference of 1 or 2 points
between two evaluations is nothing.

## 3. The prompt wording, on the dev slice

200 pool questions no run trains on, greedy, up to 3 queries.

| Prompt | First query | With turns | Fixed | Broken | Calls an episode |
|---|---|---|---|---|---|
| The SFT prompt, unchanged | 114 / 198 (57.6%) | 57.6% | 5 | 5 | 1.47 |
| `turns` (the run's) | 115 / 197 (58.4%) | **59.4%** | 9 | 7 | 1.41 |
| `turns` plus a hint to check the result | 120 / 197 (60.9%) | 59.9% | 7 | 9 | 1.40 |

The three are within noise of each other. `turns` has the best balance of fixed against broken
and is kept.

With the `turns` prompt, by dataset:

| | First query right | With turns | Fixed | Broken |
|---|---|---|---|---|
| CypherBench | 35 / 50 | 38 / 50 | 3 | 0 |
| gretel | 31 / 49 | 34 / 49 | 3 | 0 |
| text2cypher | 27 / 50 | 28 / 50 | 2 | 1 |
| SQaLe | 22 / 48 | 17 / 48 | 1 | 6 |

- **Turns help three datasets and hurt SQaLe.** Without SQaLe: fixed 8, broken 1. SQaLe alone:
  fixed 1, broken 6.
- **Why SQaLe:** three of its six broken answers followed `0 rows` for a query that was right
  (the look-around copy is a random fill; section 4), three followed rows that the model then
  second-guessed.
- **What the model does after a first result:** it calls again after every error (15 of 15),
  after most empty results (13 of 15) and after 14% of results with rows.
- **What it gets wrong about the protocol:** 27 of 200 episodes try a fourth call, and 26 of
  those end wrong, mostly because the last query that ran was an exploring one
  (`SELECT * FROM t LIMIT 5`), not the answer. 8% run the same query twice.

### SQaLe on one query (decisions R16)

Rishi's call after the table above: SQaLe keeps run 1's protocol (the original prompt, one
query, nothing shown). The dev slice again under that protocol, which is the run's:

| | First query right | Final answer right | Fixed | Broken | Calls an episode |
|---|---|---|---|---|---|
| CypherBench | 36 / 50 | 39 / 50 | 3 | 0 | 1.26 |
| gretel | 31 / 50 | 35 / 50 | 4 | 0 | 1.26 |
| text2cypher | 27 / 50 | 28 / 50 | 2 | 1 | 1.44 |
| SQaLe (one query) | 21 / 49 | 21 / 49 | | | |
| **All** | 115 / 199 (57.8%) | **123 / 199 (61.8%)** | 9 | 1 | 1.32 (the three with turns) |

- The three datasets with turns repeat the first table within rerun noise (a rerun moves one
  or two questions a dataset).
- Every SQaLe episode ended at its first call and ran nothing on a look-around copy. One
  replied without a query.
- After a first result the model called again after every error (8 of 8), most empty results
  (8 of 9) and 11% of results with rows (14 of 133).
- 11 of 150 episodes with turns tried a fourth call (7%); 4% ran a query twice; no final query
  was caught by the constants rule; one copied a literal off a result.

## 4. Does a correct query return rows on the look-around copy?

Each reference run on the database the model would look at (a check for the scoring side only:
the builder never sees a reference). "Says something" is the checker's own rule: a row with a
value that is not NULL, zero or empty.

| Dataset | Dev slice: returns rows | Dev slice: says something | Evaluation set: returns rows | Evaluation set: says something |
|---|---|---|---|---|
| CypherBench | 50 of 50 | 50 | 50 of 50 | 50 |
| gretel | 50 of 50 | 50 | 40 of 50 | **37** |
| text2cypher | 46 of 50 | 45 | 27 of 50 | **18** |
| SQaLe (one query in run 2) | 36 of 50 | 35 | 40 of 50 | 39 |

The SQaLe misses filter on a key value or a constant the question doesn't spell out; a fill
built without the reference can't know those.

**The dev slice and the evaluation set differ on gretel and text2cypher, and the first table of
this section (dev slice only) hid it until the evaluation set was run.** Section 8 says why.

## 5. The drift reading's offset

200 episodes sampled from the start adapter, with the server's log-probabilities; the trainer
then scores the same tokens under the same adapter. The true drift is zero.

| | |
|---|---|
| Offset, first-turn tokens (22,536) | 0.0099 |
| Per batch of 32 episodes (one step's probe) | 0.005 to 0.016, standard deviation 0.0035 |
| Later turns | 0.0057 |
| Drift from the SFT model on the same tokens | 0.079 (run 1b's step 30: 0.078) |

The 0.05 pause bound is about 14 standard deviations above it. Measured again under the final
protocol (section 10, text2cypher on one query): **0.0084** on 24,301 first-turn tokens, 0.005
to 0.012 a batch, standard deviation 0.0021. The run subtracts 0.0084.

## 6. The update's memory, and a loop test

| Episode length (prompt + all turns) | Peak GPU memory |
|---|---|
| 2,048 tokens | 10.4 GB |
| 3,072 | 12.2 GB |
| 3,650 | 13.5 GB |
| 4,096 | 14.5 GB (fails when 2,048 of them are generated) |
| 4,608 | doesn't fit |

`max_seq` is 3,840. In 328 sampled episodes (the 200 above and the loop test's 128) none was
longer than 3,650.

The loop test (`smoke-3`: 2 steps, 8 questions x 8 episodes from the pool's reserve): both
steps trained, saved with verified hashes and merged; 5.1 minutes a step at that size; the
trainer's rendering of the prompt equals the server's.

The loop test again with SQaLe on one query (`smoke-4`, the same questions and seeds):

| | Step 1 | Step 2 |
|---|---|---|
| Episodes (with turns / one query) | 48 / 16 | 48 / 16 |
| Trained sequences, of them distilled | 48, 25 | 20, 7 |
| Loss and gradient | finite; gradient norm 0.21 | finite; 0.31 |
| Drift reading, raw (the offset is 0.0099) | 0.0112 | 0.0178 |
| Peak GPU memory | 10.5 GB | 13.9 GB |
| Minutes | 4.9 | 5.2 |

- A wrong SQaLe episode with a right sibling is distilled as a whole and has no later turn for
  the reward update. Step 1 held two of them; the update handled the empty part.
- The trainer rendered the same prompt as the server for all 128 episodes: SQaLe's reads "ONE
  read-only query", the other three datasets' read "one query at a time".
- In step 2 six of one SQaLe question's eight episodes replied without a query. The question
  asks for a "domain analysis", not for data; the earlier loop test (SQaLe with turns) did the
  same on it. It is a quirk of that pool question, not of the protocol.

## 7. The evaluation set with turns, no training

200 questions, greedy, the run's protocol (three datasets with up to 3 queries, SQaLe with
one). "First query" is the greedy episode's first query, counted as every earlier number was.

| Model | First query | With turns | Strict, with turns | Fixed | Broken | Calls an episode |
|---|---|---|---|---|---|---|
| **Step 30** (the start) | 99 / 189 (52.4%) | **91 / 189 (48.1%)** | 45.5% | 4 | 10 | 1.64 |
| Step 30, thoughts stripped from the history | 101 / 189 (53.4%) | 95 / 188 (50.5%) | 47.5% | 5 | 10 | 1.54 |
| Step 26 | 96 / 185 (51.9%) | 88 / 189 (46.6%) | 44.0% | 5 | 13 | 1.67 |
| The SFT model | 70 / 186 (37.6%) | 72 / 194 (37.1%) | 36.0% | 12 | 11 | 2.09 |

- Step 30's kept single-query evaluation is 102 / 187 (54.5%). Its first query here, 52.4%, is
  that number within rerun noise (section 2). Its answer with turns is lower, question by
  question: 7 better, 18 worse (sign test p 0.04).
- "Fixed" and "broken" count questions where both the first and the final query are decided, so
  they don't always add up to the difference of the two columns: a final query can be one the
  checker can't decide. For step 26 and SFT the two columns also have different totals.
- Step 26 and step 30 are the same with turns (13 better, 13 worse). The SFT model is far
  behind (17 better, 35 worse than step 30, p 0.02). Under this prompt its first query is often
  an exploring one (37.6% against 44.5% with the original prompt), and it uses 2.1 calls.
- Stripping thoughts from the history changes little (10 better, 6 worse, p 0.45). A client
  that drops them is not worse off.

Step 30 by dataset:

| | First query | With turns | Fixed | Broken | Tried a fourth call |
|---|---|---|---|---|---|
| CypherBench | 21 / 50 | 23 / 50 | 2 | 0 | 14 of 50 |
| gretel | 36 / 50 | 30 / 50 | 2 | **8** | 4 |
| text2cypher | 23 / 42 | 19 / 42 | 0 | 2 | 9 |
| SQaLe (one query) | 19 / 47 | 19 / 47 | | | |

Step 30, 8 sampled episodes a question, with turns: **mean@8 43.9, pass@8 69.2** (its kept
single-query evaluation: 43.4 and 67.5; sampled sign test 64 better, 52 worse, p 0.31; pass@8
15 gained, 11 lost). So the headroom with turns is what it was with one query. By dataset
(mean@8 / pass@8): CypherBench 43.4 / 78.0, gretel 61.0 / 86.0, SQaLe 34.4 / 52.1, text2cypher
36.5 / 60.0. This evaluation is the run's start reference (`phase0/heldout-step30`).

Reward hacking signs in 1,800 episodes: 2 final queries caught by the constants rule (neither
right), under 1% with a literal copied off a result, no look-around leak.

**Against D10's table:** net -6 questions (-3.2 points) and 12.8% of right first answers broken
is its third row: "results derail it. Fix the wording or the rendering; cold start; do not
start RL." It does retry after errors (9 of 9), which is the first row's other condition.

## 8. Why: a right query whose result says nothing

Every greedy episode of step 30 whose **first query was right**, split by what that query's
result showed (`phase0_score firstresult`; "says something" is the checker's rule: a value
that is not NULL, zero, false or empty):

| The first result | Evaluation set, greedy: broken | Evaluation set, 8 samples: broken | Dev slice, greedy: broken |
|---|---|---|---|
| Rows that say something | 1 of 53 (2%) | 5 of 388 (1%) | 0 of 93 |
| `0 rows`, or only NULL, zero or false | **9 of 25 (36%)** | **69 of 153 (45%)** | 1 of 1 |

Step 26 is the same (1 of 54, and 12 of 21); so is step 30 with thoughts stripped (1 of 55, and
9 of 25).

- **Shown real values, the model keeps a right answer.** That is the behaviour turns need.
- **Shown nothing, it breaks a third to a half of them.** And 25 of the evaluation set's 78
  right first answers (32%) are shown nothing; on the dev slice 1 of 94 is.

**The same evaluations, split by the reference instead of by the model's query.** A question is
on the first line when its reference's result says something on the look-around copy (144 of
200: CypherBench 50, gretel 37, SQaLe 39, text2cypher 18), on the second when it says nothing
(56). Greedy:

| | First query | With turns | Fixed | Broken | Step 30's kept single-query score |
|---|---|---|---|---|---|
| Step 30, the reference says something | 70 / 142 (49.3%) | **73 / 142 (51.4%)** | 4 | 1 | 72 / 142 (50.7%) |
| Step 30, the reference says nothing | 29 / 47 (61.7%) | **18 / 47 (38.3%)** | 0 | 9 | 30 / 45 (66.7%) |
| Step 26, says something | 72 / 141 (51.1%) | 75 / 140 (53.6%) | 4 | 1 | |
| Step 26, says nothing | 24 / 44 (54.5%) | 13 / 49 (26.5%) | 1 | 12 | |
| SFT, says something | 58 / 140 (41.4%) | 66 / 141 (46.8%) | 11 | 4 | |
| SFT, says nothing | 12 / 46 (26.1%) | 6 / 53 (11.3%) | 1 | 7 | |

Where a look can inform, turns add 2 to 5 points with no training, as on the dev slice (D10's
first row: build as designed). All of the loss is on the 56 questions where the right result is
nothing. Those questions also score far higher with one query (66.7% against 50.7%), which is
the weak check described below, not an easier task.

**Where the difference comes from: the pool's filter.** `run1/pool.py` keeps a training question
only when its reference "can earn the reward": the checker must decide it, which needs a result
that says something, and for gretel the reference's result on the dataset's own rows must say
something too. The dev slice is cut from the pool, so it has the same filter. The evaluation
set is unfiltered. The design page named this gap in D9's trade-off; the dev slice could not
measure it. By dataset:

- **gretel.** The pool rejected 627 of 2,646 candidates because the reference's result said
  nothing (24%). On the evaluation set 13 of 50 references say nothing on the dataset's own
  rows, which is what the model looks at (section 4).
- **text2cypher.** 31 of the evaluation set's 50 prompts describe a graph with **no
  relationships** (`Relationships: (none)`); 30 of them come from one source of the dataset
  (`neo4jLabs_functional_cypher`) and most ask about paths ("is there a path between Journal X
  and Article Y"). The look-around graph follows the schema, so it has no relationships, and
  the right answer is "no" or 0: 24 of those 31 references say nothing. The dev slice has 14
  such prompts, and 12 of their references still say something (property lookups). The pool
  rejected 422 of 1,426 text2cypher candidates whose reference returned no rows on the
  checker's graphs, which is where a path question over such a schema ends up (the 422 were
  not sorted by kind here). The evaluation set's
  text2cypher is 41 of 50 from that source; the dev slice's is 14 of 50. Six more evaluation
  references (2 on the dev slice) filter on a value the question names but the fill did not
  plant, such as an unquoted lower-case phrase. That part a better fill could reach.

**How the 9 greedy breaks (and the 69 sampled ones) happened:**

| | Greedy | 8 samples |
|---|---|---|
| Explored the data, ran out of calls: the last query that ran was an exploring one | 5 | 22 |
| Rewrote the answer until rows came back, and stopped | 4 | 47 |

Three gretel examples, read by hand:

- *Maximum fuel consumption in the Caribbean in January 2022.* The first query is right and
  returns NULL: the table has no such rows. The model looks for the date format, then the
  areas, then the whole table, and has no call left. Its answer is `SELECT ... GROUP BY Area`.
- *Campaigns in South America.* The reference filters `country = 'South America'` and returns
  nothing; the data lists Argentina, Brazil, Chile. The model sees that and filters on the
  five countries. On the data that is the better answer; against the reference it is wrong.
- *Average satisfaction in the last 3 years.* The data ends in 2021, so the right query returns
  NULL. The model moves the window until it gets a number.

So two things are mixed:

1. **A protocol habit** (first row): exploring is sensible, but here the last query that ran is
   the answer. Wording or training can reach this.
2. **The reference disagrees with the data** (second row). The model's rewrite is often the more
   useful query. Nothing learned on the pool fixes it, because in the pool an empty result is
   always a wrong query: training there teaches "nothing came back, change the query", which
   makes these evaluation questions worse.

**A weakness of the yardstick that this exposes.** On gretel the checker runs the reference and
the answer on the dataset's own rows and compares. When the reference returns nothing, any
answer that returns nothing passes. About a quarter of the evaluation set's gretel questions
are checked that weakly, in every number so far (the 54.5 included).

**Wrong first queries, for completeness** (step 30, greedy, fixed of all): after an error 1 of
9; after a result that says nothing 3 of 25; after rows 0 of 28. On the dev slice: 2 of 8, 6 of
16, 1 of 32. The untrained model fixes little on the evaluation set; that part is what RL is
for.

**How to repeat these counts.** `python -m sqlrl.run2.phase0_score lookcheck --rows evalset`
(and `--rows dev`) runs the references on the look-around copies; `firstresult --dir <folder>`
splits a scored evaluation by what the first result showed. Both are CPU only; the first loads
graphs, so nothing else may use the Neo4j.

## 9. What follows (Rishi's choice)

By the rule used for SQaLe in R16 ("we don't want it to regress"), gretel and text2cypher also
fail on the evaluation set; only CypherBench gains (+2 questions). Nothing was changed after
this result: no setting, no environment code, no rerun.

What the run would need before it can be read fairly:

- **Training questions where nothing coming back is right.** The 627 gretel questions the pool
  rejected are still there (their ids in `pool-v2.checked.jsonl`, their rows in the training
  split). They give a second dev slice for this case and training questions for it. Without
  them any fix can only be tuned against the evaluation set, which must not happen.
- **A measure that separates the two cases.** The evaluation set's score with turns on the
  questions whose reference says something on the look-around copy, beside the full score. On
  its own this is not enough: the trained habit would still hurt the other questions.
- **A decision on text2cypher**, where 32 of 50 evaluation references say nothing on the
  look-around graph, 24 of them because the prompt's schema has no relationships. No fill
  changes that: looking can only mislead there. One query, like SQaLe, is the plain reading.
- **The success line.** "At least +5 over 54.5" (D11) is +11 from 48.1.

**Rishi chose (2026-11-19): "Go with option C"** (decisions R17). Section 10.

## 10. Option C: what was built, and how its tests will be read

Written before any of its GPU tests were run.

**Built** (decisions R17):

- **text2cypher keeps one query**, like SQaLe. Turns are for gretel and CypherBench, the two
  datasets where the model looks at real data: 20 of a step's 32 questions, 160 of 256 episodes.
- **The returned questions.** `pool-v2.returned.jsonl`: the 627 gretel questions run 1's pool
  rejected because the reference's result says nothing, each checked again; 604 kept (for 23
  no filled copy could judge the reference). Three of a step's ten gretel questions come from
  it (120 in 40 steps); its last 50 are a second dev slice.
- **What kind of nothing.** 452 of the 604 have a database with **no rows at all** (the
  dataset gave tables and no data); 152 have data that the filter doesn't match. The evaluation
  set's 13 such gretel questions are 10 and 3. The dev slice: 31 and 19.
- **A stricter reward for them.** The checker compares a gretel answer with the reference on the
  dataset's own rows, where every query that returns nothing equals a reference that returns
  nothing: "it showed nothing, stop" would earn the reward whatever the query. A returned
  question's answer is right only when the checker's full SQL check also passes on filled
  copies of the schema (`pool.strict_check`). Training and the dev slices only; the evaluation
  set's yardstick is untouched.
- **"The same query twice"** now means twice in a row. Running the answer again after a look at
  the data is what the answer rule asks for, so it is counted apart (`rerun`).

**How the no-training tests will be read.** By the letter of D10's table the evaluation set
lands in its third row again: with text2cypher on one query, step 30's own numbers above give
fixed 4, broken 8. That row was written for a case training could not reach. Under option C
the case is in training, so the reading is:

| Number | Where | Means |
|---|---|---|
| Net fixed minus broken, and the broken rate | Evaluation questions with turns whose reference says something (87: CypherBench 50, gretel 37) | D10's first row there (net +2 or more, broken at most 3%): a look helps where it can inform. Below that: not ready, whatever the rest says |
| The same | The pool's dev slice, its gretel and CypherBench questions | The baseline for "turns help on the pool" (was fixed 7, broken 0) |
| Broken rate | The returned dev slice | **The number training must lower.** No bar for the untrained model; it is the baseline |
| Of the returned answers the checker passes, the share the filled copies reject | The returned dev slice | How large the reward hole would be without the stricter reward |
| The full evaluation score with turns, and the three-way split | Evaluation set | Reported. The replacement for "+5 over 54.5" is Rishi's to set |

**One wording test, on the two dev slices only.** `keep` adds to the turn rule: "A result with
no rows can be the correct result: if your query says what the question asks, keep it. If you
look at the data first, run your answer query again as your last call." Adoption rule, fixed
here: `keep` is recommended only if, against `turns`, it breaks at least 5 fewer right first
answers on the returned slice, and on the pool's dev slice loses no more than 2 in fixed minus
broken and no more than 2 fixes after a result that says nothing. With 50 and 100 questions
only a gap that size counts. The run's prompt stays `turns` unless Rishi takes the
recommendation.

### The results (step 30, no training, 2026-11-19)

**The two dev slices, greedy.** The pool's slice counts its gretel and CypherBench questions
(the ones with turns). On the returned slice "right" means the stricter check.

| | First query right | With turns | Fixed | Broken | Calls an episode |
|---|---|---|---|---|---|
| Pool's dev slice, `turns` | 66 / 99 | 73 / 99 | 7 | 0 | 1.25 |
| Pool's dev slice, `keep` | 67 / 99 | 71 / 99 | 5 | 1 | 1.24 |
| Returned slice, `turns` | 25 / 50 | **11 / 50** | 0 | **14** (56%) | 2.02 |
| Returned slice, `keep` | 24 / 50 | 5 / 50 | 0 | 19 (79%) | 2.12 |

- **`keep` is not adopted.** It broke 5 more right answers on the returned slice, not 5 fewer.
  Telling the model that an empty result can be right did not make it keep one. The run's
  prompt stays `turns`.
- **The baseline training has to lower: 14 of 25 right first answers broken on the returned
  slice** (11 rewrote the query, 3 ran out of calls). By kind: 8 of 13 where the database has
  no rows, 6 of 12 where it has rows the filter doesn't match.
- **The reward hole, measured.** Of the 42 first answers the checker passes on the returned
  slice, the filled copies confirm 25 and reject 17 (40%); of the 20 final answers it passes
  they reject 9. Without the stricter reward 4 in 10 "right" answers would be queries that
  merely returned nothing.
- The whole pool's dev slice with this protocol: 114 / 198 on the first query, 121 / 198 with
  turns (57.6% to 61.1%).

**The evaluation set, the new start reference** (`phase0/heldout-step30`; gretel and CypherBench
with turns, SQaLe and text2cypher with one query):

| | First query | With turns | Fixed | Broken |
|---|---|---|---|---|
| **All 200** | 98 / 187 (52.4%) | **98 / 187 (52.4%)** | 7 | 7 |
| With turns, the reference says something (87) | 45 / 86 (52.3%) | **52 / 86 (60.5%)** | 7 | 0 of 45 |
| With turns, the reference says nothing (13 gretel) | 10 / 13 | 3 / 13 | 0 | 7 of 10 |
| One query (100) | 43 / 88 (48.9%) | 43 / 88 | | |
| CypherBench | 21 / 50 | 25 / 50 | 4 | 0 |
| gretel | 34 / 49 | 30 / 49 | 3 | 7 |

Strict count with turns: 49.0%. Against step 30's kept single-query evaluation, question by
question: greedy 10 better, 14 worse (p 0.54). **With 8 samples: mean@8 47.6, pass@8 71.8**
(kept: 43.4 and 67.5); the sampled sign test is 69 better, 41 worse (p 0.01), pass@8 14 gained,
7 lost. By dataset (mean@8 / pass@8): CypherBench 46.8 / 76.0, gretel 63.8 / 88.0, SQaLe
32.3 / 52.1, text2cypher 46.9 / 70.2.

**Read by the rules set above:**

| Number | Result | Reading |
|---|---|---|
| Evaluation questions with turns whose reference says something | net +7, 0 of 45 broken | D10's first row: a look helps where it can inform |
| Pool's dev slice, gretel and CypherBench | fixed 7, broken 0 | As before |
| Returned dev slice, broken | 14 of 25 (56%) | The baseline; training has to lower it |
| Returned answers the checker passes that the filled copies reject | 17 of 42 (40%) | The stricter reward is needed |
| Full evaluation score with turns | 52.4%, equal to the first query | Reported. All of the gap to the 60.5% above is the 13 questions whose right result is nothing |

**One thing the returned questions showed, not built:** 10 of those 13 evaluation questions,
and three quarters of the returned file, have a **database with no rows at all**. Six of the
seven gretel breaks are on those ten. A rule that needs no reference ("a database with no rows
gives one query: there is nothing to look at") would, on these same episodes, put step 30 at
about 104 / 187 with turns (55.6%) with no training; this is an estimate that takes the first
query as the one-query answer. It is a protocol change, so it is Rishi's.

**The loop test under option C** (`smoke-5`, 2 steps, 8 questions x 8 episodes: 2 of each
dataset, one gretel question a returned one):

| | Step 1 | Step 2 |
|---|---|---|
| Episodes with turns / with one query | 32 / 32 | 32 / 32 |
| Trained sequences, of them distilled | 48, 22 | 28, 13 |
| Loss and gradient | finite; gradient norm 0.23 | finite; 0.16 |
| The trainer's prompt against the server's, every dataset | equal | equal |
| Returned question: right first query, right final answer | 8 of 8, 2 of 8 | 8 of 8, 0 of 8 |
| Minutes, peak GPU memory | 4.0, 10.8 GB | 4.8, 13.2 GB |

- The one-query episodes ran nothing and loaded no graph; nothing in the rollout file marks a
  returned question.
- The returned question shows the problem in miniature: every first query right, almost every
  one rewritten.
- Step 1's raw drift reading was 0.022 on its 8 probe episodes, so the offset was measured
  again under this protocol: 0.0084 (section 5). A real step probes 32 episodes.
