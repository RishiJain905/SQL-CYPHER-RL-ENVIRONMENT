# Run 2, phase 0: what the untrained model does with turns

No step is trained here. These are the checks and measurements `run2/design.md` asks for before
run 2 starts (`scripts/run2/phase0.py`). **Part done (2026-11-19):** the held-out evaluations
with turns are still to run. The model is run 1b's step-30 adapter merged into the SFT model,
unless a row says otherwise.

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

## 4. Does a correct query return rows on the look-around copy?

Each dev-slice reference run on the database the model would look at (a check for the scoring
side only: the builder never sees a reference).

| Dataset | Returns rows |
|---|---|
| gretel | 50 of 50 |
| CypherBench | 50 of 50 |
| text2cypher | 46 of 50 |
| SQaLe | 36 of 50 |

The SQaLe misses filter on a key value or a constant the question doesn't spell out; a fill
built without the reference can't know those.

## 5. The drift reading's offset

200 episodes sampled from the start adapter, with the server's log-probabilities; the trainer
then scores the same tokens under the same adapter. The true drift is zero.

| | |
|---|---|
| Offset, first-turn tokens (22,536) | 0.0099 |
| Per batch of 32 episodes (one step's probe) | 0.005 to 0.016, standard deviation 0.0035 |
| Later turns | 0.0057 |
| Drift from the SFT model on the same tokens | 0.079 (run 1b's step 30: 0.078) |

The run subtracts 0.0099. The 0.05 pause bound is about 14 standard deviations above it.

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

## Still to run

1. SQaLe on run 1's protocol (`single_turn_sets`): Rishi said that is fine if it is better, and
   section 3 says it is. Set it, then:
2. The evaluation set with turns: step 30 (greedy and 8 samples: the reference every later
   evaluation is compared with), step 26, the SFT model, and step 30 with thoughts stripped
   from the history.
3. Rishi reads the result against the design page's table (D10) and says go, or not.
