# Run 2 results: multi-turn SRPO from run 1b's step 30, 40 steps, step 40 kept

Run `run2`, 2026-11-19 to 2026-11-21. Stage B: the model sees what its query returned and may
fix it, up to 3 queries, the last one being the answer (`run2/design.md`). It continues run 1b's
step 30 adapter (R15) under option C (R17): SQaLe and text2cypher keep one query; gretel and
CypherBench get turns; 3 of each step's 10 gretel questions are "returned" questions, whose right
result is nothing, scored with the stricter check. SRPO, 32 questions x 8 episodes a step,
learning rate 5e-6, KL to the start adapter on first-turn tokens. **All 40 steps ran. The
adapter kept is step 40's** (Rishi, 2026-11-21).

- Design and how it was run: `run2/design.md`, `run2/RUNBOOK.md`; phase 0 (the no-training
  tests that led to option C): `results/run2-phase0.md`; live page: `run2/dashboard/README.md`
- Decisions along the way: R19 (the three pauses and the settings changed after them)
- The kept adapter: `run2-step40/` in the folder of kept models (`lora_state.safetensors`,
  `adapter/` in PEFT form, `adapter.gguf`, the run's config and the step-40 evaluation)

## The evaluations

The 200-question evalset with turns (the run's own protocol), checker 4. **Greedy with turns is
the measure the kept step is chosen by** (R14); "strict" counts undecided and no-query answers as
wrong; "first query" is the first query alone, before any turn. A quick evaluation is greedy
only; the full ones (steps 20 and 40) add 8 sampled answers a question at temperature 1.0.

| | With turns | Strict | First query | Fixed / broken by the turns | mean@8 | pass@8 |
|---|---|---|---|---|---|---|
| Start (run 1b step 30) with turns | 52.4 | 49.0 | 52.4 | 7 / 7 | 47.6 | 71.8 |
| Start on one query | 54.5 | | | | 43.4 | 67.5 |
| After step 5 | 52.1 | 49.0 | 55.6 | 3 / 10 | | |
| After step 10 | 56.8 | 54.0 | 58.4 | 5 / 8 | | |
| After step 15 | 56.8 | 54.0 | 57.9 | 4 / 6 | | |
| After step 20 | 57.6 | 55.0 | 59.7 | 3 / 7 | 49.4 | 74.1 |
| After step 25 | 56.0 | 53.5 | 59.7 | 2 / 9 | | |
| After step 30 | 57.9 | 55.0 | 59.5 | 5 / 8 | | |
| After step 35 | 58.9 | 56.0 | 60.0 | 7 / 9 | | |
| **After step 40** | **60.2** | **57.5** | **62.3** | 4 / 8 | **50.7** | **74.6** |

Sign tests, question by question (better / worse, p):

| | Greedy with turns | Sampled (right of 8) | pass@8 |
|---|---|---|---|
| Step 20 against the start with turns | 19 / 11 (0.20) | 54 / 37 (0.09) | 11 / 5 (0.21) |
| Step 20 against the start on one query | 19 / 15 (0.61) | **73 / 40 (0.002)** | **19 / 6 (0.015)** |
| Step 35 against the start with turns | 19 / 8 (0.052) | | |
| **Step 40 against the start with turns** | **20 / 6 (0.009)** | 66 / 46 (0.07) | 17 / 10 (0.25) |
| **Step 40 against the start on one query** | 23 / 14 (0.19) | **76 / 34 (0.0001)** | **20 / 6 (0.009)** |

By dataset at step 40 (greedy / mean@8 / pass@8): CypherBench 60.0 / 49.6 / 88, gretel 60 / 62.0
/ 82, SQaLe 40.4 / 31.9 / 51.0, text2cypher 81.8 / 59.3 / 77.1. About 50 questions a dataset:
one question is 2 points.

**Split by what a look at the result can tell** (the reference's result on the copy the model
sees; `results/run2-phase0.md` section 8):

| | Start with turns | Step 20 | Step 35 | Step 40 |
|---|---|---|---|---|
| A look can inform (87 questions): first query to final | 45 to 52 | 51 to 54 (3 fixed, 0 broken) | 52 to 58 (6 / 0) | 53 to 56 (3 / 0) |
| The right result is empty (13): first query to final | 10 to 3 | 11 to 4 (0 / 7) | 11 to 3 (1 / 9) | 11 to 4 (1 / 8) |
| One query (100) | 43 of 88 decided | 52 | 51 | 55 |

**The returned dev slice** (50 gretel questions whose right result is nothing, no step trains on
them; strict check), right first / right after the turns / broken: start 25 / 11 / 14; step 5
22 / 8 / 14; step 10 25 / 10 / 15; steps 15 and 20 25 / 8 / 17; step 25 25 / 5 / 20; steps 30,
35 and 40 23 / 6 / 17.

## The training side

Means over groups of five steps. Every step's 32 questions are new, so one step's solve rate
swings by 0.1-0.2 with its batch.

| Steps | Solve rate | First query | Calls | `kl_start` (start to end) | KL weight | Entropy (first turn) | Returned solved | Returned: right first answers broken | Largest grad norm |
|---|---|---|---|---|---|---|---|---|---|
| 1-5 | 0.444 | 0.441 | 1.60 | 0.003 to 0.009 | 0.05 | 0.265 | 0.025 | 93% | 0.21 |
| 6-10 | 0.478 | 0.494 | 1.52 | 0.012 to 0.047 | 0.05 | 0.227 | 0.117 | 81% | 0.22 |
| 11-15 | 0.504 | 0.508 | 1.49 | 0.065 to 0.072 | 0.05; 0.10 from step 12 | 0.222 | 0.051 | 86% | 0.18 |
| 16-20 | 0.452 | 0.446 | 1.52 | 0.115 to 0.060 | 0.10; 0.20 from step 17 | 0.205 | 0.108 | 83% | 0.19 |
| 21-25 | 0.514 | 0.533 | 1.51 | 0.078 to 0.074 | 0.20 | 0.192 | 0.059 | 90% | 0.45 |
| 26-30 | 0.546 | 0.559 | 1.54 | 0.063 to 0.031 | 0.20 | 0.219 | 0.042 | 89% | 0.20 |
| 31-35 | 0.502 | 0.514 | 1.55 | 0.036 to 0.033 | 0.20 | 0.235 | 0.233 | 57% | 0.17 |
| 36-40 | 0.515 | 0.510 | 1.47 | 0.042 to 0.046 | 0.20 | 0.214 | 0.077 | 64% | 0.24 |

- **The first query got better and the model kept using turns.** About 1.5 calls an episode
  throughout; the share of answers broken by a later query fell to under 5% in the last ten
  steps; the other gretel questions fixed about 3 a step and broke almost none.
- **Drift rose for 16 steps, then the doubled and redoubled penalty held it.** `kl_start` (less
  the measured offset) went 0.009 at step 5 to 0.115 at step 16, then 0.03-0.08 for the rest of
  the run under the 0.20 weight. What moved first: shorter first turns (median 245 to 208
  characters by step 11) written on one line (multi-line first queries 51 of 256 at step 1, 15
  at step 11).
- **The returned questions stayed the weak spot.** Most steps' returned groups lost all 8 tries,
  and a group that scores the same everywhere teaches SRPO nothing. Steps 31-35 look better
  (23% solved) because steps 31 and 32 drew returned questions that are counts: their right
  answer is a row with 0 in it, which the model keeps. The one GROUP BY question of step 32,
  whose right result really is empty, failed all 8 tries.
- **No reward hacking.** Final queries that return constants stayed at 0-1.2% (they earn
  nothing); copied literals 0-6%; leaks and look-leaks 0.

## Changes made during the run (R19)

| When | What fired | Change | Who |
|---|---|---|---|
| After step 11 | first-turn KL 0.065 above 0.05 | KL weight and bound 0.05 to 0.10 | the agent, Rishi having delegated the call |
| After step 12 | `SELECT *` in 5.9% of final queries | none: 7 of the 15 were one question asking for whole rows (5 right); the other 8 probed an empty table and earned nothing | the agent, same delegation |
| After step 16 | KL 0.115 above 0.10; DISTINCT 39% against 24% at the start | KL weight 0.20, bound 0.15, DISTINCT margin 0.10 to 0.20 (DISTINCT is CypherBench's own style: right 38 of 65 with it, 2 of 15 without) | Rishi |

## What went wrong on the machine (nothing lost)

- **Two rollout failures** (steps 4 and 23): llama-server answered HTTP 500 for 1 and 2
  episodes ("does not match the expected Content-only format"). The run stopped as designed;
  one restart each retried only those episodes.
- **Stopped after step 34 for other GPU work** (Rishi's request, `pause.py --after-step`).
  The PC restarted while it was stopped; Docker Desktop and the checker's Neo4j
  (`asktrainer-neo4j`) do not start by themselves after a reboot and were started by hand
  before the resume. Step 35's rollouts ran with no errors.

## The reading

- **Run 2 works on what a look can help with.** Step 40 beats the start adapter with turns by
  about 8 points greedy (60.2 against 52.4; 20 / 6 questions, p 0.009) and the start on one
  query by about 6 (60.2 against 54.5), with mean@8 and pass@8 up as well. The gain comes from
  better first queries (62.3 against 52.4) and from fixing a query when its result shows
  values (none broken there at steps 20-40).
- **Option C's own target did not move.** When the right result is empty the model still
  probes the empty tables and rewrites a right query: 8 of 11 right first answers broken at
  step 40 on the evalset, 17 broken on the returned dev slice (start 14).
- **Headroom:** pass@8 minus greedy was 14.4 at step 40 against 19.4 for the start with turns:
  more of what the model can do now comes out in its greedy answer.

## What a next run must take from this

1. **The returned questions need signal SRPO can use.** An all-wrong group gives none. Options:
   more returned questions a step with easier ones among them, or a reward for ending on the
   first query when its result is empty and the check passes, or the dev slice's own split
   (count questions apart from row questions).
2. **Start the KL weight at about 0.20,** not 0.05: the drift needed it from step 12 on, and the
   run's best steps came under it.
3. **The DISTINCT rule should be measured per dataset** (CypherBench's references use it).
4. **After a reboot,** start Docker Desktop and `asktrainer-neo4j` before resuming.

## The tool protocol (for the model card)

What the step-40 adapter was trained and measured under (`src/sqlrl/run2/prompt.py`); a model
card must show it exactly, or the model is used under a protocol it never saw:

- **The system prompt is the SFT model's,** with two edits: "with ONE read-only <language>
  query" becomes "with one read-only <language> query at a time", and one rule is added before
  the "If the database can't answer the question" rule:
  "After each call you get its result: the rows, or an error. If the result shows the query is
  wrong, call the tool again with a corrected query; you have 3 calls in all. When the result
  answers the question, stop: reply with one short sentence and no tool call. The last query you
  ran is your answer."
- **Up to 3 calls; the last query run is the answer.** A result is shown as a table of at most
  10 rows (in training also 5 or 20 rows and JSON, at random), or the error text; Neo4j queries
  run read-only with a time limit.
- **SQaLe and text2cypher questions were kept on one query** (the SFT prompt unchanged, no
  result shown), so the adapter is measured on those two the single-turn way.
- Thinking stays on (the SFT model's "high" setting, a 1,024-token budget a turn).

## The test set

Once, on the kept adapter, with Rishi's OK (`scripts/run2/testset.py run --run run2 --step 40`;
R20): the 1,000 questions (250 a dataset), greedy, the same yardstick as the evaluations. Three
answer sets, each under its own server: step 40 with turns, the start adapter with turns (the
run's protocol), and the start adapter on one query with the released prompt (the anchor's
setting, which matched the kept single-turn evaluation 101 against 102 of 187). No episode
failed; every CypherBench graph and SQL database loaded.

| | With turns (decided) | Strict | First query | Fixed / broken by the turns |
|---|---|---|---|---|
| Start on one query | 53.0 (495 of 934) | 49.5 | 53.0 | |
| Start with turns | 54.2 (505 of 932) | 50.5 | 54.5 | 25 / 28 |
| **Step 40 with turns** | **60.8 (571 of 939)** | **57.1** | **62.3** | 17 / 32 |

Sign tests, question by question, greedy: **step 40 against the start with turns 100 better /
43 worse (p 0.000002); against the start on one query 124 / 59 (p 0.000002)**. The start with
turns against the start on one query: 53 / 46 (p 0.55).

By dataset (greedy with turns; first query in brackets where turns apply):

| | CypherBench | gretel | SQaLe | text2cypher |
|---|---|---|---|---|
| Start on one query | 46.3 | 67.6 | 43.1 | 54.5 |
| Start with turns | 50.8 (47.2) | 63.2 (68.0) | 45.7 | 57.0 |
| **Step 40 with turns** | **63.6** (61.3) | 62.8 (**70.9**) | 43.8 | **73.7** |

About 250 questions a dataset: one question is 0.4 points.

- **The evalset's result holds on the test set:** step 40 is about 7 points above the start with
  turns (60.8 against 54.2) and 8 above the start on one query (53.0), and with five times the
  questions the sign tests are no longer close.
- **The gain is the first query.** Step 40's first query alone scores 62.3; its turns then cost
  1.5 points (17 fixed, 32 broken), almost all on gretel, where the first query is 70.9 and the
  final answer 62.8. CypherBench gains from its turns (61.3 to 63.6); text2cypher (one query)
  carries the largest gain, 57.0 to 73.7.
- **SQaLe did not move** (43.1 to 45.7 to 43.8: within a few questions).
- So on unseen questions **the turns still break more right gretel answers than they fix**, as
  the returned slice said: what training must change next is the same thing option C aimed at.

## Not done

- A release export of step 40 (merged into the Q8_0 model) and its model card (the protocol
  above is the part a card must carry).
- What runs next: Rishi's call (the options in "What a next run must take from this").
