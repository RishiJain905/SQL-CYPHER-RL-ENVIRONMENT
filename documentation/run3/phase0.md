# Run 3, phases 0 and 1: what is measured, and how it is read

**Written 2026-11-23, before any phase 0 measurement** (Rishi approved phases 0 and 1 the same
day, R27 fixed the splits first). Nothing below may be changed after its numbers are seen without
a new decision saying why. The design is `design.md`; the splits `src/sqlrl/run3/splits.py`.

## Phase 0

### 0.1 The anchor (design 5.1)

Step 40 as the PC serves it (its float32 merge to Q8_0 on llama-server, run 2's file) answers run
2's evalset greedy, with turns, through run 2's own evaluation path. **Passes** when the score is
within noise of run 2's 60.2 (at least 57.0) and it agrees with run 2's recorded answers on at
least 95% of questions (run 2's own anchor: 183 of 186). A failure stops phase 0: every later
number would be measured on the wrong model.

### 0.2 pass@8 of step 40: the useful band (design 8.1)

- **Questions:** training split only (`splits.trainable`). Never the held-out databases, the dev
  slices, BIRD dev or CypherBench test: the band chooses what RL trains on.
- **A group:** one (question, engine) pair, 8 episodes. The episode setting (schema shown, tools,
  row count, result style, hints, tool names) is drawn once per group from the seed
  `run3-p0:<question id>:<engine>`, so the 8 episodes differ only by sampling, and is recorded.
- **Sampling:** run 2's settings: temperature 1.0, top-p 1.0, a thought budget of 1,024 tokens,
  2,048 for the answer part, a different seed for each of the 8. Run 3's protocol (submit /
  decline / clarify, up to 8 calls).
- **The sample:** stratified by pool and engine (BIRD on SQLite, DuckDB, PostgreSQL, MySQL;
  CypherBench by graph on Neo4j; the non-answers). Its size is set from a timed pilot of 50 groups
  so that the whole measurement fits about 10 hours on the PC, and is written into this file
  before the main measurement starts.
- **Set 2026-11-24 from the pilot** (50 groups, 400 episodes in 9.8 minutes, no crashes; about 25
  BIRD episodes a minute, non-answers faster): **200 BIRD questions per SQL engine, 80 per
  CypherBench training graph, 40 write requests and 40 wrong-database questions per SQL engine**,
  1,440 groups, 11,520 episodes, about 7 hours. The pilot's groups were drawn from the smaller plan
  and are kept apart (`pass8-pilot`), not mixed in.
- **Read as:** a group is *never* (0 of 8 right), *useful* (1 to 7) or *always* (8 of 8). Reported
  per pool, engine and harness variant (schema shown x tools), with mean@8 and pass@8.
- **The gate to RL (design 5.2):** a pool and engine is used by RL when at least 15% of its
  groups are *useful*. Below that, the warm start has to move it first (re-measured on the dev
  slice after phase 1), or it is left out of phase 2 and the report says so.

### 0.3 What else phase 0 reports

The gold check per engine (design 3.2) as built; the non-answers' decline rates (write requests,
wrong database); calls an episode and the explore share (an episode whose first call is not its
answer) by schema shown; the episodes' token lengths against the trainer's limit on 16 GB (3,840
tokens in run 2); the teacher pilot, if Rishi approves its spend (design 3.5).

## Phase 1: the warm start (design 5.2)

- **Data:** right episodes only, training split only. Step 40's own right episodes from 0.2 (at
  most two different ones a question, so easy questions don't dominate), and the teacher's right
  episodes if approved, mixed so step 40's own make up at least half (its style stays). Every
  tool result is context, never trained on; the loss is on the model's own turns only.
- **Model:** step 40's adapter trained further (R25); the result is a new adapter, `run2-step40`
  is never overwritten. Served only through the float32 merge to Q8_0.
- **Gates** (all on the PC, greedy):
  1. **No loss on run 2's evalset:** at least 57.0, and a question-by-question sign test against
     step 40 does not show it worse (p >= 0.05).
  2. **Better on the run-3 dev slice** than step 40 on the same (question, engine, setting)
     triples: more right answers, with the sign test reported.
  3. **Behaviour on the dev slice:** at most 5% of episodes malformed; write requests declined at
     least 90%, wrong-database questions at least 80%; answerable questions falsely declined at
     most 5%; when the prompt shows no schema, at least 80% of episodes look before submitting.
- **Tracked live** on a dashboard (as runs 1 and 2) and in W&B (`launched-by:agent`).
- A gate that fails stops before phase 2, for Rishi to read.
