# Run 3, phase 0: the anchor, step 40 on the dev set, and pass@8 (2026-11-23 / 24)

Read by the rules in `../run3/phase0.md`, which were committed before any of this was measured
(`fad8715`; the sample size in `3ca9219`, before the main measurement). Everything on the PC.

## The anchor (0.1): passes

Step 40 as the PC serves it (run 2's float32-merged Q8_0), run 2's evaluation set, greedy, with
turns, through run 2's own path: **114 of 192 decided, 59.4%** (run 2's evaluation after step 40:
115 of 191, 60.2%). Question by question against run 2's: 2 better, 3 worse (sign test p = 1.0),
97.5% agreeing. Bar: 57.0 and 95%.

## Step 40 on the run-3 dev set (the phase-1 baseline)

300 greedy episodes (`<data>/run3/phase0/dev-step40`): 40 BIRD dev-slice questions per SQL engine,
15 per CypherBench training graph, 10 write requests and 10 wrong-database questions per engine.

| | Step 40 | Phase 1 gate |
|---|---|---|
| Right, answerable questions | 94 / 220 (42.7%) | must improve |
| Malformed endings | 18.7% (54 over the 8-call limit, 2 with no ending) | <= 5% |
| Write requests declined | 97.5% | >= 90% |
| Wrong-database questions declined | 47.5% | >= 80% |
| Answerable questions falsely declined | 0.9% | <= 5% |
| Looks before submitting when no schema is shown | 95% | >= 80% |

Two gates fail before any training: the call limit and wrong-database declines.

## pass@8 (0.2): the useful band

1,440 groups (a question on an engine, 8 episodes each at temperature 1.0, the setting drawn once
a group), **11,520 episodes in 5 h 21 m, 0 crashes after 5 retries** (server errors on very long
episodes). Training split only.

| Pool / engine | Groups | Never | Useful | Always | Useful % | mean@8 | pass@8 |
|---|---|---|---|---|---|---|---|
| BIRD · SQLite | 200 | 70 | 104 | 26 | **52.0** | 37.6 | 65.0 |
| BIRD · DuckDB | 200 | 78 | 96 | 26 | **48.0** | 36.4 | 61.0 |
| BIRD · PostgreSQL | 200 | 67 | 100 | 33 | **50.0** | 40.2 | 66.5 |
| BIRD · MySQL | 200 | 79 | 89 | 32 | **44.5** | 37.3 | 60.5 |
| CypherBench art | 80 | 26 | 48 | 6 | **60.0** | 38.3 | 67.5 |
| CypherBench biology | 80 | 18 | 56 | 6 | **70.0** | 45.3 | 77.5 |
| CypherBench soccer | 80 | 14 | 50 | 16 | **62.5** | 51.7 | 82.5 |
| CypherBench terrorist_attack | 80 | 25 | 45 | 10 | **56.2** | 43.6 | 68.8 |
| Write requests (4 engines) | 160 | 0 | 51 | 109 | 22.5-42.5 | 87.8-94.7 | 100 |
| Wrong database (4 engines) | 160 | 7 | 98 | 55 | 52.5-72.5 | 69.1-73.1 | 92.5-97.5 |
| **All** | **1,440** | **384** | **737** | **319** | **51.2** | **49.1** | **73.3** |

(Everything per stratum and variant: `<data>/run3/phase0/pass8-step40/summary.json`.)

**The gate to RL (15% useful): every pool and engine passes, the lowest by three times** (MySQL,
44.5%). The Cypher graphs have the widest band (56-70%), where step 40 did worst on the real
deployment (12.5%). 384 groups (27%) are never right: RL can't learn from them; examples can.

**By harness variant** (episodes, mean@8, useful %, explore share): full schema 59-61% right,
names only 40-43%, no schema 32-33%. Hiding the schema halves the right rate even though the
model explores in 89% of those episodes: what it finds, it uses badly. Calls an episode 3.64.

**Endings:** submit 7,315, decline 2,167, **over the 8-call limit 1,959 (17%)**, no ending 79.
Why answers are wrong: a different result 2,935, an engine error 575, a submit on a write request
101 or an unanswerable question 135, a false decline 84.

**Lengths:** the longest context an episode reaches, median 1,544 tokens, 90th percentile 2,900,
max 10,359; 4.4% over the 16 GB trainer's 3,840 (each turn alone, which is what the warm start
trains on, is shorter). Spark's template drops earlier turns' thinking, so episodes stay short.

## What it means for phase 1 and phase 2

- RL has signal everywhere: half of all groups are useful. The warm start is not needed to make RL
  possible; its job is the call-limit habit (17-19% of episodes) and the wrong-database declines,
  cheaply, on the PC.
- The hidden-schema variants are the weakest: a target for both the warm start (step 40's own
  right "none" episodes) and RL.
- Found on the way (fixed for later runs): the sampler wrote results in submission order and
  Neo4j holds one graph at a time, so at the soccer / terrorist_attack switch 26 soccer episodes
  waited about 25 minutes behind the other graph and nothing was written meanwhile (no work lost).

## Phase 1: the warm start (2026-11-24)

Two passes on the PC (R28). Pass 1: step 40's own 1,659 right episodes (6,182 examples, 379
steps, lr 1e-5, 16 examples a step, loss on the model's own tokens). Pass 2, from pass 1's step
129: 8-call episodes left out (343), the teacher pilot's right episodes within the cap added
(42; 1,550 of step 40's own), 112 steps. Each checkpoint merged in float32 into Q8_0 and scored
greedy on run 2's evaluation set and the run-3 dev set (the same 300 triples as step 40's).

| Checkpoint | Evaluation set | Dev right | Malformed | Wrong-db declined | Write declined | False decline | Looks (no schema) |
|---|---|---|---|---|---|---|---|
| Step 40 | 59.4 | 42.7 | 18.7 | 47.5 | 97.5 | 0.9 | 95.0 |
| **Pass 1 · 129** | **64.2** (12 / 4, p 0.08) | 46.4 (17 / 9) | 11.3 | 55.0 | 92.5 | 1.8 | 91.7 |
| Pass 1 · 257 | 62.7 | 43.2 | 12.0 | 60.0 | 95.0 | 0.9 | 92.0 |
| Pass 1 · 379 | 58.8 | 45.9 | 14.0 | 55.0 | 95.0 | 0.9 | 88.0 |
| **Pass 2 · 56** | 63.5 (12 / 6) | **46.8** (19 / 10) | 12.7 | 55.0 | 95.0 | 0.5 | 92.3 |
| Pass 2 · 112 | 62.0 | 45.9 | 15.0 | 52.5 | 100 | 1.4 | 94.7 |

- **The best checkpoints** are pass 1's 129 and pass 2's 56, level within noise. Both beat step
  40 on run 2's evaluation set (which the warm start never trained on) and on the run-3 dev set.
  After them, more steps over-fit: the training loss falls while both scores slide back.
- **Two gates fail at every checkpoint:** malformed endings (11-15%, the 8-call limit) and
  wrong-database declines (52-60%). Leaving out the 8-call episodes did not move the first. The
  call-limit episodes are struggles, not a copied habit: 34 of 38 with the schema partly or fully
  hidden, 63 engine errors and 33 empty results among them, 84 repeated identical calls, and a
  further tool call after the "No tool calls left" notice. Knowing when to stop and commit is
  shaped by reward, which right-episode SFT cannot show: RL's reward targets it directly
  (running out -0.1, below a wrong answer; a right decline 1; a cost a call after the second).

The teacher pilot (approved, $0.29 of a $2 cap): 200 episodes of DeepSeek V4.1 Flash at max
effort in run 3's environment, about $0.0014 each, no crashes. Right on 31 of 140 questions step
40 never got right (22%), 34 of 40 wrong-database questions (85%), 12 of 20 useful-band ones;
6.9 calls an episode; reasoning short (median 25 tokens).
