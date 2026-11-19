# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | ✅ done (2026-11-01) | `baseline-sft1500-t1.0`: greedy 43.2%, mean@8 36.9%, **pass@8 62.6%**; 94 sometimes / 74 never / 30 always (`results/step1-baseline.md`) |
| 2. Method pilots | ✅ both run (2026-11-02); the choice is Rishi's | two pilots of 10 steps (R4). GRPO: greedy 45.4, mean@8 37.5, pass@8 62.6; SRPO: 45.4 / 39.7 / 64.2 (baseline 43.2 / 36.9 / 62.6); none outside noise; `pilots/TRACKER.md` |
| 3. Run 1 (stage A, single turn) | ⏹️ stopped after step 28 of 100 (2026-11-13); **step 19's adapter kept** (R12) | SRPO, 32 questions x 8 a step. After step 19: greedy 45.6, mean@8 40.8, pass@8 66.2 (baseline 43.2 / 36.9 / 62.6; sampled sign test 60 / 41, p 0.07). Worse at steps 25 and 28 (step 28 against 19: 42 / 63, p 0.05): the settings overshoot. `results/run1.md`; R10-R12. Not done: the test set on step 19, a release export |
| 4. Run 1b (the trainer, fixed; the official stage-A run) | ✅ done, 40 of 40 steps (2026-11-17); **step 30's adapter kept** (R14) | After step 30: greedy **54.5** / mean@8 43.4 / pass@8 67.5 (SFT through the same path 44.5 / 36.4 / 59.9; greedy sign test against the baseline 32 / 14, p 0.01). text2cypher 16 to 62.5 greedy. Step 40: 48.6 / 39.6 / 66.0, after the KL weight hit its cap and pulled `kl_sft` from 0.078 to 0.010. Greedy picks the step (R14). `results/run1b.md`, `run1b/design.md`, R13-R14. Not done: the test set on step 30, a release export |
| 5. Run 2 (stage B, multi-turn) | 🔨 built and loop-tested, **not started**; the no-training tests are done (2026-11-19) and say **not ready** | The model sees what its query returned and may fix it, up to 3 queries (SQaLe keeps one: R16); continues step 30's adapter (R15, `run2/design.md`, `run2/RUNBOOK.md`, `guides/run2-environment.md`). Checks: the template round trip (18 of 18 equal), the anchor (the engine's first query 101 of 186 against the kept 102 of 187), the drift offset (0.0099), two 2-step loop tests. **Dev slice: turns add 4 points** (57.8% to 61.8%; fixed 9, broke 1). **Evaluation set: turns cost 4 points** (step 30: first query 52.4%, with turns 48.1%; fixed 4, broke 10; mean@8 43.9, pass@8 69.2), which is the "do not start RL" row of the design page's table (D10). Cause: a right query whose result says nothing gets rewritten (9 of 25 broken, against 1 of 53 when the result shows values), and the pool's filter keeps such questions out of training and out of the dev slice (gretel: references whose result says nothing; text2cypher: path questions over schemas with no relationships, 31 of the evaluation set's 50 prompts). `results/run2-phase0.md` sections 7-9. **Rishi chose option C (R17), built and tested the same day:** text2cypher on one query; 3 of a step's 10 gretel questions are ones the pool had rejected because the right result is nothing (604 taken back, 50 held as a second dev slice), scored with a stricter check. Under it, untrained: evaluation set 52.4% first query and 52.4% with turns (fixed 7, broke 7; mean@8 47.6, pass@8 71.8); where a look can inform 52.3% to 60.5% with nothing broken; the returned dev slice 25 of 50 right to 11 (14 broken: the number training must lower). A third loop test passed. Section 10. **Open, Rishi's:** the go; and whether a database with no rows should mean one query (an estimated 55.6% untrained) |

## Session log

- **2026-10-31 / 11-01:** Repo created (public, `RishiJain905/SQL-CYPHER-RL-ENVIRONMENT`), plan
  copied from the trainer (`rl.md`), CLAUDE.md / AGENTS.md / Codex config / the secret hook. Rishi
  said to start the baseline. Built step 1 (`src/sqlrl/step1_baseline/`: sample, score, feed, run;
  `passk.py`), smoke-tested the scorer (43.17% on the greedy answers, as released) and sampling
  (seeds reproduce), ran it (two stray Ctrl+C interruptions, both resumed with nothing lost: R3).
  Live page built by the sonnet-xhigh subagent at Rishi's request (`documentation/dashboard/`).
  Results, guide (`guides/sampling-baseline.md`), decisions R1-R3.
- **2026-11-01:** Rishi asked for each step's code in its own folder (`step1_baseline/`,
  `step2_pilots/`), a doc on the two RL runs (sonnet-xhigh: `rl-runs.md`, published as "The Two
  RL Runs") and a step-2 tracker (sonnet-xhigh: `pilots/TRACKER.md` and the "Spark RL Pilots"
  page). Built step 2: the pool, rollouts on llama-server's raw endpoint, the WSL update with
  GRPO / DAPO / GSPO / SRPO losses, the LoRA to GGUF, the evaluation, `pause.py`, the queue, the
  tracker feed. `p0-smoke` passed (2 steps). Rishi cut the pilots to GRPO and SRPO at 10 steps
  with a pause at any moment (R4); `p1-grpo` started, the pause was tested on it mid-rollout and
  it resumed; the queue runs the SRPO smoke and `p2-srpo` after it.
- **2026-11-01 / 02:** two power cuts (each left Docker and the checker's Neo4j down: training
  and scoring now refuse to run without Neo4j) and two pauses for Rishi's GPU; every resume kept
  its work. `p1-grpo` and `p2-srpo` trained and evaluated. Results in `pilots/TRACKER.md`.
- **2026-11-05:** discussed the environments with Rishi: what the model reads and what decides
  the reward match deployment, the plumbing is a sandbox (decisions R5). Runs are designed one at
  a time; run 1's draft is `run1-design.md`, waiting on Rishi's three choices (dataset mix,
  learning rate, length).

- **2026-11-12:** Rishi asked for run 1 in its own folders, a review of its design, and a faster
  llama.cpp harness, with the run itself left for him to start. Review: SRPO kept; a floor under
  the group std (R8); the importance weights dropped and KL measured from the server, after
  finding that the trainer's bf16 forward pass under-reads a small LoRA by about half (R7: the
  pilots' rollouts and evaluations were right, their logged KL was too low); lr 1e-5 kept.
  Harness (R9): the shared KV pool was the brake (208 -> 542 tokens/s at 16 slots with a KV
  per slot); the LoRA is merged into the Q8_0 rollout model each step (22 s, 23% faster than
  `--lora`); a worker per slot; one step's rollouts 292 s against 504-540 s; WSL kept awake
  during a run (the update's model load 17-22 s instead of ~200 s cold). Built
  `src/sqlrl/run1/` (R6), 20 CPU tests, loop-tested (2 steps, SQL questions), pause and resume
  tested mid-rollout. Later the same day, after Rishi had Docker restarted: `pool-v2` built
  (6,600 questions; a first build was discarded because the checker passes an exact copy of
  the reference without running it, so the filter had tested nothing); the evaluation path
  run end to end on the SFT model (greedy 44.5, mean@8 36.4, pass@8 59.9: the baseline within
  noise); `RUNBOOK.md`, `dashboard-brief.md` and the dashboard's feed written. Rishi chose
  the recommended mix, learning rate and length, and has the run started and watched from an
  Opus 5.5 session with a Sonnet agent building the dashboard.
- **2026-11-13:** run 1 started and watched from that session, the dashboard built by the
  sonnet-xhigh subagent and fed after every step. The KL bound was raised twice with Rishi's OK
  (R10, R11); evaluations after steps 19, 25 and 28. Step 19 beat the SFT model, 25 and 28 gave
  the gain back, and Rishi stopped the run with step 19's adapter kept (R12). Results and the
  reading in `results/run1.md`, with one finding for the environment: the model learned to drop
  `DISTINCT` and the checker's data can't show the difference. Next is Rishi's choice (row 4).
- **2026-11-14:** Rishi chose to fix the trainer and the `DISTINCT` gap first, then a short run
  1b. Built (R13): the KL penalty and its weight rule, the duplicate probe for CypherBench
  (tested on run 1's 1,045 passed answers), evaluations compared with the best earlier one,
  `BEST.json`, the turn rule, the `run1b` preset, run 1 drawn behind run 1b in the dashboard's
  feed. The `DISTINCT` finding was measured properly and corrected: the gap is CypherBench's
  sparse graphs (82% of references), and the model dropped `DISTINCT` even where the checker
  punished it. 27 run-1 tests (49 in all), ruff clean, a 2-step loop test on the GPU
  (`smoke-1b`). `run1b/design.md`, `RUNBOOK.md`, `dashboard-brief.md` written. The PC crashed
  mid-session (nothing lost; the checker's Neo4j was started again).
- **2026-11-14 (later):** at run 1b's start Docker wouldn't come up: the crash had corrupted the
  data drive's exFAT structures inside Docker Desktop's disk folder. A `chkdsk /f` with Rishi's OK
  (interrupted once by a restart, then run to the end) fixed the drive (0 bad sectors) but could
  not keep Docker's data disk, so every container and image was lost. The checker's Neo4j keeps
  nothing (graphs are loaded per check) and was recreated with the `docker run` in the trainer's
  `docs/traces.md`; a 10-reference probe ran with 0 errors. The RL data read back in full, WSL's
  disk mounted clean, the run-1 tests pass. The NFL engine's two Neo4j containers (their data is
  in folders on the drive, not in Docker's disk) are not recreated yet: Rishi's call. Run 1b
  started 19:59 UTC.
- **2026-11-14 / 15:** paused after step 7 for Rishi's GPU, resumed 01:34 UTC. Steps 8-12 healthy
  (`kl_sft` about 0.001, a tenth of run 1's at the same steps; `kl_beta` still 0.05). The
  evaluation after step 10 (greedy 46.7, mean@8 36.9, pass@8 58.4; sign tests 8/4, 38/49, 7/16
  against the baseline) reads as flat. The PC restarted at 03:38 UTC while step 13 was saving its
  update (no `TRAINED`, so step 13 is redone from its kept rollouts and scores). The machine's
  event log shows the same kind of unexplained restart about once or twice a week since August,
  most of them with no RL job running, so the run's code isn't the cause; heavy load makes them
  more likely. Rishi resumes the run himself after a restart (no auto-resume) and chose not to
  change the machine's settings. Marked paused by hand; resume on another day.
- **2026-11-15 / 16:** resumed; a NaN update at step 14 (fixed: 3e5d2eb) and pages of another
  process inside a saved `optim.pt` (fixed: every state file hashed and checked on the disk,
  6ae275b). A second crash during step 20's save to the USB data drive (bugcheck 0x13C, like the
  first) damaged Docker's disk and step 19's files; after `chkdsk` (Rishi's OK) Docker's disk was
  recreated (the checker's Neo4j and the NFL engine's two containers started again) and the run's
  data root was moved to the internal drive. Resumed from step 18, step 19 redone.
- **2026-11-16 / 17:** steps 19-40 and the evaluations after 20, 26, 30 and 40. Pauses by the
  length rule (steps 23 and 26) and the entropy rule (step 31) were read with Rishi, who loosened
  the rules and chose to run to step 40 (R14). Step 26 beat step 20 on every measure with half
  the thinking; step 30 had the best greedy (54.5); from step 34 the KL weight sat at its cap and
  pulled the model back toward SFT. A one-off GPU freeze at step 35 (resumed once with Rishi's
  OK). Rishi kept step 30 (R14): `results/run1b.md`. Copying the run back to the USB drive as
  history crashed the PC (bugcheck 0x12C, exFAT), so that drive fails under heavy writes from
  Windows too.

- **2026-11-17 / 18:** the USB data drive repaired (`chkdsk`, with Rishi's OK; 0 bad sectors),
  run 1b's folder copied to it as history (throttled; every file's hash checked) and removed
  from the internal drive; Docker's disk moved to the internal drive. Run 2's design page
  written for Rishi (14 choices, each with a recommendation).
- **2026-11-18 / 19:** Rishi took every recommendation and asked for the environment and the
  harness to be built and hardened, with the run left for him to start (R15). Built
  `src/sqlrl/run2/` (the environment, the episode engine, the scorer, the turn-aware trainer,
  the evaluation with turns, the stop rules, the orchestrator, the no-training tests), 81 CPU
  tests, ruff clean. On the GPU: the round trip, the anchor, the wording test, the drift
  offset, the update's memory by episode length, and a 2-step loop test (`smoke-3`: both steps
  trained, saved and merged). Three things the tests caught and fixed before any run: a right
  answer written twice would have been paid 0 (SQLite keeps prepared statements, so the table
  tracker saw nothing the second time); the trainer's 6,144-token limit doesn't fit the card
  (measured: 14.5 GB at 4,096 tokens, 4,608 fails; the limit is now 3,840); and several stop
  rules would have paused step 1 for what the untrained model already does (now measured
  against the run's own start). Paused for the day with the held-out evaluations just begun
  (stopped; the GPU is free).
- **2026-11-19:** Rishi said to continue. SQaLe put on one query (R16), with what that needed:
  the turn rates the stop rules read now count only episodes with turns, an evaluation records
  its protocol and the look-around data's version and is never continued or used as a start
  reference under another, and the trainer checks its prompt against the server's for every
  dataset (83 CPU tests). GPU: the dev slice again (57.8% to 61.8%), a second loop test
  (`smoke-4`), then the evaluation set with turns on step 30 (with 8 samples), step 26, the SFT
  model and step 30 with thoughts stripped. **The evaluation set disagrees with the dev slice:**
  turns cost step 30 four points there. Traced to right queries whose result says nothing,
  which the pool's filter keeps out of training and out of the dev slice
  (`results/run2-phase0.md`). Nothing was changed after that result; what to do is Rishi's.
  Also: Docker Desktop had stopped and would not start (a stale socket file it could not
  rename); with Rishi's OK the folder holding it was renamed and Docker came back.
- **2026-11-19, later:** Rishi chose option C (R17). Built: the returned-questions file (604 of
  the 627 the pool rejected), the scheduler (3 of gretel's 10 a step, the offset still in run
  1b's count), the stricter reward for them, text2cypher on one query, per-kind metrics, the
  evaluation's three-way split, 87 CPU tests. The reading rules were written down before the
  GPU tests. Then: both dev slices with two wordings (the new `keep` wording lost and is not
  used), the new start reference on the evaluation set (step 30, 8 samples), a loop test
  (`smoke-5`), and the drift offset again (0.0084). Found on the way: three quarters of the
  returned questions have a database with no rows at all.

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
- **2026-11-01:** Rishi told the agent to begin step 2's pilots (run with `--launched-by agent`).
- **2026-11-12:** Rishi told the agent to begin run 1 with the recommended choices (run with
  `--launched-by agent`); started 2026-11-13 18:14 UTC from an Opus 5.5 session following
  `run1/RUNBOOK.md`. After step 10 the KL stop bound was raised to 0.10 with his OK (R10). It paused after step 19 (KL 0.115); evaluated there it beat the SFT model (greedy 45.6, mean@8 40.8, pass@8 66.2 against 43.2 / 36.9 / 62.6) and resumed with the bound at 0.30 and evaluations after steps 35 and 75 added (R11). At step 28 (`kl_sft` 0.29) Rishi OK'd an extra evaluation there; after it he stopped the run for good and kept step 19's adapter (R12).
- **2026-11-14:** Rishi told the agent to begin run 1b (his kickoff from `run1b/RUNBOOK.md`, from
  an Opus 5.5 session); run with `--preset run1b --launched-by agent`, started 19:59 UTC.
  Paused after step 7 for his GPU and resumed when he said to (2026-11-15 01:34 UTC); after the
  PC restarted during step 13 he asked for it to stay paused until another day.
- **2026-11-16:** Rishi had the run resumed after each crash ("resume from the last saved step"),
  OK'd the evaluation after step 26 and the loosened length, entropy and KL rules, said to run on
  to step 40, OK'd one resume after the GPU freeze at step 35, and kept step 30's adapter (R14).
- **2026-11-18:** Rishi told the agent to build run 2 with every recommendation of the design
  page (D1-D14) and **not to start the main run**. The no-training tests and the loop test were
  run with `--launched-by agent` as part of that build (D10's recommendation). 2026-11-19: he
  said SQaLe may stay single-turn if that is better, and paused the session before the
  held-out evaluations. Later that day he said to continue: the rest of the no-training tests
  (the dev slice again, a loop test, the held-out evaluations) were run with
  `--launched-by agent`. The main run stays his to start. He then chose option C ("Go with
  option C"): its no-training tests, start reference and loop test were run the same way.
