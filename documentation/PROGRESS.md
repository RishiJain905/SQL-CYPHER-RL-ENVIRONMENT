# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | ✅ done (2026-11-01) | `baseline-sft1500-t1.0`: greedy 43.2%, mean@8 36.9%, **pass@8 62.6%**; 94 sometimes / 74 never / 30 always (`results/step1-baseline.md`) |
| 2. Method pilots | ✅ both run (2026-11-02); the choice is Rishi's | two pilots of 10 steps (R4). GRPO: greedy 45.4, mean@8 37.5, pass@8 62.6; SRPO: 45.4 / 39.7 / 64.2 (baseline 43.2 / 36.9 / 62.6); none outside noise; `pilots/TRACKER.md` |
| 3. Run 1 (stage A, single turn) | ⏹️ stopped after step 28 of 100 (2026-11-13); **step 19's adapter kept** (R12) | SRPO, 32 questions x 8 a step. After step 19: greedy 45.6, mean@8 40.8, pass@8 66.2 (baseline 43.2 / 36.9 / 62.6; sampled sign test 60 / 41, p 0.07). Worse at steps 25 and 28 (step 28 against 19: 42 / 63, p 0.05): the settings overshoot. `results/run1.md`; R10-R12. Not done: the test set on step 19, a release export |
| 4. Run 1b (the trainer, fixed) | 🧑 built and loop-tested (2026-11-14); **Rishi starts it** | run 1 again for 40 steps, held near the SFT model: a KL penalty whose weight follows the measured drift (target 0.03), learning rate 5e-6, evaluations after 10 / 20 / 30 / 40 with the best step kept, no reward for CypherBench answers that repeat rows where the reference doesn't. About 10.5 h. `run1b/design.md`, `run1b/RUNBOOK.md`, R13 |
| 5. Run 2 (stage B, multi-turn) | not designed | after run 1b: its settings if the trainer holds (R5 for the environment) |

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

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
- **2026-11-01:** Rishi told the agent to begin step 2's pilots (run with `--launched-by agent`).
- **2026-11-12:** Rishi told the agent to begin run 1 with the recommended choices (run with
  `--launched-by agent`); started 2026-11-13 18:14 UTC from an Opus 5.5 session following
  `run1/RUNBOOK.md`. After step 10 the KL stop bound was raised to 0.10 with his OK (R10). It paused after step 19 (KL 0.115); evaluated there it beat the SFT model (greedy 45.6, mean@8 40.8, pass@8 66.2 against 43.2 / 36.9 / 62.6) and resumed with the bound at 0.30 and evaluations after steps 35 and 75 added (R11). At step 28 (`kl_sft` 0.29) Rishi OK'd an extra evaluation there; after it he stopped the run for good and kept step 19's adapter (R12).
