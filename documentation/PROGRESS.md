# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | ✅ done (2026-11-01) | `baseline-sft1500-t1.0`: greedy 43.2%, mean@8 36.9%, **pass@8 62.6%**; 94 sometimes / 74 never / 30 always (`results/step1-baseline.md`) |
| 2. Method pilots | ✅ both run (2026-11-02); the choice is Rishi's | two pilots of 10 steps (R4). GRPO: greedy 45.4, mean@8 37.5, pass@8 62.6; SRPO: 45.4 / 39.7 / 64.2 (baseline 43.2 / 36.9 / 62.6); none outside noise; `pilots/TRACKER.md` |
| 3+. The real RL run | 📝 run 1 being designed | stage A single-turn, stage B multi-turn, optional stage C; environments per R5; run 1's draft `run1-design.md` |

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

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
- **2026-11-01:** Rishi told the agent to begin step 2's pilots (run with `--launched-by agent`).
