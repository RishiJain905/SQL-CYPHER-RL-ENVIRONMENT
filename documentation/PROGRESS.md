# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | ✅ done (2026-11-01) | `baseline-sft1500-t1.0`: greedy 43.2%, mean@8 36.9%, **pass@8 62.6%**; 94 sometimes / 74 never / 30 always (`results/step1-baseline.md`) |
| 2. Method pilots | 🟡 running | two pilots of 10 steps (R4): `p1-grpo` running, then `p0-srpo-smoke` and `p2-srpo` (queued); `pilots/TRACKER.md` |
| 3+. The real RL run | ⏳ not started | stage A single-turn, stage B multi-turn, optional stage C (`rl-runs.md` has the open questions) |

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

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
- **2026-11-01:** Rishi told the agent to begin step 2's pilots (run with `--launched-by agent`).
