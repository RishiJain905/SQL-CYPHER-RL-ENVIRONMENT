# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | ✅ done (2026-11-01) | `baseline-sft1500-t1.0`: greedy 43.2%, mean@8 36.9%, **pass@8 62.6%**; 94 sometimes / 74 never / 30 always (`results/step1-baseline.md`) |
| 2. Method pilots | ⏳ not started | GRPO family, SDPO, SRPO |
| 3+. The real RL run | ⏳ not started | stage A single-turn, stage B multi-turn, optional stage C |

## Session log

- **2026-10-31 / 11-01:** Repo created (public, `RishiJain905/SQL-CYPHER-RL-ENVIRONMENT`), plan
  copied from the trainer (`rl.md`), CLAUDE.md / AGENTS.md / Codex config / the secret hook. Rishi
  said to start the baseline. Built step 1 (`src/sqlrl/step1_baseline/`: sample, score, feed, run;
  `passk.py`), smoke-tested the scorer (43.17% on the greedy answers, as released) and sampling
  (seeds reproduce), ran it (two stray Ctrl+C interruptions, both resumed with nothing lost: R3).
  Live page built by the sonnet-xhigh subagent at Rishi's request (`documentation/dashboard/`).
  Results, guide (`guides/sampling-baseline.md`), decisions R1-R3.

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
