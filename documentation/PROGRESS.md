# Progress

## Status

| Step | Status | Notes |
|---|---|---|
| 1. Sampling baseline | 🟡 running | `baseline-sft1500-t1.0`: 200 evalset questions x 8 sampled answers, checker 4 (R1) |
| 2. Method pilots | ⏳ not started | GRPO family, SDPO, SRPO |
| 3+. The real RL run | ⏳ not started | stage A single-turn, stage B multi-turn, optional stage C |

## Session log

- **2026-10-31:** Repo created (public, `RishiJain905/SQL-CYPHER-RL-ENVIRONMENT`), plan copied from
  the trainer (`rl.md`), CLAUDE.md / AGENTS.md / Codex config / the secret hook. Rishi said to
  start the baseline. Built step 1 (`src/sqlrl/baseline/`: sample, score, feed, run; `passk.py`),
  smoke-tested the scorer (43.17% on the greedy answers, as released) and sampling (seeds
  reproduce), launched the run detached. Live page built by the sonnet-xhigh subagent at Rishi's
  request (`documentation/dashboard/`).

## Rishi-run steps and waivers

- **2026-10-31:** Rishi started the baseline by telling the agent to begin it (run with
  `--launched-by agent`).
