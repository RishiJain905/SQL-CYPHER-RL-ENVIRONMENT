# Decisions

Every choice that isn't obvious from `rl.md`, and every place the build differs from it. Newest
last.

## R1: the sampling baseline's settings (2026-10-31)

- **The model:** the released merged Q8_0 GGUF, served by the trainer's `serve.ps1` (the same
  llama.cpp build, port 8092, thinking budget 1,024) as the greedy evalset scores. This makes
  greedy and sampled answers directly comparable, and it's the setup RL's rollouts will use.
- **Sampling:** temperature 1.0, top-p 1.0, top-k off, **min-p 0** (llama.cpp's default min-p
  is 0.05, which would quietly cut the tail; RL samples from the full distribution). The thinking
  level is each question's own (all 200 are `high`), as in the greedy run. Max 2,048 tokens.
- **Seeds:** one per answer, `1000 * (k + 1) + i` for sample k and question i. The same seed
  gives the same answer (checked), so any answer can be reproduced.
- **Scoring:** the trainer's `publiceval.verify` with checker 4, imported as a library; no
  trainer code changed. A smoke test re-scored the release's greedy answers: 43.17% on 183
  decided questions, exactly the release number.
- **Undecided answers:** an answer the checker can't decide (`exec_ok` None) counts neither way.
  A question's pass@k uses its decided answers only; a question with none is left out (as in the
  greedy scoring). With fewer than k decided answers, pass@k is 1 when any is right.
- **pass@k:** the unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021).

## R2: where runs and logs go (2026-10-31)

- **Runs** go under the data root's `rl/` folder (`<data>/rl/baseline/<run>/`), set in the
  git-ignored `configs/paths.local.toml` (`configs/paths.example.toml` is the template).
- **W&B project:** `sql-cypher-rl` (separate from the trainer's), job type `baseline`, tags
  `baseline`, `launched-by:<who>`, `checker:<n>`.
- **For now, the code runs with the trainer's WSL venv** (it has the checker's dependencies,
  W&B, pytest and ruff). This repo gets its own environment when the RL trainer needs
  packages the trainer's venv doesn't have.
- **W&B entity:** the stored login's default (the same place as the trainer's runs).

## R3: the baseline run's two interruptions, and how long jobs start (2026-11-01)

- **What happened:** the first launch went through `cmd /c` (for its output redirect); a Ctrl+C
  reached that console and ended the driver ~10 minutes into sampling (298 answers kept). The
  relaunch was a detached process with no console and finished sampling (1,600 answers, 0
  errors), but its `wsl.exe` child for scoring exited with 0xC000013A (Ctrl+C) after 6 s.
  Scoring was then run directly inside WSL and finished; the run was marked done by hand (an
  event in `events.jsonl` says so). Nothing was lost or recomputed: every resume keeps what's
  done.
- **The source of the Ctrl+C isn't known yet.** Before the multi-day RL runs, long jobs will be
  started from inside WSL (`setsid nohup`, logs in the run folder), and the cause looked for.
- **Resumes are safe:** `sample.drop_partial_line` cuts a half-written last line before
  appending; the driver writes its own `driver.log`.
