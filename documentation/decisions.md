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
- **Update (step 2):** the likely cause is a console window. A process started without a
  console gets a new, visible one for each console child (`wsl.exe`, `powershell`), and closing
  such a window ends it with exactly this exit code. Step 2 starts every process hidden
  (`CREATE_NO_WINDOW`, also for the children); its runs have had no interruption since.

## R4: two pilots, 10 steps, and how they're built (2026-11-01)

- **Rishi's call:** 10 steps per pilot (not 40), and only the two methods that matter most in
  theory: **GRPO** (the reference; learns from "sometimes" questions) and **SRPO** (learns from
  failures too, through a right sibling's solution; reported to beat GRPO and SDPO without
  SDPO's collapse). DAPO and GSPO stay defined but aren't run. Pilots can be paused at any
  moment to free the GPU (`pause.py`), and resumed with the command that started them.
- **The loop on 16 GB:** llama-server (Windows) and the trainer (WSL) take turns on the GPU.
  WSL can't reach a server on Windows (NAT networking, the server listens on localhost only),
  so a Windows orchestrator runs the rollouts and calls one WSL process per update (it loads
  the model, ~80 s, trains, saves, converts the LoRA to GGUF, exits, freeing the GPU).
- **Exact tokens:** rollouts use the raw `/completion` endpoint with the server's own template
  rendering (`/apply-template`) and tokenizer, and return token ids; the trainer computes
  log-probabilities on those same ids. The thinking budget is the yardstick's: 1,024 tokens,
  then `</think>` (one token) is appended and masked out of the loss.
- **Same questions for both pilots:** `pool-v1`, 1,280 `gold-full` training questions (320 per
  set, interleaved, evalset and test questions excluded by id and text), 32 a step. The model
  saw these questions in SFT (as thinking-off gold calls); `rl-runs.md` raises whether a fresh
  pool would be better for the real run.
- **GRPO here** is Dr. GRPO's form (r - mean, constant normaliser) with a k3 KL to the SFT
  model at beta 0.01 (the adapter switched off is the reference: no second model).
- **SRPO here** follows the paper's configuration where it fits: routing per rollout (a wrong
  one with a right sibling distils, the rest take GRPO's branch), an EMA teacher of the LoRA at
  0.05, the teacher shown the sibling's query as the correct solution (no environment
  feedback), JSD over the student's top-100 tokens plus a tail bucket, entropy weights
  exp(-H) normalised to mean 1, one token-level average, no KL. **Departures:** the same lr as
  GRPO (1e-5, LoRA; the paper tunes per method), no warmup or weight decay (10 steps), no
  rollout importance-sampling clip (the trainer's own old log-probs are the ratio's base).
- **Groups with no signal** are left out of the update for every method (DAPO's dynamic
  sampling); for GRPO this also leaves their KL term out.
- **Rollout-vs-trainer mismatch** is logged every step (the server's log-probs of the sampled
  tokens against the trainer's): 0.013 a token on the smoke run.

## R5: how close to production each run's environment is (2026-11-05)

Agreed with Rishi after the pilots. The question was "mimic production (the real harness and
databases) or an artificial sandbox?". The rule: **what the model reads, and what decides its
reward, must match deployment; the plumbing behind it may be artificial.**

| Must match deployment | May be artificial |
|---|---|
| The prompt: chat template, system prompt, how the schema is written | Where the databases run (containers, local files) |
| The tool's call format and how results come back (row cap, truncation, error text) | Isolation: read-only, no network, reset after each episode |
| Real database engines, so errors are the real ones | Timeouts and concurrency sized for training speed |
| Realistic data wherever the model gets to look at it | Live system or frozen snapshot |

- **Run 1 (single turn): a sandbox.** The model never sees what its query returns, so the
  environment is a reward function. What matters is the prompt matching deployment and a checker
  that is hard to fool. Production databases wouldn't help: production has no reference answers.
- **Run 2 (multi-turn): a sandbox that matches deployment in everything the model reads.** One
  tool protocol, trained on and documented in the model card (for a public model, we define
  "production"). Real engines and real data where the model looks; random-fill sets (SQaLe,
  text2cypher) either get real data or stay single-turn. Frozen snapshots, read-only, short
  timeouts. What can't be pinned down across other people's harnesses (row caps, schema layout,
  result rendering) is randomised.
- **Stage C (the NFL harness): matches production fully.** The same guard, schema and result
  format as Ask the Engine, on a frozen copy of the NFL databases (not the weekly-appended live
  ones), with training questions kept apart from Ask's golden and held-out sets.
- **The runs are designed one at a time:** run 1 first (`run1-design.md`), then run 2.
