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

## R6: run 1 lives in its own folders (2026-11-12)

Rishi: "begin run 1 inside a separate folder so that way nothing is intertwined". Run 1's code
is `src/sqlrl/run1/` and `scripts/run1/`, its tests `tests/run1/`, its docs
`documentation/run1/`, its data `<data>/run1/`. It imports nothing from `step2_pilots/`, and
the pilots import nothing from it. This replaces the shared `src/sqlrl/rl/` module the design
draft proposed.

- The piloted loss, reward, log-prob and SRPO code is **copied**, not shared, so a change to
  run 1 can't alter what the pilots' record says they ran. `tests/run1` proves the copies give
  the pilots' numbers on the same inputs.
- What stays shared is repo-level and not a step's: `paths.py`, `passk.py`, and step 1's
  sampler and scorer, because the yardstick must be the same measurement every time.
- Run 1 starts llama-server itself (`run1/server.py`), with its own port, logs and pid file, so
  it doesn't depend on the trainer repo's `serve.ps1`.

## R7: the rollout model is the LoRA merged into Q8_0, and the trainer under-reads a small LoRA (2026-11-12)

**What was found.** The pilots logged a gap between the server's and the trainer's
log-probabilities that grew with the LoRA (0.013 a token with no LoRA, 0.031 after 10 SRPO
steps). Three measurements on `p2-srpo`'s step-10 rollouts explain it:

| Compared with the server's log-probs | Mean gap a token |
|---|---|
| Trainer, LoRA on | 0.032 |
| Trainer, LoRA off | 0.037 |
| Trainer, LoRA at 2x its strength | **0.016** |
| (Trainer, LoRA on against LoRA off) | 0.011 |

A model with the LoRA merged into its weights and quantised (no runtime adapter at all) shows
the same 0.030 gap against the trainer, so the server is right and **the trainer's forward pass
under-reads the LoRA, by about half at this size**. The cause is bf16: each adapted layer's
output is a bf16 number, the LoRA's small addition is computed in fp32 and the sum is rounded
back to bf16, so additions smaller than half a bf16 step vanish. The gradients are unaffected
(the backward pass doesn't round), and the effect fades as the LoRA grows.

**What it means.**
- The pilots' rollouts and evaluations were right: they ran the real policy. What was off is
  the trainer's own view: the pilots' logged KL to the SFT model (0.0002 a token for GRPO)
  understates how far the policy moved. Measured from the server's side, SRPO's policy was
  0.004 a token from the SFT model after 10 steps.
- **No importance weighting.** The design draft proposed weighting tokens by the trainer's
  probability over the server's. That would pull the update toward the trainer's rounded view,
  away from the policy that is deployed. Dropped.
- **Drift is measured from the server.** `kl_sft` is the mean of (the server's log-prob of its
  own token) minus (the SFT model's, adapter off), on the rollouts that carry log-probs. That is
  KL(rollout policy || SFT) on the policy's own samples. `mismatch` stays logged as a reading of
  how far the trainer's view lags; its stop bound is 0.10.
- **The learning rate stays 1e-5.** The draft's case for 2e-5 was that GRPO "barely moved"; the
  real policy moved more than the logs showed.

**The rollout model.** Each step the update merges the LoRA into the 180 adapted matrices
(W + scaling x B x A, in fp32 on the GPU), quantises them to Q8_0 and writes them over the same
bytes of a copy of the released GGUF (`run1/gguf_merge.py`). Same shapes, type and offsets, so
it takes ~22 s instead of a conversion. This replaces `--lora`:
- it is what a release does (merge, then quantise), so rollouts and evaluations run the model
  that would ship (R5);
- a runtime adapter costs 23% of the decoding speed (415 against 542 tokens/s).
Checked: quantising the trainer's own base weights reproduces the released file (99.91% of the
int8 values equal, the rest differ by 1: rounding ties). A marker file says which step's LoRA
the file holds; a merge that was cut is redone from the saved LoRA.

## R8: a floor under the group's standard deviation (2026-11-12)

SRPO's GRPO branch divides each advantage by its group's standard deviation. A group with no
right answer and one malformed call has rewards 0 and -0.1 (std 0.03), which the division turns
into advantages as large as a right-versus-wrong group's: seven wrong answers pushed up at full
strength for being well-formed. Run 1 divides by max(std, 0.25). A group of eight with one right
answer has std 0.33, so every right-versus-wrong group is unchanged; the malformed-only group's
advantages become 0.05 and -0.35, in proportion to the 0.1 its rewards differ by.

## R9: the rollout harness, measured (2026-11-12)

Rishi asked for the llama.cpp harness to be made faster before run 1. Measured on the 7800 XT
with the fork build, 16 slots all decoding (`scripts/run1/bench_decode.py`):

| Setting | Tokens/s |
|---|---|
| The pilots': one shared KV pool (`-kvu`), `--lora`, log-probs on | ~170-200 |
| Shared KV pool, no LoRA | 208 |
| **A KV per slot** (no `-kvu`), no LoRA | **542** |
| A KV per slot, `--lora` | 415 |
| A KV per slot, log-probs on every rollout (`n_probs`) | 423 |

- **The shared KV pool was the brake.** With `-kvu` every decoding step attends over all slots'
  tokens, so a step's cost grows with the square of the slot count: 80 tokens/s at 1 slot, 247
  at 8, 208 at 16, 148 at 32. With a KV per slot: 392 at 8, 542 at 16, 535 at 32 (12, 20 and
  24 slots are slower than 16). Run 1 uses 16 slots of 8,192 tokens each.
- **Not the cause:** the sampler chain (the same speed with only the temperature sampler),
  CPU threads (4, 8, 16), the context size. Sampling on the GPU (`-bs`) isn't available: this
  ROCm build lacks an operation the sampler chain needs.
- **Scheduling** (`run1/rollouts.py`): one worker per slot, a question's eight rollouts one
  after another on one slot (its prompt is computed once, not eight times), long questions
  first, idle workers take over waiting rollouts. A rollout's two requests always find their
  own cache; in the pilots a quarter of the second requests had lost it to another rollout.
- **Log-probs on one rollout per group.** They cost a fifth of the speed and are only needed
  for the canaries (R7).
- **One full step of rollouts** (the pilots' step-1 questions and seeds, 256 rollouts, 117k
  tokens): **292 s**, against 504 s and 540 s in the two pilots.

The update side: scoring runs in its own process while the model loads. **WSL is kept awake
for the whole run:** it shut itself down during every rollout phase, so each update started
cold and its model load took 195-210 s in the first loop tests; with a keep-alive process and
the model's copy on WSL's own disk held in the file cache, 17-22 s. Padding several sequences
into one pass was tried and was far slower on this GPU, so the update keeps one sequence per
pass.

## R10: run 1's KL stop bound raised from 0.05 to 0.10 after step 10 (2026-11-13)

**What:** `stop.kl_max` in run 1's `config.json`, 0.05 to 0.10, with the run stopped cleanly
between steps 10 and 11 (a `STOP` file) and resumed. Nothing else changed; the old file is kept
next to it as `config.json.before-kl010`.

**Why:** at the full learning rate (from step 5) `kl_sft` grew steadily, 0.0006 at step 5 to
0.0040 at step 9 (0.0033 at step 10), close to the square of the steps taken, which is what a
constant learning rate moving the LoRA in one direction gives. Extended, it crossed 0.05 near
step 25, and the stop rules are checked before the evaluation, so the run would have paused
without its first held-out measurement. KL to the SFT model is expected to grow (RL is meant to
change the answers); it says how far the model moved, not whether the move helped. The
evaluations after steps 25, 50 and 100 judge that, and the pass@8 gate still pauses on a
significant loss. A sudden jump between two steps is still read as a fault. Rishi agreed.

**Not changed:** `mismatch_max` (0.10). The trainer under-reads the LoRA (R7) and `mismatch`
(0.014 at step 1, 0.034 at step 10) says how far its view has fallen behind the policy that
samples; it is a question of whether the updates are still right, not of distance. If it
nears 0.10 before step 25, Rishi decides.
## R11: run 1 continues past the KL bound after a step-19 evaluation (2026-11-13)

**What:** the KL rule (0.10, R10) paused run 1 after step 19 (`kl_sft` 0.115). With Rishi's OK
the run was evaluated there (step 19 added to `eval_steps`, a `STOP` file so it stopped before
step 20), then resumed with `stop.kl_max` 0.30 and two more evaluations, after steps 35 and 75
(`eval_steps` 19, 25, 35, 50, 75, 100). The earlier files are kept beside `config.json`
(`.before-eval19`, `.before-kl030`).

**Why:** the evaluation after step 19, on the 200-question evalset (195 scored):

| | greedy pass@1 | mean@8 | pass@8 | always / sometimes / never |
|---|---|---|---|---|
| SFT, the baseline | 43.2 | 36.9 | 62.6 | 30 / 94 / 74 |
| SFT, measured again through run 1's path | 44.5 | 36.4 | 59.9 | 29 / 89 / 79 |
| **after step 19** | **45.6** | **40.8** | **66.2** | 37 / 92 / 66 |

Sign tests against the baseline (better / worse, p): greedy 21 / 16 (0.51), sampled 60 / 41
(0.07), pass@8 14 / 9 (0.40); the SFT model measured again gave a sampled 37 / 44. By dataset
(greedy / mean@8 / pass@8): CypherBench, tested on graphs it never trains on, 44.9 / 37.9 / 80
to 53.1 / 42.2 / 78; gretel 64 / 59.5 / 80 to 62 / 64 / 86; text2cypher 15.4 / 15.2 / 32.7 to
25.0 / 22.0 / 46.8; SQaLe 42.2 / 34.3 / 57.1 to 36.2 / 33.7 / 52.1 (down, within the noise of
48 questions; watched as a possible overfit). pass@8 rose and "never" fell, so the gain is not
the policy narrowing. The drift that fired the KL rule bought accuracy, so the bound became a
backstop and the evaluations the main check.

## R12: run 1 stopped after step 28, step 19's adapter kept (2026-11-13)

**What:** run 1 is closed at 28 of its 100 planned steps. Rishi's decision, after the
evaluations at steps 25 and 28 (step 28 was added to `eval_steps` with his OK when `kl_sft`
reached 0.29, close to the 0.30 bound; the earlier config is kept as `.before-eval28`). The
adapter that is kept is step 19's (`step-0019/lora_state.safetensors`). Nothing is deleted:
every step's rollouts, scores and LoRA stay.

**Why:** every evaluation after step 19 was worse on the held-out questions
(`results/run1.md`):

| | greedy pass@1 | mean@8 | pass@8 | sampled sign test against the baseline |
|---|---|---|---|---|
| SFT, the baseline | 43.2 | 36.9 | 62.6 | |
| after step 19 | 45.6 | 40.8 | 66.2 | 60 better / 41 worse (p 0.07) |
| after step 25 | 46.8 | 39.7 | 62.6 | 61 / 52 (p 0.45) |
| after step 28 | 44.6 | 39.4 | 63.5 | 56 / 60 (p 0.78) |

Step 28 against step 19: 42 better / 63 worse (p 0.05). The training solve rate fell back too
(0.53 over steps 11-19, 0.45 after), SQaLe ended 13 points under the SFT model on pass@8, and
`kl_sft` went from 0.115 to 0.294 over those nine steps. Nothing in the settings changes after
step 28 (a constant learning rate, `kl_beta` 0), so 72 more steps would be 12 hours of the same
drift. R11's reading ("the drift bought accuracy") held up to step 19 and not beyond it.

**What it means for the next run:** the settings overshoot, which is a different finding from
"single-turn RL is done". A next run needs something that holds the model near the SFT model
(a KL penalty or a lower, decaying learning rate), evaluations from step 5-10 on with the best
step kept, and a fix for a blind spot the run exposed in the reward: the model's use of
`DISTINCT` fell from 22% to 9% of its held-out answers while the checker's score on the
questions whose reference uses `DISTINCT` went up, because the checker's data rarely contains
the duplicates that would show the difference (`results/run1.md`). Step 19's adapter carries
part of that habit.

**Not decided:** the test set on step 19's adapter, a release export, and what runs next.

## R13: run 1b, run 1's trainer with the model held near SFT (2026-11-14)

Rishi, after run 1: "let's fix the trainer and the blind spot, along with a KL penalty etc
first. Then let's run a short 1B". Built and loop-tested; Rishi starts it
(`run1b/RUNBOOK.md`). The design, with the reasons in full: `run1b/design.md`.

**What run 1b is:** run 1 again (SRPO, the same pool, mix and questions in the same order, a
new LoRA on the SFT model) for 40 steps, with:

| Change | Setting | Why |
|---|---|---|
| A KL penalty | KL(policy \|\| SFT) a token on every trained position (the policy's top 100 tokens plus the rest, `losses.topk_kl`), times a weight, added to the loss of both SRPO branches | Run 1 had nothing pulling back toward the SFT model (`kl_beta` 0): `kl_sft` 0.115 at step 19, 0.294 at step 28 |
| The weight follows the drift | starts at 0.05; doubles while the measured `kl_sft` is above 0.045, halves (not below 0.05) while under 0.02; at most 6.4. Target 0.03 | A fixed weight would be a guess. This is PPO's adaptive penalty, driven by the server-side measurement, which is exact (R7) |
| Learning rate | 5e-6 (was 1e-5) | Half the distance a step |
| Stop bound | `kl_max` 0.10 | Run 1's level at its last good evaluation |
| The duplicate probe | every passed CypherBench answer is run again, with the reference, on the graph with every relationship twice; when they disagree there the answer gets no reward (undecided, not wrong) | Below |
| Evaluations | after steps 10, 20, 30, 40 | Run 1's turn fell between two evaluations |
| The best step | `BEST.json`: the evaluated step with the highest mean@8 | The last step was not the best in run 1 |
| The turn rule | pause when an evaluation is worse than the best earlier one on the sampled sign test with p < 0.10 | Run 1 would have paused at step 28 |

**Not changed:** SRPO's parts, two optimizer steps a step, the batch (32 x 8), the mix, the
warm-up, the LoRA, the rollout model, the other stop rules, the gate, the yardstick. A larger
batch was the other way to a less noisy step; halving the learning rate does about the same
for no extra rollouts. A decaying learning rate was left out: it would make "the run levelled
off" unreadable in a 40-step test.

**The `DISTINCT` finding, corrected.** R12 said the checker's data rarely contains the
duplicates that would show a missing `DISTINCT`. Measured on run 1's rollouts, that is true of
CypherBench's sampled graphs only: for 82% of the training references that use `DISTINCT` (271
of 280 do) the graph returns the same rows without it. SQaLe's and text2cypher's random data
does punish it. And on the 18% of CypherBench questions where the graph shows the difference,
the model dropped `DISTINCT` anyway (81% of answers used it in steps 1-9, 31% in steps 20-28,
and without it they were right 7 times in 99): the habit eroded where nothing objected and
took the needed cases with it. `results/run1.md` is corrected.

**Why "no reward" and not "wrong":** the doubled graph is artificial. Tested on run 1's 1,045
passed CypherBench answers it withholds 523 (522 of them answers without `DISTINCT` to a
reference with it), keeps 485 (146 of those avoid repeats without the keyword), and stays
silent on 37 where the doubled graph changes what the reference itself returns. A cheaper
probe (every row twice on the unchanged graph) also withheld 104 answers that are fine, and
was rejected. The rule "rewards come from execution only" holds: this is one more database the
answer must agree on to be paid.

**Where the code is:** run 1's (`src/sqlrl/run1/`), with the new parts switched on by
`config.PRESETS["run1b"]` (`run.py --preset run1b`). Without the preset every new setting is
off and the code does what run 1 did; `tests/run1` checks that (27 tests). R6 kept run 1 apart
from the pilots so a change couldn't alter what the pilots' record says they ran; run 1 is
closed and its `config.json` and results are kept, so its record doesn't depend on the code
staying frozen. A second copy of the whole harness for a 40-step test would be the worse trade.

**What it will show:** `run1b/design.md` section 4. In short: if `kl_sft` levels off near the
target and the evaluations hold at or above run 1's step 19, the trainer is fixed and these
settings go into run 2.

## R14: run 1b closed, step 30 kept; greedy picks the step (2026-11-17)

Run 1b ran all 40 steps (`results/run1b.md`). Rishi kept **step 30's adapter**: greedy 54.5 /
mean@8 43.4 / pass@8 67.5, against 44.5 / 36.4 / 59.9 for the SFT model through the same path
and 48.6 / 39.6 / 66.0 at step 40.

**Greedy pass@1 chooses the step** (Rishi): it is how the deployed model answers, and the person
asking can't tell a wrong answer from a right one. mean@8 and pass@8 measure headroom (the gap
between pass@8 and greedy is what is left to learn) and decide whether a run should go on.
`BEST.json` still picks by mean@8 (step 26 here); read greedy yourself.

**Settings changed during the run, each with Rishi's OK:**

| When | Change | Why |
|---|---|---|
| Length pause at step 23 | evaluation after step 26 added | see whether the shorter answers were better or worse |
| After step 26's evaluation | `length_ratio` 2.0 to 3.0 | step 26 was better than step 20 on every measure (p about 0.01) with answers half as long: the rule read improvement as collapse |
| Entropy pause at step 31 | `entropy_min_frac` 0.5 to 0.25, `kl_max` 0.10 to 0.15 | run on to step 40 and choose between steps 30 and 40 there |

Also with his OK: resumed once after a GPU freeze at step 35 (not in the RUNBOOK's table), and
the run's data root moved to the internal drive after two crashes during saves to the USB
drive (`results/run1b.md`, "What went wrong on the machine").

**For run 2:** the gains lived at `kl_sft` 0.03-0.08 against the SFT model, above the 0.03
target, and the weight (doubling above 0.045, halving only below 0.02, cap 6.4) spent steps
34-40 pulling the model back. Run 2 starts from step 30's adapter and measures drift against
that start, or uses a looser target.

## R15: run 2, the multi-turn agent: built, not started (2026-11-18)

Rishi read the run 2 design page (14 open choices, D1 to D14, each with a recommendation) and
said: "go with the recommendations for all of them", build the environment and the harness,
harden it against entropy collapse, reward hacking and cheating, and **do not start the run**.
The design as built, with the reasons in full: `run2/design.md`.

**What run 2 is:** the same task, datasets, checker and evaluation set. The model sees what its
query returned and may fix it: up to 3 queries, and the last one that ran is its answer. It
continues run 1b's step-30 adapter.

| | Choice taken | In short |
|---|---|---|
| D1 | A look-around copy built without the reference; per dataset after the no-training test | The checker's fills plant the reference's literals. The model looks at a separate copy made from the schema and the question (SQaLe, text2cypher) or at the real rows (gretel, CypherBench). The reward is still decided on the checker's own databases |
| D2 | The constants rule and a canary | A final query that reads nothing, or returns only literals, earns 0. Literals read off a result are counted |
| D3 | A plain reply ends the episode; 3 queries | The answer is the last query that ran |
| D4 | Vary the result's form in training; keep thoughts in history | Row cap 5 / 10 / 20, a table or JSON lines; evaluation fixed at 10 rows as a table |
| D5 | 0.03 off a right answer per extra call | A wrong answer pays 0 whatever it took |
| D6 | SRPO; only the first turn is distilled | Later turns of a distilled episode take the reward update |
| D7 | Continue the adapter; drift against the frozen start adapter; a fixed 0.05 penalty on first-turn tokens; pause at 0.05; learning rate 5e-6 | Nothing pulls the model back by itself |
| D8 | Cypher questions take turns on the one Neo4j | Loads measured: 0.1 to 5 s a graph |
| D9 | Skip run 1b's 40 pool steps; a 200-question dev slice; run 1b's mix | The pool file is untouched, so the offset skips the right questions |
| D10 | The no-training tests first | `results/run2-phase0.md`. Reading them against the design page's table, and what follows, is Rishi's |
| D11 | Greedy with turns every 5 steps, full at 20 and 40; keep by greedy with turns | Success: at least +5 over 54.5 |
| D12 | Start from step 30; step 26 measured without training too | |
| D13 | A new `src/sqlrl/run2/` | Run 1's loss, reward and merge code copied, not shared (R6); `tests/run2` proves the copies equal |
| D14 | Read-only inputs stay on the data drive | Everything a run writes is on the internal drive |

**A new headline beside the one yardstick.** Every earlier number is one greedy query. Run 2's
headline is **greedy with turns**: the final query of a greedy episode, counted the same way
(right / decided). The greedy episode's first query is reported beside it, so the old number
is still measured at every evaluation. A strict count (undecided and no-query answers as wrong)
is reported too, because the old count would let a model hide by not answering.

**The yardstick was anchored first.** The start model was rebuilt from the kept adapter and
measured against the kept step-30 evaluation (102 of 187, 54.5%): through the old path 99 of
187 (agreeing on 183 of 186 questions), and the new engine's first query 101 of 186 (agreeing
on 184 of 186).

**Hardening beyond the design page** (found while building; each is tested):

- A result is tokenised as plain text, so it can't carry the template's turn markers.
- The reference is dropped in one function before a question reaches the rollout side; a test
  shows the random fill never holds a value only the reference names.
- Read-only is enforced by the engines (SQLite's authorizer and `query_only`, Neo4j's read
  transaction), with time limits and size caps, tested against the real engines.
- Entropy is measured before the update on a fixed sample of episodes, per turn, so the number
  doesn't follow the routing; a group's count of different final queries is a second collapse
  signal.
- The first query's solve rate is a stop rule: outcome-level credit can teach a careless first
  query whenever a retry fixes it.
- Loops, calls past the limit and broken answers already occur in the untrained start model, so
  their rules are measured against the run's own first five steps.
- A dataset can be put back on run 1's protocol (`single_turn_sets`). Off when built; on for
  SQaLe since R16.

**A bug the no-training test caught:** SQLite keeps prepared statements, so a query run a
second time asked the authorizer nothing and looked like it read no table. The constants rule
would have paid 0 for a right answer written twice. Fixed (the tables are remembered per query
text) and tested.

**Not built:** an entropy bonus or any automatic pull-back (method changes, Rishi's to make); a
held-out second copy for the constants rule (the design's option C, only if the canary rises);
a live dashboard for run 2.

## R16: run 2 keeps SQaLe on one query (2026-11-19)

**Rishi's call**, after the no-training test on the dev slice: "If SQaLe is better left as
single turn that is fine - as we don't want it to regress."

**The evidence** (`results/run2-phase0.md`, step 30's adapter, 200 dev questions, greedy):

| | Turns fixed | Turns broke |
|---|---|---|
| CypherBench, gretel, text2cypher | 8 | 1 |
| SQaLe | 1 | 6 |

SQaLe's look-around copy is a random fill built from the schema and the question (D1: it may
not see the reference). A correct query returns rows on it for 36 of 50 dev questions; for the
other 14 it shows `0 rows`, because the query filters on a key or a constant the question
doesn't spell out. Three of the six broken answers followed such a `0 rows`; the other three
followed rows the model second-guessed.

**What changes:** `single_turn_sets` is `["sqale"]` by default. A SQaLe question gets the
original SFT prompt and one query, and nothing is shown: run 1b's episode. With SQaLe on one
query the same dev slice reads 115 of 199 on the first query and 123 of 199 with turns (57.8%
to 61.8%; fixed 9, broken 1).

**What follows from it:**

- The headline, greedy with turns, is three datasets with turns plus SQaLe with one query. The
  start adapter's reference evaluation was made under this protocol.
- The cause is the environment (a fill made without the reference can't hold what a hidden
  constant needs), not the model. Run 2 therefore neither teaches nor measures turns on SQaLe's
  kind of schema. A real copy of data for SQaLe would lift this; none exists.
- The rule goes by dataset; real use has no dataset label. A deployment picks one prompt.
- A quarter of a step's episodes have one turn. The rates the stop rules read for turn
  behaviour (calls, a call too many, loops, broken answers, copied literals, the look-around
  leak) are counted over episodes with turns only, so their bounds mean what they meant.
- **A protocol record.** An evaluation writes the protocol it was made under next to its
  episodes: the prompt, the call limit, the one-query datasets, the result format and the
  version of the look-around data (`env.LOOK_VERSION`). A file is never continued under another
  protocol, and the run refuses to start when the start adapter's reference was made under
  another one. (A partial reference made before this decision, with SQaLe on turns, was sitting
  in the reference's folder; the episode writer resumes whatever a file holds, so it would have
  been mixed in silently.)

**What the evaluation set then showed (same day; `results/run2-phase0.md` sections 7-9).** The
dev-slice numbers above do not carry over. On the evaluation set the untrained step 30 goes
from 52.4% on its first query to 48.1% with turns (fixed 4, broke 10; gretel fixed 2 and broke
8, text2cypher 0 and 2, CypherBench 2 and 0). The cause is a right query whose result says
nothing (`0 rows`, or only NULL or zero): the model breaks 9 of 25 such answers and 1 of 53
others. The pool's filter keeps those questions out of training and out of the dev slice; the
evaluation set has them. gretel: the pool rejected 627 of 2,646 candidates whose reference's
result said nothing, and 13 of 50 evaluation references are like that. text2cypher: 31 of 50
evaluation prompts describe a graph with no relationships (path questions from one source), so
the right answer is "nothing" for 24 of them; the pool rejected 422 of 1,426 text2cypher
candidates whose reference returned no rows on the checker's graphs. By the design page's
table (D10) this reads "do not start RL". So, by the rule Rishi used for SQaLe, two more
datasets fail on the evaluation set. **Nothing further was changed: what to do is open, and
Rishi's** (the options are in the phase-0 results, section 9).

## R17: run 2 trains on questions whose right result is nothing; text2cypher keeps one query (2026-11-19)

**Rishi's call**, from three options (stop at step 30; turns for CypherBench only; fix the
training questions): "Go with option C".

**Why:** trained on the pool as it was, the model can only learn "nothing came back, so change
the query until something does". The pool holds no question where nothing is the right result;
the evaluation set does (13 of 50 gretel questions), and so does a real database, where an
empty result is often the true answer.

**What changes:**

- **text2cypher keeps one query**, like SQaLe (`single_turn_sets: ["sqale", "text2cypher"]`).
  31 of the evaluation set's 50 text2cypher prompts describe a graph with no relationships; a
  look can only mislead there. Turns are now for gretel and CypherBench, the two datasets
  where the model looks at real data: 20 of a step's 32 questions.
- **The returned questions** (`pool-v2.returned.jsonl`, `pool.build_returned`): the 627 gretel
  questions run 1's pool rejected because the reference's result says nothing, checked again;
  604 kept. **3 of a step's 10 gretel questions** come from it (120 in 40 steps), a fixed
  count so the stop rules' rates don't swing. Its last 50 are a second dev slice. The pool file
  and its order are untouched, and the offset still counts run 1b's 10 gretel questions a step.
- **A stricter reward for them** (`returned_guard`): the checker compares a gretel answer with
  the reference on the dataset's own rows, and there every query that returns nothing equals a
  reference that returns nothing. A returned question's answer is right only when the checker's
  full SQL check also passes on filled copies of the schema. Measured on the second dev slice:
  of 42 first answers the checker passes, the filled copies confirm 25 and reject 17. Without
  the stricter reward 4 in 10 "right" answers there would be wrong queries that merely
  returned nothing. Training and dev slices only: **the evaluation set's yardstick is
  unchanged**, which also means its score on such questions was always this generous.
- **Kept apart in the metrics:** the returned questions' solve, fixed and broken numbers beside
  the other gretel questions' (`returned_*`, `gretel_other_*`).
- **"The same query twice" means twice in a row.** Running the answer again after a look at
  the data is what "the last query that ran is the answer" asks for.
- **The evaluation's split:** greedy with turns is also reported for the questions with turns
  whose reference says something on the copy the model looks at, for those where it says
  nothing, and for the one-query questions.

**Found while building, and Rishi's to weigh:**

- **Three quarters of the returned questions have a database with no rows at all** (452 of
  604; 10 of the evaluation set's 13). The dataset gave tables and no data. So what the model
  mostly meets is "the tables are empty", not "the filter matches nothing". A rule that needs
  no reference could treat these differently (nothing to look at: one query). Not built.
- **A wording that says an empty result can be right did not help.** `keep`, tested on the two
  dev slices by a rule set beforehand, broke more right answers on the returned slice (19
  against 14). The run's prompt stays `turns`.
- **The untrained start model breaks 14 of 25 right first answers on the returned slice.**
  That is the number training has to lower; nothing before training lowers it.

The numbers, and how they are read against D10's table: `results/run2-phase0.md`, section 10.
**Run 2 is still not started.**
