# Run 1b design: run 1, held near the SFT model

**Status:** built and loop-tested 2026-11-14. **Not started: Rishi starts it** (from an Opus
session following [`RUNBOOK.md`](RUNBOOK.md), with a Sonnet agent building the dashboard, as for
run 1). Decision record: R13 in `../decisions.md`.

Run 1b is a short test of the trainer, not a new stage. Run 1 improved the model for about 19
steps and then drifted (`../results/run1.md`). Run 1b repeats it with the fixes that reading
called for, on **the same questions in the same order**, for 40 steps. It answers one thing
before run 2 is built on this trainer: **held near the SFT model, does the gain stay, and does
it grow?**

## What run 1 showed, in four numbers

| | After step 19 | After step 28 |
|---|---|---|
| Sampled answers right (mean@8; the SFT model: 36.9) | 40.8 | 39.4 |
| Questions better / worse than the SFT model (sampled) | 60 / 41 | 56 / 60 |
| Distance from the SFT model (`kl_sft`, a token) | 0.115 | 0.294 |
| Training solve rate (mean of the steps before) | 0.52 | 0.45 |

Nothing held the model near the SFT model (`kl_beta` 0, a constant learning rate), each step
learned from about 16-20 questions, and the first look at held-out questions was planned for
step 25.

## What changes

| | Run 1 | Run 1b | Why |
|---|---|---|---|
| Learning rate | 1e-5 | **5e-6** | Half the distance a step. Each step's direction is noisy; a smaller step follows the average of more of them |
| KL penalty | none | **KL(policy \|\| SFT) a token x a weight, added to the loss** | Section 1 |
| The weight | | starts at 0.05, **follows the measured drift** toward a target of 0.03 | Section 1 |
| Stop bound on `kl_sft` | 0.05, then 0.10, then 0.30 | **0.10** | Run 1's level at step 19, the last good evaluation |
| CypherBench reward | the checker's verdict | **no reward when the answer repeats rows where the reference doesn't** | Section 2 |
| Evaluations | planned after 25, 50, 100 | **after 10, 20, 30, 40** | The turn in run 1 came between two evaluations |
| Which step is kept | the last | **the best evaluated step** (`BEST.json`) | Section 3 |
| New pause rule | | **an evaluation worse than the best earlier one** (sampled sign test, p < 0.10) | Run 1 would have paused at step 28 instead of running on |
| Length | 100 planned | **40** | Enough to see whether the hold works (below) |

**What doesn't change:** SRPO as piloted (std advantages with the floor, clip-higher, the
self-distillation branch, the EMA teacher), two optimizer steps per RL step, 32 questions x 8
answers, the mix (10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher), `pool-v2`, the 5-step
warm-up, the LoRA (rank 16), the rollout model (the LoRA merged into Q8_0), every other stop
rule, the gate on pass@8, and the yardstick (the 200-question evalset, checker 4). The code is
run 1's (`src/sqlrl/run1/`) with the new parts switched on by a preset
(`config.PRESETS["run1b"]`); without the preset it does what run 1 did, and the tests check
that.

## 1. The KL penalty

**What it is.** For every position of every trained answer the trainer has two next-token
distributions: the policy's (LoRA on) and the SFT model's (LoRA off). The penalty is the KL
divergence from the SFT model's to the policy's, averaged over the trained tokens and
multiplied by a weight. Its gradient pulls each position back toward what the SFT model would
have written there. The reward pulls the other way where a different token earns more. A
change the reward keeps asking for survives; a change it asked for once, by the luck of one
batch, is pulled back. That is the difference from run 1, where every push stayed.

- It compares whole distributions (the policy's 100 likeliest tokens plus one bucket for the
  rest: `losses.topk_kl`), not the one token that was sampled, so it carries no sampling noise.
- It is applied to both of SRPO's branches (the rewarded answers and the distilled ones).
- Cost: one extra forward pass per trained answer with the LoRA off, and one more projection
  onto the vocabulary: about 2 minutes a step ("Measured" below).

**The weight follows the drift.** A fixed weight would be a guess: too small and run 1b is run
1 at half speed, too large and nothing is learned. So the weight is set each step from the
drift measured on that step's rollouts (`kl_sft`, from the server's own log-probabilities,
before the update): PPO's adaptive penalty (Schulman et al., 2017).

| Measured `kl_sft` | The weight |
|---|---|
| above 0.045 (1.5 x the target) | doubles (at most 6.4) |
| 0.02 to 0.045 | stays |
| below 0.02 (the target / 1.5) | halves, never below 0.05 |

**Why a target of 0.03.** Turning a question from "right half the time" into "right every
time" costs about 0.7 nats once, at the token where the answer is decided: about 0.003 a
token over a 200-token answer. Run 1 reached 0.03 around step 14-15, inside the stretch where
its training solve rate was highest (steps 11-19), and was at 0.115 by step 19. Most of that
distance was not decisions changing: the answers got a quarter shorter and `DISTINCT` was
dropped. A budget of 0.03 leaves room for every decision to change several times over and
none for a change of style.

**One known weakness.** The trainer's forward pass under-reads a small LoRA (R7): while the
true `kl_sft` is below about 0.005 the trainer sees almost none of it, at 0.02 it sees about
60%, at 0.05 about two thirds (run 1's log, `kl_sft_trainer` against `kl_sft`). So the penalty is weak exactly while it isn't needed and grows into
its job as the model moves. The weight reacts to the server's measurement, which is exact.
`kl_pen` in the log is what the penalty itself saw.

## 2. The duplicate probe (CypherBench)

**What was found.** During run 1 the share of the model's held-out answers that use `DISTINCT`
fell from 22% to 9%. The cause was looked for on run 1's own training rollouts:

- **CypherBench references use `DISTINCT` almost always** (271 of run 1's 280 CypherBench
  training questions). On the checker's sampled graph the reference returns the same rows
  without it for **82%** of them: the graph is too sparse for two paths to lead to the same
  node. There an answer that drops `DISTINCT` is passed.
- On the other **18%** the graph does show the difference, and the model was punished for
  dropping it. It dropped it anyway: on those questions its answers used `DISTINCT` 81% of the
  time in steps 1-9 and 31% in steps 20-28, and without it they were right 7 times in 99.
- The other datasets don't have this gap. SQaLe's and text2cypher's random data is full of
  repeats (an answer without the reference's `DISTINCT` was right 3-10% of the time on SQaLe).
  gretel's own tables are small, but only 1 gretel question in 20 has `DISTINCT` in its
  reference.

So the habit was lost on the 82% where nothing objected, and that cost reward on the 18%.

**What the probe does** (`run1/dupprobe.py`). Every CypherBench answer the checker passed is
run once more, with the reference, on **the same graph with every relationship stored twice**.
Two paths now lead wherever one did. A reference that removes repeats returns what it did
before; an answer that doesn't returns rows twice. When the two disagree there, the answer
gets **no reward**: it is left out of its group, like an answer the checker can't decide. It
is not marked wrong, because the doubled graph is artificial: it can withhold a reward the
sampled graph can't back, never take a right answer for a wrong one.

- The probe stays silent (`unknown`, the checker's verdict stands) when the doubled graph
  changes what the reference itself returns: references that count paths, and patterns like
  "other attacks on the same target", where Cypher's rule that one relationship is never used
  twice keeps an attack from matching itself and the twin relationship lifts that.
- **Tested on run 1's 1,045 passed CypherBench answers:** 523 withheld, 485 kept, 37 unknown.
  Of the 523, 522 are answers without `DISTINCT` to a reference with it, and the last one
  counts paths where the reference counts distinct nodes. No answer that uses `DISTINCT` like
  its reference was withheld. Answers that avoid repeats another way (`WHERE EXISTS { ... }`,
  an aggregation) are kept: 146 of them.
- The simpler idea, every row twice on the unchanged graph (`UNWIND [0, 1]` in front of both
  queries), was tried on the same answers and rejected: it also withheld 104 answers the
  doubled graph shows to be fine (an existence test in place of `DISTINCT`; one node looked up
  by name, where the reference's `WITH DISTINCT n` changes nothing).
- **The share withheld is a new canary** (`dup_differ_frac`): 0.48 in run 1's steps 1-9, 0.60
  in steps 20-28. A stop rule pauses run 1b when it is more than 0.15 above the first five
  steps' mean for three steps in a row.

**What it costs.** About half of CypherBench's right answers lose their reward at the start, so
fewer CypherBench groups carry a signal. What is left is the signal that can be trusted. The
solve rate in the log stays the checker's own verdict, so it compares with run 1's.

**What it doesn't touch:** the yardstick. Evaluations score with checker 4 on the sampled
graphs, as every measurement before.

## 3. Evaluations, the best step, the turn

- **After steps 10, 20, 30 and 40** (about 28 minutes each).
- **`BEST.json`** names the evaluated step with the highest mean@8 (the measure with the most
  samples behind it: 1,450 answers). Every step's LoRA is kept, so "keep the best" costs
  nothing. Picking the best of four on the evalset flatters it a little; the test set, run once
  on the kept adapter, is the honest number.
- **`vs_best`:** each evaluation is also compared, question by question, with the best earlier
  one. If its sampled answers are worse with p < 0.10 the run pauses for a look. In run 1 that
  test read 42 better / 63 worse (p 0.05) at step 28 against step 19.
- The gate on pass@8 against the SFT baseline is unchanged.

## 4. What to expect

Run 1b moves about half as far a step as run 1, so without the penalty it would retrace run 1
at half speed: step 40 would sit about where run 1's step 21 did.

| Run 1b step | About where run 1 was at step | Run 1's `kl_sft` there | Evaluation |
|---|---|---|---|
| 10 | 6 | 0.001 | first |
| 20 | 11 | 0.007 | second |
| 28-30 | 15-16 | 0.046-0.061: the weight starts to double | third |
| 40 | 21 (0.139) | **held near 0.03-0.05 if the penalty works** | fourth |

(Run 1 took 3 full-rate steps' worth of distance in its warm-up and one a step after it; run
1b takes half of each. This is arithmetic on the learning rate, not a forecast: the optimiser
doesn't move in a straight line.)

How to read the end:

| What happens | What it means | Next |
|---|---|---|
| `kl_sft` levels off under about 0.06, and the evaluations at 30 and 40 are at or above run 1's step 19 (mean@8 40.8) and not falling | The trainer holds. The gain doesn't need the drift | These settings go into run 2; a longer single-turn run is an option |
| `kl_sft` levels off, evaluations stay at the SFT model's level | The budget is too tight, or 40 half-size steps are too few | Extend to 80 steps before changing anything (the settings allow it) |
| `kl_sft` keeps rising through the band with the weight at its ceiling | The penalty can't hold against this reward | The trainer isn't fixed: lower the learning rate again, look at what the reward keeps pushing |
| The evaluations rise to 20 or 30 and then fall while `kl_sft` is held | The harm isn't distance from the SFT model | Read the rollouts: something the reward teaches is wrong |

**Time:** 40 steps of 12-13 minutes and four evaluations of about 28: **about 10.5 hours**.

## 5. Limits

- **Same questions as run 1.** Deliberate: the two runs differ in the trainer only, and their
  solve rates pair step by step. It also means run 1b tells nothing new about the pool.
- **Four things change at once** (the learning rate, the penalty, the probe, the evaluation
  schedule). If run 1b holds, it won't say which of the first three did it. The log separates
  them partly: the penalty's weight stays at its floor until `kl_sft` reaches the band, so
  before that any difference from run 1 is the learning rate and the probe.
- **SQaLe is unchanged.** It was the one dataset that ended run 1 below the SFT model. Its mix
  share and its random-fill databases are as before; run 1b shows whether it still falls when
  the model is held.
- **gretel's small tables** have the same gap as CypherBench's graphs in principle. It is not
  probed: 5% of its references use `DISTINCT`.
- **One evaluation is good to about 3 points.** The sign tests are the reading, not the points.

## Measured (the loop test, 2026-11-14)

`smoke-1b`: 2 steps of 8 questions (4 CypherBench, 2 gretel, 2 SQaLe) with the preset, no
evaluation. Both steps trained and the run finished.

| | Step 1 | Step 2 |
|---|---|---|
| Sequences trained (of 64 rollouts) | 52 | 34 |
| `kl_sft` (server) / `kl_pen` (what the penalty saw) | 0.0007 / 0.00007 | 0.0011 / 0.00017 |
| `kl_beta` | 0.05 | 0.05 |
| The probe on the passed CypherBench answers | 12 same, 4 withheld | 1 same, 10 withheld, 1 unknown |
| Update, seconds a trained sequence | 2.8 | 3.2 |
| Peak GPU memory | 10.5 GB | 11.2 GB |

- The penalty is computed and tiny, as it should be on a LoRA two steps old (and under-read:
  the server sees 0.0007-0.0011).
- **The update costs about 40-50% more a sequence than run 1's** (2.1 s there): the extra
  forward pass and projection. A full step's update goes from about 5 to about 7 minutes, a
  step from 10-11 to **12-13 minutes**. Peak memory is where run 1's was (10.5-12 GB).
- The format check on the rollout model passed (`template_equal_share` 0.999).
