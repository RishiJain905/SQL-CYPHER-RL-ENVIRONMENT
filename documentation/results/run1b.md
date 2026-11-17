# Run 1b results: single-turn SRPO held near the SFT model, 40 steps, step 30 kept

Run `run1b`, 2026-11-14 to 2026-11-17. Run 1 again with the trainer fixed (decisions R13): the
same questions, mix and SRPO method, with a KL penalty whose weight follows the measured drift,
a learning rate of 5e-6 and evaluations after steps 10 / 20 / 26 / 30 / 40. **All 40 steps ran.
The adapter that is kept is step 30's** (R14). Run 1b is the official stage-A run; run 1
(`results/run1.md`) is the first attempt it fixed.

- Design and how it was run: `run1b/design.md`, `run1b/RUNBOOK.md`; live page:
  `run1b/dashboard/README.md`
- Decisions along the way: R14 (below)
- The kept adapter: `step-0030/` of the run (`lora_state.safetensors`, `adapter/` in PEFT
  form, `adapter.gguf`), also kept with step 40's as the runner-up in the folder of kept models.
  Not merged for release or run on the test set.

## The evaluations

The 200-question evalset, greedy plus 8 sampled answers at temperature 1.0, checker 4: the same
measurement as step 1 and run 1. **Greedy pass@1 is the measure the kept step is chosen by**
(Rishi, R14): it is how the model answers in use, where nobody can tell a wrong answer from a
right one. mean@8 and pass@8 measure how much is left to learn: the gap between pass@8 and
greedy is headroom, and they decide whether a run should go on.

| | Greedy pass@1 | mean@8 | pass@8 | Headroom (pass@8 - greedy) | Always / sometimes / never | `kl_sft` at that step |
|---|---|---|---|---|---|---|
| SFT, the baseline | 43.2 | 36.9 | 62.6 | 19.4 | 30 / 94 / 74 | 0 |
| SFT again | 44.5 | 36.4 | 59.9 | 15.4 | 29 / 89 / 79 | 0 |
| Run 1, step 19 | 45.6 | 40.8 | 66.2 | 20.6 | 37 / 92 / 66 | 0.115 |
| After step 10 | 46.7 | 36.9 | 58.4 | 11.7 | 29 / 86 / 82 | 0.001 |
| After step 20 | 43.0 | 40.8 | 65.5 | 22.5 | 34 / 95 / 68 | 0.006 |
| After step 26 | 49.5 | **43.5** | **73.1** | 23.6 | 36 / 108 / 53 | 0.025 |
| **After step 30** | **54.5** | 43.4 | 67.5 | 13.0 | **38** / 95 / 64 | 0.078 |
| After step 40 | 48.6 | 39.6 | 66.0 | 17.4 | 29 / 101 / 67 | 0.010 |

Sign tests against the baseline, question by question (better / worse, p):

| | Greedy | Sampled (right of 8) | pass@8 |
|---|---|---|---|
| After step 10 | 8 / 4 (0.39) | 38 / 49 (0.28) | 7 / 16 (0.09) |
| After step 20 | 9 / 11 (0.82) | **59 / 37 (0.03)** | 18 / 13 (0.47) |
| After step 26 | 24 / 13 (0.10) | **73 / 40 (0.002)** | **26 / 6 (0.001)** |
| After step 30 | **32 / 14 (0.01)** | 63 / 47 (0.15) | 23 / 14 (0.19) |
| After step 40 | **15 / 5 (0.04)** | 51 / 40 (0.29) | 13 / 7 (0.26) |

Between the run's own steps: step 26 against 20 was better on every measure (p about 0.01);
step 30 against 26: greedy 14 / 6 (p 0.12), sampled 49 / 52, pass@8 4 / 16 (p 0.01, worse);
step 40 against 30: greedy 21 / 35 (p 0.08), sampled 51 / 57 (0.63), pass@8 15 / 18 (0.73).

By dataset (greedy / mean@8 / pass@8):

| | CypherBench | gretel | SQaLe | text2cypher |
|---|---|---|---|---|
| SFT again | 48.0 / 36.4 / 74 | 64 / 60.5 / 76 | 42.2 / 32.7 / 55.3 | 16.2 / 15.6 / 34.0 |
| Run 1, step 19 | 53.1 / 42.2 / 78 | 62 / 64.0 / 86 | 36.2 / 33.7 / 52.1 | 25.0 / 22.0 / 46.8 |
| After step 10 | 46.9 / 33.4 / 72 | 66 / 61.3 / 76 | 42.2 / 36.2 / 54.2 | 25.0 / 16.2 / 30.6 |
| After step 20 | 34.7 / 39.9 / 82 | 66 / 65.6 / 84 | 34.8 / 33.3 / 50.0 | 32.4 / 23.8 / 44.9 |
| After step 26 | 40.8 / 40.3 / 82 | 68 / 63.2 / 84 | 40.4 / 33.2 / 57.1 | 47.5 / 36.7 / 68.8 |
| **After step 30** | 44.0 / 36.7 / 76 | **70** / 63.5 / 84 | 42.6 / 32.3 / 47.9 | **62.5** / 40.6 / 61.2 |
| After step 40 | **54.0** / 40.1 / 80 | 64 / 63.3 / 86 | 42.6 / 34.3 / 55.1 | 26.5 / 20.0 / 41.7 |

About 50 questions a dataset: one question is 2 points.

- **text2cypher carries the run.** Its greedy score went from 16 to 62.5 by step 30, almost four
  times the SFT model's, and fell back to 26.5 by step 40.
- **gretel gained a little and SQaLe held,** where run 1 lost SQaLe. **CypherBench** dipped in
  the middle of the run and ended at its best (54) at step 40, the one set the late steps helped.
- **`BEST.json` names step 26** because the run picks by mean@8. By greedy, step 30 is ahead of
  every evaluated step, and it is the one kept.

## The training side

Means over groups of five steps (median for the response length). Every step's 32 questions are
new, so one step's solve rate swings by 0.1-0.2 with its batch.

| Steps | Solve rate | Entropy | `kl_sft` (start to end) | KL weight | Response tokens (median) | Queries with `DISTINCT` | Largest grad norm |
|---|---|---|---|---|---|---|---|
| 1-5 (warm-up) | 0.49 | 0.46 | 0.000 to 0.001 | 0.05 | 223 | 28% | 0.06 |
| 6-10 | 0.43 | 0.44 | 0.000 to 0.001 | 0.05 | 211 | 26% | 0.07 |
| 11-15 | 0.49 | 0.43 | 0.001 to 0.004 | 0.05 | 220 | 25% | 0.09 |
| 16-20 | 0.49 | 0.42 | 0.004 to 0.006 | 0.05 | 181 | 25% | 0.10 |
| 21-25 | 0.48 | 0.34 | 0.008 to 0.028 | 0.05 | 124 | 26% | 0.14 |
| 26-30 | 0.50 | 0.28 | 0.025 to 0.078 | 0.05 to 0.4 | 95 | 21% | 0.18 |
| 31-35 | 0.51 | 0.27 | 0.075 to 0.044 | 0.8 to 6.4 | 95 | 25% | 3.11 |
| 36-40 | 0.42 | 0.35 | 0.028 to 0.010 | 6.4 to 1.6 | 180 | 28% | 1.26 |

- **For 25 steps the model barely moved:** `kl_sft` 0.006 at step 20, a twentieth of run 1's at
  step 19. The evaluations after steps 10 and 20 were flat on greedy.
- **Then it learned to think less.** From step 21 the median answer went from about 200 tokens
  to about 95. On the same held-out questions, step 26's thinking ran 24-66 tokens where run 1's
  ran 94-188, while the evaluations rose. This is the stretch (steps 21-30, `kl_sft` 0.01 to
  0.08) where the run's gain was made.
- **Then the penalty took it back.** Above 1.5 x the 0.03 target the weight doubles every step;
  from step 28 it went 0.1, 0.2, 0.4, 0.8, 1.6, 3.2 and hit its cap of 6.4 at step 34. The
  gradient norm went past the 1.0 clip at steps 33-36 (1.75, 3.11, 2.0, 1.26): most of the update was
  the penalty. `kl_sft` fell from 0.078 to 0.010 by step 40, the answers grew back to 160-280
  tokens and the solve rate fell to 0.42 over steps 36-40. The weight only halves below
  0.03 / 1.5 = 0.02, so it kept pulling for six steps after the drift was already back under
  the target.
- No stop rule other than the ones loosened below fired at the end. Malformed calls stayed at
  0-1.6%, `leak_frac` at 0-0.8%, `mismatch` at 0.04-0.06; the duplicate rule touched its line
  once (step 33, 0.74) and fell back.

## Changes made during the run (R14, each with Rishi's OK)

| When | Change | Why |
|---|---|---|
| After step 22 | Evaluation after step 26 added (`eval_steps` 10 / 20 / 26 / 30 / 40) | the length rule paused the run at step 23; an evaluation to see whether the shorter answers were better or worse |
| After step 26's evaluation | `length_ratio` 2.0 to 3.0 | step 26 was better than step 20 on every measure with answers half as long: the rule read improvement as collapse |
| After step 31 | `entropy_min_frac` 0.5 to 0.25, `kl_max` 0.1 to 0.15 | entropy at 0.18 paused the run; with step 30 the best greedy so far, Rishi chose to run to step 40 and then pick |

## What went wrong on the machine (nothing lost)

- **NaN at step 14.** An update produced non-finite weights; the attempt is kept in
  `step-0014/nan-attempt/`. Fixed in 3e5d2eb (a minibatch with a non-finite loss or gradient is
  skipped, non-finite weights are refused).
- **Damaged saves.** Pages of another process were found inside an `optim.pt`. Fixed in 6ae275b:
  every saved state file has a sha256 sidecar, is checked on the disk after the save, and the
  step is trained again if it doesn't match.
- **Three PC crashes during writes to the USB data drive (exFAT).** Two (2026-11-14 and
  -15) were bugcheck 0x13C during the update's save from WSL; the second left step 19's files
  and Docker's disk damaged, and `chkdsk` removed step 19's contents. The run was moved to the
  internal drive on 2026-11-16 and resumed from step 18 (step 19 redone from its kept
  rollouts, step 20's earlier files set aside). The third (2026-11-16, after the run, bugcheck
  0x12C EXFAT_FILE_SYSTEM) came during a plain Windows copy of the finished run back to the USB drive, so
  the drive itself, not only the WSL path, is what fails under heavy writes. Runs now write to
  the internal drive and keep read-only inputs on the USB drive.
- **A GPU freeze at step 35.** The update hung for 40 minutes loading the optimizer state onto
  the GPU (GPU idle, no reads). The trainer was stopped, the state files' hashes checked, and
  the step resumed once with Rishi's OK; it trained normally.

## The reading

- **Run 1b worked where run 1 overshot.** Its best step beats the SFT model by 10 points on
  greedy (54.5 against 44.5 re-measured, 32 / 14 questions, p 0.01) and run 1's best by 9, with
  `kl_sft` lower than run 1's step 19 (0.078 against 0.115).
- **The gain came with drift of about 0.03-0.08, not near zero.** Steps 1-20 at `kl_sft` under
  0.01 changed nothing measurable. The useful change, shorter and more direct thinking, needed
  the model to move.
- **The penalty's target (0.03 against the SFT model) was set below where the gains lived,** so
  the last ten steps were spent pulling the model back. Step 40 kept CypherBench's late gain but
  lost most of text2cypher's.
- **Headroom:** the gap between pass@8 and greedy was 13.0 at step 30, against 15.4 for the SFT
  model and 23.6 at step 26. Step 30 turned much of step 26's sampled ability into its greedy
  answer. Some of what is left is the kind of miss a second look at the result could fix
  (below).

## What a next run must take from this

1. **Measure drift against the run's own starting point** (step 30's adapter for run 2), or
   allow more drift against the SFT model; and let the weight fall as fast as it rises.
2. **Choose by greedy pass@1;** report the headroom gap at each evaluation.
3. **The length rule must allow shorter answers** (or measure collapse another way).
4. **Write the run's data to the internal drive;** move it to the USB drive only as history.

Step 30's 200 greedy answers: 102 right, 85 wrong, 13 undecided (greedy is 102 / 187). The
undecided are left out of the score: 10 text2cypher questions whose reference returns no rows on
the checker's data, 2 answers with no query, 1 other. Of the 85 wrong, 15 hit an error when run
(8 SQaLe, 6 CypherBench, 1 gretel), the clearest thing a second turn can see; 70 ran and
returned other rows, about 20 of them with a different number of columns from the reference
(counted from the query text). Run 2's design page estimates what turns add from this split.

## Not done

- The test set (1,000 questions) on step 30's adapter: needs Rishi's OK (about 2.5 hours of GPU).
- A release export of step 30 (merged into the Q8_0 model) and a model card.
