# Run 1 results: single-turn SRPO, stopped after step 28, step 19 kept

Run `run1`, 2026-11-13. SRPO on the released SFT model with a new LoRA (rank 16), 32 training
questions a step (10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher) with 8 answers each,
learning rate 1e-5 after a 5-step warm-up, no KL penalty (`kl_beta` 0). Planned for 100 steps;
**stopped for good after step 28 (decisions R12). The adapter that is kept is step 19's.**

- Design, harness and how it was run: `run1/design.md`, `run1/harness.md`, `run1/RUNBOOK.md`
- Decisions along the way: R10 (KL bound 0.05 to 0.10), R11 (evaluated at step 19, bound 0.30),
  R12 (stopped, step 19 kept)
- Live page: `run1/dashboard/README.md`
- Files: `<data>/run1/run1/` (`step-00kk/` with each step's rollouts, scores and
  `lora_state.safetensors`; `eval-0019/`, `eval-0025/`, `eval-0028/`; `train_log.jsonl`;
  `config.json` with its `changes` list and the earlier copies beside it)
- The kept adapter: `<data>/run1/run1/step-0019/lora_state.safetensors`. Not yet exported,
  merged for release or run on the test set.

## The evaluations

The 200-question evalset, greedy plus 8 sampled answers at temperature 1.0, checker 4: the
same measurement as step 1. "SFT again" is the unchanged SFT model through run 1's evaluation
path: what no change at all looks like.

| | Greedy pass@1 | mean@8 | pass@8 | Always / sometimes / never | `kl_sft` at that step |
|---|---|---|---|---|---|
| SFT, the baseline | 43.2 | 36.9 | 62.6 | 30 / 94 / 74 | 0 |
| SFT again | 44.5 | 36.4 | 59.9 | 29 / 89 / 79 | 0 |
| **After step 19** | 45.6 | **40.8** | **66.2** | **37 / 92 / 66** | 0.115 |
| After step 25 | **46.8** | 39.7 | 62.6 | 37 / 87 / 74 | 0.208 |
| After step 28 | 44.6 | 39.4 | 63.5 | 31 / 94 / 72 | 0.294 |

Sign tests against the baseline, question by question (better / worse, p):

| | Greedy | Sampled (right of 8) | pass@8 |
|---|---|---|---|
| SFT again | 6 / 4 (0.75) | 37 / 44 (0.51) | 8 / 14 (0.29) |
| After step 19 | 21 / 16 (0.51) | **60 / 41 (0.07)** | 14 / 9 (0.40) |
| After step 25 | 23 / 15 (0.26) | 61 / 52 (0.45) | 18 / 18 (1.0) |
| After step 28 | 22 / 19 (0.76) | 56 / 60 (0.78) | 16 / 15 (1.0) |

Step 28 against step 19, sampled: 42 better / 63 worse (p = 0.05). That is the clearest test in
the run, and it says the model got worse between the two.

By dataset (greedy / mean@8 / pass@8):

| | CypherBench | gretel | SQaLe | text2cypher |
|---|---|---|---|---|
| SFT, the baseline | 44.9 / 37.9 / 80 | 64 / 59.5 / 80 | 42.2 / 34.3 / 57.1 | 15.4 / 15.2 / 32.7 |
| SFT again | 48.0 / 36.4 / 74 | 64 / 60.5 / 76 | 42.2 / 32.7 / 55.3 | 16.2 / 15.6 / 34.0 |
| After step 19 | 53.1 / 42.2 / 78 | 62 / 64.0 / 86 | 36.2 / 33.7 / 52.1 | 25.0 / 22.0 / 46.8 |
| After step 25 | 51.0 / 44.7 / 86 | 64 / 59.1 / 78 | 40.4 / 30.3 / 47.9 | 29.5 / 24.3 / 38.0 |
| After step 28 | 49.0 / 42.9 / 80 | 64 / 59.4 / 82 | 31.9 / 29.7 / 43.8 | 30.0 / 25.1 / 46.9 |

About 50 questions a dataset: one question is 2 points, so only the larger moves mean anything.

- **Both Cypher sets ended above the SFT model and stayed there.** text2cypher's greedy score
  doubled (15 to 30) and CypherBench gained about 5 points on graphs it never trains on.
- **Both SQL sets fell after step 19.** gretel's mean@8 went 64.0 back to 59; SQaLe fell at
  every evaluation, to 10 points under the SFT model on greedy and 13 on pass@8.
- So the overall decline after step 19 is the SQL half giving back its gain and SQaLe going
  below where it started, while the Cypher half held.

## The training side

Means over groups of steps. Every step's questions are new, so the solve rate is a test on
unseen training-split questions, not a memory test. One step swings by 0.1-0.2 with its batch.

| Steps | Solve rate | Entropy | `kl_sft` (start to end) | Response tokens (median) | Queries with `DISTINCT` |
|---|---|---|---|---|---|
| 1-5 (warm-up) | 0.48 | 0.44 | 0.0002 to 0.0006 | 233 | 27% |
| 6-10 | 0.44 | 0.44 | 0.001 to 0.003 | 220 | 24% |
| 11-15 | 0.53 | 0.42 | 0.007 to 0.046 | 208 | 22% |
| 16-19 | 0.52 | 0.34 | 0.061 to 0.115 | 193 | 17% |
| 20-24 | 0.45 | 0.32 | 0.124 to 0.186 | 186 | 15% |
| 25-28 | 0.45 | 0.35 | 0.208 to 0.294 | 177 | 8% |

- **The training solve rate never climbed past its level at steps 11-19.** It was 0.53 there and
  0.45 afterwards, the same shape as the evaluations. This is not the textbook overfit (training
  up, held-out down): after about step 19 the model was moving without getting better anywhere.
- **`kl_sft` grew faster than the square of the steps:** 0.003 at step 10, 0.115 at step 19,
  0.294 at step 28. Nothing in the loss pulled the model back toward the SFT model.
- **Narrowing:** entropy fell by a quarter, answers got a quarter shorter (less thinking).
- No stop rule other than KL fired. Malformed calls stayed at 0-2%, `leak_frac` at 0-1%,
  `SELECT *` at 0, `mismatch` at 0.01-0.06 (bound 0.10).

## The model learned to drop DISTINCT, and the checker can't see it

The same 200 held-out questions, all 8 sampled answers:

| Answers that use `DISTINCT` | All | CypherBench | gretel | SQaLe | text2cypher |
|---|---|---|---|---|---|
| SFT, the baseline | 21.9% | 53.2% | 7.5% | 12.0% | 15.0% |
| After step 19 | 13.2% | 36.0% | 2.2% | 5.5% | 9.0% |
| After step 28 | 9.2% | 21.7% | 2.0% | 5.8% | 7.2% |

On the 56 Cypher questions whose reference query uses `DISTINCT`, the model's answers used it
52% of the time before the run and 22% after step 28, and the checker's right-rate on those
questions went *up* (36.6% to 40.6%). So on the checker's graphs the rows come out the same with
or without it, and the reward can't tell the two apart.

Why the reward pushes this way: rows are compared as multisets, so an unneeded `DISTINCT` is
marked wrong whenever the reference returns duplicates, while a missing `DISTINCT` is marked
wrong only when the test data happens to contain duplicates. Small sampled graphs and random
fills rarely do. The safe habit under this checker is to leave `DISTINCT` out.

This is a gap in the environment, not a proven fault in the answers: an answer that returns the
right rows on the checker's data may return duplicates on real data. Step 19's adapter already
carries part of the habit (36% against 53% on CypherBench). It has to be fixed in the checker's
data (duplicates where the reference uses `DISTINCT`) before another run trusts the reward.

## The reading

- **Run 1 worked for about 15-20 steps and then overshot.** By step 19 it had moved the sampled
  right-rate from 36.9 to 40.8 (about a fifth of the 19-point gap between greedy and pass@8),
  with pass@8 up and "never" down, so it was not narrowing yet. The evidence for that gain is
  p = 0.07 on 198 questions: likely real, not proven.
- **After step 19 it drifted.** The training solve rate fell back, the held-out sampled test went
  from 60 / 41 to 56 / 60, the SQL sets lost what they had gained and SQaLe ended below the SFT
  model. The model was three times further from the SFT model at step 28 than at step 19.
- **Why, as far as the run can show.** Each step learns from about 16-20 groups with both right
  and wrong answers. That is a noisy direction. The optimiser (Adam at a constant 1e-5) takes a
  step of about the same size however noisy the direction is, and nothing held the model near
  the SFT model. So the distance travelled grew every step while the useful signal did not.
- **The peak is only bracketed.** Evaluations exist at 19, 25 and 28. The training solve rate
  puts the best stretch at steps 11-19. Steps 13-17 were never evaluated.
- **What the run does not show:** that single-turn RL is finished at +4 points. It shows that
  these settings overshoot. A slower run could keep more of the gain or find more.

## What a next run must change

1. **Hold the model near the SFT model:** a KL penalty in the loss, or a lower learning rate
   (3e-6 to 5e-6) with decay, so `kl_sft` at step 100 is where this run's was at step 15-19.
2. **Evaluate early and often, and keep the best step,** not the last. The first evaluation was
   planned for step 25, six steps after the best one that was measured. A 100-question quick
   evaluation every 5-10 steps would have shown the turn.
3. **A less noisy step:** more questions a step, or two steps' rollouts per update.
4. **Fix the reward's blind spot for `DISTINCT`** (above) and look for others of its kind.
5. **Look at SQaLe before training on it again:** its databases are random fills, it is the one
   dataset that ended clearly below the SFT model, and 24-28 of its 384 sampled answers can't
   be decided at all.

## Not done

- The test set (1,000 questions) on step 19's adapter: needs Rishi's OK (about 2.5 hours of GPU).
- A release export of step 19 (the LoRA merged into the Q8_0 model) and a model card: only if
  the adapter is to be published or used as run 2's starting point.
