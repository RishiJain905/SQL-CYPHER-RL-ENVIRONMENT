# Step 1 results: the sampling baseline

Run `baseline-sft1500-t1.0`, 2026-11-01. The released model (step 1,500 of `sft-main`, merged,
Q8_0) answered each of the 200 evalset questions 8 times at temperature 1.0 (top-p 1, top-k and
min-p off, thinking budget 1,024; decisions R1), and checker 4 scored all 1,600 answers. 198
questions had at least one answer the checker could decide.

- Live page: [Spark RL Baseline](https://claude.ai/artifact/JcR5uM76GHeJUCf1UkHooJ) (private)
- W&B: project `sql-cypher-rl`, run `baseline-sft1500-t1.0` (job type `baseline`)
- Files: `<data>/rl/baseline/baseline-sft1500-t1.0/` (`preds_s0..7.jsonl`, `scores_s0..7.json`,
  `summary.json`)

## The numbers

| Set | Questions | Greedy pass@1 | Sampled mean@8 | pass@2 | pass@4 | pass@8 | Gap (pass@8 − greedy) | Always | Sometimes | Never |
|---|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 198 | **43.2%** | 36.9% | 47.0% | 55.7% | **62.6%** | **+19.4** | 30 | **94** | 74 |
| CypherBench | 50 | 44.9% | 37.9% | 52.1% | 66.2% | 80.0% | **+35.1** | 5 | 35 | 10 |
| gretel | 50 | 64.0% | 59.5% | 68.6% | 75.4% | 80.0% | +16.0 | 17 | 23 | 10 |
| SQaLe | 49 | 42.2% | 34.3% | 44.7% | 52.4% | 57.1% | +14.9 | 6 | 22 | 21 |
| text2cypher | 49 | 15.4% | 15.2% | 22.0% | 28.1% | 32.7% | +17.3 | 2 | 14 | **33** |

- Greedy pass@1 is the released model's greedy evalset score (183 decided questions; per set,
  over the questions decided in both runs). pass@k is the unbiased estimate from the 8 samples.
- **Always / sometimes / never:** questions with all, some or none of their decided answers right.
- **How many of 8 were right** (questions with all 8 answers decided, 152 of them):
  0: 44, 1: 18, 2: 9, 3: 7, 4: 12, 5: 9, 6: 14, 7: 9, 8: 30.
- Answers the checker couldn't decide: 148 of 1,600, most of them text2cypher (97) and SQaLe (41).
- Each sample file on its own: 36.0-43.0% (mean 39.5%). Well-formed calls 96.5-98.0%.

## What it means for RL

- **RL has a lot of room.** pass@8 is 62.6% against 43.2% greedy: one question in five that the
  model can answer but doesn't answer first. A GRPO-style method can, at best, move pass@1
  toward pass@8.
- **The learning pool is large: 94 of 198 questions (47%) are "sometimes".** Those are the
  questions a GRPO group learns from. At group size 8, roughly half of a step's questions carry
  a gradient, which sets the step cost (the plan's estimate assumed about that).
- **43 questions the greedy answer gets wrong have at least one right sample:** the easiest wins.
  Only 1 question greedy gets right has no right sample. When greedy is right, the samples are
  right 74% of the time.
- **CypherBench has the most room (+35 points, 35 "sometimes").** That's a surprise: the plan
  guessed text2cypher. CypherBench's answers are checked on the dataset's own graphs, so its
  reward is the cleanest of the four.
- **text2cypher is mostly "never" (33 of 49).** GRPO can't learn from those; it's where SDPO /
  SRPO's learning from error feedback would have to show its worth. Its many undecided answers
  (random graphs that can't tell the queries apart) also make it the noisiest reward.
- **Temperature 1.0 is usable.** Sampling costs 6.3 points of first-try accuracy (43.2% greedy,
  36.9% sampled). That's a moderate drop: the samples are varied but not wild (well-formed calls
  stay at ~97%, 1 answer of 1,600 hit the token cap). The pilots can still try 0.8.
- **The failures are mostly wrong answers, not broken ones:** 613 of 879 wrong answers ran and
  returned different rows (on the real rows or the filled copies), 194 errored at run time, 48
  didn't parse. That's what an execution reward is good at shaping.

## For step 2

- Start the pilots on stage A with all four datasets, weighted toward the "sometimes" pool;
  CypherBench gives the clearest signal.
- Judge pilots on pass@1 (greedy) rising toward 62.6% and pass@8 not falling below it (the
  sign test against this run's per-question counts).
- SDPO / SRPO's case rests on the 74 "never" questions (44 with all 8 decided); watch them
  separately.
