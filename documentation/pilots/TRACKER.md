# Step 2 tracker: the method pilots

The written record of step 2 (`../rl.md`, "Step 2: the method pilots"). It is updated by hand as
each pilot runs and finishes. The same numbers stream into the live page while a pilot trains.

- **Live page:** [Spark RL Pilots](https://claude.ai/artifact/V9TGXUsRRRw493gZE74QuC) (private; only the
  owner and people given access can open it). How it is fed: `README.md` in this folder.
- **Results written up per step:** `../results/`. The baseline every pilot is compared with:
  `../results/step1-baseline.md`.

## What step 2 does

Each candidate method trains a **new LoRA on top of the merged, released SFT model**, on
**training-split questions only**, for the same short budget. The pilot model is then scored with
step 1's measurement on the 200-question evalset. The pilots pick the method for the real run
(step 3) and settle its remaining settings. They also time a real step on the 7800 XT, which
decides whether a stage runs locally or on Modal.

### The budget, the same for every pilot

| Setting | Value |
|---|---|
| Stage | A: single turn, execution reward (checker 4) |
| RL steps | 40 |
| Questions a step | 32 (training split only) |
| Rollouts a question (group size) | 8 |
| Sampling temperature | 1.0 |
| Start | the released SFT model (step 1,500, merged), new LoRA |
| Held out | the 200-question evalset and the 1,000-question test set never reach a rollout, a reward or a filter |

`p0-smoke` is the exception: 2-3 steps, only to test the loop (generate, check, train, convert
the adapter). It is not a candidate and is not compared.

### The pilots

| Id | Method | What is different |
|---|---|---|
| `p0-smoke` | Loop test | 2-3 steps, only to prove the loop works end to end |
| `p1-grpo` | Dr. GRPO | Advantage is reward minus the group mean, no std normalisation; token-level loss; KL penalty to the SFT model |
| `p2-dapo` | DAPO | Clip-higher, dynamic sampling (skip groups all right or all wrong), token-level loss, no KL |
| `p3-gspo` | GSPO | Sequence-level importance ratios |
| `p4-sdpo` | SDPO | Self-distillation from feedback: the model, shown the error or the wrong rows of a failed attempt, teaches itself |
| `p5-srpo` | SRPO | GRPO for right samples, SDPO for failed ones |

## The yardstick: the baseline

Run `baseline-sft1500-t1.0` (step 1): the released model, evalset of 200 questions, 198 decided.

| Set | Questions | Greedy pass@1 | Mean@8 | pass@8 | Always / sometimes / never |
|---|---|---|---|---|---|
| **Overall** | 198 | **43.2%** | **36.9%** | **62.6%** | 30 / 94 / 74 |
| CypherBench | 50 | 44.9% | 37.9% | 80.0% | 5 / 35 / 10 |
| gretel | 50 | 64.0% | 59.5% | 80.0% | 17 / 23 / 10 |
| SQaLe | 49 | 42.2% | 34.3% | 57.1% | 6 / 22 / 21 |
| text2cypher | 49 | 15.4% | 15.2% | 32.7% | 2 / 14 / 33 |

One evalset question is 0.5 points of pass@1, so changes of a point or two are inside noise unless
the sign test says otherwise.

## How a pilot is judged

1. **pass@1 up, toward pass@8.** Greedy pass@1 and sampled mean@8 should rise above 43.2% and
   36.9%. That is RL turning "sometimes right" into "right first time".
2. **pass@8 not down by more than noise.** The question-by-question sign test against the baseline
   decides. A real drop means the method narrows the model's answers instead of teaching it. A
   method that lowers pass@8 by more than noise does not move on.
3. **The canaries stay steady:** entropy, KL to the SFT model, thought length (the thinking budget
   stays capped), the well-formed-call rate, the "can't answer, say so" cases and thinking-off
   calls. A pilot that drifts is stopped and reported, not tuned around.
4. **Reward hacking is a finding.** Format tricks, `SELECT *`, `DISTINCT` everywhere, empty results
   matching empty references: report any sign of them.
5. **Where the room is:** CypherBench has the most room (+35 points), and text2cypher is mostly
   "never" (33 of 49), which is where SDPO and SRPO would have to show their worth. Watch the
   "never" bucket of each pilot on its own.
6. **Cost:** the mean step time on the 7800 XT (generate, check, train, convert), because it
   decides local against Modal. The evalset picks between pilots; the test set is run only to
   confirm the finished stage.

## Status

Updated by hand. Dates are the pilot's start and finish. "Final reward" is the mean training
reward of the last step (training questions, not the evalset).

### Where each pilot stands

| Pilot | Method | Status | Started | Finished | Steps | Final reward | Mean step time |
|---|---|---|---|---|---|---|---|
| `p0-smoke` | Loop test | not started | - | - | - / 3 | - | - |
| `p1-grpo` | Dr. GRPO | not started | - | - | - / 40 | - | - |
| `p2-dapo` | DAPO | not started | - | - | - / 40 | - | - |
| `p3-gspo` | GSPO | not started | - | - | - / 40 | - | - |
| `p4-sdpo` | SDPO | not started | - | - | - / 40 | - | - |
| `p5-srpo` | SRPO | not started | - | - | - / 40 | - | - |

Status values: not started, running, evaluating, done, failed, stopped.

### How each pilot scored

Evalset, 200 questions. Each cell is the pilot's number, then its change against the baseline in
points. Sign test: questions better / worse than the baseline, and p. Baseline: greedy 43.2%,
mean@8 36.9%, pass@8 62.6%.

| Pilot | Greedy pass@1 | Mean@8 | pass@8 | Sign test | Canaries | Verdict |
|---|---|---|---|---|---|---|
| `p0-smoke` | (not evaluated) | | | | | smoke only |
| `p1-grpo` | - | - | - | - | - | not started |
| `p2-dapo` | - | - | - | - | - | not started |
| `p3-gspo` | - | - | - | - | - | not started |
| `p4-sdpo` | - | - | - | - | - | not started |
| `p5-srpo` | - | - | - | - | - | not started |

Verdict values: **candidate** (pass@1 up, pass@8 held, canaries steady), **check pass@8**
(pass@1 up but pass@8 down inside noise), **narrowed** (pass@8 down by more than noise),
**no gain**, **drifted** (canaries moved, stopped), **failed**.

## Per-pilot sections

Copy this template into the pilot's section when it starts, and fill it as it runs.

```
### <id>: <method>

- **Status and dates:** started ..., finished ..., evaluated ...
- **Config:** steps 40, questions a step 32, group size 8, temperature 1.0, lr ..., KL beta ...,
  clip ..., LoRA rank ..., anything else the method needs
- **W&B run:** ...
- **What happened:** how the run went, interruptions, anything odd (a few sentences)
- **Training:** reward first 3 steps -> last 3 steps; solve rate; informative share; mean step
  time (rollouts / scoring / training); KL, entropy and response length, start -> end
- **Evaluation** (greedy pass@1, mean@8, pass@1..8, always / sometimes / never):

  | Set | Greedy pass@1 | Mean@8 | pass@8 | Never | Sometimes | Always |
  |---|---|---|---|---|---|---|
  | Overall | | | | | | |
  | CypherBench | | | | | | |
  | gretel | | | | | | |
  | SQaLe | | | | | | |
  | text2cypher | | | | | | |

- **Sign test against the baseline:** ... better, ... worse, p = ...
- **Canaries:** entropy, KL, thought length, well-formed calls, "can't answer" cases
- **Reward-hacking watch:** what was looked at, what was found
- **Verdict:** ...
```

### p0-smoke: loop test

Not started. Fill from the template when it runs. It is only a test of the loop; no evaluation.

### p1-grpo: Dr. GRPO

Not started. Fill from the template when it runs.

### p2-dapo: DAPO

Not started. Fill from the template when it runs.

### p3-gspo: GSPO

Not started. Fill from the template when it runs.

### p4-sdpo: SDPO

Not started. Fill from the template when it runs.

### p5-srpo: SRPO

Not started. Fill from the template when it runs.

## Decision: which method goes to the real run, and why

Not decided yet. Fill this in when every pilot that will run has been evaluated. Log the choice in
`../decisions.md` (the next `R` number) and name it here.

- **Chosen method:**
- **Why (the numbers it rests on):**
- **Settings carried forward** (temperature, group size, KL weight, learning rate, clip):
- **What the others showed** (including any that narrowed pass@8 or drifted):
- **Step time on the 7800 XT, and so local or Modal for stage A:**
- **Risks and what to watch in the real run:**
- **Decision entry:** `../decisions.md`, R...
- **Date:**
