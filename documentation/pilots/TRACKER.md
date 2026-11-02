# Step 2 tracker: the method pilots

The written record of step 2 (`../rl.md`, "Step 2: the method pilots"). It is updated by hand as
each pilot runs and finishes. The same numbers stream into the live page while a pilot trains.

- **Live page:** [Spark RL Pilots](https://claude.ai/artifact/V9TGXUsRRRw493gZE74QuC) (private; only the
  owner and people given access can open it). How it is fed: `README.md` in this folder.
- **Results written up per step:** `../results/`. The baseline every pilot is compared with:
  `../results/step1-baseline.md`.
- **The two runs and the methods explained:** `../rl-runs.md`.

## What step 2 does

Each method trains a **new LoRA on top of the merged, released SFT model**, on **training-split
questions only**, for the same short budget. The pilot model is then scored with step 1's
measurement on the 200-question evalset. The pilots pick the method for the real run (step 3)
and time a real step on the 7800 XT, which decides whether a stage runs locally or on Modal.

**Two pilots, 10 steps each** (Rishi, 2026-11-01; decisions R4): start short, and only the two
methods that matter most in theory.

### The budget, the same for every pilot

| Setting | Value |
|---|---|
| Stage | A: single turn, execution reward (checker 4) |
| RL steps | **10** |
| Questions a step | 32 (8 per dataset; training split only, the same questions in the same order for both pilots: `pool-v1`) |
| Rollouts a question (group size) | 8 (256 a step) |
| Sampling | temperature 1.0, top-p 1, top-k and min-p off; thinking budget 1,024 tokens, then `</think>` is forced; 2,048 tokens in all |
| Reward | 1 right (checker 4), 0 wrong, -0.1 no parsable call; undecided answers get none |
| LoRA | new, rank 16, alpha 32, on the SFT run's modules; lr 1e-5, AdamW, grad clip 1.0, 2 minibatches a step |
| Start | the released SFT model (step 1,500, merged); rollouts from its Q8_0 + the pilot's LoRA |
| Held out | the 200-question evalset and the 1,000-question test set never reach a rollout, a reward or a filter |

`p0-smoke` and `p0-srpo-smoke` are the exceptions: 2 steps of 8 questions, only to test the loop.
They are not candidates and are not compared.

### The pilots

| Id | Method | What it does |
|---|---|---|
| `p1-grpo` | GRPO (Dr. GRPO form) | The standard. Advantage = reward minus the group mean (no std division), clipped token ratios [0.8, 1.2], the token sum divided by a constant (8 x 2,048 per group), KL to the SFT model (k3, beta 0.01). Learns from the "sometimes" questions only. |
| `p2-srpo` | SRPO | GRPO's branch for right answers and for wrong ones with no right sibling (std advantage, clip-higher [0.8, 1.28]); **self-distillation for wrong answers that have a right sibling**: an EMA copy of the LoRA (rate 0.05), shown the question plus the sibling's query, re-scores the wrong answer's tokens, and the student moves toward it (JSD over its top-100 tokens, weighted by the teacher's confidence). One token-level average, no KL. Learns from failures too. |

Why these two: GRPO is the reference every RL method is measured against; SRPO (arXiv
2604.02288) is the one built for what GRPO can't do (the 74 "never" questions get a signal from
a right sibling's solution) and reports beating both GRPO and plain SDPO without SDPO's late
collapse. DAPO and GSPO are refinements inside GRPO's family; they stay in the code
(`config.METHODS`) but aren't run.

## Running, pausing, resuming

| What | Command (Windows, start detached and hidden) |
|---|---|
| A pilot, start to finish (training, then the evaluation) | `C:\Python314\python.exe -I scripts\step2_pilots\run.py --pilot p1-grpo` |
| Pilots one after another | `C:\Python314\python.exe -I scripts\step2_pilots\pilot_queue.py --after p1-grpo p0-srpo-smoke p2-srpo` |
| **Free the GPU now** | `C:\Python314\python.exe -I scripts\step2_pilots\pause.py --pilot <id>` |
| Stop after the current step | `C:\Python314\python.exe -I scripts\step2_pilots\pause.py --pilot <id> --after-step` |
| Resume | the same command that started it (`run.py --pilot <id>`) |
| A tick for the live page | `C:\Python314\python.exe -I scripts\step2_pilots\feed.py --pilot <id>` |

`pause.py` stops the pilot's process tree, the queue, the pilots' llama-server and any pilot
process inside WSL, checks that none is left, and writes `PAUSED`. A resume redoes at most the
update of the step that was cut (its rollouts and scores are kept). Tested on `p1-grpo` mid-
rollout (2026-11-01): GPU freed, resumed with its 20 rollouts kept.

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
the sign test says otherwise. The pilots' 8 sampled answers use the baseline's own seeds, so every
answer pairs with the baseline's.

## How a pilot is judged

1. **pass@1 up, toward pass@8.** Greedy pass@1 and sampled mean@8 should rise above 43.2% and
   36.9%.
2. **pass@8 not down by more than noise** (the question-by-question sign test).
3. **The canaries stay steady:** entropy, KL, thought length and the forced-`</think>` share, the
   well-formed-call rate, the rollout-vs-trainer log-prob mismatch.
4. **Reward hacking is a finding** (format tricks, `SELECT *`, `DISTINCT` everywhere, empty
   results matching empty references).
5. **The "never" bucket on its own:** SRPO's case rests on it.
6. **Cost:** the mean step time on the 7800 XT.

With 10 steps the changes may be small; a pilot that moves nothing outside noise is "no gain at
10 steps", not proof the method can't work.

## Status

### Where each pilot stands

| Pilot | Method | Status | Started | Finished | Steps | Final reward | Mean step time |
|---|---|---|---|---|---|---|---|
| `p0-smoke` | GRPO, loop test | done | 2026-11-01 | 2026-11-01 | 2 / 2 | 0.368 | 6.8 min (8 questions) |
| `p1-grpo` | GRPO | done | 2026-11-01 | 2026-11-02 | 10 / 10 | 0.495 | 15.3 min |
| `p0-srpo-smoke` | SRPO, loop test | running | 2026-11-02 | - | - / 2 | - | - |
| `p2-srpo` | SRPO | queued (after the SRPO smoke) | - | - | - / 10 | - | - |

Status values: not started, queued, running, evaluating, done, failed, stopped / paused.

### How each pilot scored

Evalset, 200 questions. Each cell is the pilot's number, then its change against the baseline in
points. Sign test: questions better / worse than the baseline, and p.

| Pilot | Greedy pass@1 | Mean@8 | pass@8 | Sign test (greedy) | Canaries | Verdict |
|---|---|---|---|---|---|---|
| `p1-grpo` | 45.4% (+2.2) | 37.5% (+0.6) | 62.6% (0.0) | 10 better / 6 worse, p = 0.45 (sampled 43 / 41, p = 0.91) | steady | **no gain at 10 steps** (pass@8 held) |
| `p2-srpo` | - | - | - | - | - | queued |

Verdict values: **candidate** (pass@1 up, pass@8 held, canaries steady), **check pass@8**
(pass@1 up but pass@8 down inside noise), **narrowed** (pass@8 down by more than noise),
**no gain at 10 steps**, **drifted** (canaries moved, stopped), **failed**.

## Per-pilot sections

Template:

```
### <id>: <method>

- **Status and dates:** started ..., finished ..., evaluated ...
- **Config:** (the budget above) + the method block
- **W&B run:** project sql-cypher-rl, run <id>
- **What happened:** how the run went, interruptions, anything odd
- **Training:** reward first 3 steps -> last 3 steps; solve rate; informative share; mean step
  time (rollouts / scoring / training); KL, entropy and response length, start -> end
- **Evaluation:** the per-set table (greedy pass@1, mean@8, pass@8, never / sometimes / always)
- **Sign tests against the baseline:** greedy ... better / ... worse, p = ...; sampled ...
- **Canaries / reward-hacking watch / verdict**
```

### p0-smoke: loop test (GRPO)

- **2026-11-01, 2 steps of 8 questions x 8 rollouts.** Every part of the loop ran: llama-server
  rollouts with exact token ids, the checker, the update in WSL, the LoRA saved and converted to
  GGUF, and step 2's rollouts served with step 1's LoRA (the server confirms it loaded, scale 1).
- **Step 1:** reward 0.313 (gretel 0.88, CypherBench 0.13, SQaLe 0.13, text2cypher 0.0), 5 of 8
  groups informative, 39 sequences trained, 15.6% of answers undecided, 23% hit the thinking
  budget; peak GPU memory 11.3 GB; 7.3 min (rollouts 2.7, scoring 0.8, model load 1.3, update
  2.2). Rollout-vs-trainer log-prob mismatch 0.013 a token. **Step 2:** reward 0.368, 6.2 min.
- **Not evaluated** (a loop test).

### p1-grpo: GRPO

- **Status and dates:** trained 2026-11-01 (steps 1-10; the PC lost power during step 2 and the
  run resumed with nothing lost), evaluated 2026-11-02 (paused once mid-evaluation to free the
  GPU, resumed).
- **Config:** the budget above; Dr. GRPO (r - mean, constant normaliser), clip [0.8, 1.2], KL
  beta 0.01. W&B: `sql-cypher-rl`, run `p1-grpo`.
- **Training:** reward by step 0.356, 0.415, 0.415, 0.577, 0.464, 0.285, 0.308, 0.388, 0.425,
  0.495 (each step new questions, 8 per set, so it swings with the batch). Informative groups
  38-63%. Entropy 0.45-0.53, KL 0.0001-0.0002 a token (the LoRA moved the model very little),
  answers 295-436 tokens, forced `</think>` 16% -> 4-10%, malformed calls ~3%. Mean step 15.3 min
  (rollouts ~8.5, scoring ~1.2, load ~1, update ~5); peak GPU 12.2 GB; rollout-vs-trainer
  mismatch ~0.014 a token.
- **Evaluation** (baseline in brackets):

  | Set | Greedy pass@1 | Mean@8 | pass@8 | Never | Sometimes | Always |
  |---|---|---|---|---|---|---|
  | Overall | 45.4 (43.2) | 37.5 (36.9) | 62.6 (62.6) | 74 (74) | 92 (94) | 32 (30) |
  | CypherBench | 44.0 (44.9) | 36.1 (37.9) | 78.0 (80.0) | 11 | 36 | 3 |
  | gretel | 64.0 (64.0) | 62.5 (59.5) | 80.0 (80.0) | 10 | 21 | 19 |
  | SQaLe | 42.2 (42.2) | 34.4 (34.3) | 55.1 (57.1) | 22 | 20 | 7 |
  | text2cypher | 27.5 (15.4) | 16.5 (15.2) | 36.7 (32.7) | 31 | 15 | 3 |

  pass@1-8: 37.5, 47.4, 52.9, 56.4, 59.0, 60.6, 61.7, 62.6 (baseline 36.9 ... 62.6).
- **Sign tests:** greedy 10 better / 6 worse (p = 0.45); sampled 43 / 41 (p = 0.91).
- **Reading:** a small greedy gain inside noise, pass@8 exactly held (no narrowing), canaries
  steady. text2cypher's greedy +12 points is the one large move; text2cypher is the noisiest set
  (random graphs), so it's a lead, not a result. With a KL this small, 10 steps at lr 1e-5 barely
  moved the LoRA: a longer run or a higher learning rate is what would show GRPO's effect.
- **Verdict:** no gain at 10 steps (pass@8 held, canaries steady).

### p0-srpo-smoke: loop test (SRPO)

Queued: tests the self-distillation branch (routing, the EMA teacher's pass, the JSD loss) on
the GPU before `p2-srpo`.

### p2-srpo: SRPO

Queued.

## Decision: which method goes to the real run, and why

Not decided yet. Fill this in when both pilots have been evaluated. Log the choice in
`../decisions.md` (the next `R` number) and name it here.

- **Chosen method:**
- **Why (the numbers it rests on):**
- **Settings carried forward** (temperature, group size, KL weight, learning rate, clip):
- **Step time on the 7800 XT, and so local or Modal for stage A:**
- **Risks and what to watch in the real run:**
- **Decision entry:** `../decisions.md`, R...
- **Date:**
