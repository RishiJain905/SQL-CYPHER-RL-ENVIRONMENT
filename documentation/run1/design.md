# Run 1 design (stage A, single turn)

**Status:** reviewed and built, 2026-11-12. **Not started: Rishi starts it.** The first draft
(2026-11-05) was reviewed line by line before anything was built; "Review" below says what held
and what changed. The three choices that are Rishi's are in section 8, each with a default the
code already uses. Operating it: [`README.md`](README.md). The rollout harness and its
measurements: [`harness.md`](harness.md). Run 2 gets its own design after this one.

Run 1's job (`../rl-runs.md` section 2): turn "sometimes right" into "right first time", with one
query per answer and an execution reward. It starts from the released SFT model with a new LoRA,
like the pilots.

## Review: what held, what changed

| Draft choice | Verdict | Why |
|---|---|---|
| SRPO as piloted | **Kept** | Still the best-supported choice; section 3 says what the pilots do and don't show |
| Checker 4, rewards 1 / 0 / -0.1 | **Kept**, with a floor under the group std (R8) | Without it a group with no right answer and one malformed call pushes its wrong answers up at full strength |
| Learning rate 1e-5 with 5 warm-up steps | **Kept, and now with a reason** | The real policy moved further in the pilots than their logs showed (R7); 2e-5 is not needed |
| Importance weights for the server/trainer gap | **Dropped** | The gap is the trainer's rounding, not the server's (R7); weighting would pull the update the wrong way |
| KL to SFT logged | **Kept, measured differently** | From the server's own log-probs, because the trainer under-reads a small LoRA |
| Stop the run if the gap passes 0.05 | **Changed** to a KL bound (0.05 a token) and a looser gap bound (0.10) | The gap is expected and understood; drift is the thing to bound |
| Rollouts from the Q8_0 model + `--lora` | **Changed** to the LoRA merged into the Q8_0 weights | It is the model that would ship, and 23% faster (R7) |
| Shared `src/sqlrl/rl/` module | **Changed** to self-contained `run1/` folders | Rishi's instruction (R6) |
| Dataset mix, 100 steps, no probe pass, `high` thinking, pool filters | **Kept** | Nothing in the review speaks against them; the mix and the length are Rishi's calls |
| New: a "teacher leak" canary | **Added** | SRPO's teacher reads a solution; a student that starts citing one is copying the wrong thing |

## 1. The environment (R5: a sandbox, with a faithful prompt)

The model never sees what its query returns, so the environment is the prompt plus the reward.

**What the model reads is deployment's:**
- **The prompt:** the tool-call prompt SFT trained on (the trainer's `tool_prompt`: schema, up to
  two tested examples, the question), rendered by llama-server itself with the release's chat
  template (`/apply-template`). Every step checks that the trainer's rendering of a prompt
  equals the server's (`template_match`).
- **To check before release, not before the run:** the model card shows this exact system prompt
  and tool schema.
- **Thinking level:** `high` everywhere, as in the pilots and the yardstick.
- **Thinking budget:** 1,024 tokens, then a forced `</think>` that carries no loss (4-16% of
  answers hit it in the pilots).

**What samples the answers is the model that would ship:** the SFT weights with the LoRA merged
in, quantised to Q8_0 (R7). The pilots loaded the LoRA at run time instead.

**What decides the reward:** checker 4 (`publiceval.score`), imported unchanged.
- **Rewards as piloted:** 1 right, 0 wrong, -0.1 for no parsable call; an undecided answer is
  left out of its group and the loss.
- **Every question is used once.** The checker's filled databases use fixed seeds, so a question
  seen often could be answered for those copies only. `pool-v2` never repeats a question.
- **Questions that can't teach are left out:** each reference query is scored by the checker as
  if the model had written it (with its first keyword in the other case: the checker passes
  an exact copy of the reference without running it). One that the checker can't decide (it returns nothing, or only
  zeros and nulls, or no rule proves it) would make every rollout "undecided"; for gretel the
  reference's rows must also say something.
- **Canaries, logged every step** (`canaries.py`): the share of queries with `DISTINCT`, with
  `SELECT *`, replies with several calls, thoughts that cite a "correct / provided / given
  solution" (the teacher leak), the median query and response length. Not built: comparing the
  columns a query returns with the reference's (it needs a second execution of every rollout).

## 2. The questions

**What the pilots showed:** 38-63% of groups carried a signal (mean about 50%). The rest were
all right or all wrong and cost rollout time only.

- **The pool:** `pool-v2` (`run1/pool.py`), from the training split, the filters of section 1
  applied, evalset and test questions excluded by id and text, duplicates of a question kept
  once, prompts of at most 4,000 tokens. Built 2026-11-12: 6,600 questions (2,000 CypherBench,
  2,000 gretel, 1,600 SQaLe, 1,000 text2cypher), enough for 200 steps at the recommended mix.
  The filter rejected 30% of the text2cypher candidates, 24% of gretel's, 7% of SQaLe's and
  4% of CypherBench's.
- **No probe pass.** Probing 2,000 questions x 8 samples costs about what it would save.
  Groups with no signal are skipped and the informative share is logged.
- **The mix per step of 32** is a real choice (section 8). The pilots used 8 per dataset.
  CypherBench has the most room (pass@8 80 against greedy 45) but trains on 4 graphs and is
  measured on 7 others, and it slipped a little in both pilots (inside noise); text2cypher's
  reward is the weakest (random graphs, half its questions undecidable).

## 3. The method

**SRPO, as piloted** (R4): routed per rollout. A wrong rollout with a right sibling learns by
self-distillation from an EMA teacher (rate 0.05) that is shown the sibling's query; every
other rollout takes GRPO's branch (std advantages, clip [0.8, 1.28]); one token-level average;
no KL penalty.

**What the pilots do and don't show.** SRPO led GRPO on every sampled measure (mean@8 +2.8
against +0.6), none of it outside noise at 10 steps. Two caveats the draft didn't state:
- The two pilots differ in more than the method. GRPO ran in Dr. GRPO's form, whose loss is
  divided by a constant about 6 times larger than the tokens actually trained, so its gradient
  was ~12 times smaller. With Adam that changes less than it sounds (Adam normalises step
  sizes), but the comparison is "SRPO's whole recipe against Dr. GRPO's", not the distillation
  alone.
- SRPO gives the "never" questions no direct signal: distillation needs a right sibling. What
  it adds is a richer signal for the wrong answers of mixed groups (a token-by-token target
  instead of "less of this").

It stays the choice because it is the only recipe here that learns from failures, it matched or
beat GRPO everywhere, and its canaries were steady. Run 1 logs what would show it going wrong:
the distilled share, the teacher's JSD, the leak canary.

- **Learning rate: 1e-5, 5 warm-up steps** (0.2, 0.4, ... of the rate over steps 1-5). The
  pilots' logs said the policy barely moved; measured from the server it was 0.004 a token from
  the SFT model after 10 SRPO steps (R7). Raise it only if the step-25 evaluation and `kl_sft`
  both show no movement.
- **A floor of 0.25 under the group's standard deviation** (R8).
- **Drift from SFT, `kl_sft`,** from the server's log-probs on one rollout per group.

## 4. Speed

One step took ~15.2 minutes in the pilots. What changed (R9, `harness.md`):

| Part | Pilots | Run 1 | How |
|---|---|---|---|
| Rollouts (256) | 504-540 s | **292 s** measured | a KV per slot, merged weights, a worker per slot, prompts computed once per question |
| Scoring | 50-118 s | hidden behind the model load | its own process |
| Model load | 55-90 s | **17-22 s** measured (loop test) | WSL kept awake for the run, the model held in its file cache |
| Update | 190-380 s | the same, plus ~25 s for the probe | one sequence per pass (batching was slower) |
| Next rollout model | 5-8 s (adapter to GGUF) | ~22 s | merged and quantised in place |

A full step is expected at about **11 minutes** (to be confirmed by the run's first steps), so
100 steps are about 18 GPU hours instead of 26, plus three evaluations.

## 5. Length, evaluations and the gate

- **100 steps first,** evaluated after steps 25, 50 and 100 on step 1's yardstick (evalset,
  temperature 1.0, 8 samples with the baseline's seeds, checker 4, plus greedy), served from
  the merged model.
- **Extend to 200 steps** only if mean@8 is still rising at 100 and the gate holds. 300 stays
  the ceiling.
- **The gate:** pass@8 not down. The test is question by question: questions that had a right
  answer among the baseline's 8 and have none now, against the reverse. More lost than gained
  with p < 0.05 pauses the run and keeps the last good adapter.
- **The test set** is used once, on the adapter we keep.

## 6. Stop rules (any one pauses the run for a look)

Checked after every step from `train_log.jsonl`; "the start" is the mean of the run's first
five steps. The bounds are in the run's `config.json` (`stop`).

| Canary | Pilot values | Pause when |
|---|---|---|
| Mean token entropy | 0.43-0.53 | below half of the start |
| Informative share | 0.38-0.63 | below 0.25 for 5 steps in a row |
| Malformed calls | 0-3.5% | above 5% for 3 steps in a row |
| KL to SFT (`kl_sft`) | 0.004 after 10 SRPO steps | above 0.05 a token |
| Server/trainer gap (`mismatch`) | 0.013 with no LoRA, 0.031 after 10 steps | above 0.10 |
| Median response length | ~160-400 tokens | more than 2x or less than half of the start |
| `DISTINCT` share | 15-22% | 10 points above the start |
| Thoughts citing a given solution | 0% | above 2% |

## 7. Where the code is

Run 1 is self-contained (R6): `src/sqlrl/run1/`, `scripts/run1/`, `tests/run1/`,
`documentation/run1/`, `<data>/run1/<run name>/`. It shares only `paths.py`, `passk.py` and step
1's sampler and scorer (the yardstick).

| File | What it does |
|---|---|
| `config.py` | the settings, the mix, the warm-up, a step's questions |
| `pool.py` | `pool-v2` |
| `server.py` | starts and stops llama-server |
| `rollouts.py` | the scheduler and the two-phase rollout |
| `train_step.py` | scoring, the probe, the update, saving, the merge |
| `rewards.py`, `losses.py`, `logprobs.py`, `srpo.py` | the piloted method (copies) |
| `gguf_merge.py` | the LoRA into the Q8_0 file |
| `canaries.py` | the text canaries and the stop rules |
| `orchestrate.py`, `pause.py` | the run loop, evaluations, the gate; freeing the GPU |
| `eval_score.py` | an evaluation's summary and sign tests |
| `tools.py` | rebuild the rollout model; measure the server/trainer gap |

## 8. Decisions for Rishi

Each has a default the code uses unless the run is created with another value.

1. **The dataset mix:** 10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher (default,
   `--mix recommended`), or the pilots' even 8s (`--mix even`).
2. **The learning rate:** 1e-5 with warm-up (default), or another (`--lr`). The review found no
   case for 2e-5.
3. **The length:** 100 steps, then decide on 200 (default), or a fixed number (`--steps`).
