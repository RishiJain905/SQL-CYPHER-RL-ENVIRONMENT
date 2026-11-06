# Run 1 design (stage A, single turn): draft

**Status:** draft for Rishi, 2026-11-05. Nothing here is built or run yet. Each section ends with
a recommendation; the ones marked **Your call** are listed together at the end. Run 2 gets its own
design once this one is settled (decisions R5).

Run 1's job (`rl-runs.md` section 2): turn "sometimes right" into "right first time", with one
query per answer and an execution reward. It starts from the released SFT model with a new LoRA,
like the pilots.

## 1. The environment (R5: a sandbox, with a faithful prompt)

The model never sees what its query returns, so the environment is the prompt plus the reward.

**What the model reads must be deployment's:**
- **The prompt:** the tool-call prompt SFT trained on, which the pilots and the yardstick already
  use (the trainer's `tool_prompt`: schema, up to two tested examples, the question), with the
  release's chat template rendered by llama-server itself (`/apply-template`).
- **To check before the run:** the model card shows this exact system prompt and tool schema. If
  it doesn't, add it, so people who run the model send what it was trained on.
- **Thinking level:** `high` everywhere, as in the pilots and the yardstick (all 200 evalset
  questions are `high`). Mixing levels would make the model robust to a user's choice, but it
  changes what the yardstick measures. Recommendation: keep `high` for run 1.
- **Thinking budget:** 1,024 tokens, then a forced `</think>` that carries no loss (as in the
  pilots; 4-16% of answers hit it).

**What decides the reward:** checker 4 (`publiceval.verify`), imported unchanged.
- **Rewards as piloted:** 1 right, 0 wrong, -0.1 for no parsable call; an undecided answer is left
  out of its group and the loss.
- **Fixed copies.** The checker's filled databases use fixed seeds, so a question sees the same
  copies on every visit, and a query tuned to them could pass without being right. Low risk here:
  100 steps visit 3,200 questions of a ~95,000 pool. Rule: no question is visited more than twice
  in run 1.
- **Questions that can't teach:** drop training questions whose reference returns nothing (or only
  zeros and NULLs) on the checker's copies, and those whose reference the checker can't decide.
  The evalset already drops both kinds.
- **Hacking canaries** (logged every step, from `rl-runs.md` section 5): share of rollouts using
  `DISTINCT`, using `SELECT *`, columns returned against the reference's, several calls in one
  reply. Plus reading the step's highest-reward and most-changed samples at each evaluation.

The infrastructure stays as piloted: llama-server on Windows, one WSL process per update, the
checker's Neo4j in Docker (training refuses to run without it), and `pause.py`.

## 2. The questions

**What the pilots showed:** 38-63% of groups carried a signal (mean about 50%). The rest were
all right or all wrong and cost rollout time only (about half of the ~8 rollout minutes per step).

- **The pool:** a fresh `pool-v2` from the training split, the filters of section 1 applied,
  evalset and test questions excluded by id and text as before. The pilots' 320 questions can
  stay in it.
- **No probe pass.** Probing 2,000 questions × 8 samples takes about as long as the rollout time
  it would save over 100 steps. Groups with no signal are skipped (as in the pilots) and the
  informative share is logged.
- **The mix per step of 32** is a real choice. The pilots used 8 per dataset. CypherBench has the
  most room (pass@8 80 against greedy 45) but trains on 4 graphs and is measured on 7 others;
  text2cypher's reward is the weakest (random graphs, half its questions undecidable).
  Recommendation: **10 CypherBench, 10 gretel, 8 SQaLe, 4 text2cypher** (decidable ones only).
  **Your call.**

## 3. The method

**SRPO, as piloted** (decisions R4): routed per rollout, an EMA teacher at 0.05, the right
sibling's query as the correct solution, top-100 JSD, entropy weights, token-mean loss. It led
on every sampled measure; none of it was outside noise.

- **Learning rate.** GRPO barely moved in 10 steps (KL to SFT 0.0002). SRPO's update norm was
  about 12× GRPO's at the same 1e-5 (0.06 against 0.005 per step), so it already moves much
  more. Recommendation: **keep 1e-5** with 5 warm-up steps, and raise it only if the 25-step
  evaluation shows no movement. The alternative is 2e-5 from the start. **Your call.**
- **Log KL to SFT even with no KL penalty.** SRPO has no KL term, so the pilots couldn't see how
  far it drifted. The reference is free (the adapter switched off), so log it as a canary.
- **Correct the rollout/trainer gap.** The mismatch between the server's and the trainer's
  log-probabilities grew from 0.013 to 0.031 a token over SRPO's 10 steps (GRPO: 0.014 to 0.019).
  Over 100 steps that can make the "on-policy" update quietly off-policy. Recommendation: weight
  each token by the trainer/server probability ratio, capped at 2 (truncated importance
  sampling, as Yao, Liu et al. propose). The server's log-probs are already collected, so this
  costs nothing. Stop the run if the mismatch passes 0.05.

## 4. Length, evaluations and the gate

- **100 steps first** (about 26 hours of GPU at the pilots' ~15.5 min a step, plus pauses),
  evaluated at steps 25, 50 and 100 on step 1's yardstick (evalset, temperature 1.0, 8 samples,
  checker 4, plus greedy).
- **Extend to 200 steps** only if mean@8 is still rising at 100 and the gate holds. The original
  plan's 300 steps stays the ceiling. **Your call** (100 then decide, or a fixed number now).
- **The gate:** pass@8 not down by a question-by-question sign test. A significant drop stops the
  run and keeps the last good adapter.
- **The test set** is used once, on the adapter we keep.

## 5. Stop rules (any one pauses the run for a look)

| Canary | Pilot values | Stop when |
|---|---|---|
| Mean token entropy | 0.43-0.53 | below 0.22 (half) |
| Informative share | 0.38-0.63 | below 0.25 for 5 steps in a row |
| Malformed calls | 0-3.5% | above 5% for 3 steps |
| Rollout/trainer mismatch | 0.013-0.031 | above 0.05 |
| KL to SFT | (not logged for SRPO) | to be set from the first 10 steps |
| Response length | 295-439 tokens | median changes by more than 2× |
| `DISTINCT` share | to log | up 10 points from step 1 |

## 6. Where the code goes

Rishi's rule: each step's code in its own folder, and pilots don't pollute the runs.
Recommendation: move the tested, general pieces (rollouts, rewards, losses, log-probs, SRPO,
pause) out of `step2_pilots/` into a shared `src/sqlrl/rl/`; the pilots import them from there,
with behaviour unchanged (the 22 tests prove it). Run 1 lives in `src/sqlrl/run1/` (its config,
pool builder and orchestrator) and `scripts/run1/`, with runs under `<data>/run1/`.

## 7. Decisions for Rishi

1. **The dataset mix:** 10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher, or the pilots' even 8s?
2. **The learning rate:** 1e-5 with warm-up, raised only if step 25 shows no movement; or 2e-5?
3. **The length:** 100 steps, then decide on 200; or a fixed number now?
4. **Agreed as recommended unless you say otherwise:** `high` thinking, the pool filters, no probe,
   KL logged, the importance-sampling correction, the stop rules, the code layout.
