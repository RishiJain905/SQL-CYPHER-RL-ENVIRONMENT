# The two RL runs

Status (2026-11-01): step 1 (the sampling baseline) is done, step 2 (the method pilots) hasn't
started. This note lays out the two RL runs the plan calls stage A and stage B (**run 1** and
**run 2** here) plainly enough to argue with. It is not a decision document: where the plan
(`documentation/rl.md`) is silent, the note says so, and the open decisions end the page. Paper
claims were checked against the arXiv pages, with links inline; what couldn't be read is marked.

**In short.** Run 1 is the task as it is today: the model writes one query, an execution check
scores it, RL pushes it toward the answers that checked out. Run 2 is the task as a real agent has
it: the model sees what its query returned, may look around or retry for 3-4 turns, and is scored
on the final query. The baseline says run 1 has room (greedy 43.2% against pass@8 62.6%) and where
it doesn't (74 of 198 questions are never right in 8 tries). It says little about run 2, because
the baseline is single-turn.

## 1. Where we start

Released model (step 1,500 of `sft-main`, merged, Q8_0); 200 evalset questions, 198 with a decidable
answer; 8 sampled answers each at temperature 1.0 (`results/step1-baseline.md`):

| Set | Greedy pass@1 | Sampled mean@8 | pass@8 | Gap | Always | Sometimes | Never |
|---|---|---|---|---|---|---|---|
| **Overall** (198) | **43.2%** | 36.9% | **62.6%** | +19.4 | 30 | **94** | 74 |
| CypherBench (50) | 44.9% | 37.9% | 80.0% | **+35.1** | 5 | 35 | 10 |
| gretel (50) | 64.0% | 59.5% | 80.0% | +16.0 | 17 | 23 | 10 |
| SQaLe (49) | 42.2% | 34.3% | 57.1% | +14.9 | 6 | 22 | 21 |
| text2cypher (49) | 15.4% | 15.2% | 32.7% | +17.3 | 2 | 14 | **33** |

- **Buckets.** Of the 152 questions with all 8 answers decided, 44 were right 0 times, 30 right 8
  times, 78 in between. Methods that compare answers within a group (GRPO and kin, section 4)
  learn only from the in-between ones.
- **How the 879 wrong answers fail:** 613 ran and returned different rows, 194 errored at run
  time, 48 didn't parse. About a quarter are errors a retry could see.
- **Noise:** eight single-sample runs of the unchanged model scored 36.0% to 43.0%. One rate on
  200 questions is good to about ±3.5 points. Temperature 1.0 costs 6.3 points against greedy and
  96.5-98.0% of answers are still well-formed calls.
- **Test set** (1,000 questions): 43.7%, against 32.5% for the base model.

## 2. The two runs as planned

| | **Run 1: stage A, single turn** | **Run 2: stage B, multi-turn agent** |
|---|---|---|
| Goal | "Sometimes right" becomes "right first time" | Read a result, look around, retry |
| Model sees | Schema, up to two tested examples, the question | The same, then after each query the rows (capped) or the error |
| Tool | One read-only query tool, one call | The same tool, up to 3-4 calls |
| Databases | The checker's: gretel's real rows, SQaLe's and text2cypher's random fills, CypherBench's sampled Neo4j graphs | Live databases the model queries (the plan names gretel's rows, SQaLe's fills, CypherBench's graphs; text2cypher's are schema-built) |
| Reward | 1 if the query passes checker 4, else 0; small penalty for a malformed call | The same, from the final query only |
| Starts from | The released SFT model (new LoRA) | Run 1's adapter, after the gate |
| Compute guess | 15-20 min a step (32 questions × 8); 300 steps ≈ 3-4 days | 2-3× per step: roughly 6-12 days for 300 steps |

Both runs train a **new LoRA on the merged SFT model**, so the KL reference is the same model with
the adapter off and SFT stays intact. Rollouts come from llama-server (Q8_0 plus the adapter as a
GGUF LoRA); training is bf16 PEFT. Only training-split questions are used; the evalset and test set
never reach a rollout, reward or filter.

### Run 1: single turn, execution reward

```
question ─► llama-server (Q8_0 + RL adapter) ─► 8 sampled answers
         ─► sandboxed checker: run each query, compare rows with the reference's
         ─► reward 1 / 0 each ─► groups with a mix of right and wrong carry a gradient
         ─► server stops, trainer updates the LoRA ─► new GGUF adapter ─► next step
```

- **The task:** unchanged. The model thinks (budget 1,024 tokens), writes one query, never learns
  what it returned.
- **Checker 4:** SQL runs on gretel's real rows and on randomly filled copies (plus near-miss and
  rescue copies); Cypher on CypherBench's sampled graphs, or random graphs built from the schema
  for text2cypher. Rows compare as multisets, column order is free, ties at a final `LIMIT` count
  neither way. Answers no copy can decide are *undecided*.
- **Which questions:** the plan says the ones the model sometimes gets right. Two things it
  leaves open. The model **saw these training questions in SFT** with their reference queries, so
  the training split's "sometimes" pool probably differs from the evalset's (my inference; a probe
  of a few hundred training questions would show). And 300 steps × 32 questions visits ~9,600 of a
  ~95,000-question pool, so which ones is a real choice.
- **What it can move.** pass@1 can climb toward pass@8, not far past it: RL mostly reweights what
  the model can already produce ([Yue et al., 2025](https://arxiv.org/abs/2504.13837): RL-trained
  models gain at small k and lose to their base models at large k). Closing a quarter of the
  19.4-point gap is about +5 points, half about +10: scale, not a forecast. Per dataset the pass@8
  ceilings are CypherBench and gretel 80.0%, SQaLe 57.1%, text2cypher 32.7%. CypherBench has the
  most room, with a catch: **its training questions cover 4 graphs (art, biology, soccer,
  terrorist_attack), the evalset's cover 7 others**, so the +35 is measured on graphs RL never
  saw. The 74 "never" questions are out of reach of a group-comparison method; text2cypher's 33
  are mostly unverifiable or hopeless on random graphs.
- **Done looks like:** a step budget or a plateau; then step 1's measurement. Pass = pass@1 up,
  pass@8 not down, canaries steady.

### Run 2: multi-turn agent in real databases

```
question ─► turn 1: think, call query tool ─► sandbox runs it on the real database
         ─► rows (capped) or the error ─► turn 2: think, call again or answer ... (3-4 turns)
         ─► final query ─► checker ─► reward ─► trainer (loss on the model's own tokens only)
```

- **What changes:** the observation step. SFT never showed the model a tool result, so early
  trajectories are new territory. The release card lists this as a limitation, and SFT answers
  sometimes spend their one query looking around (`SELECT DISTINCT home_type FROM ...`), which a
  single-turn score marks wrong; a smoke run on the base model put exploration at ~8% of failures
  (11% on gretel). Tool output is context, masked from the loss, as in
  [Search-R1](https://arxiv.org/abs/2503.09516) ("retrieved token masking").
- **Why the databases matter.** gretel's rows and CypherBench's sampled graphs are real, so seeing
  them tells the model something. SQaLe's and text2cypher's checker databases are *random fills built
  from the schema*; looking at them teaches little about the question's real values. The plan
  doesn't say how those two get real data (SQaLe's populated databases exist upstream but weren't
  downloaded).
- **What it can move.** A retry is an extra attempt, so run 2 isn't capped by single-turn pass@8.
  The clean win is the ~27% of failures that are errors or parse failures. The 70% that ran and
  returned different rows are harder: rows reveal a wrong filter only if the result is empty,
  absurd or the wrong shape. Published multi-turn SQL RL reports gains from small data:
  [SkyRL-SQL](https://neurips.cc/virtual/2025/128018) (7B, ~600 training examples, up to +9.2
  points on five Spider benchmarks; I couldn't open its blog for the reward details) and
  [MTIR-SQL](https://arxiv.org/abs/2510.25510) (a 4B model, 64.4% on BIRD dev, GRPO modified, KL
  dropped). Expect gains to concentrate on gretel and CypherBench.
- **Not settled in the plan:** how the model ends (a plain reply, or a submit tool); which call
  is the answer (the last, presumably); per-turn thinking budget; row cap; query timeout (the
  checker's Neo4j allows 20 s, too long when a slot waits on it); context length. SFT trained at a
  4,096-token maximum, and four turns with up to 1,024 thinking tokens each can exceed that.
- **A claim to check.** The plan says a first-try right answer "is still the best outcome, so it
  doesn't unlearn stage A". With a reward on the final query only, first-try and third-try successes
  pay the same. It holds only with some pressure for early answers (section 3, E).
- **Done looks like:** a step budget or plateau, then two measurements: a multi-turn score against a
  multi-turn baseline of the *released* model (which doesn't exist yet), and the old single-turn
  score, to show nothing was unlearned.

### The measurement and the gate

Same questions and settings (temperature 1.0, 8 samples, checker 4) on the released model, each
pilot, after each run, and on the final model; the test set only confirms a finished stage.

- **The gate:** pass@8 must not fall by more than noise, by a question-by-question sign test. A
  drop means RL is narrowing the model rather than teaching it.
- **What the gate can see.** On 198 questions a sign test needs about 15 questions changed right
  against 5 changed wrong (p ≈ 0.04), a net swing of 10, about 5 points. Pilots will often differ
  by less. Per-question mean right-rates over all ~1,450 decided samples are more sensitive than
  greedy.
- **pass@8 is a ceiling at 8 tries:** an RL model can improve at 8 and look unchanged at 64.

## 3. Alternatives worth discussing

**A. Run 2 only (multi-turn from the start).** Pro: one run; single-turn is the special case of
answering on turn 1; multi-turn SQL RL is published as a direct recipe. Con: the cold start meets a
model that has never seen a tool result, with no single-turn gain banked and costlier steps; it
loses the cheap, clean run-1 check that the pipeline works.

**B. Interleave single- and multi-turn.** Each batch mixes `max_turns=1` and `max_turns=4`
questions (or lets the model stop after turn 1). Pro: one adapter, no handover, single-turn skill
keeps being rewarded. Con: more moving parts, uneven step times, harder attribution.

**C. Per-dataset curricula.** E.g. CypherBench first (biggest gap, cleanest reward), gretel for real
rows, SQaLe and text2cypher later or only in run 1. Pro: compute goes where there is room and a
trustworthy reward. Con: can overfit one dataset's habits (CypherBench is template-like, on 4 graphs).

**D. Run 1 as a short warm-up.** The pilots are stage-A runs: continue the winner briefly and put
the budget into run 2. Pro: no repeated work; run 1's gains are capped by pass@8 anyway. Con: gives
up the large single-turn gap (CypherBench +35) if run 2 can't exploit it.

**E. Reward shapes for run 2.**
- *Final query only* (the plan): simplest, nothing to game; no pull toward early answers.
- *Per-turn cost* (say reward × 0.9 per extra turn): makes first-try best, as the plan intends; too
  large and the model stops looking. A *first-try bonus* does the same and is easier to log.
- *A small score for a query that runs.* [Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315)
  paid 1 for a match, 0.1 for valid executable SQL, 0 otherwise, and reports that finer-grained
  rewards produced "lazy" behaviour. The plan's malformed-call penalty is close to this.

**F. A cold start for run 2.** One round of rejection-sampling fine-tuning first: run the model as a
multi-turn agent, keep trajectories that end right, train on them (as the trace stage did for
thinking). Pro: the model learns the observation format before RL spends rollouts on it. Con:
another stage and dataset to keep off the evalset.

**G. Different methods per run.** A GRPO-family method for run 1; SDPO or SRPO in run 2, where their
feedback is natural (section 4). Con: changes the method and the task at once.

**My recommendation (mine, not the plan's).** Keep two runs, overlapped: (1) treat the pilots as the
first slice of run 1 and continue the winner; (2) measure the released model as a multi-turn agent
*before* building run 2's trainer, which shows how much of the error-type failures a retry fixes
with no training; (3) in run 2, mix single- and multi-turn episodes with a small per-turn cost, and
give it observations only where the data is real (gretel, CypherBench); (4) add a rejection-sampling
cold start only if the zero-shot multi-turn run is clearly broken.

## 4. The methods

Notation: for a question, sample *G* answers *o₁..o_G* from the policy, score them with the checker
for rewards *r_i* (0 or 1). *π_θ* is the policy being trained, *π_old* the one that produced the
samples, *π_ref* the SFT model (adapter off).

**Why not the classics.** *PPO* needs a critic, a value network about the policy's size, which
doesn't fit next to a 4B model on 16 GB; [DeepSeekMath](https://arxiv.org/abs/2402.03300) drops it
for the same reason, and Arctic-Text2SQL-R1 found GRPO ahead of PPO (64.9 vs 63.0 BIRD dev, with the
caveat that PPO may need more tuning). *RLHF* with a learned reward model is unneeded when the
checker verifies answers. *DPO* (offline preference pairs) could be built from right and wrong
samples with no rollouts in the loop, but isn't on-policy and uses the checker's signal indirectly.
A cheap control is **rejection sampling** (train on the right samples); it isn't in the pilot list.

**GRPO** (DeepSeekMath, 2024). No critic: the baseline is the group's own average.

> A_i = (r_i − mean(r)) / std(r), the same for every token of answer *i*
> loss = −(1/G) Σ_i (1/|o_i|) Σ_t min(ρ_it A_i, clip(ρ_it, 1−ε, 1+ε) A_i) + β·KL(π_θ ‖ π_ref)

ρ is the per-token ratio π_θ/π_old; the paper used β = 0.04, KL in the loss. An answer is good when
it beats its siblings; a group all right or all wrong has zero advantage and teaches nothing. Suits
run 1 directly.

**Dr. GRPO** ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)) removes two terms GRPO added:
the 1/|o_i| (long wrong answers cost less per token, so they drift longer) and the division by std
(questions with nearly all-right or all-wrong groups get bigger weights). The advantage is just
*r_i − mean(r)*, and the sum is divided by a constant such as the token budget.

**DAPO** ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) makes four changes: **clip-higher**
(ε_low 0.2, ε_high 0.28, so unlikely tokens can rise and entropy doesn't collapse); **dynamic
sampling** (drop all-right and all-wrong groups, keep sampling until the batch is informative);
**token-level loss** (divide by the batch's tokens, not per answer); **overlong shaping** (a soft
length penalty near the cap). It also **drops the KL term**. On a 32B math model the ablation climbs
from 30 (naive GRPO) to 50 on AIME 2024. Dynamic sampling matters most here: rollouts go only where
there is a gradient.

**GSPO** ([Zheng et al., Qwen, 2025](https://arxiv.org/abs/2507.18071)) uses one ratio per whole
answer instead of per token:

> s_i = (π_θ(o_i | q) / π_old(o_i | q))^(1/|o_i|), clipped at a tiny ε (3e-4 / 4e-4 in their MoE runs)

The argument: the reward belongs to the sequence, and per-token ratios, each from one sample, add
noise that accumulates. It is also more tolerant of small likelihood differences between rollout
engine and trainer, which matters here (section 5).

**SDPO** ([Hübotter et al., ICML 2026](https://arxiv.org/abs/2601.20802); arXiv title "Reinforcement
Learning via Self-Distillation"). GRPO's signal is one number per answer. SDPO uses the *text* the
environment returns (an error, a failed test). The same model, shown that feedback in its prompt,
becomes a self-teacher, and the policy (without the feedback) is pulled toward the teacher's
token predictions:

> loss = Σ_t KL( π_θ(· | q, o_<t) ‖ stopgrad π_θ(· | q, feedback, o_<t) )

The teacher's prompt holds the question, the feedback and, if one exists, a correct answer from
the same group. Credit is dense: tokens the teacher likes more than the student get pushed up. In
the paper (Qwen3-8B, 8 rollouts, temperature 1.0): 48.8% vs 41.2% for GRPO on LiveCodeBench v6,
reaching GRPO's final accuracy in 4× fewer generations, and answers ~3× shorter on its science
tasks. The implementation uses a Jensen-Shannon divergence and an EMA or trust-region teacher for
stability, and an extra forward pass. Caveats from the paper: it needs a model strong enough to
use in-context feedback (it lost to GRPO on a 1.5B model); misleading feedback hurts; and
**agentic and long-horizon use is listed as untested**. For us the question is what counts as
feedback: the model's own error or rows, plus a correct sibling, are fair; the *reference's* rows
would leak an answer the deployed model never sees.

**SRPO** ([Li et al., 2026](https://arxiv.org/abs/2604.02288)). The authors say SDPO learns fast,
then often collapses in long training, and blame two things: distilling on answers already right is
ambiguous, and the self-teacher gets less reliable over time. SRPO **routes** each sample: right
ones to GRPO's reward update, failed ones to SDPO's correction, with distillation targets weighted
down when the model is uncertain (entropy-aware). Reported over five benchmarks with Qwen3-8B: +3.4%
over GRPO, +6.3% over SDPO, up to 17.2% less compute per step. Read from the abstract only; a
very recent preprint.

| Method | Needs | Good for | Watch |
|---|---|---|---|
| GRPO / Dr. GRPO / DAPO / GSPO | the 0/1 reward | run 1, the safe default | zero gradient on "never" and "always"; entropy collapse |
| SDPO | feedback text | the 74 "never" questions via error text; run 2, where the model sees errors anyway | collapse; untested multi-turn; 4B scale |
| SRPO | feedback and reward | the same, with a collapse guard | newest, least reproduced |

My reading of SDPO in run 2 (not a published result): turn 2's model-with-feedback is already a
natural teacher for turn 1, which is the "right the first time" effect run 2 wants. The pilots are
single-turn, so they test run 1's side only.

## 5. RL facts that matter here

**On-policy, and the rollout/trainer mismatch.** The maths assumes samples come from the policy being
updated. Here two programs are involved: rollouts from llama.cpp (Q8_0 plus a GGUF LoRA) and
training in bf16 PEFT in transformers, which define slightly different distributions. The release
measured the quantisation part: Q8_0 against BF16 has a KL divergence of 0.0005 and the same top
token 99.45% of the time. Small, but the same kind of gap between other stacks (vLLM against FSDP)
has been shown to make "on-policy" RL quietly off-policy
([Yao, Liu et al., 2025](https://liyuanlucasliu.github.io/publication/feng-off-2025), which proposes an
importance-sampling fix; I couldn't read the full text). Practical consequences: recompute
log-probs in the trainer, not the server; take one update per rollout batch so the ratio clip
rarely fires; log the mean server-vs-trainer log-prob gap as a canary; GSPO's sequence ratio
tolerates this better. Also check how the thinking budget's forced end-of-thought is handled: those
tokens weren't sampled and shouldn't carry loss.

**Group size and informative groups.** A group teaches only if it holds a right and a wrong answer.
If a question's success chance is *p*, an 8-group is informative with probability 1 − p⁸ − (1−p)⁸:
99% at p = 0.5, 66% at 0.125, 57% at 0.1, 15% at 0.02. A baseline "never" may be rare-but-possible,
and larger groups reach further (Arctic-Text2SQL-R1 used 16 and saw no gain from 24 or 32). Eight
matches the baseline's samples. Log the informative share: it is the run's real throughput.

**Advantage normalisation.** With one right answer in eight, GRPO's advantages are about +2.5 for
the right one and −0.35 for each wrong one (the exact values depend on how std is computed); without
the std division (Dr. GRPO), +0.88 and −0.13. Dividing by std weights the hardest and easiest
questions more; whether that helps or only adds variance is something the pilots can show.

**KL to the SFT reference, free here.** The reference is the policy with its RL adapter off: no
second copy in memory, just an extra no-grad forward pass per training sequence. Published choices
run from β = 0.04 (DeepSeekMath) to 0.001 (Arctic-Text2SQL-R1) to none (DAPO, MTIR-SQL). A small
β is cheap insurance for a low-rank adapter and keeps the output format near SFT's.

**Entropy collapse.** RL tends to sharpen the next-token distribution until the model stops
exploring ([Cui et al., 2025](https://arxiv.org/abs/2505.22617)). Signs: mean token entropy falling
fast; a group's eight answers becoming near-identical (so the informative share drops); pass@1 up
while pass@8 is flat or down. Defences: clip-higher, dynamic sampling, KL, a low learning rate,
stopping early. Log distinct queries per group beside entropy.

**Reward hacking for SQL and Cypher.** The repo's rule is execution-only reward, and any sign of
these is a finding:
- *`DISTINCT` everywhere:* passes when the check copies hold few duplicates. Canary: share of
  rollouts using `DISTINCT`.
- *`SELECT *` and extra columns:* the strict match fails extra columns, but on gretel's one-to-
  three-table schemas `*` can equal the answer. Track columns returned.
- *Empty results matching empty references:* a copy where the reference returns nothing (or only
  zeros and NULLs) proves nothing and isn't counted; the evalset drops such questions; apply the same
  filter to training questions.
- *Format tricks:* several calls in a reply, no call, a call placed inside the thought. The
  parser's strictness is the defence.
- *Overfitting to the checker's copies:* if a question's databases are identical on every visit, a
  query tuned to them can pass without being right. Fresh seeds per visit, and different copies
  for evaluation, test for it.
- *In run 2, the sandbox itself:* read-only connections, timeouts, row caps, no network, scratch
  copies. A model that can run `ATTACH`, `PRAGMA` or `LOAD CSV` has more to try.

Read the highest-reward and most-changed samples by hand each time; counters miss new tricks.

**Undecided answers.** Baseline scoring counts them neither way (148 of 1,600, 97 of them
text2cypher). In training it matters: counting them wrong punishes right answers on unverifiable
questions, counting them right rewards anything. Options: remove them from the group, or drop
questions with too few decided answers. About half of text2cypher's questions can't be decided at
all.

**Thinking length.** The 1,024 budget stays capped, so thoughts can't grow forever, but can pile up
at the cap or shrink to nothing. Arctic-Text2SQL-R1 saw a U-shape (shorter, then longer as accuracy
rose); SDPO's answers were ~3× shorter than GRPO's; Dr. GRPO's critique is that GRPO inflates wrong
answers. Canaries: median thought length, the share cut by the budget (1 of 1,600 baseline answers
hit the token cap), thinking-off calls, the "can't answer, say so" rate.

**LoRA for RL.** [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines,
Sept 2025) reports LoRA matching full fine-tuning in RL even at rank 1, which the authors argue is
because policy gradients carry about one bit per episode, and recommends LoRA on all layers, MLP
included. Their RL experiments were math-style on Llama and Qwen, not ours. The SFT adapter is rank
16, alpha 32 on every linear layer; reusing that keeps the GGUF conversion path proven, and a lower
rank is an option. **Learning rate:** they find the best LoRA rate is about 10× the full
fine-tune rate. Full fine-tune RL papers use around 1e-6 (SDPO's feedback runs, GRPO's default in
its comparison), which points to roughly 1e-5 here against SFT's 1e-4 on the same adapter: my
reading, to be settled by the pilots.

**Evaluation and noise.** Since eight unchanged samples spanned 7 points, a method difference under
~4 points is noise on 198 questions. RL runs also vary by seed and a 16 GB run takes days, so each
pilot will be one seed: treat rankings between close methods as weak evidence.

**On 16 GB.** llama-server (~6 GB with 16 slots) and the trainer (~10-12 GB) can't both be resident,
so each step is phases: generate and score; stop the server; train; convert the adapter; restart.
From the SFT run: GPU memory sat near 15.5 GB and the trainer crashed in WSL's GPU runtime; the fix
was capping PyTorch at 80% of the card and tuning the allocator (the largest step needed 12.4 GB).
Other risks: system RAM pressure as weights move between RAM and GPU each step; the checker's Neo4j
and Docker on the same machine; the Ctrl+C that ended the baseline's jobs twice (R3: start long
jobs inside WSL, detached); run 2's longer sequences; and the need for a checkpoint/PAUSE protocol
like SFT's. On the Modal H100 fallback the same loop talks to vLLM instead of llama.cpp; no cost
estimate exists yet, and any spend needs Rishi's OK.

## 6. Open questions

1. **Two runs, or another structure?** Run 1 then run 2, multi-turn only, or interleaved
   (section 3, A and B)?
2. **How big is run 1?** A full ~300-step stage, or the pilots plus a short continuation of the
   winner (section 3, D)?
3. **Which training questions?** Probe a few hundred training questions with the SFT model first
   (it saw them in SFT) and take the informative pool? How should the four datasets be weighted,
   given CypherBench's 4 training graphs against 7 evaluation graphs and text2cypher's weak reward?
4. **Undecided answers in training:** drop them from the group, count them wrong, or pre-filter
   questions whose reference can't be decided?
5. **Run 2's databases:** real observations for gretel and CypherBench only, or also download
   SQaLe's populated databases? Does text2cypher stay in run 2 on random graphs? Should the
   exploration database always be the one the reward is computed on?
6. **Run 2's protocol:** how does the model finish (plain reply or a submit tool); 3 or 4 turns;
   row and error-text caps; query timeout; per-turn thinking budget; how to stay inside the
   trainer's 4,096 tokens (e.g. drop earlier turns' thoughts from the context)?
7. **Run 2's reward:** final query only, a per-turn cost, a first-try bonus, or a small score for a
   query that runs? How strongly should the first-try answer be favoured?
8. **Run 2's cold start:** straight RL from run 1's adapter, or a short rejection-sampling round on
   multi-turn trajectories first?
9. **Run 2's yardstick:** add a multi-turn baseline of the released model (a "step 1b") and a
   multi-turn pass@k? Does the gate use single-turn or multi-turn pass@8?
10. **One method or two:** the same winner in both runs, or a GRPO-family method for run 1 and SDPO
    or SRPO for run 2? If SDPO, what counts as feedback (the model's own error or rows plus a correct
    sibling; never the reference's rows)?
11. **Are the pilots powered?** Judge them on per-question mean@8 rather than greedy? Add a
    rejection-sampling run as a control? Accept one seed per method?
12. **KL penalty:** on (0.04, 0.001) or off, given the adapter-off reference is free?
13. **Compute and money:** which parts run locally (the pilots will time a step), and what budget
    and approval rule applies to Modal H100 time?
14. **Stopping and canary rules:** step budget, plateau test, and the thresholds (entropy, thought
    length, informative-group share) that stop a run?
15. **Stage C** (the NFL harness behind Ask the Engine's guard): still optional, decided after run 2?

## Sources

[DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO) · [Dr. GRPO](https://arxiv.org/abs/2503.20783) ·
[DAPO](https://arxiv.org/abs/2503.14476) · [GSPO](https://arxiv.org/abs/2507.18071) ·
[SDPO](https://arxiv.org/abs/2601.20802) ([ICML 2026 proceedings](https://proceedings.mlr.press/v306/hubotter26a.html)) ·
[SRPO](https://arxiv.org/abs/2604.02288) · [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) ·
[Yue et al.](https://arxiv.org/abs/2504.13837) · [Cui et al.](https://arxiv.org/abs/2505.22617) ·
[Yao, Liu et al.](https://liyuanlucasliu.github.io/publication/feng-off-2025) ·
[Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315) · [MTIR-SQL](https://arxiv.org/abs/2510.25510) ·
[SkyRL-SQL](https://neurips.cc/virtual/2025/128018) · [Search-R1](https://arxiv.org/abs/2503.09516).
In this repo: `documentation/rl.md`, `results/step1-baseline.md`, `guides/sampling-baseline.md`,
`decisions.md`; the trainer's
[`docs/release.md`](https://github.com/RishiJain905/nfl-ask-trainer/blob/main/docs/release.md) and
[`docs/real-run.md`](https://github.com/RishiJain905/nfl-ask-trainer/blob/main/docs/real-run.md).
