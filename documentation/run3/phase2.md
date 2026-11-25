# Run 3, phase 2: the RL run, its settings and its stop rules

**Written 2026-11-24, before any phase 2 code** (Rishi's approvals the same day: start from the
best warm-start checkpoint, the two failing gates waived, the KL reference as suggested, build the
RL environment and start the run on Colab when ready). Nothing below may change after its numbers
are seen without a new decision saying why.

## Start, reference, gates

- **Start:** the warm start's pass 1 checkpoint 129 (`phase1/ckpt-0129`): run 2's evaluation set
  64.2% (step 40: 59.4%), the run-3 dev set 46.4% (42.7%). One adapter, step 40's, trained on
  (R25), never merged into bf16 (design 5.1).
- **KL reference:** the start checkpoint itself (design 5.1 said step 40, 5.3 the warm-start
  model; Rishi chose the start), a frozen copy of its adapter.
- **Waived gates** (Rishi): malformed endings <= 5% (the start: 11.3%) and wrong-database
  declines >= 80% (55%). RL's reward is aimed at both; the stop rules below watch them.

## The method (design 5.3, as built)

- A group is one question on one engine, **8 episodes** at temperature 1.0 (thought budget
  1,024, answer part 2,048, run 2's), the setting drawn once a group; the reward is design 4's
  (`episode.grade`).
- **Advantage** = (reward - group mean) / max(group std, 0.25) (run 2's SRPO advantage); groups
  where every episode scored the same give no gradient.
- **Loss:** every turn of an episode is one example (the prompt as the server rendered it, the
  model's own output as target, as the warm start), all its tokens sharing the episode's
  advantage, token mean over the step. Tokens the model did not choose (a `</think>` closed for
  it at the thought budget) are not trained. **KL penalty** to the reference, weight 0.20 (run
  2's from step 12), the k3 estimate on the sampled tokens.
- **Not ported from run 2, said here so it is not silent (R29):** SRPO's distillation branch (a
  wrong episode with a right sibling distilled toward an EMA teacher shown that sibling's query)
  and the EMA teacher. Run 2's pilots had GRPO and SRPO within noise (R4); the branch assumes one
  submitted query to show, which run 3's multi-turn episodes and non-answers don't map onto
  cleanly. Ambiguous (clarify) questions don't exist yet (they need the teacher, design 3.4): run
  3's RL has none.
- AdamW, lr 5e-6 (run 2's), gradient clip 1.0, sequences over 3,840 tokens left out and counted.

## The questions a step draws

Training split only (`splits.trainable`, asserted by the driver); the held-out databases are not
in the VM's bundle at all. Per step, by share of groups:

- 55%: phase 0's useful-band groups on BIRD (1-7 of 8 right for step 40: 389 over the four SQL
  engines), each drawn once before any repeats;
- 20%: BIRD training questions phase 0 never measured, on the bundle's databases and a random
  engine (the run is longer than 737 useful groups; the warm start moved the band);
- 15%: one CypherBench training graph a step, in turn, its useful-band groups first;
- 10%: non-answers (write requests and wrong-database questions), half each.

The bundle carries the BIRD databases richest in useful groups per MB (up to about 500 MB), the
four CypherBench training graphs and the gold checks. The questions a step (Q) are set from step
1's time so the run fills `max_hours`.

## Budget

Colab's run-3 cap is 6 hours (R24); the smokes used about 1. **`max_hours` = 4.0** of RL after
setup (about 30 minutes); the driver saves and stops there whatever step it is on, and the VM is
stopped by a watchdog on the PC at 4 h 45 m after it attaches.

## Before the first rollout (the smoke is the first steps)

The served start checkpoint on the VM answers the run-3 dev set's 300 triples greedy and must
agree with the PC's reading (46.4%) within 3 points; a failure stops the VM. Step 1's metrics are
the baseline every relative stop rule reads.

## Stop rules (5-step means against step 1's reading, from step 6)

| Watch | Stops when |
|---|---|
| Malformed endings (waived gate) | above step 1 + 0.10, or above 0.30 |
| False declines on answerable questions | above max(3 x step 1, 0.10) |
| Wrong-database declines (waived gate) | below the start's dev reading (0.55) - 0.15 (R31; was step 1 - 0.15) |
| KL to the reference (per token) | above 0.10 |
| Episode length (generated tokens) | above 2 x step 1 |
| Constants caught by the hard-coding guard | above 0.05 of right answers |
| Mean reward | below step 1 - 0.15 |

A stop saves the step and pauses for Rishi.

## Evaluations and keeping a step

Every 10 steps the PC (idle while Colab trains) merges the latest adapter in float32 into Q8_0
and runs the run-3 dev set and run 2's evaluation set greedy (`scripts/run3/phase1.py`'s
`evaluate`). **The kept step is the best greedy dev-set score** that is not worse than the start
on run 2's evaluation set by the sign test. The held-out databases, BIRD dev and CypherBench test
are read once, on the kept step, in phase 3.

## Tracking

W&B (project `sql-cypher-rl`, group `run3`, run `run3-rl`, `launched-by:agent`) is logged **from
the PC** (WSL's stored login) from the metrics the watcher pulls each step, so no key goes to the
VM; its URL is checked before step 1 is called tracked. A new dashboard ("Run 3 RL") is live and
stub-tested before step 1.
