# Run 3, phase 2: the RL run (2026-11-24 / 25)

Run by the rules in `../run3/phase2.md` (SRPO core, start and KL reference = phase 1's pass-1
checkpoint 129, the malformed and wrong-database gates waived by Rishi). Colab A100 40 GB, rollouts
on llama.cpp b11779 with CUDA, the update in transformers + PEFT on the same VM; the PC pulled
every step (adapters hash-checked), logged W&B (`sql-cypher-rl`, group `run3`, run `run3-rl`) and
ran the held-out checks.

**Result: no gain. Stopped by Rishi after step 21; the start checkpoint stays the best (R33).**

## What ran

- 21 steps, 4.25 hours of training (about 12 minutes a step: 2-4 minutes of rollouts, about 9 of
  update on about 630 examples). 16 questions x 8 samples a step.
- Paused once, after step 6, by the wrong-database stop rule read against step 1's single-question
  reading; resumed with the rule read against the start's dev reading (R31).
- Extended at Rishi's request to 30-40 steps (R32); stopped at his request after the step-20
  check (R33). The VM was stopped 2026-11-25 03:46 UTC.

## Training steps

| Step | Reward | Solve | Malformed | KL | Seconds |
|---|---|---|---|---|---|
| 1 | .538 | .518 | .125 | .00009 | 844 |
| 2 | .561 | .527 | .102 | .00019 | 521 |
| 3 | .486 | .455 | .109 | .00022 | 759 |
| 4 | .472 | .491 | .078 | .00020 | 853 |
| 5 | .536 | .491 | .086 | .00029 | 702 |
| 6 | .560 | .606 | .133 | .00039 | 749 |
| 7 | .484 | .473 | .078 | .00054 | 681 |
| 8 | .526 | .500 | .102 | .00049 | 678 |
| 9 | .436 | .429 | .242 | .00058 | 948 |
| 10 | .588 | .571 | .117 | .00061 | 900 |
| 11 | .366 | .298 | .192 | .00073 | 810 |
| 12 | .393 | .375 | .133 | .00082 | 796 |
| 13 | .687 | .664 | .042 | .00087 | 438 |
| 14 | .622 | .625 | .070 | .00078 | 620 |
| 15 | .507 | .509 | .109 | .00080 | 774 |
| 16 | .393 | .330 | .109 | .00076 | 820 |
| 17 | .291 | .289 | .167 | .00070 | 827 |
| 18 | .536 | .500 | .094 | .00065 | 687 |
| 19 | .599 | .571 | .094 | .00072 | 622 |
| 20 | .468 | .429 | .109 | .00074 | 656 |
| 21 | .542 | .509 | .047 | .00086 | 630 |

The reward has no trend: it follows which questions a step draws (steps 11 and 17 drew mostly hard
BIRD DuckDB / Postgres / MySQL groups and 0-for-16 Cypher groups; step 13 drew easy ones). Every
malformed episode checked (steps 6-9) hit the 8-call limit. The KL stayed under 0.001: the policy
barely moved.

## Held-out checks (on the PC, greedy)

| Checkpoint | Dev set (300) | vs start | Eval set | Eval right (of 200) | vs start |
|---|---|---|---|---|---|
| Start (ckpt-0129) | 46.4% | | 64.2 | 124 | |
| Step 5 | 43.2% | 6 / 13, p 0.17 | 62.2 | 120 | 1 / 5, p 0.22 |
| Step 6 | 43.2% | 6 / 13, p 0.17 | 62.7 | 121 | 1 / 4, p 0.38 |
| Step 10 | 44.1% | 7 / 12, p 0.36 | 62.7 | 121 | 1 / 4, p 0.38 |
| Step 15 | 44.1% | 6 / 11, p 0.33 | 60.1 | 116 | 1 / 9, **p 0.02** |
| Step 20 | 44.1% | 7 / 12, p 0.36 | 61.1 | 118 | 2 / 8, p 0.11 |

(better / worse questions, sign test.) Steps 5 and 6 tie at 43.2% but differ on 12 questions. At
step 15, 4 of the 9 newly wrong eval-set answers were CypherBench, 3 gretel; almost all ran and
returned the wrong rows.

## Reading

Nothing beat the start, on either set, at any check. With the KL under 0.001 after 21 steps the
update is too weak to move the policy within a Colab budget, and what it did move went slightly
the wrong way on the eval set. More steps of the same setup were judged not worth the compute.

## Kept

Nothing new: phase 1's pass-1 checkpoint 129 stays the best run-3 model. Steps 1-21's adapters
and optimiser states are on the data drive (`<data>/rl/run3/phase2/remote/step-00NN`), with the
episodes, metrics, events and evaluations.

## Cost

About 4.3 hours of training plus setup, the idle pause after step 6 (about 35 minutes) and the
evaluations' time on the VM side: roughly 6 hours of A100 40 GB at about 8.2 units an hour.

## If run 3 is tried again

Change the update, not the length: a higher learning rate or a weaker KL weight (the policy hardly
moved), and a fixed per-step mix of question difficulty so the training reward can be read across
steps. Fix the BIRD DuckDB group whose gold fails on the VM (8 skipped episodes in several steps).
