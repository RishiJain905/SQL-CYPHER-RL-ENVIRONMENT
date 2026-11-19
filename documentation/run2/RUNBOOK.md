# Run 2 runbook

How to start, watch, pause and close run 2. Read `design.md` first: it says what the run is.
Run 1b's runbook (`run1b/RUNBOOK.md`) is the model for this one; what differs is marked.

**Run 2 is not started by an agent on its own.** Rishi says when. Phase 0 and the loop test are
done; their results are in `results/run2-phase0.md`.

**As of 2026-11-19 the setup is option C** (decisions R17): gretel and CypherBench with turns,
SQaLe and text2cypher with one query, 3 of a step's 10 gretel questions from the returned
file. Its no-training tests, its loop test and its start reference are done
(`results/run2-phase0.md`, section 10). The go is still Rishi's. If the look-around data or
the protocol changes, raise `env.LOOK_VERSION` where it applies and make the start reference
again; the run refuses a reference made under another protocol.

## 0. Before starting

Run from the repo root on Windows. Stop and tell Rishi if a line fails and the fix isn't listed.

| Check | How | Expect |
|---|---|---|
| The GPU is free | `tasklist /FI "IMAGENAME eq llama-server.exe" /NH` and, in WSL, `pgrep -af "sqlrl\|train"` | nothing. If something of Rishi's is on the GPU, ask him |
| Docker and the checker's Neo4j | `docker ps` | `asktrainer-neo4j` is up. It has no restart policy: after a reboot, `docker start asktrainer-neo4j` |
| The start adapter | `<models>/run1b-step30/lora_state.safetensors` and `ema_state.safetensors` | both there |
| The database scripts | `<data>/run2/pool-v2.db.jsonl`, `<data>/run2/evalset.db.jsonl` | both there (`python -m sqlrl.run2.pool build-db`, in WSL, rebuilds them) |
| The returned questions | `<data>/run2/pool-v2.returned.jsonl` (604 lines) and `pool-v2.returned.db.jsonl` | both there. **Never rebuild the first one once a run has used it:** its order is what the steps count in |
| The references | `<data>/run2/refs/run1b-eval-0030/`, `<data>/run2/phase0/heldout-step30/` (`summary.json` and `protocol.json`) | both there: the evaluations are compared with them. The run refuses to start when the second one is missing or was made under another protocol (another prompt, call limit or set of one-query datasets): `phase0.py heldout --samples 8` makes it again |
| Tests | in WSL: `PYTHONPATH=src <venv>/bin/python -m pytest tests/run2 -q` | green |
| No run named `run2` yet | `<data>/run2/run2/` | doesn't exist (if it does, this is a resume) |

Where things are: `configs/paths.local.toml` (`data`, `models`). Run 2 writes only to those two
folders, both on the internal drive.

## 1. Commands

```
C:\Python314\python.exe -I scripts\run2\run.py --run run2 --launched-by agent
C:\Python314\python.exe -I scripts\run2\pause.py --run run2              (frees the GPU now)
C:\Python314\python.exe -I scripts\run2\pause.py --run run2 --after-step
```

Start it hidden and detached (`Start-Process ... -WindowStyle Hidden`), as run 1b was. The
start command also resumes: a step without `TRAINED` starts again from its kept episodes and
scores. Settings come from `config.py` and are frozen into `<run>/config.json` when the run is
created; `--steps`, `--lr`, `--mix`, `--init-from`, `--prompt-variant` and `--single-turn`
change them for a new run only.

SQaLe and text2cypher questions keep run 1's protocol (one query, nothing shown; decisions R16,
R17), so 12 of a step's 32 questions have one turn; gretel and CypherBench have turns.
`--single-turn none` gives every dataset turns; the start reference then has to be made again
under that protocol.

Three of a step's ten gretel questions are **returned questions**: ones whose right result is
nothing (decisions R17). They are scored more strictly (`design.md`, section 3) and reported
apart.

A loop test (2 steps of 8 questions from a reserve no real run uses, no evaluation):
`run.py --run <name> --smoke`.

## 2. What a step does

1. **Rollouts** (about 2 to 5 minutes): `step-000k/rollouts.jsonl`, 256 episodes. Before step 1
   the start adapter is merged into the run's rollout model (`merge.log`).
2. **Update** (about 5 to 8 minutes, in WSL): scores (`scored.jsonl`), trains, saves
   (`lora_state.safetensors`, `optim.pt`, `ema_state.safetensors`, each with a `.sha256`),
   merges the new adapter into the rollout model, writes `metrics.json` and `TRAINED`.
3. After steps 5, 10, ... : **greedy with turns** on the evaluation set (about 5 minutes).
   After steps 20 and 40 also 8 sampled episodes a question (about 40 minutes more).

`events.jsonl` has a line per phase; `train_log.jsonl` a line of metrics per step;
`BEST.json` the step to keep so far.

## 3. What to read after each step

| Field | Healthy | Look closer when |
|---|---|---|
| `solve_rate`, `first_solve_rate` | near the no-training numbers at first; one step swings by 0.1 | the first query's rate falls while the final rate holds: the model is leaning on retries |
| `calls_mean`, `calls`, `retry_after` | 1.2 to 1.7 calls; a second call mostly after an error or an empty result | calls after `rows` rise: it is exploring, or second-guessing right answers. In the pool an empty result is always a wrong query; on the evaluation set it is often a right one, so a rising retry rate after `empty` helps training scores and hurts held-out ones |
| `fixed`, `broken`, `broken_frac` | fixed above broken | broken near fixed: turns aren't paying |
| `informative_frac`, `distinct_final_mean` | about 0.4 to 0.6; several different final queries a group | both falling: the groups are collapsing onto one answer |
| `entropy_t1`, `t1_tokens_median` | level | entropy falling for several steps |
| `kl_start`, `kl_start_raw`, `kl_step1` | step 1's raw reading near 0.008 (0.005 to 0.012); `kl_start` (raw less the 0.0084 offset) within +-0.005 of 0 at first, then rising slowly | `kl_start` above 0.05 pauses the run; a step-1 raw reading above 0.025 pauses it |
| `mismatch` | about 0.04 to 0.06 | above 0.10 pauses |
| `hardcoded_frac`, `copied_frac`, `look_leak_frac` | 0, near 0, 0 | any rise. Read the episodes |
| `loop_frac`, `limit_frac`, `broken_frac` | the untrained model's own level and falling. `loop_frac` is the same query twice in a row; `rerun` in the scores also counts the answer run again after a look, which is fine | rising: the model is learning to spend calls, not to fix |
| `returned_solve`, `returned_broken_frac`, `returned_calls_mean` beside `gretel_other_fixed`, `gretel_other_broken` | the returned questions start near 0.2 solved with over half of the right first answers broken (the untrained model on the second dev slice: 11 of 50, 14 of 25 broken, 2.0 calls); the other gretel questions fix more than they break | **this is what option C trains:** `returned_broken_frac` should fall while `gretel_other_fixed` holds. If the first falls because the model stopped retrying everywhere, `gretel_other_fixed` falls with it |
| `returned_weak_only_frac` | about 0.4 to 0.5 untrained: answers the checker passes that the filled copies reject | rising toward 1: the model is ending on queries that merely return nothing. They earn 0, so it should not pay; read the episodes |
| `grad_norm` | under the clip of 1.0 | at the clip for several steps (run 1b's sign that a penalty was driving) |
| `look_error_frac` | 0 | a database that didn't build: read `look_error` in the episodes |
| `step_s` | 9 to 14 minutes | twice that: something else is on the GPU, or Neo4j is slow |

`calls_mean`, `calls`, `retry_after`, `fixed`, `broken`, `limit_frac`, `loop_frac`,
`copied_frac` and `look_leak_frac` count only the episodes that had turns (gretel and
CypherBench, 160 of a step's 256 episodes): one-query episodes can't retry, loop or copy, and
would only make a problem look smaller. The scores, `informative_frac`, the entropy and `hardcoded_frac`
count every episode.

Read some episodes by hand every ten steps: the highest-reward ones, the ones a retry fixed,
the ones it broke, and any the canaries flag. An episode line holds every turn's text, every
query and what came back.

## 4. What you may and may not do on your own

- **May:** read files, pause for Rishi's GPU, resume after a pause you made, restart after a
  crash section 5 explains, start the checker's Neo4j container.
- **Ask Rishi first:** changing any setting (`config.json`, a stop bound, the penalty, the
  learning rate, the step count, `single_turn_sets`), resuming after a stop rule, the gate or
  the turn rule paused the run, stopping the run for good, running the test set, anything that
  costs money.
- **Never:** edit the run's code while it runs, start another GPU job, `wsl --shutdown` or
  `--terminate`, stop Docker, touch `.env` or credential files, put local paths in tracked
  files.

## 5. When it fails (`FAILED`, or an `error` event)

Read the event's text, then the step's `train.log` (update) or `server.err.txt` (rollouts).

| Cause | Fix |
|---|---|
| "the checker's Neo4j doesn't answer" | `docker start asktrainer-neo4j`, wait 30 s, the start command again |
| "another llama-server is running" | find whose it is; a leftover of this run is cleared by `pause.py`; otherwise ask Rishi |
| "N episodes failed; run again to retry them" | the start command again (finished episodes are kept) |
| "the rollout model for step k could not be written" | read `merge.log`; in WSL `python -m sqlrl.run2.tools merge --run-dir <dir> --step <k-1>`, then start again |
| "the start adapter is missing" | `<models>/<init_from>/lora_state.safetensors` must exist |
| "no start reference", or "made under another protocol" | the start adapter's evaluation with turns is missing or doesn't match this run's settings: `phase0.py heldout --samples 8` (about 40 minutes), then start again |
| "holds episodes made under another protocol" | an evaluation folder has episodes from other settings; move the folder aside, never mix them |
| A power cut or reboot | start Docker and the Neo4j container, then the start command; at most one update is redone |
| The update hangs (GPU idle, the log silent for 15 minutes; run 1b step 35) | stop only that trainer inside WSL (`kill <pid>`), check the step before's state files against their `.sha256`, then ask Rishi before resuming |
| Out of memory in the update | check nothing else holds the GPU; start again. If it repeats on the same step, stop and tell Rishi |
| Anything else | don't guess: leave the run stopped, tell Rishi what the log says |

## 6. When it pauses by itself (`PAUSED` with a reason)

A stop rule, the gate or the turn rule fired (`design.md` sections 6 and 7). Don't resume on
your own. Read the step's episodes for what the rule points at, and bring Rishi: the rule, the
numbers, what the episodes look like, and your recommendation (resume as is; change a setting
and resume; stop and keep the best step).

Two pauses have a known next step, still Rishi's to choose:

- **"the drift reading at step 1 ... is above 0.025":** the offset was measured at 0.0084 on
  the start adapter. A step-1 reading this far from it means the rollout model isn't the start
  adapter merged into the SFT model (check `merge.log` and `init_from`), or the offset has
  changed (`phase0.py offset` measures it again).
- **Reward hacking rules:** read the flagged episodes first. `hardcoded` episodes already earn
  nothing; a rising `copied_frac` may be the model learning how a value is spelled (fine) or
  reading answers off rows (not fine).

A setting can only change with the run stopped: the config is read once at the start.

## 7. Closing the run

1. `BEST.json` names the step with the highest greedy with turns. Rishi confirms the step.
2. The test set (1,000 questions), once, on the kept adapter, with Rishi's OK.
3. Copy the kept adapter to `<models>/run2-step<k>/`; move the run's folder to the data drive
   as history, throttled, and check every file's hash before deleting the working copy.
4. `results/run2.md`, a model-card section for the tool protocol, `PROGRESS.md`, a decision
   entry, commit and push.
