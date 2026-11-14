# Run 1b runbook: start it, show it, watch it

For the agent that starts and watches run 1b (Rishi's plan, 2026-11-14: as for run 1, this
session runs on Opus 5.5 and a Sonnet agent builds the dashboard). Read this file top to bottom
before doing anything. It assumes nothing from an earlier conversation.

**What Rishi decided (2026-11-13 / 14):** run 1 is closed with step 19's adapter kept (R12).
Before run 2, fix the trainer (a KL penalty, a lower learning rate, earlier evaluations, the
best step kept) and the reward's gap on `DISTINCT`, then run a short run 1b to see that the
trainer no longer drifts. The fixes are built and loop-tested (R13). **Run 1b: 40 steps, about
10.5 hours.**

## 0. Read first (15 minutes)

| File | Why |
|---|---|
| `CLAUDE.md` | the rules: security, one GPU job at a time, never `wsl --shutdown`, public repo (no local paths in tracked files) |
| `documentation/run1b/design.md` | what run 1b changes, why, and how to read its end (sections 1-4) |
| `documentation/results/run1.md` | what run 1 did: the numbers run 1b is read against |
| `documentation/run1/README.md` | commands, the files a run leaves, what the metrics mean (run 1b uses run 1's code) |
| `documentation/run1b/dashboard-brief.md` | the dashboard task |
| `configs/paths.local.toml` | this machine's paths (`data` is `<data>` below); untracked, never copy its values into tracked files |

## 1. Before the start: every line must be true

Run from the repo root. Stop and tell Rishi if a line fails and the fix isn't listed.

| Check | Command | Expect |
|---|---|---|
| Docker and the checker's Neo4j | `docker ps --format "{{.Names}} {{.Status}}"` | `asktrainer-neo4j Up ...` (else `docker start asktrainer-neo4j`; if Docker itself is down, start Docker Desktop and wait). **Run 1b needs it every step:** the CypherBench score and the duplicate probe |
| The GPU is free | `tasklist /FI "IMAGENAME eq llama-server.exe" /NH` and, in WSL, `pgrep -af "sqlrl\|train"` | no llama-server, no training process. If something of Rishi's is on the GPU, ask him |
| The pool | `<data>/run1/pool-v2.stats.json` | exists (run 1b uses the first 400 CypherBench, 400 gretel, 320 SQaLe and 160 text2cypher questions: run 1's, in run 1's order) |
| Run 1 is where it was left | `<data>/run1/run1/STOPPED`, `<data>/run1/run1/eval-0019/summary.json` | both exist (the dashboard draws run 1 behind run 1b) |
| No run named `run1b` yet | `<data>/run1/run1b/` | doesn't exist (if it does, this is a resume: skip to section 2's command and say so) |
| Tests | in WSL: `PYTHONPATH=src <venv>/bin/python -m pytest -q tests/run1` | all pass (27) |

## 2. Start the run

```
Start-Process C:\Python314\python.exe -ArgumentList '-I','scripts\run1\run.py','--run','run1b','--preset','run1b','--launched-by','agent' -WindowStyle Hidden
```

(PowerShell, from the repo root. Hidden and detached: a console window that is closed kills its
process. `--preset run1b` is the whole of run 1b's settings; no other flags. The preset is read
once, when the run's `config.json` is written.)

Within two minutes `<data>/run1/run1b/events.jsonl` must show `run1b (srpo): 40 steps` and
`step 1: rollouts`. Then:

- Check `<data>/run1/run1b/config.json`: `lr` 5e-06, `steps` 40, `eval_steps` [10, 20, 30, 40],
  `method_cfg.kl` `{beta 0.05, target 0.03, beta_max 6.4, top_k 100}`, `dup_probe` true,
  `stop.kl_max` 0.1. If any is missing the preset wasn't applied: stop the run
  (`scripts\run1\pause.py --run run1b`), tell Rishi.
- Record the start in `documentation/PROGRESS.md` (status row 4 and "Rishi-run steps and
  waivers": Rishi told the agent to begin run 1b, run with `--launched-by agent`).
- Step 1 takes about 14 minutes (the first update also checks the rollout model's format;
  `template_equal_share` must be above 0.99); later steps 12-13.

**Never start a second copy.** One GPU job at a time. To free the GPU for Rishi at any moment:
`C:\Python314\python.exe -I scripts\run1\pause.py --run run1b`; the start command resumes
(`--preset` is ignored on a resume: the run's own `config.json` rules).

## 3. The dashboard

1. **Spawn the `sonnet-xhigh` subagent** (Rishi named a Sonnet agent for this) with the task in
   `documentation/run1b/dashboard-brief.md`: tell it to read that file and `CLAUDE.md`, build
   and publish the page, and return the Artifact URL and the exact push calls. Spawn it right
   after the run starts; the run doesn't wait for it.
2. **Check what it returns:** open the page, push one tick yourself, confirm a chart matches
   `train_log.jsonl` for a step and that run 1's line is behind it. Give Rishi the URL. Commit
   the page and its README.
3. **A tick** (after every step, and when the status changes):
   - `C:\Python314\python.exe -I scripts\run1\feed.py --run run1b` (prints the files it wrote
     and `stop_rules_now`);
   - push each file in `<data>/run1/run1b/feed/` to the page's store with `ArtifactData`:
     collection `run1b`, document id = the file name without `.json`, `set` from the file (one
     batch). `state` every tick; only the `steps-n` file that changed; `compare` once.
   - **A write to a document that exists needs `if_version`** (the version the last `set` or a
     `list` reported), or the store refuses it.

## 4. Watching: what to do after each step

Wait for the next step instead of polling in a tight loop: one background wait on
`<data>/run1/run1b/step-<k+1>/TRAINED` or on a `PAUSED` / `FAILED` / `DONE` file. Keep each
step's work small; the run is about 10.5 hours and your context has to last.

**Every step (a minute of work):** run the tick; read the step's row in `train_log.jsonl`;
check the table below; say nothing unless something is off or it is a reporting step.

| Look at | Healthy | Act when |
|---|---|---|
| `events.jsonl`, `status` | `step k done`, then the next step's rollouts | `failed`: section 5. `paused`: section 6 |
| **`kl_sft`** | far slower than run 1: about 0.001 at step 10, 0.005-0.01 at step 20, reaching 0.03-0.05 around step 28-32 and **levelling off there** | it passes 0.06 and is still climbing three steps later with `kl_beta` rising: tell Rishi with the last 10 steps' `kl_sft` and `kl_beta`. 0.10 pauses the run |
| **`kl_beta`** | 0.05 until `kl_sft` passes 0.045, then doubling step by step until `kl_sft` stops rising; halving again when `kl_sft` falls under 0.02 | at 6.4 (its ceiling) with `kl_sft` still above 0.045: the penalty can't hold; tell Rishi |
| `kl_pen` | below `kl_sft` (the trainer under-reads a small LoRA), rising with it | zero or missing while `kl_beta` is above 0: the penalty isn't being computed; stop and tell Rishi |
| `solve_rate`, per dataset | the same questions as run 1's step: compare with run 1's row (`<data>/run1/run1/train_log.jsonl`, or the dashboard). Steps swing by 0.1-0.2 with the batch | a 5-step mean more than 0.08 under run 1's at the same steps |
| `informative_frac` | 0.35-0.6 (a little under run 1's: the probe withholds some CypherBench rewards) | below 0.25 for several steps (5 in a row pauses the run) |
| **`dup_differ_frac`** | 0.4-0.55 at the start (run 1: 0.48), flat or falling | rising: the model repeats rows where the reference doesn't. 0.15 over the first five steps' mean for 3 steps pauses the run |
| `entropy` | near the first steps' (0.40-0.50 in run 1), not falling steadily | a steady fall: the policy is narrowing; watch pass@8 at the next evaluation |
| `resp_tokens_median` | near the first steps' (about 230 in run 1; run 1 fell to 177) | a steady fall of more than 15%: report it |
| `distinct_frac` | near the first steps' (0.27 in run 1; run 1 fell to 0.08) | a steady fall: report it with `dup_differ_frac` |
| `mismatch` | 0.012-0.03 | above 0.06: report; 0.10 pauses the run |
| `malformed_frac`, `multi_call_frac`, `leak_frac` | 0-3%, 0, 0-1% | any rising trend; `leak_frac` above 1%: read those thoughts |
| `sdpo_rollouts`, `distill_jsd` | 50-90, about 0.03-0.05 | zero for several steps, or the JSD climbing |
| `grad_norm`, `loss` | 0.05-0.15; the loss is no longer comparable with run 1's (it includes the penalty) | an order-of-magnitude change |
| `step_s`, `load_s`, `peak_gb` | about 720-780 s (the update is about 2 minutes longer than run 1's), about 20 s, under 13 GB | `load_s` above 100 s after a step that wasn't an evaluation (the keep-alive died?), `peak_gb` above 14 |

**Every 5 steps (a short note to Rishi, 3-5 lines):** the 5-step solve rate overall and per
dataset against run 1's at the same steps, `kl_sft` and `kl_beta`, `dup_differ_frac`, entropy,
anything off, hours left.

**Reading rollouts** (every 10 steps, and whenever a canary moves): open the latest
`step-000k/rollouts.jsonl` with `scored.jsonl`; read five right answers and five wrong ones,
and five CypherBench answers with `"dup": "differ"` in `scored.jsonl`. Right answers must be
right for the right reason; thoughts must still read like the model's own reasoning; a
withheld answer should be one that would repeat rows on a denser graph.

### After each evaluation (steps 10, 20, 30, 40)

Report to Rishi: greedy / mean@8 / pass@8 against the SFT baseline (43.2 / 36.9 / 62.6), the
SFT model re-measured (44.5 / 36.4 / 59.9) and **run 1's step 19 (45.6 / 40.8 / 66.2)**; per
dataset; the three sign tests against the baseline; `vs_best` (against the best earlier
evaluation); never / sometimes / always; `kl_sft` at that step; `BEST.json`; and your reading.

| Reading | The evidence |
|---|---|
| Learning and holding | sampled sign test leaning "better", pass@8 not down, `kl_sft` level, later evaluations not below earlier ones |
| Flat | everything inside what "SFT re-measured" shows (a sampled test like 37 / 44, 2-3 points either way) |
| Drifting again | `kl_sft` through 0.06, entropy and response length falling, an evaluation below the one before |
| Narrowing | mean@8 up, pass@8 down, entropy falling |

One evaluation is good to about 3 points; one question is 0.5. Read the sign tests.

`design.md` section 4 has the table of what each ending means. Don't decide what comes next
yourself: that is Rishi's.

### What you may and may not do on your own

- **May:** run ticks, read files, pause for Rishi's GPU, resume after a pause you made, restart
  after a crash that section 5 explains, start the checker's Neo4j container.
- **Ask Rishi first:** changing any setting (`config.json`, a stop bound, the KL target or
  weight, the learning rate, the step count), resuming after a stop rule, the gate or the turn
  rule paused the run, stopping the run for good, extending past 40 steps, running the test
  set, anything that costs money.
- **Never:** edit the run's code while it runs, start another GPU job, `wsl --shutdown` or
  `--terminate`, stop Docker, touch `.env` or credential files, put local paths in tracked
  files.

## 5. When it fails (`FAILED`, or an `error` event)

Read the event's text, then the step's `train.log` (update) or `server.err.txt` (rollouts).
`documentation/run1/RUNBOOK.md` section 5 has the table of causes and fixes; it applies as
written with `run1b` for `run1`. Two more for run 1b:

| Cause | Fix |
|---|---|
| "the checker's Neo4j doesn't answer" during the duplicate probe | `docker start asktrainer-neo4j`, wait 30 s, run the start command again (the step's scores are redone) |
| Out of memory in the update | the penalty holds the SFT model's states of every trained answer in RAM and adds a projection per chunk. Check nothing else holds the GPU; start again. If it repeats on the same step, stop and tell Rishi (don't change `max_seq` or the penalty yourself) |

The start command always resumes: a step without `TRAINED` starts again from its kept rollouts
and scores.

## 6. When it pauses by itself (`PAUSED` with a reason)

A stop rule, the gate, or the turn rule ("worse than the evaluation after step k") fired. Don't
resume on your own. Run a tick so the dashboard shows it, read the step's rollouts, and bring
Rishi: which rule, the numbers for the last 10 steps, what the rollouts look like, and your
recommendation (resume as is; change a setting and resume; stop and keep the best step).

A setting can only change with the run stopped: the config is read once at the start. The run
removes `STOP`, `STOPPED`, `PAUSED` and `FAILED` when it starts, so a `STOP` file meant to end
it after an evaluation has to be written after the restart.

## 7. When it finishes (`DONE`)

Run a last tick. Report the step-40 evaluation as in section 4, and the whole run against
`design.md` section 4's table. Then, with Rishi: which adapter is kept (`BEST.json`, against
run 1's step 19); extend or stop; `documentation/results/run1b.md` (the numbers, the sign
tests, the canaries, run 1 beside it, the reading); `PROGRESS.md`; the test set once, on the
adapter that is kept; whether these settings go into run 2.

## Kickoff message for Rishi to paste

> Read `documentation/run1b/RUNBOOK.md` in `NFL-RL` and follow it: check section 1, start run
> 1b, spawn the sonnet-xhigh subagent for the dashboard (section 3), then watch the run and keep
> the dashboard updated after every step.
