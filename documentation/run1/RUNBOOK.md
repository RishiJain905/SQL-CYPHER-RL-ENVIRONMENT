# Run 1 runbook: start it, show it, watch it

For the agent that starts and watches run 1 (Rishi's plan, 2026-11-12: this session runs on
Opus 5.5; a Sonnet agent builds the dashboard). Read this file top to bottom before doing
anything. It assumes nothing from an earlier conversation.

**What Rishi decided (2026-11-12):** start run 1 with the three choices as recommended (mix
10 CypherBench / 10 gretel / 8 SQaLe / 4 text2cypher, learning rate 1e-5 with warm-up, 100
steps then decide). A Sonnet agent builds a dashboard that tracks everything and updates after
every step. The watching agent keeps the dashboard fed and makes sure the run does what it
should: learning, not drifting, not overfitting.

## 0. Read first (10 minutes)

| File | Why |
|---|---|
| `CLAUDE.md` | the rules: security, one GPU job at a time, never `wsl --shutdown`, public repo (no local paths in tracked files) |
| `documentation/run1/README.md` | commands, files a run leaves, what the metrics mean, what a pause means |
| `documentation/run1/design.md` | what run 1 is and the stop rules (sections 3, 5, 6) |
| `documentation/run1/dashboard-brief.md` | the dashboard task and the data it shows |
| `configs/paths.local.toml` | this machine's paths (`data` is `<data>` below); untracked, never copy its values into tracked files |

## 1. Before the start: every line must be true

Run from the repo root. Stop and tell Rishi if a line fails and the fix isn't listed.

| Check | Command | Expect |
|---|---|---|
| Docker and the checker's Neo4j | `docker ps --format "{{.Names}} {{.Status}}"` | `asktrainer-neo4j Up ...` (else `docker start asktrainer-neo4j`; if Docker itself is down, start Docker Desktop and wait) |
| The GPU is free | `tasklist /FI "IMAGENAME eq llama-server.exe" /NH` and, in WSL, `pgrep -af "sqlrl\|train"` | no llama-server, no training process. If something of Rishi's is on the GPU, ask him |
| The pool | `<data>/run1/pool-v2.stats.json` | at least 1,000 CypherBench, 1,000 gretel, 800 SQaLe and 400 text2cypher questions accepted (100 steps at the mix) |
| The yardstick dry run | `<data>/run1/evalcheck/eval-0000/summary.json` | exists (done 2026-11-12: greedy 44.5, mean@8 36.4, pass@8 59.9; the baseline is 43.2 / 36.9 / 62.6) |
| No run named `run1` yet | `<data>/run1/run1/` | doesn't exist (if it does, this is a resume: skip to section 2's command and say so) |
| Tests | in WSL: `PYTHONPATH=src <venv>/bin/python -m pytest -q tests/run1` | all pass |

## 2. Start the run

```
Start-Process C:\Python314\python.exe -ArgumentList '-I','scripts\run1\run.py','--run','run1','--launched-by','agent' -WindowStyle Hidden
```

(PowerShell, from the repo root. Hidden and detached: a console window that is closed kills its
process. The defaults are Rishi's choices; no other flags.)

Within two minutes `<data>/run1/run1/events.jsonl` must show `run1 (srpo): 100 steps` and
`step 1: rollouts`. Then:

- Record the start in `documentation/PROGRESS.md` under "Rishi-run steps and waivers":
  "2026-11-12: Rishi told the agent to begin run 1 with the recommended choices (run with
  `--launched-by agent`)", and set the status row to running.
- Step 1 takes about 11 minutes: ~5 of rollouts, ~6 of update (the first update also checks the
  rollout model's format; `template_equal_share` must be above 0.99).

**Never start a second copy.** One GPU job at a time. To free the GPU for Rishi at any moment:
`C:\Python314\python.exe -I scripts\run1\pause.py --run run1`; the start command resumes.

## 3. The dashboard

1. **Spawn the `sonnet-xhigh` subagent** (Rishi named a Sonnet agent for this) with the task in
   `documentation/run1/dashboard-brief.md`: tell it to read that file and `CLAUDE.md`, build and
   publish the page, and return the Artifact URL and the exact push calls. Spawn it right after
   the run starts; the run doesn't wait for it.
2. **Check what it returns:** open the page, push one tick yourself, confirm the charts match
   `train_log.jsonl` for a step. Give Rishi the URL. Commit the page and its README.
3. **A tick** (after every step, and when the status changes):
   - `C:\Python314\python.exe -I scripts\run1\feed.py --run run1` (prints the files it wrote
     and `stop_rules_now`);
   - push each file in `<data>/run1/run1/feed/` to the page's store with `ArtifactData`:
     collection `run1`, document id = the file name without `.json`, `set` from the file (one
     batch). `state` every tick; only the `steps-n` file that changed.

## 4. Watching: what to do after each step

Wait for the next step instead of polling in a tight loop: one background wait on
`<data>/run1/run1/step-<k+1>/TRAINED` or on a `PAUSED` / `FAILED` / `DONE` file (a Monitor
until-loop, or a scheduled wake-up of ~11 minutes). Keep each step's work small; the run is ~18
hours and your context has to last.

**Every step (a minute of work):** run the tick; read the step's row in `train_log.jsonl`;
check the table below; say nothing unless something is off or it is a reporting step.

| Look at | Healthy | Act when |
|---|---|---|
| `events.jsonl`, `status` | `step k done`, then the next step's rollouts | `failed`: section 5. `paused`: section 6 |
| `informative_frac` | 0.4-0.6 | below 0.25 for several steps: the batch has too few "sometimes" questions; tell Rishi |
| `entropy` | near the first steps' (0.43-0.59 in tests) | falling steadily: the policy is narrowing; watch pass@8 at the next evaluation |
| `kl_sft` | ~0.0005 at step 1, rising slowly (0.004 after 10 pilot steps) | a jump between two steps, or a path that will cross the bound well before step 100: tell Rishi before the rule fires (the bound is 0.30 since step 19, R10-R11; a steady rise is expected, the evaluations judge it: after steps 19, 25, 35, 50, 75, 100) |
| `mismatch` | ~0.012-0.015 at step 1, rising with the LoRA | above 0.06: report; 0.10 pauses the run |
| `malformed_frac`, `multi_call_frac`, `leak_frac` | 0-3%, 0, 0 | any rising trend; `leak_frac` above 0: read those thoughts |
| `distinct_frac`, `select_star_frac`, `query_chars_median` | flat | a trend: read the highest-reward rollouts for a trick |
| `sdpo_rollouts`, `distill_jsd` | 30-80, ~0.03 | zero for several steps, or JSD climbing |
| `grad_norm`, `loss` | SRPO pilot: 0.05-0.08, loss around -0.6 | an order-of-magnitude change |
| `step_s`, `rollout_tok_s`, `load_s` | ~660 s, ~400, ~20 s | `load_s` above 100 s (WSL's keep-alive died?), rollouts twice as slow (something else on the GPU?) |

**Every 5 steps (a short note to Rishi, 3-5 lines):** the 5-step rolling solve rate overall and
per dataset against the steps before, `kl_sft`, entropy, anything off, hours left.

**Reading rollouts** (every 10 steps, and whenever a canary moves): open the latest
`step-000k/rollouts.jsonl` with `scored.jsonl`; read five right answers and five wrong ones.
Right answers must be right for the right reason (no `SELECT *`, no blanket `DISTINCT`, no
query shaped to the checker); thoughts must still read like the model's own reasoning.

### Is it learning, generalising, or overfitting?

- **Learning:** every step's questions are new training questions, so the rolling solve rate is
  already a held-in-distribution test, not a memory test. It should drift up over tens of
  steps. One step proves nothing: steps swing by 0.1-0.2 with the batch.
- **Generalising:** the evaluations after steps 25, 50 and 100 on the 200 held-out questions
  (`<data>/run1/run1/eval-00kk/summary.json`, also on the dashboard). Good: mean@8 and greedy
  pass@1 above the baseline (36.9 and 43.2), pass@8 not below 62.6, and the sampled sign test
  leaning "better". One question is 0.5 points; read the sign tests, not the points.
- **How much is noise:** the SFT model itself, measured again through run 1's evaluation path
  on 2026-11-12, scored greedy 44.5, mean@8 36.4, pass@8 59.9 (against the baseline: greedy 6
  better / 4 worse, sampled 37 / 44, pass@8 8 / 14, p = 0.29). So a pass@8 that is 2-3 points
  off, or a sampled test of 37 / 44, is what no change at all looks like. Compare every
  evaluation with both: the baseline and this re-measurement (`baseline_run1_path` in the feed).
- **Overfitting:** training solve rate up while the evalset is flat or down. Watch CypherBench
  most: it trains on 4 graphs and is tested on 7 others, and it slipped in both pilots. A
  CypherBench training rate that climbs while its evalset pass@8 falls is the signal to bring
  to Rishi with the numbers.
- **Narrowing:** mean@8 up but pass@8 down, with entropy falling: the model is getting more
  consistent by giving up answers it used to find sometimes. The gate catches the significant
  case; report the trend earlier.

**After each evaluation, report to Rishi:** the three numbers against the baseline and the last
evaluation, per dataset, the three sign tests, never / sometimes / always, the canaries'
state, and your reading (learning / flat / overfitting / narrowing) with the evidence.

### What you may and may not do on your own

- **May:** run ticks, read files, pause for Rishi's GPU, resume after a pause you made, restart
  after a crash that section 5 explains, start the checker's Neo4j container.
- **Ask Rishi first:** changing any setting (`config.json`, a stop bound, the learning rate, the
  step count), resuming after a stop rule or the gate paused the run, stopping the run for
  good, extending to 200 steps, running the test set, anything that costs money.
- **Never:** edit run 1's code while it runs, start another GPU job, `wsl --shutdown` or
  `--terminate`, stop Docker, touch `.env` or credential files, put local paths in tracked files.

## 5. When it fails (`FAILED`, or an `error` event)

Read the event's text, then the step's `train.log` (update) or `server.err.txt` (rollouts).

| Cause | Fix |
|---|---|
| "the checker's Neo4j doesn't answer" | `docker start asktrainer-neo4j`, wait 30 s, run the start command again |
| "another llama-server is running" | find whose it is; if it is a leftover of this run (`<data>/run1/run1/step-*/server.pid`), `pause.py` clears it; otherwise ask Rishi |
| "N rollouts failed; run again to retry them" | run the start command again (finished rollouts are kept) |
| "N evaluation answers failed after 3 tries" | the server rejected many answers; read `eval-00kk/server.err.txt`. Up to 8 unparseable answers are kept as empty answers and only logged as a warning |
| "the rollout model for step k could not be written" | read `merge.log`; `python -m sqlrl.run1.tools merge --run-dir <dir> --step <k-1>` in WSL, then start again |
| A power cut or reboot | start Docker and the Neo4j container, then the start command; at most one update is redone |
| Out of memory in the update | check nothing else holds the GPU; start again. If it repeats on the same step, stop and tell Rishi (don't change `max_seq` yourself) |
| Anything else | don't guess: leave the run stopped, tell Rishi what the log says |

The start command always resumes: a step without `TRAINED` starts again from its kept rollouts
and scores.

## 6. When it pauses by itself (`PAUSED` with a reason)

A stop rule or the gate fired. Don't resume on your own. Run a tick so the dashboard shows it,
read the step's rollouts, and bring Rishi: which rule, the numbers for the last 10 steps, what
the rollouts look like, and your recommendation (resume as is; change a bound and resume; stop
and keep step X's LoRA: every step's `lora_state.safetensors` is kept).

## 7. When it finishes (`DONE`)

Run a last tick. Report the step-100 evaluation as in section 4. Then, with Rishi: extend to 200
or stop; `documentation/results/run1.md` (the numbers, the sign tests, the canaries, the
reading); the model card; `PROGRESS.md`; the test set once, on the adapter that is kept.

## Kickoff message for Rishi to paste

> Read `documentation/run1/RUNBOOK.md` in `NFL-RL` and follow it: check section 1, start run 1,
> spawn the sonnet-xhigh subagent for the dashboard (section 3), then watch the run and keep the
> dashboard updated after every step.
