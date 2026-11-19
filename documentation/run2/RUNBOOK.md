# Run 2 runbook: start it, show it, watch it

For the agent that starts and watches run 2 (Rishi's plan, as for runs 1 and 1b: an Opus
session runs this file, a Sonnet agent builds the dashboard, and for run 2 a second Sonnet
agent explains what changed). Read this file top to bottom before doing anything. It assumes
nothing from an earlier conversation.

**Run 2 is not started by an agent on its own.** Rishi says when. The kickoff message at the
end of this file, pasted by Rishi, is that go: it covers starting the run with
`--launched-by agent`, spawning the two `sonnet-xhigh` subagents, and watching until the run
ends or pauses. It doesn't cover anything section 6 lists under "ask Rishi first".

**What run 2 is, in short:** the model may run up to 3 queries on a question, sees each
result, and its last query is its answer. It starts from run 1b's step-30 adapter. **40 steps,
about 9 to 10 hours.** The setup is **option C** (decisions R17): gretel and CypherBench have
turns, SQaLe and text2cypher keep one query, and 3 of a step's 10 gretel questions are
"returned questions" whose right result is nothing. Its no-training tests, loop tests and
start reference are done (`results/run2-phase0.md`, section 10).

## 0. Read first (20 minutes)

| File | Why |
|---|---|
| `CLAUDE.md` | the rules: security, one GPU job at a time, never `wsl --shutdown`, never stop Docker, public repo (no local paths in tracked files) |
| `documentation/run2/design.md` | what the run is: an episode (2), what decides the reward (3, 4), the evaluations (6), the stop rules (7) |
| `documentation/results/run2-phase0.md`, section 10 | the numbers the run is read against: the untrained start under option C |
| `documentation/decisions.md`, R16 and R17 | why SQaLe and text2cypher keep one query, and what the returned questions are |
| `documentation/run2/dashboard-brief.md` | the dashboard task, and every field of the feed |
| `configs/paths.local.toml` | this machine's paths (`data` is `<data>` below, `models` is `<models>`); untracked, never copy its values into tracked files |

**This machine, so you don't lose time:**

- The Bash tool may print nothing here. Use PowerShell. For a command inside WSL, write a
  `.sh` file and run `wsl -d Ubuntu -- bash <file>`.
- The GPU is AMD: there is no `nvidia-smi`. "The GPU is free" means no `llama-server.exe`
  and no training process in WSL.
- Don't use `uv run` in this repo (it makes a `.venv` and a `uv.lock`). Windows Python is
  `C:\Python314\python.exe -I`; the tests run in WSL with the repo's venv.
- The secrets hook refuses any command with `docker inspect` in it.
- A commit message with double quotes in it: write it to a file and use `git commit -F`.

## 1. Before the start: every line must be true

Run from the repo root on Windows. Stop and tell Rishi if a line fails and the fix isn't
listed.

| Check | How | Expect |
|---|---|---|
| The GPU is free | `tasklist /FI "IMAGENAME eq llama-server.exe" /NH` and, in WSL, `pgrep -af "sqlrl\|train"` | nothing. If something of Rishi's is on the GPU, ask him |
| Docker and the checker's Neo4j | `docker ps --format "{{.Names}} {{.Status}}"` | `asktrainer-neo4j` is up. It has no restart policy: after a reboot, `docker start asktrainer-neo4j`. **The run needs it every step** (CypherBench's look-around graph, the scores, the duplicate probe) |
| The start adapter | `<models>/run1b-step30/lora_state.safetensors` and `ema_state.safetensors` | both there |
| The database scripts | `<data>/run2/pool-v2.db.jsonl`, `<data>/run2/evalset.db.jsonl` | both there (`python -m sqlrl.run2.pool build-db`, in WSL, rebuilds them) |
| The returned questions | `<data>/run2/pool-v2.returned.jsonl` (604 lines) and `pool-v2.returned.db.jsonl` | both there. **Never rebuild the first one once a run has used it:** its order is what the steps count in |
| The references | `<data>/run2/refs/run1b-eval-0030/`, `<data>/run2/phase0/heldout-step30/` (`summary.json` and `protocol.json`) | both there: the evaluations are compared with them. The run refuses to start when the second one is missing or was made under another protocol (another prompt, call limit or set of one-query datasets): `phase0.py heldout --samples 8` makes it again |
| Disk | the drive that holds `<data>` and `<models>` | at least 50 GB free (the run writes about 0.65 GB a step, 26 GB over 40 steps, and a 4 GB rollout model) |
| Tests | in WSL: `PYTHONPATH=src <venv>/bin/python -m pytest tests/run2 -q` | green |
| No run named `run2` yet | `<data>/run2/run2/` | doesn't exist (if it does, this is a resume: skip to section 2's command and say so) |

Where things are: `configs/paths.local.toml` (`data`, `models`). Run 2 writes only to those
two folders, both on the internal drive.

## 2. Start the run

```
Start-Process C:\Python314\python.exe -ArgumentList '-I','scripts\run2\run.py','--run','run2','--launched-by','agent' -WindowStyle Hidden
```

(PowerShell, from the repo root. Hidden and detached: a console window that is closed kills
its process, and a background tool task can be cut by its time limit. No other flags: the
settings are `config.py`'s defaults, frozen into `<data>/run2/run2/config.json` when the run
is created.)

Before step 1 the start adapter is merged into the run's rollout model (a 4 GB copy and a
merge: a few minutes, `merge.log`). **Within about five minutes**
`<data>/run2/run2/events.jsonl` must show `run2 (srpo, from run1b-step30): 40 steps` and
`step 1: rollouts`. Then:

- Check `<data>/run2/run2/config.json`: `steps` 40, `init_from` `run1b-step30`,
  `single_turn_sets` `["sqale", "text2cypher"]`, `returned_per_step` `{"gretel": 3}`,
  `returned_guard` true, `prompt_variant` `turns`, `max_queries` 3, `lr` 5e-06,
  `quick_eval_steps` 5 to 40 in fives, `eval_steps` `[20, 40]`, `method_cfg.kl` with `beta`
  0.05 and `offset` 0.0084, `stop.kl_step1_max` 0.025, and **no** `smoke` key. If one
  differs, stop the run (`pause.py`, below) and tell Rishi.
- Record the start in `documentation/PROGRESS.md` (the status row and "Rishi-run steps and
  waivers": Rishi told the agent to begin run 2, run with `--launched-by agent`).
- After step 1 (its row in `train_log.jsonl`): `template_match` true,
  `template_equal_share` above 0.99, `kl_step1` between 0.004 and 0.014 (the run pauses
  itself above 0.025).

Other commands:

```
C:\Python314\python.exe -I scripts\run2\pause.py --run run2              (frees the GPU now)
C:\Python314\python.exe -I scripts\run2\pause.py --run run2 --after-step
C:\Python314\python.exe -I scripts\run2\feed.py --run run2               (the dashboard's data)
```

**Never start a second copy.** One GPU job at a time. The start command also resumes: a step
without `TRAINED` starts again from its kept episodes and scores, and the run's own
`config.json` rules (`--steps`, `--lr`, `--mix`, `--init-from`, `--prompt-variant` and
`--single-turn` change the settings of a new run only).

SQaLe and text2cypher questions keep run 1's protocol (one query, nothing shown; decisions
R16, R17), so 12 of a step's 32 questions have one turn; gretel and CypherBench have turns.
Three of a step's ten gretel questions are **returned questions**: ones whose right result is
nothing (decisions R17). They are scored more strictly (`design.md`, section 3) and reported
apart.

A loop test (2 steps of 8 questions from a reserve no real run uses, no evaluation):
`run.py --run <name> --smoke`; with `--smoke-eval` also a greedy evaluation after each step.

## 3. The two Sonnet agents

Spawn both right after the run has started; the run doesn't wait for them. Rishi named the
`sonnet-xhigh` subagent for both in his kickoff message. Neither may touch the run, the GPU
or the run's code.

### 3a. The dashboard

1. **Spawn `sonnet-xhigh`** with the task in `documentation/run2/dashboard-brief.md`: tell it
   to read that file and `CLAUDE.md`, build and publish the page, and return the Artifact URL
   and the exact push calls.
2. **Check what it returns:** open the page, push one tick yourself, confirm a chart matches
   `train_log.jsonl` for a step, that the returned-questions panel and the start reference
   show, and that the collection held no test rows before your first push. Give Rishi the
   URL. Commit the page and its README (`[RL]` prefix; no local paths in either).
3. **A tick** (after every step, after every evaluation, and when the status changes):
   - `C:\Python314\python.exe -I scripts\run2\feed.py --run run2`. It prints one line:
     `changed` (the files that differ from the last tick), `status`, `stop_rules_now`,
     `hours_left`, and `watch`: the latest step's numbers for the table in section 5 and
     their 5-step means. **Read that line instead of the step's whole row.**
   - push each file in `changed` from `<data>/run2/run2/feed/` to the page's store with
     `ArtifactData`: collection `run2`, document id = the file name without `.json`, `set`
     with `file_path` (one batch). `state` changes every tick; one `steps-n` a step (10 steps
     a document); `evals` after an evaluation; `refs` once.
   - **A write to a document that exists needs `if_version`** (the version the last `set` or
     a `list` reported), or the store refuses it.

### 3b. The explainer

Rishi asked for a Sonnet agent that explains the key changes made for this run (option C).

1. **Spawn `sonnet-xhigh`** with this task: "Read `CLAUDE.md`, then
   `documentation/decisions.md` (R15 to R17), `documentation/results/run2-phase0.md`
   (sections 9 and 10) and `documentation/run2/design.md` (sections 1 to 3). Explain to
   Rishi, in plain words, the key changes made for run 2 under option C. He knows run 1b and
   chose option C, but did not follow every detail. Cover, in this order: what the first
   no-training test showed and why (one worked example helps); the three changes (two
   datasets back on one query, the returned questions, the stricter score) and what each is
   for; what the untrained model scores under the new setup; what to watch on the dashboard
   to see whether training is doing what option C wants; and what is still open (the
   empty-database rule that was not built, and that no success line is set). Under 500
   words, short sentences, one small table at most, every number taken from those files and
   none invented. Write no files, run nothing on the GPU, and don't touch the run. Return the
   explanation as your answer."
2. **Check five of its numbers** against `results/run2-phase0.md` section 10, fix any that
   are off, and **give Rishi the explanation in full in your first report** (a subagent's
   answer isn't shown to him).

## 4. What a step does

1. **Rollouts** (about 2 to 5 minutes): `step-000k/rollouts.jsonl`, 256 episodes.
2. **Update** (about 5 to 8 minutes, in WSL): scores (`scored.jsonl`), trains, saves
   (`lora_state.safetensors`, `optim.pt`, `ema_state.safetensors`, each with a `.sha256`),
   merges the new adapter into the rollout model, writes `metrics.json` and `TRAINED`.
3. After steps 5, 10, ... : **greedy with turns** on the evaluation set (about 4 minutes), and
   on the returned dev slice (50 questions, about a minute; `eval-000k/returned/`, and
   `returned_dev` in the evaluation's summary: untrained it is 25 right on the first query, 11
   after the turns, 14 broken). After steps 20 and 40 also 8 sampled episodes a question on
   the evaluation set (about 15 minutes more).

`events.jsonl` has a line per phase; `train_log.jsonl` a line of metrics per step;
`BEST.json` the step to keep so far.

## 5. Watching

Wait for the next step instead of polling in a tight loop. One background wait, which ends
when the step is trained, the run ends, or 45 minutes pass (`<k>` is the step you wait for,
four digits):

```
$r='<data>\run2\run2'; $t=(Get-Date).AddMinutes(45); while((Get-Date) -lt $t){ if((Test-Path "$r\step-<k>\TRAINED") -or (Test-Path "$r\PAUSED") -or (Test-Path "$r\FAILED") -or (Test-Path "$r\DONE")){break}; Start-Sleep 20 }
```

After a step with an evaluation (5, 10, ...), wait the same way on
`$r\eval-<k>\returned\summary.json` first: the evaluation's result is there a few seconds
later. If a wait ends with nothing new (its 45 minutes ran out, or the tool cut it short),
run a tick and start the wait again. The feed says `dead` when the run's process is gone
(section 7). **`running` with no new event and no growing log for 45 minutes is unhealthy
too**: read `driver.log` and the step's `train.log`, and see "the update hangs" in section 7.

Keep each step's work small: the run is about 9 to 10 hours and your context has to last.

**Every step (a minute of work):** run the tick and push it; read `watch` in the feed's line
against the table below; say nothing unless something is off or it is a reporting step.

| Field | Healthy | Look closer when |
|---|---|---|
| `solve_rate`, `first_solve_rate` | near the no-training numbers at first; one step swings by 0.1 | the first query's rate falls while the final rate holds: the model is leaning on retries |
| `calls_mean`, `calls`, `retry_after` | 1.2 to 1.7 calls; a second call mostly after an error or an empty result | calls after `rows` rise: it is exploring, or second-guessing right answers. In the pool an empty result is always a wrong query; on the returned questions and on the evaluation set it is often a right one |
| `fixed`, `broken`, `broken_frac` | fixed above broken | broken near fixed: turns aren't paying |
| `informative_frac`, `distinct_final_mean` | about 0.4 to 0.6; several different final queries a group | both falling: the groups are collapsing onto one answer |
| `entropy_t1`, `t1_tokens_median` | level | entropy falling for several steps |
| `kl_start`, `kl_start_raw`, `kl_step1` | step 1's raw reading near 0.008 (0.004 to 0.014: 32 probe episodes read within 0.0023 of the offset, and never above 0.02 in 20,000 draws from the offset's own episodes); `kl_start` (raw less the 0.0084 offset) within 0.006 of 0 at first, then rising slowly | `kl_start` above 0.05 pauses the run; a step-1 raw reading above 0.025 pauses it |
| `mismatch` | about 0.04 to 0.06 | above 0.10 pauses |
| `hardcoded_frac`, `copied_frac`, `look_leak_frac` | 0, near 0, 0 | any rise. Read the episodes |
| `loop_frac`, `limit_frac`, `broken_frac` | the untrained model's own level and falling. `loop_frac` is the same query twice in a row; `rerun` in the scores also counts the answer run again after a look, which is fine | rising: the model is learning to spend calls, not to fix |
| `returned_solve`, `returned_broken_frac`, `returned_calls_mean` beside `gretel_other_fixed`, `gretel_other_broken` | the returned questions start near 0.2 solved with over half of the right first answers broken (the untrained model on the second dev slice: 11 of 50, 14 of 25 broken, 2.0 calls); the other gretel questions fix more than they break. 24 episodes a step: read the 5-step mean | **this is what option C trains:** `returned_broken_frac` should fall while `gretel_other_fixed` holds. If the first falls because the model stopped retrying everywhere, `gretel_other_fixed` falls with it |
| `returned_weak_only_frac` | about 0.4 to 0.5 untrained: answers the checker passes that the filled copies reject | rising toward 1: the model is ending on queries that merely return nothing. They earn 0, so it should not pay; read the episodes |
| `grad_norm` | under the clip of 1.0 | at the clip for several steps (run 1b's sign that a penalty was driving) |
| `look_error_frac` | 0 | a database that didn't build: read `look_error` in the episodes |
| `step_s`, `peak_gb` | 9 to 14 minutes; under 14.5 GB | twice the time: something else is on the GPU, or Neo4j is slow |

`calls_mean`, `calls`, `retry_after`, `fixed`, `broken`, `limit_frac`, `loop_frac`,
`copied_frac` and `look_leak_frac` count only the episodes that had turns (gretel and
CypherBench, 160 of a step's 256 episodes): one-query episodes can't retry, loop or copy, and
would only make a problem look smaller. The scores, `informative_frac`, the entropy and
`hardcoded_frac` count every episode.

**Read some episodes by hand every ten steps**, and whenever a canary moves: the
highest-reward ones, the ones a retry fixed, the ones it broke, five returned questions
(`"kind": "returned"` in `scored.jsonl`), and any the canaries flag. An episode line holds
every turn's text, every query and what came back. Right answers must be right for the right
reason; on a returned question, a kept answer should be the right query, not any query that
returns nothing.

### After each evaluation (every 5 steps): a note to Rishi

Five to ten lines, no more:

- **Training, the last 5 steps:** solve rate and the first query's; calls an episode; the
  returned questions' broken share against the untrained 56%, with the other gretel
  questions' fixed and broken beside it; `kl_start`; anything off.
- **The evaluation:** greedy with turns against **the start with turns (52.4%)** and **the
  start on one query (54.5%)**; the first query (the floor is 51.5%); fixed and broken; the
  three rows of `by_look` against the start's (a look can inform: 45 to 52 of 86; the right
  result is nothing: 10 to 3 of 13; one query: 43 of 88); the returned dev slice against
  25 first, 11 right, 14 broken; the sign tests in `vs`; `BEST.json`.
- **After steps 20 and 40 also:** mean@8 and pass@8 against the start with turns (47.6 and
  71.8) and the start on one query (43.4 and 67.5), and the gate.
- Hours left, and your reading.

| Reading | The evidence |
|---|---|
| Learning what option C teaches | the returned dev slice breaks fewer than 14 and gets more than 11 right; on the evaluation set the "right result is nothing" row breaks fewer than 7 of 10; the "look can inform" row still fixes more than it breaks |
| Learning to stop retrying everywhere | broken falls on both rows, but so does fixed (`gretel_other_fixed`, the 87-question row), and calls an episode fall toward 1.0 |
| Leaning on retries | the first query's score falls (under 51.5% the step isn't kept) while greedy with turns holds |
| Flat | everything inside 3 points of the start; sign tests like 5 better, 4 worse |
| Drifting | `kl_start` rising toward 0.05, the entropy falling, an evaluation below the one before |

One evaluation is good to about 3 points; one question is 0.5. Read the sign tests.

**Rishi hasn't set the line that counts as "run 2 worked".** The design's "+5 over 54.5" was
written before the no-training tests. Report against the two start numbers; don't call the
run a success or a failure yourself, and don't decide what comes next.

## 6. What you may and may not do on your own

- **May:** run ticks, read files, pause for Rishi's GPU, resume after a pause you made,
  restart after a crash section 7 explains, start the checker's Neo4j container.
- **Ask Rishi first:** changing any setting (`config.json`, a stop bound, the penalty, the
  learning rate, the step count, `single_turn_sets`), resuming after a stop rule, the gate or
  the turn rule paused the run, stopping the run for good, extending past 40 steps, running
  the test set, deleting anything, anything that costs money.
- **Never:** edit the run's code while it runs, start another GPU job, `wsl --shutdown` or
  `--terminate`, stop Docker, touch `.env` or credential files, put local paths in tracked
  files.

## 7. When it fails (`FAILED`, an `error` event, or `dead`)

Read the event's text, then the step's `train.log` (update) or `server.err.txt` (rollouts),
and `driver.log`.

| Cause | Fix |
|---|---|
| "the checker's Neo4j doesn't answer" | `docker start asktrainer-neo4j`, wait 30 s, the start command again |
| "another llama-server is running" | find whose it is; a leftover of this run is cleared by `pause.py`; otherwise ask Rishi |
| "N episodes failed; run again to retry them" | the start command again (finished episodes are kept) |
| "the rollout model for step k could not be written" | read `merge.log`; in WSL `python -m sqlrl.run2.tools merge --run-dir <dir> --step <k-1>`, then start again |
| "the start adapter is missing" | `<models>/<init_from>/lora_state.safetensors` must exist |
| "no start reference", or "made under another protocol" | the start adapter's evaluation with turns is missing or doesn't match this run's settings: `phase0.py heldout --samples 8` (about 40 minutes), then start again |
| "holds episodes made under another protocol" | an evaluation folder has episodes from other settings; move the folder aside, never mix them |
| The feed says `dead` (the run's process is gone, no `DONE`, `PAUSED` or `FAILED`) and the PC did not restart | **First make sure it is gone:** `tasklist /FI "PID eq <the number in run.pid>"` must list nothing, and `train.log` or `server.err.txt` of the current step must have stopped growing. A second copy started beside a live one puts two jobs on the GPU. Then read the tail of `driver.log` and the last event, and run `pause.py --run run2` once: a driver that was killed can leave its server, its trainer or its keep-alive loop in WSL behind, and `pause.py` clears them. Check the GPU is free (section 1), then the start command once. If it dies again, leave it and tell Rishi |
| A power cut or reboot | start Docker Desktop and wait, `docker start asktrainer-neo4j`, then the start command; at most one update is redone. Tell Rishi it happened |
| Docker Desktop won't start ("initializing Secrets Engine ... engine.sock") | ask Rishi: the fix that worked before changes a folder outside the repo |
| The update hangs (GPU idle, the log silent for 15 minutes; run 1b step 35) | stop only that trainer inside WSL (`kill <pid>`), check the step before's state files against their `.sha256`, then ask Rishi before resuming |
| Out of memory in the update | check nothing else holds the GPU; start again. If it repeats on the same step, stop and tell Rishi |
| Anything else | don't guess: leave the run stopped, tell Rishi what the log says |

## 8. When it pauses by itself (`PAUSED` with a reason)

A stop rule, the gate or the turn rule fired (`design.md` sections 6 and 7). Don't resume on
your own. Run a tick so the dashboard shows it, read the step's episodes for what the rule
points at, and bring Rishi: the rule, the numbers for the last 10 steps, what the episodes
look like, and your recommendation (resume as is; change a setting and resume; stop and keep
the best step).

Two pauses have a known next step, still Rishi's to choose:

- **"the drift reading at step 1 ... is above 0.025":** the offset was measured at 0.0084 on
  the start adapter. A step-1 reading this far from it means the rollout model isn't the start
  adapter merged into the SFT model (check `merge.log` and `init_from`), or the offset has
  changed (`phase0.py offset` measures it again).
- **Reward hacking rules:** read the flagged episodes first. `hardcoded` episodes already earn
  nothing; a rising `copied_frac` may be the model learning how a value is spelled (fine) or
  reading answers off rows (not fine).

A setting can only change with the run stopped: the config is read once at the start. The run
removes `STOP`, `STOPPED`, `PAUSED` and `FAILED` when it starts.

## 9. When it finishes (`DONE`): closing the run

Run a last tick. Report the step-40 evaluation as in section 5, and the whole run: every
evaluation in one table, the returned dev slice across them, the canaries, what the episodes
looked like. Then, with Rishi:

1. `BEST.json` names the step with the highest greedy with turns. Rishi confirms the step.
2. The test set (1,000 questions), once, on the kept adapter, with Rishi's OK.
3. Copy the kept adapter to `<models>/run2-step<k>/`; move the run's folder to the data drive
   as history, throttled, and check every file's hash before deleting the working copy.
4. `results/run2.md`, a model-card section for the tool protocol, `PROGRESS.md`, a decision
   entry, commit and push.

## Kickoff message for Rishi to paste

Open the session in the `NFL-RL` folder (the `sonnet-xhigh` subagent is defined in its
`.claude/agents`), on Opus, and paste:

> Read `documentation/run2/RUNBOOK.md` in `NFL-RL` and follow it. This is my go to start run 2: check
> section 1, start the run with `--launched-by agent` (section 2), then spawn the
> `sonnet-xhigh` subagent twice (section 3): once to build the live dashboard from
> `documentation/run2/dashboard-brief.md`, and once to explain the key changes made for this
> run (option C). Give me the dashboard link and the explanation in your first report. Then
> watch the run, keep the dashboard updated after every step, and send me a short note after
> each evaluation. Don't change any setting, resume after a pause the run made itself, or
> start anything else on the GPU without asking me.
