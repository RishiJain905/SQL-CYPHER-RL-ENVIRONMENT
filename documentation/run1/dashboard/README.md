# Run 1 dashboard

One live page for run 1, published as a private claude.ai Artifact. It redraws after every training
step and replaces opening W&B: the series are the same numbers W&B receives as `train/*`.

- **Source:** `run1-dashboard.html` (this folder). Plain HTML, CSS and JS, no build step. No local
  paths, user names or W&B entity names in it (the repo is public).
- **Task spec:** `../dashboard-brief.md`. **Who feeds it, and when:** `../RUNBOOK.md`, section 3.

## What the page shows

A sticky bar (status, step k of N, progress, "updated ... ago"), then a header with the phase, the
step in progress (rollouts written), the time left and the six numbers that matter (solve rate,
reward, informative groups, KL to SFT, entropy, pass@8), each with its bound and an inside / outside
mark. A banner appears when the run is paused, failed or stopped (with the reasons) and when a stop
rule fires on the latest step. Then nine panels:

| Panel | Charts |
|---|---|
| Rollouts | reward and solve rate (raw + 5-step rolling mean); solve rate per dataset (four small multiples, each with the SFT model's `mean_at8` as a dotted reference); informative share (0.25 bound); malformed and undecided shares (0.05 bound); response length (2x and 1/2x of the first five steps' median); forced `</think>` share; rollout tokens/s |
| Optimizer | loss; gradient norm; learning rate (warm-up shaded); clip fraction; entropy (bound: half of the first five steps' mean); sequences trained and distilled; distillation JSD; teacher weight |
| Drift | `kl_sft` (server, the true one) and `kl_sft_trainer` on one chart with the 0.05 bound; `mismatch` with the 0.10 bound; a line on why the trainer under-reads a small LoRA |
| Canaries | DISTINCT share (start + 10 points), SELECT * share, several-tool-calls share, solution-citing thoughts (0.02 bound), median query length |
| Evaluations | greedy pass@1, mean@8 and pass@8 at the SFT baseline (filled), the SFT model re-measured through run 1's path (hollow rings) and each evaluated step, overall and per dataset; a checkpoint table with the never / sometimes / always split and differences against the baseline; the three sign tests with the gate (pass@8) marked and its verdict |
| Generalising | per dataset, the rolling training solve rate next to the evalset's `mean@8`, a numbers line (training change, evalset change) and the overfitting reading rule |
| Timing | stacked seconds per step (rollouts, scoring + load, update, merge, other) and peak GPU memory against the 16 GB card |
| Events | the last 40 driver events, newest first, warnings and errors marked in words |

Notes on reading it:

- Every chart has hover values (crosshair; left / right arrow keys when focused) and a **Table** twin
  (per card, or all at once with "Charts as: Tables"). Dashed red lines are stop bounds, dotted grey
  lines are the SFT model.
- "Steps shown" (All / Last 50 / Last 25) scopes the step charts. The evaluation and generalising
  charts always show the whole run.
- Timing: `step_s` = rollouts (`rollout_s` + `serve_s`) + update. "Scoring + load" is the longer of
  `score_s` and `load_s` (scoring runs on the CPU while the model loads), "update" is `train_s`,
  "merge" is `merge_s`, and "other" is whatever is left of `step_s`.
- The "bound starts after step 5" rule copies the stop rules: the relative bounds (entropy, response
  length, DISTINCT) need more than five steps.
- The "reading" line under a generalising chart is a reading aid, not a gate: it compares the
  training solve rate (last 5 steps against the first 5) with the evalset's `mean@8` (latest
  evaluation against the SFT baseline) and calls a gap of about one point flat.
- The feed time turns amber and says "feed may be stale" when the run is `running` and the state is
  more than 45 minutes old.
- Colours: the chart series are categorical slots 1-4 of the dataviz palette, validated in both
  themes (adjacent-pair colour-blind separation passes; the light-theme aqua and yellow sit below
  3:1 contrast on the card, so every value is also in the tooltips, the legends and the table twin).
  The page follows the viewer's light or dark theme.

## How it is fed

`scripts/run1/feed.py --run <name>` writes small JSON files into `<data>/run1/<name>/feed/`, rewritten
whole each time: `state.json` and `steps-1.json`, `steps-2.json`, ... (25 steps each). The watching
agent pushes them into the page's shared database after every step. The page fetches nothing else.

**Store layout:** collection `run1`, one document per file, document id = the file name without
`.json`: `state`, `steps-1`, `steps-2`, `steps-3`, `steps-4`. The page subscribes once to the whole
collection (`onSnapshot`) and redraws on every change. A missing `steps-n` is fine: the page shows
what it has and says "Showing x of y trained steps" when `state.step_now` is ahead of the rows.

**Access:** the page declares the `db` capability with the root rule `read: view, write: admin`:
every viewer reads, only the owner and editors can write. Viewers never write anything.

**Empty and failure states:** before the first push the page shows "Waiting for the first feed".
Before step 1 it shows the baseline and the bounds. If the store cannot be reached it says so after
10 seconds and keeps listening; it never reloads.

## Push one tick by hand

After `feed.py --run run1` has written the files, push each one with `ArtifactData`, collection
`run1`, document id = the file name without `.json`. `state` every tick; only the `steps-n` file that
changed (the newest, plus a new one when a 26th, 51st, ... step arrives).

**The store refuses a write to an existing document unless it carries `if_version`** (the version
you last read or wrote). So a tick is: list the versions, then one `batch` of `set`s.

1. Read the versions (use `out_dir` so the documents are saved instead of printed):

   ```json
   {"action": "list", "url": "<artifact url>", "collection": "run1", "out_dir": "<scratch dir>"}
   ```

   Each document is reported with its `version`. A result of a `set` also reports the new version,
   so a watcher can keep them without listing again.

2. Push, one entry per file; `if_version` is the document's current version, left out for a document
   that does not exist yet (the first push, a new `steps-n`):

   ```json
   {
     "action": "batch",
     "url": "<artifact url>",
     "writes": [
       {"op": "set", "collection": "run1", "doc_id": "state",
        "file_path": "<data>/run1/run1/feed/state.json", "if_version": 7},
       {"op": "set", "collection": "run1", "doc_id": "steps-1",
        "file_path": "<data>/run1/run1/feed/steps-1.json", "if_version": 7},
       {"op": "set", "collection": "run1", "doc_id": "steps-2",
        "file_path": "<data>/run1/run1/feed/steps-2.json"}
     ]
   }
   ```

   `file_path` is the local feed file (the tool sends its top-level object as the document). A batch
   is all or nothing; if it fails with `version_mismatch`, list again and resend with the current
   versions.

Pushing a feed again is idempotent (the files are rewritten whole). To switch the page from a test
feed to the real run, push the real run's files over the same ids: any `steps-n` document the real
run does not have yet must be deleted (`delete` on `run1/steps-n`), or the old rows would show.

## Changing the page

Edit `run1-dashboard.html`, then publish the same file to the same Artifact URL (the store and its
rules are kept). A new chart is one `makeCard(...)` call: a title, an optional legend, and a `build`
function that returns a spec (series, bounds, axis kind, formatter) from `D.rows` and `D.state`; the
chart, tooltip, keyboard reading and table twin come with it. A new field in the feed needs nothing
else: the rows are flat, and `seriesOf('<field>')` reads it.
