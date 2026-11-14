# Run 1b dashboard

One live page for run 1b, published as a private claude.ai Artifact titled "Spark RL Run 1b". It is
run 1's page (`../../run1/dashboard/`) with run 1 drawn behind run 1b and the run 1b additions
below. Read run 1's README for how the page is built (chart cards, table twins, themes, the feed
time turning amber) and `../design.md` for what run 1b changes and why. Run 1's own page stays as
run 1's record.

- **Source:** `run1b-dashboard.html` (this folder). Plain HTML, CSS and JS, no build step. No local
  paths, user names or W&B entity names in it (the repo is public).
- **Task spec:** `../dashboard-brief.md`. **Who feeds it, and when:** `../RUNBOOK.md`.

## What differs from run 1's page

- **Reads collection `run1b`** (one `onSnapshot` on the whole collection). The page works before
  `compare` arrives (the run 1 legend entries stay hidden until it does) and before the first step.
- **Run 1 behind run 1b**, a muted dashed line with its own legend entry ("run 1"): solve rate (its
  5-step rolling mean only), solve rate per dataset (rolling mean), `kl_sft`, entropy, response
  length (run 1's median), the DISTINCT share. Run 1 has 28 steps, run 1b 40. A note on the
  Rollouts panel says both runs see the same questions at every step, so the solve rates pair.
- **Drift panel.** `kl_sft` on a log axis (the early values are 0.0002-0.005, run 1 ended near 0.3)
  with three reference lines: the target (0.03, dotted teal), the band's top edge (0.045, amber
  dashed, "weight doubles above") and the stop bound (`config.stop.kl_max`, 0.10, red dashed). The
  target and the edge come from `config.method_cfg.kl.target` (edge = 1.5 x target). A `kl_beta`
  step chart on a log axis from the weight's floor to `beta_max` (0.05 to 6.4, a tick at each
  doubling). A trainer-KL chart with `kl_sft_trainer` and `kl_pen`, plus the server's `kl_sft` for
  scale. The line "The weight reacts to the server's KL; the penalty itself is computed from the
  trainer's view, which under-reads a small LoRA."
- **Header tiles.** "KL to SFT" reads inside / above the band, or over the stop bound (and "under
  the band" below 0.02, where the weight halves). A new "KL weight" tile (`kl_beta`, with "at the
  floor", "raised: Nx the floor", "at the ceiling"). A "Best step so far" tile (`state.best`). The
  solve-rate tile adds run 1's rolling mean at the same step.
- **Canaries panel.** "Answers that repeat rows (CypherBench)": `dup_differ_frac` with its bound (the
  mean of steps 1-5 plus `config.stop.dup_up`, drawn after step 5) and `dup_probed` in the tooltip
  and the table twin. The DISTINCT chart reads its bound from the config.
- **Evaluations panel.** Evaluations are after steps 10, 20, 30 and 40 (`config.eval_steps`). Run
  1's three evaluations (from `compare.evals`, after its steps 19, 25 and 28) are dashed lines with
  square marks in the charts and rows "Run 1, after step N" under a separator in the checkpoint
  table. The best step so far (`state.best.step`) is a dotted vertical line in the charts and a "best
  so far" chip in the table, with a one-line summary above the table. A second sign-test table
  under the `vs_baseline` one shows `vs_best` (each evaluation against the best earlier one, the
  sampled test highlighted) and the pause rule in words (read from `config.turn_p`). Before the
  first evaluation: "first evaluation after step 10".
- **Text.** "Run 1" in titles and notes is "run 1b" except where run 1 is the comparison; the
  bounds that quoted run 1's numbers (KL, informative, malformed, DISTINCT, the probe) read the
  config. "Steps shown" is All / Last 20 / Last 10 (40 steps).

## The Artifact's store

**Collection `run1b`, one document per feed file, document id = the file name without `.json`:**

| Document | Source file | Changes |
|---|---|---|
| `state` | `<data>/run1/run1b/feed/state.json` | every tick (phase, rollouts done, events, evaluations, `best`, status) |
| `steps-1` | `steps-1.json` (steps 1-25) | every step until step 25 |
| `steps-2` | `steps-2.json` (steps 26-40) | every step from step 26 |
| `compare` | `compare.json` (run 1) | never: pushed once, not again |

**Access:** the page declares the `db` capability with the root rule `read: view, write: admin`:
every viewer reads, only the owner and editors write. Viewers never write anything.

## Push one tick by hand

After `feed.py --run run1b` has written the files, push with `ArtifactData`, collection `run1b`.

**The store refuses a write to an existing document unless it carries `if_version`** (the version
you last read or wrote); a plain `set` on an existing document fails with `version_mismatch` and
writes nothing (checked). The result of every `set` reports the document's new version, so a
watcher can keep the versions without listing again.

**What changes each step:** `state` always, plus the one `steps-n` that holds the newest step
(`steps-1` for steps 1-25, `steps-2` for 26-40). A tick between steps (a phase change, rollouts
done, an evaluation running, its result) changes `state` only. `compare` was written once; leave it.
A new `steps-n` (step 26 creates `steps-2`) is created with no `if_version`.

1. Read the versions, once, then keep them from the `set` results (`out_dir` saves the documents
   instead of printing them):

   ```json
   {"action": "list", "url": "<artifact url>", "collection": "run1b", "out_dir": "<scratch dir>"}
   ```

2. Push, one `batch` per tick:

   ```json
   {
     "action": "batch",
     "url": "<artifact url>",
     "writes": [
       {"op": "set", "collection": "run1b", "doc_id": "state",
        "file_path": "<data>/run1/run1b/feed/state.json", "if_version": 3},
       {"op": "set", "collection": "run1b", "doc_id": "steps-1",
        "file_path": "<data>/run1/run1b/feed/steps-1.json", "if_version": 7}
     ]
   }
   ```

   The first push of a document leaves `if_version` out. A batch is all or nothing; if it fails with
   `version_mismatch`, list again and resend with the current versions.

Pushing a feed again is idempotent (the files are rewritten whole). To switch the page from a test
feed to the real run (for example `smoke-1b`'s), push the real run's files over the same ids and
`delete` any `steps-n` the real run does not have yet (`{"op": "delete", "collection": "run1b",
"doc_id": "steps-1", "if_version": <n>}`), or the old rows would show.

## Changing the page

Edit `run1b-dashboard.html`, then publish the same file to the same Artifact URL (the store and its
rules are kept). A new chart is one `stepCard(...)` call with a `build` function that returns a
`lineSpec({...})`: a series from `seriesOf('<field>')` (run 1b) or `cmpSeries('<field>')` (run 1,
from `compare.steps`), reference lines in `hlines`, and, where needed, `log: true` (log axis),
`stepLine` on a series (step path), `noPlot` on a series (tooltip and table only, like `dup_probed`)
and `vlines` (a vertical marker). The chart, tooltip, keyboard reading and table twin come with it.
