# Run 2 dashboard

One live page for run 2, published as a private claude.ai Artifact titled "Spark RL Run 2". It is
run 1b's page (`../../run1b/dashboard/`) with run 2's panels and fields. Read run 1's README
(`../../run1/dashboard/README.md`) for how the page is built (chart cards, table twins, themes, the
feed time turning amber) and `../design.md` for what run 2 is. Run 1b's page stays as run 1b's
record.

- **Source:** `run2-dashboard.html` (this folder). Plain HTML, CSS and JS, no build step. No local
  paths, user names or W&B entity names in it (the repo is public).
- **Task spec:** `../dashboard-brief.md` (every feed field). **Who feeds it, and when:**
  `../RUNBOOK.md`, section 3.

## What differs from run 1b's page

- **Reads collection `run2`** (one `onSnapshot` on the whole collection): `state`, `steps-n`,
  `evals`, `refs`. The page works before `evals` and `refs` arrive, before the first step, and when
  a `steps-n` does not exist yet. Any field may be absent or null: a gap is drawn, never an error.
- **Run 1b is not drawn behind run 2.** The protocol and the questions differ, so the steps do not
  pair. What run 2 is read against is `refs`: the start adapter with turns (the run's "step 0"), the
  start adapter on one query (run 1b's step 30) and the start adapter on the returned dev slice.
- **Header.** `dead` is its own status (red, with `why`). Tiles: step k of 40; solve rate; first-query
  solve rate; calls an episode; "Right answers broken (returned questions)" against the untrained
  share (`refs.returned_start.broken_frac`); drift from the start; latest evaluation against the
  start's 52.4%; best step so far (flagged when `best.first_query_ok` is false). The rates are
  5-step means. A cell shows a step in counts (`state.shape`).
- **"What is different in this run"**, a collapsible panel under the header (open the first time;
  the page remembers how a viewer left it).
- **Rollouts.** Solve rate and first-query solve rate on one chart with `bounds.first_solve_rate`;
  four per-dataset charts labelled "with turns" or "one query" (from `config.single_turn_sets`);
  informative share; different final and first queries a group; malformed and undecided; first- and
  later-turn length with `bounds.t1_tokens_median` low and high.
- **Turns (new).** Calls an episode with the 2.0 bound; episodes by number of calls (shares); another
  call after an empty result, an error and rows (the `_n` counts are in the tooltip); fixed and
  broken counts and `broken_frac`; `limit_frac`; `loop_frac`. Episodes with turns only.
- **Returned questions (new).** First-query and final solve rate; the broken share against the
  untrained level; the other gretel questions' fixed and broken; calls an episode for both kinds;
  `returned_weak_only_frac`; a table of the returned dev slice at the start and at each evaluation.
- **Drift.** `kl_start` with its stop bound and a zero line; the raw reading with the offset; step 1's
  reading against `stop.kl_step1_max`; `kl_sft` on its own, marked "for information"; `mismatch`;
  later turns and the trainer's view. No KL-weight chart or tile (the weight is fixed).
- **Canaries.** `hardcoded_frac`, `copied_frac`, `look_leak_frac` (often null: the card says why),
  `distinct_frac`, `select_star_frac`, `leak_frac`, `dup_differ_frac` (with `dup_probed` in the
  tooltip), `too_long_frac`, `look_error_frac`, query length. Relative bounds come from
  `state.bounds` and are drawn from step 6; before that a card says "drawn after step 5".
- **Evaluations.** A chart of greedy with turns, the first query and the strict score with the start
  as step 0, the start's two reference lines, the first-query floor and the best step as a dotted
  vertical line. A table with the start's two rows on top (fixed, broken, net, calls, the returned dev
  slice, mean@8 and pass@8 on full evaluations). By kind of question (`by_look`), per dataset, the
  sign tests (the gate on `vs.start_with_turns.pass8`, the turn rule on `vs.best.greedy`), and for
  full evaluations the pass@k curves and the never / sometimes / always split beside the start.
- **Generalising.** Per dataset, the rolling training solve rate beside the evaluation's greedy score
  (`greedy_by_set[set].decided`).
- **Optimizer, Timing, Events.** As in run 1b with run 2's fields: `entropy_t1` with its bound and
  `entropy_later`; the gradient norm with the clip at 1.0; the time split; `peak_gb` against 16 GB.

## The Artifact's store

**Collection `run2`, one document per feed file, document id = the file name without `.json`:**

| Document | Source file (`<data>/run2/run2/feed/`) | Changes |
|---|---|---|
| `state` | `state.json` | every tick (status, phase, the step in progress, time left, stop rules, bounds, settings, `best`, events) |
| `steps-1` ... `steps-4` | `steps-1.json` ... | the one that holds the newest step (10 steps a document); a new one is created when a step 11, 21 or 31 arrives |
| `evals` | `evals.json` | after an evaluation |
| `refs` | `refs.json` | never: pushed once |

**Access:** the page declares the `db` capability with the root rule `read: view, write: admin`:
every viewer reads, only the owner and editors write. Viewers never write anything.

## Push one tick by hand

After `feed.py --run run2` has written the files, push with `ArtifactData`, collection `run2`.

**The store refuses a write to an existing document unless it carries `if_version`** (the version you
last read or wrote); a plain `set` on an existing document fails with `version_mismatch` and writes
nothing (checked). The result of every `set` reports the document's new version, so a watcher can
keep the versions without listing again. A batch is all or nothing, and may mix an existing
document (with `if_version`) and a new one (without) (checked).

1. **The first push sends every file in the feed's `files` list** (`state`, every `steps-n`, `evals`,
   `refs`), none with `if_version`. Do not rely on `changed` for it: `refs` is written once and only
   `changed` lists what differs from the last tick. Before the first push the collection must be
   empty (`{"action": "list", "url": "<artifact url>", "collection": "run2"}`).

   ```json
   {
     "action": "batch",
     "url": "<artifact url>",
     "writes": [
       {"op": "set", "collection": "run2", "doc_id": "state",
        "file_path": "<data>/run2/run2/feed/state.json"},
       {"op": "set", "collection": "run2", "doc_id": "steps-1",
        "file_path": "<data>/run2/run2/feed/steps-1.json"},
       {"op": "set", "collection": "run2", "doc_id": "evals",
        "file_path": "<data>/run2/run2/feed/evals.json"},
       {"op": "set", "collection": "run2", "doc_id": "refs",
        "file_path": "<data>/run2/run2/feed/refs.json"}
     ]
   }
   ```

   Every result says `(version 1)`: keep those numbers. If a `steps-n` file exists already, add it
   with its own entry. Nothing in `files` may be skipped, even an `evals` that holds `{"evals": []}`.

2. **Every later tick: the files in `changed`, one `batch`, each with the version the last write
   reported.** `state` changes every tick; one `steps-n` a step; `evals` after an evaluation; `refs`
   never again.

   ```json
   {
     "action": "batch",
     "url": "<artifact url>",
     "writes": [
       {"op": "set", "collection": "run2", "doc_id": "state",
        "file_path": "<data>/run2/run2/feed/state.json", "if_version": 3},
       {"op": "set", "collection": "run2", "doc_id": "steps-1",
        "file_path": "<data>/run2/run2/feed/steps-1.json", "if_version": 2}
     ]
   }
   ```

   A document that does not exist yet (`steps-2` when step 11 arrives) is written without
   `if_version`. If a batch fails with `version_mismatch`, list again
   (`{"action": "list", "url": "<artifact url>", "collection": "run2", "out_dir": "<scratch dir>"}`:
   every document is reported with its `version`) and resend with the current versions.

3. **If a push failed after `feed.py` had run, the next tick's `changed` will not list those files**
   (it compares with the last file on disk, not with the store). Push the whole `files` list that
   tick instead.

Pushing a feed again is idempotent (the files are rewritten whole). To empty the store (a test feed
before the real run), `delete` every document with its current version
(`{"op": "delete", "collection": "run2", "doc_id": "steps-1", "if_version": <n>}`), then list to
confirm it is empty.

## Testing the page without the store

The page only needs `window.claude.use('db')` to return an object whose `collection('run2')
.onSnapshot(cb)` calls `cb` with `{docs: [{id, exists, data()}], metadata: {fromCache}}`. A small
static page that stubs that and serves the feed files as JSON is enough to render every state:
nothing pushed, `state` and `refs` only (the first tick), a few steps, a full evaluation, `paused`
and `dead`. The page was checked that way against `smoke-6`'s feed and a made-up 12-step feed with
bounds and one full evaluation, at phone width and in both themes.

## Changing the page

Edit `run2-dashboard.html`, then publish the same file to the same Artifact URL (the store and its
rules are kept). A new chart is one `stepCard(...)` call with a `build` function that returns a
`lineSpec({...})`: a series from `seriesOf('<field>', {roll: true})`, reference lines in `hlines`
(`relLine('<bounds key>', ...)` for a stop line read from `state.bounds`, `fixedLine('<stop key>', ...)`
for one read from `config.stop`), `noPlot` on a series (tooltip and table only, like `dup_probed`) and
`vlines` (a vertical marker). The chart, tooltip, keyboard reading and table twin come with it.
