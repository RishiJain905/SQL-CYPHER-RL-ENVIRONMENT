# The method pilots: tracker and live page

Step 2 of the RL plan (`../rl.md`): short stage-A training runs of each candidate method, same
budget, each scored with step 1's measurement. This folder holds the written tracker and the live
page that follows the runs.

| File | What it is |
|---|---|
| `TRACKER.md` | The written tracker: the budget, how pilots are judged, a status table, a section per pilot, and the decision (which method goes to the real run, and why). Updated by hand. |
| `pilot-tracker.html` | The live page, one file: SVG charts drawn by its own script, no libraries, light and dark, same look as the baseline page in `../dashboard/`. |
| `README.md` | This file: the page's address and its data contract. |

| | |
|---|---|
| Page | **Spark RL Pilots** (`pilot-tracker.html`) |
| URL | https://claude.ai/artifact/V9TGXUsRRRw493gZE74QuC (private: only the owner and people given access can open it) |
| Reads | the artifact's database (`db` capability), live: it updates as new documents arrive |
| Writes | nothing. The page only reads the data store and never writes to it. |

## What the page shows

1. **Summary.** How many pilots are in each state, the budget every pilot shares (taken from the
   pilots' own configs, with a warning if they differ), and the baseline numbers every pilot is
   compared with.
2. **All pilots.** One row per pilot: status, steps done of the budget, a reward sparkline, greedy
   pass@1, mean@8 and pass@8 with the change against the baseline, the sign test, and a rough
   reading (candidate, check pass@8, pass@8 fell, no pass@1 gain). On a phone each row is a card.
3. **Evaluated pilots against the baseline.** Grouped bars of greedy pass@1, mean@8 and pass@8 for
   every evaluated pilot, with the baseline as a dashed line in each group. A table view is one
   click away. Smoke runs are left out unless switched on.
4. **Pilot detail.** Pick a pilot (the running one by default):
   - status, live flag, config chips, progress, latest and best reward, step time, time left;
   - the training reward curve (overall and per dataset), solve rate and informative-group share;
   - small charts for KL, entropy, response length, loss, clip fraction, gradient norm, learning
     rate, and any other numeric field a method logs (for example `distill_loss`);
   - where a step's time goes (rollouts, scoring, training);
   - the canaries (entropy and response length flagged when they move by more than 50% and 40%
     since the first 3 steps; KL shown);
   - the evaluation once it exists: tiles, pass@k against the baseline, never / sometimes /
     always against the baseline, a per-dataset table with changes, and the sign test;
   - the latest events.
5. **How pilots are judged**, a short reminder of the rules in `TRACKER.md`.

Before any pilot is registered the page says "No pilot has started". It holds no sample or
placeholder numbers, and a missing value shows as a dash.

The "reading" column is a rough guide from the judging rules, not the decision. The decision is
written in `TRACKER.md`.

## The data store (the contract)

Access rules declared with the page: `{path: "pilots", read: "view", write: "admin"}` and
`{path: "reference", read: "view", write: "admin"}`. Everyone who can open the page reads; only the
owner and editors write. Claude pushes the data; the page never does.

| Path | Holds |
|---|---|
| `pilots/<id>` (collection `pilots`, doc id = the pilot id, e.g. `p1-grpo`) | `{name, method, title, status, smoke, config, started_at, last_tick_at, notes}` |
| `pilots/<id>/ticks/t00001`, `t00002`, ... (append-only) | one **full snapshot** per update, below |
| `reference/baseline` (collection `reference`, doc id `baseline`) | the baseline's numbers, below |

Pilot ids in use: `p0-smoke`, `p1-grpo`, `p2-dapo`, `p3-gspo`, `p4-sdpo`, `p5-srpo`. Each pilot
keeps the same color on the page; any other id gets a spare one.

```
pilots/<id> = {
  "name": "p1-grpo", "method": "Dr. GRPO", "title": "Dr. GRPO, token-level loss, KL to SFT",
  "status": "planned" | "running" | "evaluating" | "done" | "failed" | "stopped",
  "smoke": false,
  "config": {"steps": 40, "questions_per_step": 32, "group_size": 8, "temperature": 1.0,
             "lr": 1e-5, "kl_beta": 0.02, "clip": 0.2, "lora_r": 16, ...free-form},
  "started_at": ISO, "last_tick_at": ISO, "notes": "free text"
}
```

A tick (a **full** snapshot, so the page only ever reads the newest one):

```
{
 "seq": 7, "made_at": ISO,
 "status": "planned" | "running" | "evaluating" | "done" | "failed" | "stopped",
 "note": "short sentence about the current state",
 "step": 12, "steps_total": 40, "eta_s": 21600 | null,
 "steps": [   // every finished RL step so far
   {"step": 1, "reward_mean": 0.41,
    "reward_by_set": {"cypherbench": 0.4, "gretel": 0.6, "sqale": 0.38, "text2cypher": 0.15},
    "solve_rate": 0.41, "informative_frac": 0.5, "undecided_frac": 0.07,
    "loss": 0.01, "kl": 0.0004, "entropy": 0.62, "resp_tokens": 310, "clip_frac": 0.02,
    "grad_norm": 0.8, "lr": 1e-5,
    "rollout_s": 380, "score_s": 60, "train_s": 420, "step_s": 900}, ...],
 "eval": null | {
    "greedy_pass1": 0.45, "mean_at8": 0.40, "pass_at_k": [p1, ..., p8],
    "always": 30, "sometimes": 94, "never": 74,
    "by_set": {"<set>": {same keys, except vs_baseline}},
    "vs_baseline": {"better": 20, "worse": 9, "p": 0.06}},
 "events": [{"at": ISO, "kind": "info" | "warn" | "error", "text": "..."}]
}
```

```
reference/baseline = {
  "name": "baseline-sft1500-t1.0",
  "greedy_pass1": 0.432, "mean_at8": 0.369, "pass_at_k": [p1, ..., p8],
  "always": 30, "sometimes": 94, "never": 74,
  "by_set": {"cypherbench": {greedy_pass1, mean_at8, pass_at_k, always, sometimes, never}, ...}
}
```

Fractions are 0-1 (the page shows percentages). Every value may be `null` or missing; the page
renders what it has. A few details the page relies on:

- **Newest tick.** For each pilot the page subscribes to `pilots/<id>/ticks` ordered by `seq`
  descending, limit 1, and shows that tick. It never reads the older ticks, so each tick must hold
  everything: all the finished steps and the eval so far.
- **Status.** The tick's `status` is shown. If the pilot doc says `done`, `failed` or `stopped` and
  the newest tick still says something else, the pilot doc wins.
- **Other metrics.** Any numeric field in a step that is not in the list above (for example
  `distill_loss` for SDPO) gets its own small chart, up to 12. A metric a method does not log
  simply has no chart.
- **The `eval` block.** `vs_baseline` is the question-by-question sign test against the baseline:
  how many questions the pilot got right that the baseline did not (`better`), the reverse
  (`worse`), and a two-sided p. A p under 0.05 is shown as clear. `always`, `sometimes` and
  `never` are the question counts by how many of the 8 sampled answers were right.
- **Per-dataset changes** need `reference/baseline.by_set` with the same keys as the eval's.
- **`smoke: true`** marks a loop test: tagged on the page, left out of the comparison chart, and
  left out of the shared-budget check.
- **Live flag.** A pilot that is running or evaluating shows "Live" while its last update is
  recent (25 minutes, or twice its latest step time if that is longer), and "No update for N min"
  after that.
- **Size.** A tick holds every finished step, about 0.5 KB a step, far under the store's 256 KiB
  document limit.

## Pushing an update

Each update is a new document, never an overwrite: `ArtifactData set` on collection
`pilots/<id>/ticks`, doc id `t<seq:05d>`, with the full tick JSON. Write `pilots/<id>` once when
the pilot is registered, and `update` its `status` and `last_tick_at` with each tick. Write
`reference/baseline` once. A document that already exists is refused, so pushing a tick twice is
harmless.

## Changing the page

Edit `pilot-tracker.html` and republish it to the same URL (same file path, no `icon`), keeping the
same `capabilities`. Keep the page read-only, and keep local paths, user names and machine names
out of it: the repo is public.
