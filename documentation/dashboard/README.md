# The live RL baseline page

A claude.ai Artifact that shows the step 1 measurement (the sampling baseline in
[`../rl.md`](../rl.md)) while it runs: how far the run has got, and, as the checker scores the
samples, how much room RL has. It is built only from the run's own files, never from typed-in
numbers.

| | |
|---|---|
| Page | **Spark RL Baseline** (`rl-baseline.html`, one file: SVG charts drawn by its own script, no libraries, light and dark) |
| URL | https://claude.ai/artifact/JcR5uM76GHeJUCf1UkHooJ (private: only the owner and people given access can open it) |
| Reads | the artifact's database (`db` capability), live: it updates as new documents arrive |
| Writes | nothing. The page only reads the data store and never writes to it. |

## What the page shows

1. **Header.** Run name, status pill (starting, sampling, scoring, done, stalled, failed), a "Live"
   flag while the last update is under 10 minutes old (a "No update for N min" warning when an
   active run goes quiet), "Updated 3 min ago", and the three phases (sampling, scoring, done).
2. **Progress.** The sampling bar (answered of total), answers per minute, time left, errors, answers
   cut by length, mean completion tokens, one meter per sample file, a sparkline of answers over
   time (from the older updates), and the scoring bar (k of 8 sample files scored).
3. **Results**, once the first sample file has been scored. They are partial until the status is
   `done`, and the page says so ("Partial: based on N of 8 samples"). Headline tiles (greedy
   pass@1, mean@8, pass@8, room for RL, the learning pool), a per-dataset table (greedy pass@1,
   mean@8, pass@8, the gap, never / sometimes / always, undecided answers), the pass@k curve for
   k = 1..8 (one line per dataset plus overall, greedy as hollow markers, with a table view), the
   0..8 histogram (overall or per dataset), and a never / sometimes / always bar per dataset.
4. **Every question.** One cell per question, grouped by dataset, colored by how many of its
   answers were right. A ringed cell is an easy win: the greedy answer was wrong but at least one
   sample was right. Hover or tap a cell for its id, count and greedy result.
5. **How to read this**, a short explainer of pass@1, pass@8 and the buckets for RL.
6. **Settings** and the **latest events**.

Before the first update arrives the page says "Waiting for the first update". It holds no sample
or placeholder numbers.

## The data store (the contract)

Access rules declared with the page: `{path: "runs", read: "view", write: "admin"}`. Everyone who
can open the page reads; only the owner and editors write. Claude pushes the data; the page never
does.

| Path | Holds |
|---|---|
| `runs/<run>` (collection `runs`, doc id = the run name, e.g. `baseline-sft1500-t1.0`) | `{name, kind: "baseline", title, started_at (ISO), last_tick_at (ISO), model, notes}` |
| `runs/<run>/ticks/t00001`, `t00002`, ... (append-only) | one **full snapshot** per update, below |

The page shows the newest run (by `last_tick_at`, else `started_at`); with more than one run a
picker appears. For that run it subscribes to its ticks ordered by `seq` descending (limit 200)
and shows the one with the highest `seq`; the older ticks only feed the sparkline.

A tick:

```
{
 "seq": 12, "made_at": "2026-11-01T00:40:00Z",
 "status": "starting" | "sampling" | "scoring" | "done" | "stalled" | "failed",
 "note": "short human sentence about the current state",
 "settings": {"model", "questions": 200, "samples": 8, "temperature": 1.0, "top_p": 1.0,
              "thinking_budget": 1024, "max_tokens": 2048, "checker": 4, "seeds": [1001, ...]},
 "sampling": {"answered": 640, "total": 1600, "per_sample": [200,200,200,40,0,0,0,0],
              "answers_per_min": 31.5, "eta_s": 1820 (or null), "errors": 0,
              "cut_by_length": 3, "mean_completion_tokens": 412.0},
 "scoring": {"scored_samples": 0, "total_samples": 8},
 "results": null | {"overall": BLOCK, "by_set": {"cypherbench": BLOCK, "gretel": BLOCK,
                                                  "sqale": BLOCK, "text2cypher": BLOCK}},
 "questions": [] | [{"id", "origin", "greedy": true|false|null, "right": 3, "scored": 8}, ...],
 "events": [{"at": ISO, "kind": "info"|"warn"|"error", "text"}]   (last ~20)
}

BLOCK = {"questions": 50, "greedy_pass1": 0.43 | null, "mean_at8": 0.39,
         "pass_at_k": [p1, ..., p8], "always": 12, "sometimes": 25, "never": 13,
         "hist": [n0, ..., n8], "undecided_samples": 7}
```

Fractions are 0-1 (the page shows percentages). Every value may be `null` or missing; the page
renders what it has. `hist`, `always`, `sometimes` and `never` count only questions with all 8
samples scored, so they stay empty until the last sample file is scored and the page says "waiting
for all 8 samples". The page reads `undecided_samples` as the number of answers the checker could
not settle.

## Pushing an update

Each update is a new document, never an overwrite: `ArtifactData set` on collection
`runs/<run>/ticks`, doc id `t<seq:05d>`, with the tick JSON. Write `runs/<run>` once at the start
and `update` its `last_tick_at` with each tick, so the page can list and rank runs. A document
that already exists is refused, so pushing a tick twice is harmless.

## Changing the page

Edit `rl-baseline.html` and republish it to the same URL (same file path, no `icon`), keeping the
same `capabilities`. Keep the page read-only, and keep local paths, user names and machine names
out of it: the repo is public.
