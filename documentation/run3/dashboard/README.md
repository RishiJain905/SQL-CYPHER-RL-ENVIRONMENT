# Run 3 dashboard (phases 0 and 1)

One live page for run 3's phase 0 (step 40's pass@8, the useful band) and phase 1 (the warm
start), published as a private claude.ai Artifact titled "Run 3 Warm Start". Same colours, type
and chart rules as run 2's page (`../../run2/dashboard/`), cut down to what these phases
produce.

- **Source:** `run3-dashboard.html`. Plain HTML, CSS and JS, no build step, no local paths.
- **Feed:** `python -m sqlrl.run3.feed` writes `<data>/run3/feed/{state,phase0,train,evals}.json`
  from the run's files (`phase0/anchor-*`, `phase0/pass8-step40`, `phase0/dev-step40`,
  `phase1/metrics.jsonl`, `phase1/ckpt-*`, `phase1/eval-<ckpt>/dev`, `phase1/status.json`) and
  lists which files changed.
- **The store:** collection `run3`, one document per feed file (id = file name without `.json`),
  root rule `read: view, write: admin`. Pushed with `ArtifactData` exactly as run 2's README
  describes: one `batch`, each existing document with the `if_version` its last write reported.

## What it shows

- The bar: status and phase, the optimiser step of the total, how long ago the feed was written
  (amber after 20 minutes while running).
- Tiles: the anchor, phase 0's groups and useful share, training progress, the loss, the latest
  evaluation-set and dev-set scores, gates passed.
- **Phase 1's gates** (`documentation/run3/phase0.md`), each pass / fail / pending with its value
  and bound, from the latest checkpoint that has been evaluated.
- **Phase 0:** never / useful / always per pool and engine with the 15% line, and the same by
  harness variant (schema shown x tools).
- **Training:** loss on the model's own tokens (each step and a 10-step mean), gradient norm
  against the 1.0 clip, learning rate, peak GPU memory against 16 GB.
- **Evaluations:** run 2's evaluation set by checkpoint against the anchor and the 57.0 gate; the
  dev set's right answers against step 40; behaviour (malformed, declines, looks before
  submitting with no schema); the dev set by pool and engine.
- **Events:** the orchestrator's notes, newest first.

Checked before publishing with stubbed documents, in both themes and at phone width.
