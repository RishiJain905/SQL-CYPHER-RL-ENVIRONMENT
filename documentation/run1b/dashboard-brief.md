# Run 1b dashboard: the brief

A task for the **sonnet-xhigh** subagent (Rishi asked for the same setup as run 1: an Opus
session starts and watches the run, a Sonnet agent builds the page). The watching agent
(`RUNBOOK.md`) spawns it and then keeps the page fed.

## What to build

Run 1's page, for run 1b, with run 1 drawn behind it. Don't design a new page: **copy
`documentation/run1/dashboard/run1-dashboard.html` to
`documentation/run1b/dashboard/run1b-dashboard.html`** and change what is listed below. Read
`documentation/run1/dashboard/README.md` (how the page is built and fed) and
`documentation/run1/dashboard-brief.md` (every field of the feed) first, then
`documentation/run1b/design.md` (what run 1b changes and why).

- Publish it as a **new** private Artifact (run 1's page stays as run 1's record). Title:
  "Spark RL Run 1b".
- **Load these skills before writing:** `artifact-design`, `artifact-capabilities`, `dataviz`.
- No local paths, user names or W&B entity names in the file (the repo is public).
- A short `documentation/run1b/dashboard/README.md`: what differs from run 1's page, the
  Artifact's store layout, how to push a tick by hand.

## The feed

`scripts/run1/feed.py --run run1b` writes `<data>/run1/run1b/feed/`. **Store layout:**
collection `run1b`, one document per file, document id = the file name without `.json`:
`state`, `steps-1`, `steps-2` (25 steps each; run 1b has 40 steps), and **`compare`** (new).

What is new in the feed (everything else is as in run 1's brief):

| Where | Field | Meaning |
|---|---|---|
| `steps-n` rows | `kl_beta` | the KL penalty's weight this step (starts at 0.05; doubles while `kl_sft` is above 0.045, halves back toward 0.05 while it is under 0.02) |
| | `kl_pen` | the KL the penalty saw: the trainer's own view of KL(policy \|\| SFT) a token on the trained sequences. Lower than `kl_sft` while the LoRA is small (the trainer under-reads it) |
| | `dup_differ_frac` | of the CypherBench answers the checker passed, the share the duplicate probe withheld the reward from (answers that repeat rows where the reference doesn't). About 0.45-0.50 at the start. Falling is good; rising is the habit run 1 lost |
| | `dup_probed` | how many answers the probe judged this step |
| `state.config` | `preset` (`run1b`), `method_cfg.kl` (`beta`, `target` 0.03, `beta_max`, `top_k`), `dup_probe`, `turn_p` (0.10), `compare_run` (`run1`), `stop.dup_up` (0.15), `stop.dup_steps` (3), `stop.kl_max` (0.10) | the new settings |
| `state.evals[i]` | `vs_best` | `{step, greedy, sampled, pass8}`: the three sign tests against the run's best *earlier* evaluation (absent on the first one). `sampled` worse with p < `turn_p` pauses the run |
| `state` | `best` | `{step, mean_at8, greedy_pass1, pass_at8, evaluated, lora}`: the evaluated step to keep so far (highest mean@8) |
| `compare` | `run` (`run1`), `evals` (run 1's evaluations after steps 19, 25, 28, same blocks as `state.evals`), `steps` (one row per run 1 step: `step`, `solve_rate`, `reward_mean`, `kl_sft`, `entropy`, `informative_frac`, `resp_tokens_median`, `distinct_frac`, `lr`, `solve_<set>`) | run 1, to draw behind run 1b. Written once; it never changes |

Run 1b uses **the same questions as run 1 at every step**, so its solve rates pair with run 1's
step for step. Say so on the page.

## What to change on the page

1. **Read collection `run1b`** (one `onSnapshot` on the whole collection, as now). The page must
   work before `compare` arrives.
2. **Run 1 behind run 1b.** On these step charts add run 1's series as a muted, dashed line with
   its own legend entry ("run 1"): solve rate (the rolling mean only), solve rate per dataset,
   `kl_sft`, entropy, response length, the DISTINCT share. Run 1 has 28 steps, run 1b 40.
3. **Drift panel.** The `kl_sft` chart gets three reference lines instead of one: the target
   (0.03, labelled "target"), the band's upper edge (0.045, "weight doubles above"), the stop
   bound (`config.stop.kl_max`, 0.10). Use a log y-axis or a second zoomed chart if the early
   values (0.0002-0.005) disappear on a linear one. Add a chart of `kl_beta` (a step line; log
   y-axis from 0.05 to `beta_max`) and put `kl_pen` on the trainer-KL chart beside
   `kl_sft_trainer`. One line under them: "The weight reacts to the server's KL; the penalty
   itself is computed from the trainer's view, which under-reads a small LoRA."
4. **Header tiles.** "KL to SFT" shows the value against the target and the stop bound (inside
   the band / above the band / over the bound). Add a tile "KL weight" (`kl_beta`).
5. **Canaries panel.** Add "Answers that repeat rows (CypherBench)": `dup_differ_frac` with its
   bound (the mean of steps 1-5 plus `config.stop.dup_up`, drawn after step 5) and `dup_probed`
   in the tooltip.
6. **Evaluations panel.** Evaluations are after steps 10, 20, 30 and 40. Show run 1's three
   evaluations beside run 1b's (from `compare.evals`, marked "run 1, step 19" and so on) in the
   charts and the checkpoint table. Mark the best step so far (`state.best.step`) and add the
   `vs_best` sign tests under the `vs_baseline` ones, with the pause rule in words ("pauses when
   the sampled test is worse than the best earlier evaluation with p < 0.10"). Before the first
   evaluation: "first evaluation after step 10".
7. **Text.** Every "run 1" in titles and notes becomes "run 1b"; the bound texts that quote 0.05
   for KL read the config instead.

## Deliverable back to the caller

- The page file and the README, published as a private Artifact; the Artifact URL.
- The exact `ArtifactData` calls that push one tick into collection `run1b` (with `if_version`
  for documents that exist), tested once with a real feed: a finished loop test's feed is under
  `<data>/run1/smoke-1b/feed/` (2 steps, no evaluation, no `compare`), and
  `feed.py --run run1b` writes the real one, with `compare`, as soon as the run has started.
- A check of the page at phone width and in both themes.
