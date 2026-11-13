# Run 1 dashboard: the brief

A task for the **sonnet-xhigh** subagent (Rishi asked for a Sonnet agent to build it,
2026-11-12). The agent that watches the run (`RUNBOOK.md`) spawns it and then keeps the page fed.

## What to build

One live page, published as a private claude.ai Artifact, that shows everything about run 1 in
one place and updates after every RL step: the rollouts, the optimizer, the drift from the SFT
model, the canaries, the evaluations and the timing. Rishi and the watching agent read it to
judge whether the run is learning, generalising, overfitting or drifting. It replaces opening
W&B: the series are the same numbers W&B gets (`train/*`), drawn here with their bounds.

- **Source file:** `documentation/run1/dashboard/run1-dashboard.html` (tracked; no local paths,
  user names or W&B entity names in it: the repo is public).
- **A short `documentation/run1/dashboard/README.md`:** what the page shows, how it is fed,
  how to push a tick by hand.
- **Load these skills before writing:** `artifact-design`, `artifact-capabilities`, `dataviz`.
  Follow the dataviz rules (one axis per chart, a legend for two or more series, bounds drawn
  as reference lines, validated colours, light and dark themes).
- **Look and feel:** the same family as the repo's other pages
  (`documentation/pages/run1-design.html`, `documentation/pilots/pilot-tracker.html`): read one
  for the tokens and type. Dense but calm: it is an instrument panel, not a landing page.

## Where the data comes from

`scripts/run1/feed.py --run <name>` writes small JSON files into `<run dir>/feed/` (rewritten
whole each time). The watching agent pushes each file into the page's shared database after
every step. The page never fetches anything else.

**Store layout (use exactly this, the runbook's push step depends on it):** collection `run1`,
one document per file, the document id is the file name without `.json`:
`state`, `steps-1`, `steps-2`, `steps-3`, `steps-4` (25 steps each; later ones appear as the run
goes on). Declare the `db` capability (and `user` if the skill requires it for `db`), subscribe
with `onSnapshot`, and redraw on every change. Viewers only read. The page must render a clear
empty state before the first push and when a `steps-n` document doesn't exist yet.

### `state` (one object)

| Field | Meaning |
|---|---|
| `run`, `updated_at` | the run's name; when the feed was written (UTC) |
| `status` | `not started`, `running`, `paused`, `stopped`, `failed`, `done` |
| `why` | list of reasons when paused or failed (a stop rule, the gate, "paused by hand", the error) |
| `phase`, `phase_text`, `phase_at` | the last phase event: `rollouts`, `training`, `evaluating`, `running`, `paused`, `failed`, `done` |
| `step_now`, `steps_total` | trained steps so far, and the target |
| `in_progress` | `{step, rollouts_done, rollouts_total}` for the step being sampled |
| `timing` | `{mean_step_s, hours_left}` from the last 10 steps |
| `stop_rules_now` | the stop rules that would fire on the latest step (normally empty) |
| `config` | `method`, `mix` (questions per dataset per step), `lr`, `warmup_steps`, `group_size`, `eval_steps`, `stop` (the bounds, below), `gate_p`, ... |
| `baseline` | the SFT model on the yardstick: `overall` and `by_set` blocks (below) |
| `baseline_run1_path` | the same SFT model measured again through run 1's own evaluation path (greedy 44.5, mean@8 36.4, pass@8 59.9 against the baseline's 43.2 / 36.9 / 62.6): the yardstick's own repeat noise. Show it beside the baseline in every evaluation chart, as "SFT, re-measured" |
| `evals` | list of `{step, overall, by_set, vs_baseline}`, one per finished evaluation |
| `events` | the last 40 events `{at, kind (info / warn / error), text, phase?, step?}` |
| `step_files` | how many `steps-n` documents exist |

An evaluation block (`overall`, each of `by_set`: `cypherbench`, `gretel`, `sqale`,
`text2cypher`): `greedy_pass1`, `mean_at8`, `pass_at_k` (8 numbers, pass@1..8), `always`,
`sometimes`, `never`, `questions`. `vs_baseline` has three sign tests, `greedy`, `sampled`,
`pass8`, each `{better, worse, p}`; `pass8` is the gate.

`config.stop`: `entropy_min_frac` (0.5), `informative_min` (0.25) for `informative_steps` (5),
`malformed_max` (0.05) for `malformed_steps` (3), `mismatch_max` (0.10), `kl_max` (0.05),
`length_ratio` (2.0), `distinct_up` (0.10), `leak_max` (0.02). The relative rules compare with
the mean of the run's first five steps.

### `steps-n` (`{from, to, steps: [row, ...]}`), one flat row per trained step

| Group | Fields |
|---|---|
| Rollouts | `reward_mean`, `solve_rate`, `reward_<set>`, `solve_<set>` (four datasets), `informative_frac`, `undecided_frac`, `malformed_frac`, `resp_tokens`, `resp_tokens_median`, `think_forced_frac`, `rollout_tok_s` |
| Optimizer | `loss`, `grad_norm`, `lr`, `clip_frac`, `entropy`, `trained_seqs`, `skipped_too_long`, `sdpo_rollouts`, `sdpo_tokens`, `distill_jsd`, `teacher_weight_mean`, `template_match` |
| Drift | `kl_sft` (from the server's log-probs: the true one), `kl_sft_trainer` (the trainer's view), `mismatch`, `probe_tokens` |
| Canaries | `distinct_frac`, `select_star_frac`, `multi_call_frac`, `leak_frac`, `query_chars_median` |
| Timing | `rollout_s`, `serve_s`, `score_s`, `load_s`, `train_s`, `merge_s`, `update_s`, `step_s`, `peak_gb` |

A step's questions are new every step, so `reward_mean` and `solve_rate` swing with the batch:
always draw a rolling mean (5 steps) over the raw points.

## What the page must show

1. **Header:** run name, status (colour and words), step k of N with a progress bar, the phase
   and the in-progress step's rollouts, hours left, "updated ... ago". A banner when `status`
   is paused or failed (with `why`) or when `stop_rules_now` is not empty.
2. **Rollouts:** reward and solve rate (raw + rolling mean); solve rate per dataset as four small
   multiples, each with the baseline's `mean_at8` for that dataset as a reference line;
   informative share with its 0.25 bound; malformed and undecided shares; response length and
   the forced-`</think>` share; rollout tokens/s.
3. **Optimizer:** loss; gradient norm; learning rate (the warm-up is visible in steps 1-5);
   clip fraction; entropy with its bound (half of the first five steps' mean); sequences trained
   and how many were distilled (`sdpo_rollouts`); the distillation JSD and the teacher weight.
4. **Drift:** `kl_sft` with the 0.05 bound and `kl_sft_trainer` beside it (one chart, same
   unit); `mismatch` with the 0.10 bound. One line under the charts: the trainer under-reads a
   small LoRA, so the server's KL is the true one.
5. **Canaries:** the four shares with their bounds, the median query length.
6. **Evaluations:** greedy pass@1, mean@8 and pass@8 at the baseline and at each evaluated step,
   overall and per dataset; the never / sometimes / always split; the three sign tests with p,
   the gate marked. Before the first evaluation: the baseline alone and "first evaluation after
   step 25".
7. **Generalising or overfitting:** per dataset, the rolling training solve rate (new training
   questions) next to the evalset's `mean_at8` at each evaluation (held-out questions, and for
   CypherBench held-out graphs). Training going up while the evalset stays flat or falls is
   overfitting; say so in a one-line reading rule on the panel.
8. **Timing:** seconds per step split into rollouts, scoring + load, update, merge; peak GPU
   memory.
9. **Events:** the last events, newest first, warnings and errors marked.

Every chart: hover values, the bound drawn and labelled where one exists, a table view or
visible numbers for the latest step.

## Deliverable back to the caller

- The page file and the README, written and published as an Artifact (private).
- The Artifact URL, and the exact `ArtifactData` calls that push one tick (one `set` per feed
  file into collection `run1`, the document id = the file name without `.json`), tested once
  with a real feed (a finished loop test's feed exists under `<data>/run1/smoke-3/feed/`).
- A check of the page at phone width and in both themes.
