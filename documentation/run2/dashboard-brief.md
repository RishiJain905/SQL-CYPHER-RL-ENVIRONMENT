# Run 2 dashboard: the brief

A task for the **sonnet-xhigh** subagent (Rishi asked for the same setup as runs 1 and 1b: an
Opus session starts and watches the run, a Sonnet agent builds the page). The watching agent
(`RUNBOOK.md`, section 3) spawns it and then keeps the page fed.

## What to build

One live page for run 2, published as a **new** private claude.ai Artifact titled
"Spark RL Run 2". Don't design a new page: **copy
`documentation/run1b/dashboard/run1b-dashboard.html` to
`documentation/run2/dashboard/run2-dashboard.html`** and change what is listed below. Run 1b's
page stays as run 1b's record.

Read first, in this order:

| File | Why |
|---|---|
| `CLAUDE.md` | the rules: public repo (no local paths, user names or W&B entity names in tracked files), no credential files |
| `documentation/run1b/dashboard/README.md`, then `documentation/run1/dashboard/README.md` | how the page is built (chart cards, table twins, themes), how it is fed, the `if_version` rule |
| `documentation/run2/design.md`, sections 1 to 3 and 6 to 7 | what run 2 is: an episode with turns, what option C changed, the evaluations, the stop rules |
| `documentation/run2/RUNBOOK.md`, section 5 | what each number means and what "healthy" looks like |
| `src/sqlrl/run2/feed.py` | the feed (its docstring lists the files) |

- **Load these skills before writing:** `artifact-design`, `artifact-capabilities`, `dataviz`.
- A short `documentation/run2/dashboard/README.md`: what differs from run 1b's page, the store
  layout, how to push a tick by hand.
- **Run 1b is not drawn behind run 2.** The protocol and the questions differ, so the steps
  don't pair. Remove the `compare` document, its series and its legend entries. What run 2 is
  read against is in `refs` (below).

## Run 2 in six lines (for the page's wording)

- The model may run up to 3 queries on a question, sees each result, and its last query that
  ran is its answer. It starts from run 1b's step-30 adapter. 40 steps of 32 questions, 8
  episodes a question (256 episodes a step).
- **Only gretel and CypherBench have turns** (20 of a step's 32 questions, 160 of 256
  episodes). **SQaLe and text2cypher keep one query**, as in run 1b (12 questions).
- **Returned questions:** 3 of a step's 10 gretel questions are ones whose right result is
  nothing (no rows, NULL or zero). The untrained model rewrites a right answer when it sees
  such a result. Teaching it not to is what this run adds; the page must show it.
- Greedy with turns is the headline. A quick evaluation (greedy) runs after steps 5, 10, ...,
  40; a full one (also 8 sampled answers a question) after steps 20 and 40.
- The drift is measured against the **start adapter**, not the SFT model, with a fixed
  penalty weight.
- Rates about turns (calls, retries, fixed, broken, limit, loop, copied) count only the
  episodes that had turns.

## The feed

`scripts/run2/feed.py --run run2` writes `<data>/run2/run2/feed/` (`<data>` is `data` in the
untracked `configs/paths.local.toml`; never copy its value into a tracked file). **Store layout (use exactly
this, the runbook's push step depends on it):** collection `run2`, one document per file,
document id = the file name without `.json`.

| Document | Holds | Changes |
|---|---|---|
| `state` | status, phase, the step in progress, time left, stop rules, bounds, settings, the best step, events | every tick |
| `steps-1` ... `steps-4` | one flat row per trained step, **10 steps a document** | the one that holds the newest step |
| `evals` | `{evals: [...]}`: every evaluation of the run | after an evaluation |
| `refs` | what the run is read against, measured before it | never: pushed once |

The feed's result line lists `changed`: the files whose content differs from the last tick.
The page must render a clear empty state before the first push, and when `evals` is empty or
a `steps-n` document doesn't exist yet. **Any field may be absent or null** (a value the run
hasn't produced yet): draw a gap, never an error.

### `state`

| Field | Meaning |
|---|---|
| `run`, `updated_at` | the run's name; when the feed was written (UTC) |
| `status` | `not started`, `running`, `paused`, `stopped`, `failed`, `done`, or **`dead`** (the run's process is gone with no end file: a crash or a reboot) |
| `why` | reasons when paused, failed or dead |
| `phase`, `phase_text`, `phase_at` | the last phase event: `rollouts`, `training`, `evaluating`, `running`, `paused`, `failed`, `done` |
| `step_now`, `steps_total` | trained steps so far, and the target (40) |
| `in_progress` | `{step, episodes_done, episodes_total}` for the step being sampled |
| `timing` | `{mean_step_s, quick_eval_s, full_eval_s, evals_left, hours_left}`; `hours_left` counts the remaining steps and evaluations |
| `stop_rules_now` | the stop rules that would fire on the latest step (normally empty) |
| `bounds` | the lines the relative stop rules use, from step 6 on: per metric `{start, bound, side}` (`side`: `above` or `below` is the bad side); `t1_tokens_median` has `{start, low, high}`. Keys: `entropy_t1`, `distinct_final_mean`, `first_solve_rate`, `limit_frac`, `loop_frac`, `broken_frac`, `copied_frac`, `distinct_frac`, `dup_differ_frac`. Empty before step 6: then say "bound drawn after step 5" |
| `shape` | `{questions, with_turns, one_query, returned, group_size}`: a step in counts (32, 20, 12, 3, 8) |
| `config` | the settings; the ones the page reads: `stop` (the fixed bounds: `kl_start_max` 0.05, `kl_step1_max` 0.025, `mismatch_max` 0.10, `informative_min` 0.25, `malformed_max` 0.05, `queries_mean_max` 2.0, `too_long_max` 0.05, `hardcoded_max` 0.02, `look_leak_max` 0.05, `select_star_max` 0.05, `leak_max` 0.02), `method_cfg.kl` (`beta` 0.05, `offset` 0.0084), `quick_eval_steps`, `eval_steps`, `single_turn_sets`, `returned_per_step`, `gate_p` (0.05), `turn_p` (0.10), `first_floor` (0.515), `lr`, `warmup_steps`, `max_queries`, `mix`, `init_from` |
| `best` | `BEST.json`: `{step, greedy_with_turns, strict, first_query, first_query_ok, kind, vs, evaluated, lora}`: the evaluated step to keep so far (highest greedy with turns). `first_query_ok` false means its first query fell under `first_floor` |
| `evals_done` | the steps that have an evaluation |
| `events` | the last 40 events `{at, kind (info / warn / error), text, phase?, step?}` |
| `step_files` | how many `steps-n` documents exist |

### `steps-n` (`{from, to, steps: [row, ...]}`), one flat row per trained step

| Group | Fields |
|---|---|
| Rollouts | `reward_mean`, `solve_rate`, **`first_solve_rate`** (the first query alone), `reward_<set>`, `solve_<set>` (four datasets), `informative_frac`, `undecided_frac`, `malformed_frac`, `distinct_final_mean`, `distinct_first_mean` (different final / first queries a group of 8), `t1_tokens_median`, `later_tokens_median`, `episode_tokens_mean`, `think_forced_frac`, `rollout_tok_s` |
| Turns (episodes with turns only) | `calls_mean`, `calls_1`, `calls_2`, `calls_3` (episodes by number of calls), `ends_reply`, `ends_limit`, `ends_cut`, `limit_frac` (tried a call too many), `loop_frac` (the same query twice in a row), `retry_empty_n` / `retry_empty_again`, `retry_error_n` / `retry_error_again`, `retry_rows_n` / `retry_rows_again` (after a first result of that kind: how many, and the share followed by another call), `fixed`, `broken` (counts), `fixed_frac`, `broken_frac`, `look_error_frac` |
| Returned questions | `returned_n` (episodes, 24 a step), `returned_solve`, `returned_first_solve`, `returned_fixed`, `returned_broken`, `returned_broken_frac`, `returned_calls_mean`, `returned_weak_only_frac`; and the other gretel questions beside them: `gretel_other_solve`, `gretel_other_first_solve`, `gretel_other_fixed`, `gretel_other_broken`, `gretel_other_broken_frac`, `gretel_other_calls_mean` |
| Optimizer | `loss`, `grad_norm`, `grad_norm_max`, `lr`, `clip_frac`, `entropy`, `entropy_t1`, `entropy_later`, `trained_seqs`, `skipped_too_long`, `skipped_minibatches`, `sdpo_rollouts`, `sdpo_tokens`, `distill_jsd`, `teacher_weight_mean`, `template_match` |
| Drift | **`kl_start`** (first-turn KL to the start adapter: the server's reading less the offset; the stop rule reads this), `kl_start_raw`, `kl_offset`, `kl_step1`, `kl_start_later` (later turns), `kl_start_trainer`, `kl_sft` (to the SFT model, for information: about 0.08 from step 1, because the start adapter is already that far from SFT), `kl_beta` (fixed at 0.05), `kl_pen`, `mismatch`, `probe_tokens` |
| Canaries | `hardcoded_frac`, `hardcoded_right`, `copied_frac`, `copied_multi_frac`, `look_leak_frac` (often null), `distinct_frac`, `select_star_frac`, `leak_frac`, `dup_differ_frac`, `dup_probed`, `too_long_frac`, `query_chars_median` |
| Timing | `rollout_s`, `serve_s`, `graph_load_s`, `score_s`, `load_s`, `train_s`, `merge_s`, `update_s`, `step_s`, `peak_gb` |

A step's questions are new every step, so rates swing with the batch: always draw a 5-step
rolling mean over the raw points. The returned questions are 24 episodes a step, so their
raw points swing most: the rolling mean is the line to read.

### `evals` and `refs`

An evaluation `evals[i]`: `step`, `kind` (`quick` or `full`), `greedy` (a greedy block),
`greedy_by_set` (four datasets), `by_look` (three kinds of question, below), `vs` (sign tests),
`returned_dev`, and on a full evaluation `sampled`, `sampled_by_set`, `sampled_calls_mean`.

- **A greedy block:** `n`, `with_turns`, `right`, `wrong`, `undecided`, `no_query`, `decided`
  (right / (right + wrong): **the score**), `strict` (right / n), `first_right`,
  `first_decided` (the first query alone), `fixed`, `broken`, `net`, `kept_right`,
  `kept_wrong`, `broken_frac`, `fixed_frac`, `calls_mean`, `calls` (`{"1": n, ...}`),
  `limit_frac`, `loop_frac`, `copied_frac`, `hardcoded_frac`, `retry_<after>_n`,
  `retry_<after>_again`.
- **`by_look`** splits the evaluation set's 200 questions by what a look can tell:
  `turns_says_something` (87: turns, and the right result shows values),
  `turns_says_nothing` (13 gretel questions: turns, and the right result is nothing),
  `one_query` (100: SQaLe and text2cypher).
- **A sampled block** (run 1's shape): `questions`, `greedy_pass1`, `mean_at8`, `pass_at_k`
  (8 numbers), `always`, `sometimes`, `never`.
- **`vs`:** `single_turn` (against the start adapter on one query), `start_with_turns`
  (against the start adapter with turns), `best` (against the run's best earlier evaluation,
  with its `step`). Each holds `greedy`, and on a full evaluation also `sampled` and `pass8`,
  as `{better, worse, p}`. **The gate:** `start_with_turns.pass8` worse with p below `gate_p`
  stops the run. **The turn rule:** `best.greedy` worse with p below `turn_p` pauses it.
- **`returned_dev`:** the returned dev slice (50 questions no step trains on), greedy:
  `{n, first_right, right, fixed, broken, broken_frac, calls_mean}`.

`refs` (all measured on the start adapter before the run):

| Field | Meaning | Untrained value |
|---|---|---|
| `start` | the start adapter with turns on the evaluation set: an evaluation block with `sampled` (the run's start reference, "step 0") | greedy with turns 52.4%, first query 52.4%, fixed 7, broken 7, mean@8 47.6, pass@8 71.8 |
| `single_turn` | the start adapter on one query (run 1b's step 30): `{overall, by_set}` of sampled blocks | greedy 54.5, mean@8 43.4, pass@8 67.5 |
| `returned_start` | the start adapter on the returned dev slice, same shape as `returned_dev` | 25 right first, 11 after the turns, 14 broken |

## What to change on the page

1. **Read collection `run2`** (one `onSnapshot` on the whole collection): `state`, `steps-n`
   (10 steps each), `evals`, `refs`. The page must work before `evals` and `refs` arrive.
2. **Header.** Status with `dead` as its own state (red, with `why`). Tiles: step k of 40;
   solve rate and first-query solve rate (5-step means); calls an episode; **"Right answers
   broken (returned questions)"**: the 5-step mean of `returned_broken_frac` against the
   untrained 56% (`refs.returned_start.broken_frac`); "Drift from the start" (`kl_start`
   against `stop.kl_start_max`); "Latest evaluation" (greedy with turns against the start's
   52.4); "Best step so far" (`state.best`, flagged when `first_query_ok` is false).
3. **A panel "What is different in this run"**, short, near the top, collapsible. Use the text
   in the next section as it is.
4. **Rollouts panel.** Solve rate and first-query solve rate on one chart (raw points and
   rolling means), with `bounds.first_solve_rate` drawn. Per dataset as four small multiples,
   labelled "with turns" (gretel, CypherBench) or "one query" (SQaLe, text2cypher).
   Informative share with its 0.25 bound. Different final queries a group
   (`distinct_final_mean`, with its bound). Malformed and undecided shares. First-turn and
   later-turn length (`bounds.t1_tokens_median` low and high).
5. **A new panel "Turns"** (note on the panel: "episodes with turns only: gretel and
   CypherBench, 160 of 256"). Calls an episode with the 2.0 bound; episodes by calls
   (`calls_1/2/3` as shares); **another call after a first result**: three lines
   (`retry_empty_again`, `retry_error_again`, `retry_rows_again`, the `_n` in the tooltip),
   with the reading rule "after an error the model should retry; after rows it mostly should
   not; after an empty result it depends on the question, which is what the returned
   questions teach"; fixed and broken (counts, and `broken_frac` with its bound);
   `limit_frac` and `loop_frac` with their bounds.
6. **A new panel "Returned questions: what option C trains".** `returned_first_solve` and
   `returned_solve` on one chart (the gap between them is what the turns cost);
   `returned_broken_frac` with the untrained 56% as a reference line; beside it the other
   gretel questions: `gretel_other_fixed` and `gretel_other_broken`; calls an episode for
   both kinds; `returned_weak_only_frac` (answers the plain checker passes and the stricter
   check rejects: they earn nothing). A small table of the **returned dev slice** at the
   start (`refs.returned_start`) and at each evaluation (`evals[i].returned_dev`): right
   first, right after the turns, broken. The reading rule, in words on the panel: "Working:
   the broken share falls while the other gretel questions are still fixed. Not working:
   both fall, which means the model stopped retrying everywhere."
7. **Drift panel.** `kl_start` with the stop bound (`stop.kl_start_max`, 0.05) and a zero
   line; `kl_start_raw` with the offset (`method_cfg.kl.offset`) as a reference line; step
   1's reading as a number against `stop.kl_step1_max`. `kl_sft` on its own small chart,
   marked "for information". `mismatch` with its bound. Remove the KL-weight chart and tile
   (the weight is fixed in run 2). One line: "The drift is measured against the start adapter
   (run 1b's step 30), on first-turn tokens: the server's reading less a measured offset.
   One step's reading swings by about 0.002."
8. **Canaries panel.** `hardcoded_frac` (0.02), `copied_frac` (its bound), `look_leak_frac`
   (0.05), `distinct_frac` (its bound), `select_star_frac` (0.05), `leak_frac` (0.02),
   `dup_differ_frac` (its bound, `dup_probed` in the tooltip), `too_long_frac` (0.05).
9. **Evaluations panel.**
   - A chart over the evaluated steps: greedy with turns (`greedy.decided`), the first query
     (`greedy.first_decided`) and the strict score, with the start as step 0 (`refs.start`)
     and two reference lines: the start with turns (52.4) and the start on one query (54.5,
     `refs.single_turn.overall.greedy_pass1`). Mark `first_floor` (51.5) for the first-query
     line. The best step so far as a dotted vertical line.
   - A table, one row an evaluation, with the start's two rows on top: step, kind, greedy
     with turns, strict, first query, fixed, broken, net, calls an episode, the returned dev
     slice (right first, right, broken), and mean@8 and pass@8 where the evaluation is full.
   - **By kind of question** (`by_look`), the start beside the latest evaluation: for each
     of the three kinds, right on the first query, right after the turns, fixed, broken. One
     line: "Turns can only help where a look can inform (87 questions). On the 13 where the
     right result is nothing the untrained model breaks 7 of 10 right answers."
   - Per dataset (`greedy_by_set`, and `sampled_by_set` when full).
   - The sign tests (`vs`) with p, and the two rules in words (the gate, the turn rule).
   - Never / sometimes / always and pass@k for full evaluations, beside `refs.start` and
     `refs.single_turn`.
   - Before the first evaluation: the start alone and "first evaluation after step 5".
   - One line: "One evaluation is good to about 3 points; one question is 0.5."
10. **Generalising or overfitting.** Per dataset: the rolling training solve rate next to the
    evaluation's `greedy_by_set[set].decided` at each evaluated step.
11. **Optimizer, Timing, Events:** as in run 1b's page with run 2's fields (`entropy_t1` with
    its bound and `entropy_later` beside it; the gradient norm with the clip at 1.0; the
    timing split `rollout_s`, `score_s`, `load_s`, `train_s`, `merge_s`; `peak_gb` against
    the card's 16 GB).
12. **Text.** Every "run 1b" in titles and notes becomes "run 2" except where run 1b is named
    as the start. "Steps shown": All / Last 20 / Last 10.

## The text of "What is different in this run"

> **Run 2 lets the model look before it answers.** It may run up to 3 queries on a question,
> sees each result, and its last query is its answer. It starts from run 1b's step 30.
>
> **The first test showed a problem.** With turns, the untrained model scored 4 points lower
> on the evaluation set (52.4% to 48.1%). The cause: when a right query returns nothing (no
> rows, NULL or zero), the model takes that for a mistake and rewrites it.
>
> **Option C, the setup this run uses, changes three things:**
>
> 1. **Turns only where a look can help.** gretel and CypherBench have turns. SQaLe and
>    text2cypher keep one query, as in run 1b: their look-around data mostly shows nothing.
> 2. **Questions whose right answer is nothing are back in training.** The question pool had
>    dropped them, so the model never practised them. Now 3 of every 10 gretel questions are
>    of this kind ("returned questions").
> 3. **A stricter score for them.** On an empty database any query returns nothing, so the
>    usual check would pay for anything. These answers must also match on filled copies of
>    the same tables.
>
> **Before training, with this setup:** 52.4% on the first query and 52.4% after the turns.
> Where a look can inform, turns fixed 7 answers and broke none. Where the right result is
> nothing, they broke 7 of 10 right answers. On 50 held-back returned questions the model has
> 25 right first and 11 after its turns.
>
> **What to watch:** the share of right answers broken on returned questions should fall,
> while the other gretel questions are still fixed.

## Testing, and what to hand back

- **A fixture:** a finished loop test's feed is at `<data>/run2/smoke-6/feed/` (2 steps, 2
  quick evaluations, `refs`; `feed.py --run smoke-6` rewrites it). It has no `bounds` (they
  need 6 steps) and no full evaluation. So also **build a second, made-up fixture in your
  scratch folder** (not in the repo): 12 steps (smoke-6's rows repeated with small changes),
  a `state.bounds` filled by hand in the shape above, and one full evaluation (copy
  `refs.start`, give it `step` 10 and `kind` `full`, add a `vs` with `greedy`, `sampled` and
  `pass8` and a `returned_dev`). Render the page against both.
- Test in collection `run2`, then **delete every test document** (with `if_version`) before
  you return: the real run's first tick must start from an empty collection, or the test rows
  would show.
- Check the page at phone width and in both themes, with no console errors.

Hand back to the caller:

- the page file and the README, published as a private Artifact; the Artifact URL;
- the exact `ArtifactData` calls that push one tick into collection `run2` (the first push
  without `if_version`, later ones with it), tested once with a real feed;
- what you checked, and anything in the feed that didn't fit the page.
