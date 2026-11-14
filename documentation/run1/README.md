# Run 1: operating it

Run 1 is stage A of the real RL run: single turn, an execution reward, SRPO, a new LoRA on the
released SFT model. The design and the review behind it: [`design.md`](design.md). How the
rollout loop works and what it measured: [`harness.md`](harness.md).

**Status (2026-11-12): built and loop-tested, not started.** Starting and watching it:
[`RUNBOOK.md`](RUNBOOK.md). The dashboard's task: [`dashboard-brief.md`](dashboard-brief.md).

## Before the first start

| Check | How |
|---|---|
| The checker's Neo4j answers | Docker is running and the `asktrainer-neo4j` container is up (the update and the pool builder refuse to run without it) |
| The question pool exists | `PYTHONPATH=src <venv>/bin/python -m sqlrl.run1.pool` inside WSL (CPU, ~30 min, resumable); writes `<data>/run1/pool-v2.jsonl` and its `.stats.json` |
| The GPU is free | no other llama-server, no training process |
| `configs/paths.local.toml` has run 1's keys | `llama_bin`, `rocm_bin`, `hip_device`, `model_hf_wsl` (see `configs/paths.example.toml`) |

## Commands (Windows; start the run hidden and detached)

| What | Command |
|---|---|
| Start, or resume | `C:\Python314\python.exe -I scripts\run1\run.py --run run1` |
| A new run with other choices | add `--mix even`, `--lr 2e-5`, `--steps 200` (only when the run is created) |
| **Run 1b** (run 1 held near the SFT model: `../run1b/design.md`) | `C:\Python314\python.exe -I scripts\run1\run.py --run run1b --preset run1b` (the preset is read when the run is created; resume with the same command) |
| **Free the GPU now** | `C:\Python314\python.exe -I scripts\run1\pause.py --run run1` |
| Stop after the current step | `C:\Python314\python.exe -I scripts\run1\pause.py --run run1 --after-step` |
| The yardstick on the SFT model through run 1's evaluation path (no training, ~35 min) | `C:\Python314\python.exe -I scripts\run1\run.py --evalcheck` |
| One tick of the dashboard's data | `C:\Python314\python.exe -I scripts\run1\feed.py --run run1` (writes `<run>/feed/*.json`; `dashboard-brief.md` describes the fields) |
| A loop test (2 steps, 8 SQL questions, ~10 min) | `C:\Python314\python.exe -I scripts\run1\run.py --run smoke-<n> --smoke --pool pool-test` |
| A loop test of a preset (2 steps: 4 CypherBench, 2 gretel, 2 SQaLe questions; needs Neo4j) | `C:\Python314\python.exe -I scripts\run1\run.py --run smoke-<n> --preset run1b --smoke` |

Start it detached, for example from PowerShell:
`Start-Process C:\Python314\python.exe -ArgumentList '-I','scripts\run1\run.py','--run','run1' -WindowStyle Hidden`.

`pause.py` stops the run's process tree, its llama-server and its processes inside WSL, checks
that none is left, and writes `PAUSED`. Resuming redoes at most the update of the step that was
cut: finished rollouts, scores and steps are kept. Tested 2026-11-12 mid-rollout: GPU freed,
45 of 64 rollouts kept, the run finished after the resume.

While a run is going it keeps one small process inside WSL (so WSL doesn't shut down between
steps); `pause.py` and the run's end stop it.

## What a run leaves in `<data>/run1/<run>/`

| File | What it is |
|---|---|
| `config.json` | the run's settings, frozen when it was created |
| `events.jsonl` | one line per phase change (rollouts, update, evaluation, pause, failure) |
| `train_log.jsonl` | one line of metrics per step (also `step-000k/metrics.json`) |
| `step-000k/rollouts.jsonl` | the step's 256 rollouts: token ids, text, the server's log-probs where asked |
| `step-000k/scored.jsonl` | the checker's verdicts (with a preset that has `dup_probe`: also `dup`, the duplicate probe's `same` / `differ` / `unknown` for the passed CypherBench answers) |
| `step-000k/lora_state.safetensors` | the LoRA after the step (every step) |
| `step-000k/adapter/`, `adapter.gguf` | the adapter as a PEFT folder and a GGUF (evaluation steps and the last step) |
| `eval-000k/summary.json` | the yardstick after step k and its sign tests against the baseline (`vs_baseline`) and against the run's best earlier evaluation (`vs_best`) |
| `BEST.json` | the evaluated step with the highest mean@8 so far, and its LoRA file |
| `PAUSED`, `STOPPED`, `FAILED`, `DONE` | how the run last ended (`PAUSED` holds the reason) |

The rollout model is next to the released GGUF: `<models>/run1/<run>/rollout-model.gguf` and its
marker `rollout-model.json`.

W&B: project `sql-cypher-rl`, group `run1`, one run per run name. The x-axis is `rl_step`.

## Reading a step's metrics

| Metric | Meaning | Healthy |
|---|---|---|
| `reward_mean`, `solve_rate` | the step's rollouts (new questions each step, so it swings) | no trend needed step to step |
| `informative_frac` | groups with both right and wrong answers | 0.4-0.6 |
| `sdpo_rollouts` | wrong answers that learned from a right sibling | 30-80 |
| `entropy` | mean token entropy of the trained answers | near the start's |
| `kl_sft` | drift from the SFT model, from the server's log-probs | rises slowly; pause at 0.05 |
| `kl_beta`, `kl_pen` (run 1b) | the KL penalty's weight this step, and the KL the penalty saw (the trainer's view) | the weight at 0.05 until `kl_sft` passes 0.045, then doubling until it levels off |
| `dup_differ_frac` (run 1b) | of the passed CypherBench answers, the share that repeat rows where the reference doesn't (no reward) | 0.4-0.55 at the start, flat or falling |
| `mismatch` | server against trainer log-probs | 0.015 at step 1, rising with the LoRA |
| `distinct_frac`, `select_star_frac`, `multi_call_frac`, `leak_frac` | the reward-hacking and teacher-leak canaries | flat; `leak_frac` 0 |
| `rollout_s`, `update_s`, `step_s` | seconds | expected ~290, ~370, ~660 for a full step |

## When the run pauses by itself

`events.jsonl` and `PAUSED` say which rule fired. Look at the step's rollouts (highest reward,
most changed), then either resume with the same command (the rule is checked again after the
next step), change the bound in the run's `config.json` (`stop`) and resume, or stop there and
keep an earlier step's LoRA.

After an evaluation the gate pauses the run once if pass@8 is down by more than chance; resuming
continues past it.

## Limits

- The yardstick has its own noise. The SFT model measured again through run 1's evaluation
  path (`run.py --evalcheck`, 2026-11-12) scored greedy 44.5, mean@8 36.4, pass@8 59.9, against
  the baseline's 43.2 / 36.9 / 62.6 (no sign test below p = 0.29). A change of 2-3 points on
  one evaluation is not a result.
- The server's chat parser rejects a malformed tool call with an error instead of returning
  it (1 of 1,600 sampled answers in that check). The evaluation retries, then keeps up to 8
  such answers as empty ones and logs a warning.
- A step's questions are fixed by the pool and the mix; changing the mix mid-run would reuse or
  skip questions, so a new mix needs a new run.
