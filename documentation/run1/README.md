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
| **Free the GPU now** | `C:\Python314\python.exe -I scripts\run1\pause.py --run run1` |
| Stop after the current step | `C:\Python314\python.exe -I scripts\run1\pause.py --run run1 --after-step` |
| The yardstick on the SFT model through run 1's evaluation path (no training, ~35 min) | `C:\Python314\python.exe -I scripts\run1\run.py --evalcheck` |
| One tick of the dashboard's data | `C:\Python314\python.exe -I scripts\run1\feed.py --run run1` (writes `<run>/feed/*.json`; `dashboard-brief.md` describes the fields) |
| A loop test (2 steps, 8 SQL questions, ~10 min) | `C:\Python314\python.exe -I scripts\run1\run.py --run smoke-<n> --smoke --pool pool-test` |

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
| `step-000k/scored.jsonl` | the checker's verdicts |
| `step-000k/lora_state.safetensors` | the LoRA after the step (every step) |
| `step-000k/adapter/`, `adapter.gguf` | the adapter as a PEFT folder and a GGUF (evaluation steps and the last step) |
| `eval-000k/summary.json` | the yardstick after step k and its sign tests against the baseline |
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

- The evaluation path (greedy + 8 samples on the evalset, scoring, the gate) was not run end to
  end on 2026-11-12: the checker's Neo4j was down. Its parts are the pilots' (step 1's sampler,
  the trainer's greedy client and scorer); the first evaluation at step 25 is its first full run.
- A step's questions are fixed by the pool and the mix; changing the mix mid-run would reuse or
  skip questions, so a new mix needs a new run.
