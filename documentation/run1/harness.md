# The rollout loop on 16 GB: llama-server and the trainer taking turns

How run 1 generates its rollouts and updates the model on one 16 GB GPU, what was measured
before it was built, and how to measure it again. Numbers are from 2026-11-12 on the RX 7800 XT
(ROCm / HIP 7.1, the llama.cpp fork build with the Spark port, the released model's Q8_0 GGUF).

## The loop in one picture

```
step k
  Windows   llama-server  <- rollout-model.gguf (SFT + the LoRA after step k-1, merged, Q8_0)
            rollouts.py   -> step-000k/rollouts.jsonl   (32 questions x 8 answers, exact token ids)
            server stopped (the GPU is free)
  WSL       train_step.py
              scorer process (CPU): the checker's verdicts  -> scored.jsonl
              model load (GPU, ~20 s) at the same time
              probe, update, save the LoRA
              gguf_merge: the new LoRA into rollout-model.gguf (22 s)
            process exits (the GPU is free)
```

The GPU holds one of the two at a time: the server needs about 9 GB (4.4 GB of weights and a
4.7 GB KV cache), the update up to 12.8 GB.

**Why two programs.** llama.cpp decodes many answers at once far faster than transformers does
on this card (hundreds of tokens a second against ~40). The trainer needs gradients, which
llama.cpp doesn't give. So the server samples and the trainer learns, and the exact token ids
go from one to the other (`return_tokens`): the trainer scores the tokens the server sampled,
with no re-tokenizing.

## What made the pilots' rollouts slow

A pilot step spent 8-9 of its 15 minutes on rollouts, at 170-220 generated tokens a second.
The same build's own benchmark makes 796 at 16 requests. Where the rest went, measured with
`scripts/run1/bench_decode.py` (every slot holds a real prompt, then all generate 200 tokens):

| Slots | One shared KV pool (`-kvu`) | A KV per slot |
|---|---|---|
| 1 | 80 tokens/s | |
| 4 | 199 | |
| 8 | 247 | 392 |
| 12 | | 480 |
| 16 | **208** | **542** |
| 20 | | 417 |
| 24 | | 459 |
| 32 | 148 | 535 |

- **The shared KV pool.** `-kvu` puts every slot's tokens in one pool, and each decoding step
  then attends over all of them: with 16 slots of ~900 tokens, every new token is computed
  against ~14,000 cached ones, 15/16 of them masked out. A step's time grows with the square of
  the slot count (12 ms at 1 slot, 77 ms at 16, 216 ms at 32), so more slots made it slower.
  With a KV per slot a step is 30 ms at 16 slots. The pilots used `-kvu` so that a long prompt
  could borrow room; run 1 gives each slot 8,192 tokens instead (the longest allowed rollout is
  4,000 prompt tokens + 2,048).
- **The runtime LoRA** (`--lora`): 415 tokens/s against 542. Each of the 180 adapted weights
  gets two extra small matrix products per step. Run 1 merges the LoRA into the weights
  (decisions R7).
- **Log-probabilities** (`n_probs`): 423 against 542. The server builds and softmaxes a second
  copy of the 131,072-token distribution for every slot and step. Run 1 asks for them on one
  rollout per group.
- **Requests that lost their cache.** A pilot rollout is two requests (the thought, then the
  answer). Between them another rollout could take the slot, so a quarter of the second
  requests recomputed their whole prompt and thought, and some waited for a busy slot.

**What made no difference:** the sampler chain (the default chain against the temperature
sampler alone: the other stages do nothing at temperature 1 with no cut-offs), CPU threads (4,
8, 16), the KV size. **Not available:** sampling on the GPU (`-bs`): this ROCm build has no
`ARGSORT`, which the chain's top-p stage needs, and `post_sampling_probs` sorts the whole
vocabulary (85 tokens/s).

## Run 1's scheduler (`run1/rollouts.py`)

- **One worker per slot.** Worker i sends everything to slot i (`id_slot`).
- **A question's eight rollouts run one after another on one slot.** The prompt is computed
  once and reused from the slot's cache; the pilots computed it eight times.
- **Long questions first:** CypherBench, text2cypher, SQaLe, gretel (their mean answer lengths:
  457, 386, 307, 129 tokens), so the short groups fill the end.
- **Idle workers help.** A worker with nothing left takes a waiting rollout from the group with
  the most left (it pays one prompt). Every slot stays busy until the step's last rollouts.

**Measured, one full step** (`scripts/run1/bench_rollouts.py`: the pilots' step-1 questions and
seeds, 256 rollouts, 117k generated tokens, the SFT model): **292 s**, 399 tokens/s over the
whole step. The pilots' step 1 took 504 s (GRPO) and 540 s (SRPO) for the same questions. The
step's average is below the 542 of the table because the last minutes hold a few long thoughts
(the 1,024-token budget) on a mostly empty server.

## The rollout model (`run1/gguf_merge.py`)

After each update the LoRA is merged into the 180 matrices it adapts and written, as Q8_0, over
the same bytes of the run's copy of the released GGUF (`<models>/run1/<run>/rollout-model.gguf`).

- **Q8_0:** values in blocks of 32 along a row; per block a scale `d = max|x| / 127` (float16)
  and 32 int8 values `round(x / d)`. Computed on the GPU in the update's process.
- **Proof it's the same format:** at step 1 the update quantises the trainer's own base weights
  and compares them with the released file: 99.91% of the int8 values are equal and the rest
  differ by 1 (rounding ties). A lower share stops the run.
- **Marker:** `rollout-model.json` names the step whose LoRA the file holds. It is removed
  before a write and written after, so a write that was cut is seen and redone
  (`python -m sqlrl.run1.tools merge --run-dir <dir> --step <k>`).
- **Time:** ~22 s (4.0 GB written). Converting the adapter alone took 5-8 s in the pilots;
  the 15 s more buy 23% faster decoding.

## The update's side

- **Scoring in its own process.** The checker is CPU work (plus Neo4j for Cypher). It runs as
  `train_step --score-only` while the main process loads the model.
- **WSL is kept awake for the whole run.** While the rollouts run on Windows nothing runs
  inside WSL, and WSL shuts itself down after a minute of that. Every update then started
  cold: Python, torch and the model were read from disk again, and the "model load" took
  195-210 s in the first loop tests. The run loop now keeps one small process inside WSL
  (`orchestrate.keep_wsl_warm`), which also re-reads the model's files every 90 s so they stay
  in WSL's file cache. With it the load took **17-22 s** in the same loop test, and a
  loop-test step fell from over 6 minutes to about 3.
- **The model on WSL's own disk.** Reading 7.7 GB through `/mnt/<drive>` runs at 231 MB/s (58 s
  for the whole load) and isn't cached inside WSL; a copy on WSL's ext4 is, and loads in ~20 s
  from the cache. Read cold it can be slower than the Windows path (here WSL's disk is on the
  slower drive), so `model.model_dir` asks the kernel how much of the copy is cached
  (`mincore`) and uses it only above 90%; otherwise the Windows path. `model_hf_wsl` in
  `configs/paths.local.toml` names the copy.
- **One sequence per pass.** Padding several sequences into one forward pass was tried
  (`batch_tokens`) and was far slower on this GPU (the run was stopped after 4x the unbatched
  time), so `batch_tokens` is 0.
- **The probe** (before the update, ~25 s): on the 32 rollouts with the server's log-probs, the
  trainer's log-probs with the LoRA on and off. It gives `mismatch` and `kl_sft` (decisions
  R7).

## Why the server's and the trainer's log-probs differ

With no LoRA they differ by 0.013-0.015 a token: Q8_0 weights against bf16, fp32 against bf16
arithmetic. With a small LoRA the gap grows, and the reason is in the trainer: each adapted
layer's output is rounded to bf16 after the LoRA's small addition, so part of the addition is
rounded away. On the pilot's step-10 LoRA the trainer's forward pass saw about half of the
LoRA's effect. The server (and the merged model) apply all of it, and they are what is
evaluated and shipped, so the server's numbers are the true ones. The gradients are not
affected. `python -m sqlrl.run1.tools mismatch --rollouts <file> --lora-state <file>` measures
it on any rollout file that carries log-probs.

## Measuring again

| Command (Windows, GPU free) | What it measures | Time |
|---|---|---|
| `C:\Python314\python.exe -I scripts\run1\bench_decode.py --server <label> --slots 16 --ctx 131072` | decoding speed with every slot busy, per request variant | ~1 min |
| `... bench_decode.py --kv-unified --ctx 65536` | the same with the shared pool | ~2 min |
| `... bench_decode.py --lora <adapter.gguf>` | the cost of a runtime LoRA | ~1 min |
| `C:\Python314\python.exe -I scripts\run1\bench_rollouts.py --label <label>` | one real step of rollouts | ~5 min |

Results append to `<data>/run1/bench/decode.jsonl`; each run keeps its server log in
`<data>/run1/bench/<label>/`.

## Limits and what's next

- The server samples each slot's token on the CPU, one slot after another; at 16 slots that is
  about a third of a decoding step. GPU sampling would remove it once the ROCm backend supports
  the chain.
- The update (about 5 minutes) is now the larger half of a step. Its cost is mostly the prompt:
  a group's eight sequences share ~670 prompt tokens and differ in ~320. Computing the shared
  prefix once per group would cut the update's work by about 2.4x; it needs the model's forward
  pass to accept a cached prefix with gradients, which the remote code doesn't offer today.
- Run 2 (multi-turn) will reuse the per-slot scheduling: a turn's next request finds the whole
  conversation in its slot's cache.
