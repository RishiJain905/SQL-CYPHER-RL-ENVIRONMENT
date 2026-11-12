"""Run 1's settings (standard library only: the Windows orchestrator reads it too).

`documentation/run1/design.md` explains each choice. A run's settings are frozen into its
`config.json` when it is created (`init`); the three choices that are Rishi's (the dataset mix,
the learning rate, the length) are arguments of `scripts/run1/run.py` for a new run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

MIXES: dict[str, dict[str, int]] = {
    # questions per step from each dataset (32 in all)
    "recommended": {"cypherbench": 10, "gretel": 10, "sqale": 8, "text2cypher": 4},
    "even": {"cypherbench": 8, "gretel": 8, "sqale": 8, "text2cypher": 8},
}

DEFAULTS: dict[str, Any] = {
    "run": "run1",
    "steps": 100,
    "eval_steps": [25, 50, 100],  # the yardstick is run after these steps
    "mix": MIXES["recommended"],
    "group_size": 8,
    "pool": "pool-v2",
    # rollouts: the step-1 yardstick's sampling and thinking budget (decisions R1)
    "temperature": 1.0,
    "top_p": 1.0,
    "think_budget": 1024,  # thinking tokens before `</think>` is forced
    "max_tokens": 2048,  # thinking + call
    # the rollout server (documentation/run1/harness.md)
    "slots": 16,
    "slot_ctx": 8192,  # tokens of KV per slot (prompt + answer); the slots don't share a pool
    "rollout_model": "merged",  # the LoRA merged into a Q8_0 copy each step ("lora": --lora)
    "logprob_rollouts": 1,  # rollouts per group that ask the server for its log-probabilities
    # rewards (as piloted)
    "reward_right": 1.0,
    "reward_wrong": 0.0,
    "reward_malformed": -0.1,  # no parsable call to the tool
    "adv_std_floor": 0.25,  # decisions R8
    "checker": 4,
    # the new LoRA (the SFT run's targets)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_targets": ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
    # optimisation
    "lr": 1e-5,
    "warmup_steps": 5,  # RL steps over which the learning rate rises linearly to `lr`
    "betas": [0.9, 0.999],
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "minibatches": 2,  # optimizer steps per RL step (the second is slightly off-policy)
    "max_seq": 6144,  # prompt + answer tokens; longer rollouts are left out of the loss
    "batch_tokens": 0,  # 0: one sequence per pass (padded batches measured slower here)
    "seed": 42,
    # SRPO as piloted (decisions R4): the GRPO branch with std advantages, clip-higher and a
    # token-level average; self-distillation from an EMA teacher for wrong rollouts with a
    # right sibling; no KL penalty
    "method": "srpo",
    "method_cfg": {
        "adv": "std",
        "agg": "token_mean",
        "clip": [0.2, 0.28],
        "kl_beta": 0.0,
        "sdpo": {"top_k": 100, "ema": 0.05, "entropy_beta": 1.0},
    },
    # stop rules: any one pauses the run for a look (design.md, section 5)
    "stop": {
        "entropy_min_frac": 0.5,  # mean token entropy below this share of the first 5 steps'
        "informative_min": 0.25,  # informative share below this ...
        "informative_steps": 5,  # ... for this many steps in a row
        "malformed_max": 0.05,  # malformed calls above this ...
        "malformed_steps": 3,  # ... for this many steps in a row
        "mismatch_max": 0.10,  # rollout-vs-trainer log-prob gap (mean abs, a token)
        "kl_max": 0.05,  # KL(rollout policy || SFT) a token, from the server's log-probs
        "length_ratio": 2.0,  # median response length against the first 5 steps', either way
        "distinct_up": 0.10,  # share of queries with DISTINCT above the first 5 steps' + this
        "leak_max": 0.02,  # share of rollouts whose thought cites a "correct solution"
    },
    "gate_p": 0.05,  # pass@8 lower than the baseline's with a sign-test p below this: stop
}


def new_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    cfg["questions_per_step"] = sum(cfg["mix"].values())
    cfg["eval_steps"] = sorted({s for s in cfg["eval_steps"] if s <= cfg["steps"]}
                               | {cfg["steps"]})  # fmt: skip
    return cfg


def save(cfg: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def load(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def step_questions(pool: dict[str, list[dict[str, Any]]], cfg: dict[str, Any],
                   step: int) -> list[dict[str, Any]]:  # fmt: skip
    """Step k's questions (1-based): from each dataset, the next `mix[dataset]` of its pool
    rows. Every question is used once."""
    rows: list[dict[str, Any]] = []
    for name, n in cfg["mix"].items():
        part = pool.get(name, [])[(step - 1) * n : step * n]
        if len(part) < n:
            raise ValueError(f"the pool has {len(pool.get(name, []))} {name} questions: too "
                             f"few for step {step}")  # fmt: skip
        rows.extend(part)
    return rows


def lr_at(cfg: dict[str, Any], step: int) -> float:
    """The learning rate of RL step `step` (1-based): a linear warm-up, then constant."""
    w = int(cfg.get("warmup_steps") or 0)
    return cfg["lr"] * min(1.0, step / w) if w > 0 else cfg["lr"]


def step_dir(run_dir: Path, k: int) -> Path:
    return run_dir / f"step-{k:04d}"
