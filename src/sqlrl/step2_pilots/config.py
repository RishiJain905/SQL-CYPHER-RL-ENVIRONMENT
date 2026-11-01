"""The pilots' settings: one budget for all, one block of method settings per pilot.

Every pilot trains a new LoRA on the merged SFT model (the release) with the same questions in
the same order, the same rollout settings and the same number of steps; only the method block
differs. Standard library only (the Windows orchestrator reads it too).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# The same for every pilot (documentation/pilots/TRACKER.md: "the fixed budget").
COMMON: dict[str, Any] = {
    "steps": 10,  # Rishi, 2026-11-01: short pilots first (decisions R4)
    "questions_per_step": 32,  # 8 per dataset
    "group_size": 8,
    "pool": "pool-v1",
    # rollouts: the step-1 yardstick's sampling and thinking budget (decisions R1)
    "temperature": 1.0,
    "top_p": 1.0,
    "think_budget": 1024,  # thinking tokens before `</think>` is forced
    "max_tokens": 2048,  # thinking + call
    "parallel": 16,
    # rewards
    "reward_right": 1.0,
    "reward_wrong": 0.0,
    "reward_malformed": -0.1,  # no parsable call to the tool
    "checker": 4,
    # the new LoRA (the SFT run's targets)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_targets": ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
    # optimisation
    "lr": 1e-5,
    "betas": [0.9, 0.999],
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "minibatches": 2,  # optimizer steps per RL step (the second is slightly off-policy)
    "max_seq": 6144,  # prompt + answer tokens; longer rollouts are left out of the loss
    "seed": 42,
}

# The method blocks. `adv`: "center" (r - mean, Dr. GRPO) or "std" ((r - mean) / std, GRPO);
# `agg`: how token terms become the loss ("constant": sum / (G * max_tokens), Dr. GRPO;
# "token_mean": sum / tokens in the minibatch, DAPO; "seq": GSPO's sequence ratio);
# `clip`: [low, high] for the ratio; `kl_beta`: the k3 KL to the SFT model (adapter off).
METHODS: dict[str, dict[str, Any]] = {
    "grpo": {"adv": "center", "agg": "constant", "clip": [0.2, 0.2], "kl_beta": 0.01},
    "dapo": {"adv": "std", "agg": "token_mean", "clip": [0.2, 0.28], "kl_beta": 0.0},
    "gspo": {"adv": "std", "agg": "seq", "clip": [3e-4, 4e-4], "kl_beta": 0.0},
}

# Two pilots (decisions R4): GRPO, the standard, and SRPO, the method built to learn from failed
# answers too. DAPO and GSPO stay defined (refinements in GRPO's family) but aren't run.
PILOTS: dict[str, dict[str, Any]] = {
    "p0-smoke": {"method": "grpo", "smoke": True, "steps": 2, "questions_per_step": 8},
    "p1-grpo": {"method": "grpo"},
    # "p2-srpo": added once its loss is built (documentation/pilots/TRACKER.md)
}


def pilot_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in PILOTS:
        raise KeyError(f"unknown pilot {name!r}; known: {sorted(PILOTS)}")
    cfg = copy.deepcopy(COMMON)
    spec = copy.deepcopy(PILOTS[name])
    method = spec.pop("method")
    cfg.update(spec)
    cfg.update(overrides or {})
    cfg["name"], cfg["method"] = name, method
    cfg["method_cfg"] = copy.deepcopy(METHODS[method])
    cfg.setdefault("smoke", False)
    return cfg


def save(cfg: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def load(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def step_questions(pool: list[dict[str, Any]], cfg: dict[str, Any], step: int) -> list[dict]:
    """Step k's questions (1-based): the pool's rows in order, `questions_per_step` a step.
    The pool is interleaved by dataset, so every step has the same mix."""
    n = cfg["questions_per_step"]
    rows = pool[(step - 1) * n : step * n]
    if len(rows) < n:
        raise ValueError(f"the pool has {len(pool)} questions: too few for step {step}")
    return rows
