"""Run 2's settings (standard library only: the Windows orchestrator reads it too).

`documentation/run2/design.md` explains each choice; decisions R15 records that they are the
design page's recommendations (D1-D14), taken by Rishi as a whole. A run's settings are frozen
into its `config.json` when it is created.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

MIXES: dict[str, dict[str, int]] = {
    "recommended": {"cypherbench": 10, "gretel": 10, "sqale": 8, "text2cypher": 4},  # run 1b's
    "even": {"cypherbench": 8, "gretel": 8, "sqale": 8, "text2cypher": 8},
}

DEFAULTS: dict[str, Any] = {
    "run": "run2",
    "steps": 40,
    "quick_eval_steps": [5, 10, 15, 20, 25, 30, 35, 40],  # greedy with turns (D11)
    "eval_steps": [20, 40],  # the full evaluation: greedy and 8 sampled episodes
    "mix": MIXES["recommended"],
    "group_size": 8,
    # questions (D9): run 1's pool, past the 40 steps run 1b trained on, the last 50 of each
    # dataset held back as the dev slice
    "pool": "pool-v2",
    "pool_offset": 40,
    "dev_per_set": 50,
    # the returned gretel questions (decisions R17): the ones run 1's pool rejected because the
    # right result says nothing (`pool.build_returned`). On the pool an empty result always
    # means a wrong query; on the evaluation set 13 of 50 gretel questions are the other kind.
    # So this many of a step's gretel questions come from the returned file, in its own order
    # (a fixed count: the rates the stop rules read must not swing with the mix), and its last
    # `dev_returned` are held back as a second dev slice
    "returned_per_step": {"gretel": 3},
    "dev_returned": 50,
    # a returned question's answer earns the reward only when the checker's full SQL check also
    # passes on filled copies of the schema (`pool.strict_check`): on the dataset's own rows
    # any query that returns nothing equals a reference that returns nothing
    "returned_guard": True,
    # where it starts (D7, D12): an adapter folder under `paths.models` holding
    # `lora_state.safetensors` (and `ema_state.safetensors`); None: a new LoRA on the SFT model
    "init_from": "run1b-step30",
    # the episode (D3, D4)
    "max_queries": 3,
    "prompt_variant": "turns",  # `prompt.py`; chosen on the dev slice
    # datasets kept on run 1's protocol: the original prompt, one query, no result shown.
    # SQaLe (decisions R16): its look-around copy is a random fill that shows `0 rows` for 14
    # of 50 right queries, and on the dev slice turns fixed 1 SQaLe answer and broke 6.
    # text2cypher (R17): 31 of the evaluation set's 50 prompts describe a graph with no
    # relationships, so the right answer is "nothing" and a look can only mislead
    "single_turn_sets": ["sqale", "text2cypher"],
    "obs": {
        "rows": [5, 10, 20],  # the row cap, drawn per training episode
        "styles": ["table", "json"],  # the result's form, drawn per training episode
        "eval": {"rows": 10, "style": "table"},
    },
    "cypher_slots": 8,  # Cypher episodes side by side on the one graph (D8)
    # sampling: the step-1 yardstick's (decisions R1); budgets are per turn
    "temperature": 1.0,
    "top_p": 1.0,
    "think_budget": 1024,
    "max_tokens": 2048,
    # the rollout server
    "slots": 16,
    "slot_ctx": 8192,
    "rollout_model": "merged",
    "logprob_rollouts": 1,
    # rewards (D5): a right answer pays 1 - turn_cost x (calls - 1); a wrong one 0 whatever it
    # took; no query at all is malformed
    "reward_right": 1.0,
    "reward_wrong": 0.0,
    "reward_malformed": -0.1,
    "turn_cost": 0.03,
    "adv_std_floor": 0.25,
    "checker": 4,
    "dup_probe": True,  # run 1b's second look at passed CypherBench answers
    "hardcode_rule": True,  # D2: a final query that reads nothing or returns constants earns 0
    # the LoRA (the SFT run's targets; `init_from` must match)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_targets": ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
    # optimisation (D7): run 1b's rate; Adam starts without state, so the warm-up is needed
    "lr": 5e-6,
    "warmup_steps": 5,
    "betas": [0.9, 0.999],
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "minibatches": 2,
    # prompt + episode tokens; longer episodes are left out of the loss. Measured on the 16 GB
    # card (`tools memory`): the update peaks at 12.2 GB for 3,072 tokens, 13.5 for 3,650, 14.5
    # for 4,096, and 4,608 doesn't fit. Run 1's 6,144 was never reached there, and would not fit
    "max_seq": 3840,
    "batch_tokens": 0,
    "seed": 42,
    # SRPO as in run 1b, turn-aware (D6): a wrong episode with a right sibling distills its
    # first turn toward the teacher and takes the reward update on its later turns.
    # The penalty (D7): KL(policy || the start adapter) on first-turn tokens, a fixed weight
    "method": "srpo",
    "method_cfg": {
        "adv": "std",
        "agg": "token_mean",
        "clip": [0.2, 0.28],
        "kl_beta": 0.0,
        "sdpo": {"top_k": 100, "ema": 0.05, "entropy_beta": 1.0, "turns": "first"},
        # `offset`: what the drift reading shows when the true drift is zero, measured on 200
        # episodes of the start adapter (`phase0.py offset`: 0.0099, +-0.0035 a step)
        "kl": {
            "beta": 0.05,
            "top_k": 100,
            "turns": "first",
            "offset": 0.0099,
            "offset_for": "run1b-step30",
        },  # fmt: skip
    },
    # stop rules: any one pauses the run for a look (design.md section 7). "the start" is the
    # mean of the run's own first five steps
    "stop": {
        # collapse
        "entropy_min_frac": 0.25,  # first-turn token entropy below this share of the start
        "diversity_min_frac": 0.5,  # distinct final queries per group below this share ...
        "diversity_steps": 3,  # ... for this many steps
        "informative_min": 0.25,
        "informative_steps": 5,
        "length_ratio": 3.0,  # first-turn length against the start, either way
        # drift (D7)
        "kl_start_max": 0.05,  # first-turn KL to the start adapter: the server's reading less
        # the measured offset
        "kl_step1_max": 0.025,  # the raw reading at step 1 (the offset, about 0.010 +- 0.0035):
        # above this the run did not start from the model the offset was measured on
        "mismatch_max": 0.10,
        # the protocol
        "malformed_max": 0.05,  # episodes that never ran a query ...
        "malformed_steps": 3,
        "queries_mean_max": 2.0,  # calls an episode ...
        "limit_up": 0.15,  # ... or episodes that tried one call too many, above the start's ...
        "queries_steps": 3,
        "loops_up": 0.10,  # the same query run twice in an episode, above the start's ...
        "loops_steps": 3,
        "too_long_max": 0.05,  # episodes longer than `max_seq`
        # the first-try skill
        "broken_up": 0.08,  # right first query, wrong final one (of the right-first episodes),
        "broken_steps": 3,  # above the start's, for this many steps
        "first_right_down": 0.10,  # first-query solve rate below the start's minus this ...
        "first_right_steps": 3,
        # reward hacking
        "hardcoded_max": 0.02,  # final queries the constants rule caught
        "copied_up": 0.05,  # final queries with a literal read off a result, above the start's
        "look_leak_max": 0.05,  # random-fill episodes whose results show a reference literal
        "distinct_up": 0.10,
        "select_star_max": 0.05,
        "leak_max": 0.02,
        "dup_up": 0.15,
        "dup_steps": 3,
    },
    # after an evaluation (design.md section 6); the references are folders under <data>/run2
    "refs": {
        "single_turn": "refs/run1b-eval-0030",  # the start adapter, one query (54.5)
        "start_with_turns": "phase0/heldout-step30",  # the start adapter with turns, untrained
    },
    "gate_p": 0.05,  # pass@8 with turns lower than the start's with a sign-test p below this
    "turn_p": 0.10,  # greedy with turns worse than the run's best earlier evaluation
    "first_floor": 0.515,  # the first query's score must stay within 3 points of 54.5 to be kept
}

MERGED = ("method_cfg", "stop", "obs", "refs")
TAIL = 64  # questions per dataset, just before the dev slice, kept for loop tests


def new_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k in MERGED and isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(cfg[k].get(kk), dict):
                    cfg[k][kk].update(vv)
                else:
                    cfg[k][kk] = vv
        else:
            cfg[k] = v
    cfg["questions_per_step"] = sum(cfg["mix"].values())
    for key in ("eval_steps", "quick_eval_steps"):
        cfg[key] = sorted({s for s in cfg[key] if s <= cfg["steps"]})
    return cfg


def save(cfg: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def load(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def step_questions(pool: dict[str, list[dict[str, Any]]], cfg: dict[str, Any],
                   step: int) -> list[dict[str, Any]]:  # fmt: skip
    """Step k's questions (1-based): from each dataset, the next of its pool rows after the
    offset. Every question is used once, none is one run 1b trained on, and none is in a dev
    slice or in the reserve kept for loop tests.

    A dataset with `returned_per_step[name]` = r takes r of its `mix[name]` questions from the
    returned file (`<name>_returned`, its own order, no offset: no run has used them) and the
    other `mix[name] - r` from the pool. **The offset is counted in run 1b's questions**: run 1b
    took `mix[name]` a step for `pool_offset` steps, whatever this run takes."""
    rows: list[dict[str, Any]] = []
    off, dev = int(cfg.get("pool_offset") or 0), int(cfg.get("dev_per_set") or 0)
    back = cfg.get("returned_per_step") or {}
    for name, n in cfg["mix"].items():
        r = int(back.get(name) or 0)
        if not 0 <= r <= n:
            raise ValueError(f"{name}: {r} returned questions a step, of {n}")
        parts = [(name, n - r, off * n, dev)]
        if r:
            parts.append((name + "_returned", r, 0, int(cfg.get("dev_returned") or 0)))
        for key, m, skip, held in parts:
            have = pool.get(key, [])
            usable = len(have) - held
            if cfg.get("pool_tail"):  # a loop test: the `TAIL` questions before the dev slice,
                a, b = usable - step * m, usable - (step - 1) * m  # which no real run reaches
                if a < usable - TAIL:
                    raise ValueError(f"a loop test may use the last {TAIL} {key} questions only")
            else:
                a, b = skip + (step - 1) * m, skip + step * m
                if b > usable - TAIL:
                    raise ValueError(f"the pool has {len(have)} {key} questions ({held} held "
                                     f"back as a dev slice, {TAIL} for loop tests): too few "
                                     f"for step {step}")  # fmt: skip
            rows.extend(have[a:b])
    return rows


def lr_at(cfg: dict[str, Any], step: int) -> float:
    """The learning rate of RL step `step` (1-based): a linear warm-up, then constant."""
    w = int(cfg.get("warmup_steps") or 0)
    return cfg["lr"] * min(1.0, step / w) if w > 0 else cfg["lr"]


def step_dir(run_dir: Path, k: int) -> Path:
    return run_dir / f"step-{k:04d}"
