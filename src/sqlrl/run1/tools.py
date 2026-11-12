"""Run 1's maintenance commands (inside WSL, GPU):

    python -m sqlrl.run1.tools merge --run-dir <dir> --step <k>      # rebuild the rollout GGUF
    python -m sqlrl.run1.tools merge --lora-state <file> --out <gguf>  # any LoRA into any copy
    python -m sqlrl.run1.tools mismatch --rollouts <file> [--lora-state <file>]

`merge` rewrites the rollout model from a saved LoRA (`--step 0`: the SFT model itself); the
orchestrator calls it when a run was cut while the file was being written. `mismatch` compares
the trainer's log-probabilities of a rollout file's sampled tokens with the server's own (the
rollouts that carry them), with the LoRA on and off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlrl.run1 import config as rcfg
from sqlrl.run1 import gguf_merge
from sqlrl.run1.model import load_policy, rollout_model_path


def cmd_merge(a: argparse.Namespace) -> int:
    if a.run_dir:
        run_dir = Path(a.run_dir)
        cfg = rcfg.load(run_dir)
        state = rcfg.step_dir(run_dir, a.step) / "lora_state.safetensors" if a.step else None
        out = rollout_model_path(cfg["run"])
    else:
        cfg = rcfg.new_config()
        state = Path(a.lora_state) if a.lora_state else None
        out = Path(a.out)
    model, _tok, _b, _h = load_policy(cfg, state, grad_checkpointing=False)
    check = gguf_merge.check_template(model, out)
    stats = gguf_merge.write_merged(model, out, a.step, with_lora=state is not None)
    print(json.dumps({"template_check": check, **stats}))
    return 0


def cmd_mismatch(a: argparse.Namespace) -> int:
    import torch

    from sqlrl.run1.logprobs import sequence_logprobs

    rows = []
    with Path(a.rollouts).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("behavior_lp"):
                rows.append(r)
    rows = rows[: a.limit]
    state = Path(a.lora_state) if a.lora_state else None
    model, _tok, backbone, head = load_policy(rcfg.new_config(), state, grad_checkpointing=False)
    model.eval()
    tot = {"on": 0.0, "off": 0.0, "n": 0.0}
    with torch.no_grad():
        for r in rows:
            ids, start = r["prompt_ids"] + r["gen_ids"], len(r["prompt_ids"])
            on = sequence_logprobs(model, backbone, head, ids, start)[0].float()
            with model.disable_adapter():
                off = sequence_logprobs(model, backbone, head, ids, start)[0].float()
            known = torch.tensor([x is not None for x in r["behavior_lp"]], device=on.device)
            srv = torch.tensor([x if x is not None else 0.0 for x in r["behavior_lp"]],
                               device=on.device)  # fmt: skip
            m = known.float() * torch.tensor(r["gen_mask"], device=on.device).float()
            tot["on"] += float(((on - srv).abs() * m).sum())
            tot["off"] += float(((off - srv).abs() * m).sum())
            tot["n"] += float(m.sum())
    n = max(tot["n"], 1.0)
    print(json.dumps({"rollouts": len(rows), "tokens": tot["n"],
                      "mean_abs_gap_lora_on": round(tot["on"] / n, 5),
                      "mean_abs_gap_lora_off": round(tot["off"] / n, 5)}))  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--run-dir")
    m.add_argument("--step", type=int, default=0)
    m.add_argument("--lora-state")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_merge)
    x = sub.add_parser("mismatch")
    x.add_argument("--rollouts", required=True)
    x.add_argument("--lora-state")
    x.add_argument("--limit", type=int, default=64)
    x.set_defaults(fn=cmd_mismatch)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
