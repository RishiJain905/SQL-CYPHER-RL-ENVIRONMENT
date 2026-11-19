"""Run 2's maintenance commands (inside WSL, GPU):

    python -m sqlrl.run2.tools merge --run-dir <dir> --step <k>        # rebuild the rollout GGUF
    python -m sqlrl.run2.tools merge --lora-state <file> --out <gguf>  # any LoRA into any copy
    python -m sqlrl.run2.tools memory [--tokens 2048,4096,6144]        # the update's peak memory

`merge` rewrites a rollout model from a saved LoRA. `--step 0` is the run's start: its
`init_from` adapter (the SFT model itself when the run has none). The orchestrator calls it
before step 1 and whenever a run was cut while the file was being written. The file must be a
copy of the released Q8_0 GGUF: only the adapted tensors are overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlrl.run2 import config as rcfg
from sqlrl.run2 import gguf_merge
from sqlrl.run2.model import load_policy, rollout_model_path, start_state


def cmd_merge(a: argparse.Namespace) -> int:
    from sqlrl.paths import get_paths

    if a.run_dir:
        run_dir = Path(a.run_dir)
        cfg = rcfg.load(run_dir)
        state = (rcfg.step_dir(run_dir, a.step) / "lora_state.safetensors" if a.step
                 else start_state(cfg))  # fmt: skip
        out = rollout_model_path(cfg["run"])
    else:
        cfg = rcfg.new_config({"init_from": None})
        state = Path(a.lora_state) if a.lora_state else None
        out = Path(a.out)
    model, _tok, _b, _h = load_policy(cfg, state, grad_checkpointing=False)
    # the trainer's base weights must reproduce the released file before anything is written
    check = gguf_merge.check_template(model, get_paths().model_gguf)
    if check["equal_share"] < 0.99 or check["max_diff"] > 1:
        raise RuntimeError(f"the trainer's weights don't reproduce the GGUF: {check}")
    stats = gguf_merge.write_merged(model, out, a.step, with_lora=state is not None)
    print(json.dumps({"template_check": check, "lora": str(state.name) if state else None,
                      **stats}))  # fmt: skip
    return 0


def cmd_memory(a: argparse.Namespace) -> int:
    """The update's peak GPU memory for one episode of each length in `--tokens`: a sequence
    that takes all three loss terms at once (the reward update on every token, distillation and
    the penalty on its first half), forward and backward. `max_seq` is set from this: an
    episode that doesn't fit would fail the step for a reason that has nothing to do with the
    model."""
    import torch

    from sqlrl.run2.logprobs import answer_logprobs
    from sqlrl.run2.losses import sequence_loss, topk_jsd, topk_kl

    cfg = rcfg.new_config()
    model, _tok, backbone, head = load_policy(cfg, start_state(cfg))
    model.train()
    dev = head.device
    params = [q for q in model.parameters() if q.requires_grad]
    base = torch.cuda.memory_allocated()
    out = {"model_gb": round(base / 1e9, 2), "sizes": []}
    for n in [int(x) for x in a.tokens.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen = min(a.generated, n // 2)
        try:
            x = torch.randint(1000, 20000, (1, n), device=dev)
            h = backbone(input_ids=x, use_cache=False).last_hidden_state[0]
            lp, _ = answer_logprobs(h, head, x[0], n - gen, entropy=True)
            mask = torch.ones(gen, device=dev)
            loss, _ = sequence_loss(cfg["method_cfg"], lp, lp.detach(), mask, 1.0, float(gen))
            hs = h[n - gen - 1 : n - 1]
            n1 = gen // 2
            other = hs[:n1].detach().clone()
            loss = loss + topk_jsd(hs[:n1], other, head, 100).sum() / gen
            loss = loss + 0.05 * topk_kl(hs[:n1], other, head, 100).sum() / gen
            loss.backward()
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            del x, h, lp, loss, hs, other
        except torch.OutOfMemoryError:
            peak = None
        for q in params:
            q.grad = None
        out["sizes"].append({"tokens": n, "generated": gen, "peak_gb": peak})
        print(json.dumps(out["sizes"][-1]), flush=True)
        if peak is None:
            break
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mem = sub.add_parser("memory")
    mem.add_argument("--tokens", default="2048,3072,4096,5120,6144")
    mem.add_argument("--generated", type=int, default=2048)
    mem.set_defaults(fn=cmd_memory)
    m = sub.add_parser("merge")
    m.add_argument("--run-dir")
    m.add_argument("--step", type=int, default=0)
    m.add_argument("--lora-state")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
