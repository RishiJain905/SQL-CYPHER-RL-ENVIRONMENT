"""A kept adapter merged into the released model as a full bf16 model folder (inside WSL):

    python -m sqlrl.run2.export --adapter <models>/run2-step40/adapter \\
        --base <models>/Spark-X2.6-4B-SQL-Cypher-Agent --out <models>/<name>

Run 2's adapters are a LoRA on the released (SFT-merged) model, so the merge is
`W + (alpha / r) * B @ A` on each adapted weight, done tensor by tensor in float32.

**Save it in float32 (the default), never bf16.** The RL changes are tiny (step 40: at most
0.00017 a weight). Stored in bf16, only 7.8% of the adapted weights changed at all and what was
kept had a cosine of 0.45 with the true delta; a Q8_0 made from that bf16 folder scored 46.7%
with turns on the evalset against the adapter's 60.2% (2026-11-21, NFL-ML-AI PROGRESS), which is
about where the SFT model is. `--dtype bfloat16` exists only to reproduce that. The float32
folder (about 16 GB) is the source to quantise from (`convert_hf_to_gguf.py --outtype f32`, then
`llama-quantize`); every quant must be checked on the evalset against the adapter's number.
The run's own Q8_0 (the rollout model, `gguf_merge.py`) is quantised straight from float32.

The shards keep the base's names and index; the config (its `torch_dtype` set to the saved
dtype), the tokenizer, the chat template and the remote code are copied from the base folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."


def load_lora(adapter: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """{base weight name: (A, B)} and the scale alpha / r."""
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    parts: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            name = key.removeprefix(PREFIX)
            for tag in ("lora_A", "lora_B"):
                marker = f".{tag}.weight"
                if name.endswith(marker):
                    base = name[: -len(marker)] + ".weight"
                    parts.setdefault(base, {})[tag] = f.get_tensor(key)
    pairs = {k: (v["lora_A"], v["lora_B"]) for k, v in parts.items()}
    if len(pairs) != len(parts):
        raise ValueError("an adapted weight is missing its A or B matrix")
    return pairs, scale


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def merge(adapter: Path, base: Path, out: Path, dtype: str = "float32") -> dict:
    target = DTYPES[dtype]
    pairs, scale = load_lora(adapter)
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    out.mkdir(parents=True, exist_ok=True)
    merged, delta_max = 0, 0.0
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base / shard), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]
                    delta = (b.float() @ a.float()) * scale
                    if delta.shape != t.shape:
                        raise ValueError(
                            f"{key}: LoRA shape {tuple(delta.shape)} != {tuple(t.shape)}"
                        )
                    new = t.float() + delta
                    delta_max = max(delta_max, float(delta.abs().max()))
                    t = new
                    merged += 1
                tensors[key] = t.to(target).contiguous()
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(tensors)} tensors)", flush=True)
    if merged != len(pairs):
        missing = sorted(set(pairs) - set(index["weight_map"]))[:5]
        raise ValueError(f"merged {merged} of {len(pairs)} adapted weights (not found: {missing})")
    for item in base.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copy2(item, out / item.name)
    cfg_file = out / "config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg["torch_dtype"] = dtype
    cfg_file.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    stats = {"adapted_weights": merged, "scale": scale, "max_abs_delta": delta_max,
             "shards": len(shards), "dtype": dtype}  # fmt: skip
    (out / "rl_merge.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="a PEFT adapter folder")
    ap.add_argument("--base", required=True, help="the released bf16 model folder")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help="float32 (default); bfloat16 loses the RL adapter (see above)",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty")
        return 1
    print(json.dumps(merge(Path(a.adapter), Path(a.base), out, a.dtype)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
