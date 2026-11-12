"""The rollout model: the LoRA merged into the weights and written as Q8_0 into a copy of the
released GGUF (inside WSL, torch).

Why not `--lora` as in the pilots (decisions R7, `documentation/run1/harness.md`):
- this llama.cpp build applied the pilots' adapter at about twice its trained strength (the
  trainer's log-probabilities matched the server's best with the LoRA scaled by 2), so the
  rollouts came from a different policy than the one being updated;
- a runtime adapter costs 23% of the decoding speed (two extra small matrix products per
  adapted weight, 180 weights).

Merging is what a release does anyway (merge, then quantise), so the rollout policy is the
model that would ship. Only the 180 adapted matrices change; their Q8_0 bytes are computed on
the GPU and written over the same byte ranges of the file (same shapes, same type, same
offsets), which takes seconds instead of a full conversion.

Q8_0, as llama.cpp's `quantize_row_q8_0` does it: blocks of 32 values along a row; per block
`d = max|x| / 127` stored as float16, then 32 int8 values `round(x / d)`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sqlrl.paths import get_paths

QK = 32
BLOCK_BYTES = 2 + QK
HF2GGUF = {
    "mlp.down_proj": "ffn_down",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "self_attn.out_proj": "attn_output",
    "self_attn.q_k_v_proj": "attn_qkv",
}
_NAME = re.compile(r"layers\.(\d+)\.(" + "|".join(re.escape(k) for k in HF2GGUF) + r")$")


def gguf_name(module_name: str) -> str | None:
    """`...layers.7.mlp.up_proj` -> `blk.7.ffn_up.weight` (None for anything else)."""
    m = _NAME.search(module_name)
    return f"blk.{m[1]}.{HF2GGUF[m[2]]}.weight" if m else None


def quantize_q8_0(w: torch.Tensor) -> torch.Tensor:
    """A [rows, cols] float matrix -> its Q8_0 bytes, uint8 [blocks, 34] (cols % 32 == 0)."""
    if w.shape[-1] % QK:
        raise ValueError(f"row length {w.shape[-1]} is not a multiple of {QK}")
    x = w.reshape(-1, QK).float()
    d = x.abs().amax(dim=1) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.round(x * inv[:, None]).clamp_(-128, 127).to(torch.int8)
    out = torch.empty((x.shape[0], BLOCK_BYTES), dtype=torch.uint8, device=w.device)
    out[:, :2] = d.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(torch.uint8)
    return out


def dequantize_q8_0(raw: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """The inverse (for tests and checks): uint8 [blocks, 34] -> float32 [rows, cols]."""
    d = raw[:, :2].contiguous().view(torch.float16).float()
    q = raw[:, 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(shape)


def tensor_index(gguf_path: Path) -> dict[str, dict[str, Any]]:
    """{tensor name: {"offset", "n_bytes", "type", "shape"}} of a GGUF file (cached next to it
    as `<file>.index.json`)."""
    cache = gguf_path.with_suffix(".index.json")
    if cache.exists() and cache.stat().st_mtime >= gguf_path.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    sys.path.insert(0, str(get_paths().llama_cpp_src / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    index = {t.name: {"offset": int(t.data_offset), "n_bytes": int(t.n_bytes),
                      "type": t.tensor_type.name, "shape": [int(x) for x in t.shape]}
             for t in reader.tensors}  # fmt: skip
    del reader
    try:
        cache.write_text(json.dumps(index), encoding="utf-8")
    except OSError:
        pass
    return index


def lora_modules(model) -> list[tuple[str, Any]]:
    """(GGUF tensor name, PEFT LoRA layer) for every adapted weight."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(getattr(mod, "lora_A", None), torch.nn.ModuleDict) and len(mod.lora_A):
            g = gguf_name(name)
            if g is None:
                raise RuntimeError(f"no GGUF tensor is known for the adapted module {name}")
            out.append((g, mod))
    return out


def merged_weight(mod, adapter: str = "default", with_lora: bool = True) -> torch.Tensor:
    """W + scaling * B @ A in float32 (what PEFT's merge computes)."""
    w = mod.weight.detach().float()
    if not with_lora:
        return w
    a = mod.lora_A[adapter].weight.detach().float()
    b = mod.lora_B[adapter].weight.detach().float()
    return w + float(mod.scaling[adapter]) * (b @ a)


def check_template(model, template: Path, sample: int = 10) -> dict[str, Any]:
    """Does quantising the trainer's own base weights reproduce the released GGUF's bytes?
    Proves the name map, the layout and the quantiser before anything is overwritten. Returns
    the share of equal int8 values and the largest difference over `sample` tensors."""
    index = tensor_index(template)
    mods = lora_modules(model)
    picks = mods[:: max(len(mods) // sample, 1)][:sample]
    same = total = 0
    worst = 0
    with template.open("rb") as f:
        for name, mod in picks:
            info = index[name]
            if info["type"] != "Q8_0":
                raise RuntimeError(f"{name} is {info['type']} in the GGUF, not Q8_0")
            mine = quantize_q8_0(merged_weight(mod, with_lora=False))
            if mine.numel() != info["n_bytes"]:
                raise RuntimeError(f"{name}: {mine.numel()} bytes computed, {info['n_bytes']} "
                                   "in the file")  # fmt: skip
            f.seek(info["offset"])
            theirs = torch.frombuffer(bytearray(f.read(info["n_bytes"])), dtype=torch.uint8)
            theirs = theirs.reshape(-1, BLOCK_BYTES).to(mine.device)
            a = mine[:, 2:].contiguous().view(torch.int8).int()
            b = theirs[:, 2:].contiguous().view(torch.int8).int()
            same += int((a == b).sum())
            total += a.numel()
            worst = max(worst, int((a - b).abs().max()))
    return {"tensors": len(picks), "equal_share": same / max(total, 1), "max_diff": worst}


def write_merged(model, out: Path, step: int, index: dict[str, dict[str, Any]] | None = None,
                 with_lora: bool = True) -> dict[str, Any]:  # fmt: skip
    """Overwrite the adapted tensors of `out` (a copy of the released Q8_0 GGUF) with the
    merged weights. The marker `<out>.json` says which step's LoRA the file holds; it is
    removed first, so a file cut mid-write is never mistaken for a finished one."""
    t0 = time.time()
    index = index or tensor_index(out)
    marker = out.with_suffix(".json")
    marker.unlink(missing_ok=True)
    n_bytes = 0
    mods = lora_modules(model)
    with out.open("r+b") as f:
        for name, mod in mods:
            info = index[name]
            raw = quantize_q8_0(merged_weight(mod, with_lora=with_lora)).cpu().numpy().tobytes()
            if len(raw) != info["n_bytes"] or info["type"] != "Q8_0":
                raise RuntimeError(f"{name}: computed {len(raw)} bytes for a {info['type']} "
                                   f"tensor of {info['n_bytes']}")  # fmt: skip
            f.seek(info["offset"])
            f.write(raw)
            n_bytes += len(raw)
        f.flush()
        os.fsync(f.fileno())
    stats = {"step": step, "tensors": len(mods), "bytes": n_bytes,
             "seconds": round(time.time() - t0, 1)}  # fmt: skip
    marker.write_text(json.dumps(stats), encoding="utf-8")
    return stats
