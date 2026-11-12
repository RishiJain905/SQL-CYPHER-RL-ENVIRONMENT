"""Loading the policy for run 1's update (inside WSL): the merged SFT model in bf16 with the
run's LoRA, through the trainer's own loader (the same code that trained the release)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths


def cached_share(folder: Path) -> float:
    """The share of a folder's weight files that is in WSL's file cache right now, asked of
    the kernel page by page (`mincore`); 0.0 when it can't be told."""
    import ctypes
    import mmap

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        page = mmap.PAGESIZE
        res = total = 0
        for f in sorted(folder.glob("*.safetensors")):
            size = f.stat().st_size
            if size == 0:
                continue
            with f.open("rb") as fh:
                m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)  # a private mapping
            try:
                n = (size + page - 1) // page
                vec = (ctypes.c_ubyte * n)()
                buf = ctypes.c_char.from_buffer(m)
                rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size),
                                  vec)  # fmt: skip
                del buf
                if rc != 0:
                    return 0.0
                res += sum(b & 1 for b in bytes(vec))
                total += n
            finally:
                m.close()
        return res / total if total else 0.0
    except (OSError, ValueError, AttributeError):
        return 0.0


def model_dir() -> Path:
    """The merged SFT model's folder. The copy on WSL's own disk (`model_hf_wsl`) is used when
    it is in WSL's file cache (the run's keep-alive holds it there: 23 s to load); read cold
    it can be slower than the Windows path, which is the fallback (~58 s)."""
    p = get_paths()
    if p.model_hf_wsl:
        local = Path(p.model_hf_wsl).expanduser()
        if (local / "config.json").exists() and cached_share(local) >= 0.9:
            return local
    return p.model_hf


def load_policy(cfg: dict[str, Any], lora_state: Path | None = None, grad_checkpointing=True):
    """(model, tokenizer, backbone, head weight). A new LoRA is seeded and starts as the SFT
    model (its B matrices are zero); `lora_state` loads a saved one."""
    import torch
    from safetensors.torch import load_file

    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer.checkpoint import load_trainable
    from asktrainer.config import LoraSpec
    from asktrainer.modelio import apply_lora, backbone_of, head_weight, load_base

    torch.manual_seed(cfg["seed"])
    model, tok, _info = load_base(model_dir(), device="cuda", log=lambda *a: None)
    spec = LoraSpec(r=cfg["lora_r"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"],
                    targets=cfg["lora_targets"])  # fmt: skip
    model = apply_lora(model, spec, gradient_checkpointing=grad_checkpointing)
    if lora_state is not None:
        load_trainable(model, load_file(str(lora_state)))
    return model, tok, backbone_of(model), head_weight(model)


def rollout_model_path(run_name: str) -> Path:
    """The run's rollout GGUF, next to the released one (the fast disk)."""
    return get_paths().model_gguf.parent / "run1" / run_name / "rollout-model.gguf"


def merged_step(out: Path) -> int | None:
    """The step whose LoRA the rollout GGUF `out` holds, from the marker `gguf_merge` writes
    after a finished merge (None: no file, or a merge that was cut)."""
    marker = out.with_suffix(".json")
    if not out.exists() or not marker.exists():
        return None
    try:
        return int(json.loads(marker.read_text(encoding="utf-8"))["step"])
    except (ValueError, KeyError):
        return None
