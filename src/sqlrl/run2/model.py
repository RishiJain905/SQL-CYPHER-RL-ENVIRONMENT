"""Loading the policy for run 2's update (inside WSL), and where the run's models are. Run 1's
module with the start adapter added (decisions R15).

Run 2 continues an adapter instead of starting a new one: `init_from` names a folder under
`paths.models` with the kept adapter's `lora_state.safetensors` (run 1b's step 30). Step 1 loads
it; the same file, never changed, is the reference the drift is measured against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths


def models_dir() -> Path:
    """Where a run writes its models and kept adapters live: `paths.models`, named on its own so
    a 4 GB file rewritten every step never follows the released model onto another drive."""
    p = get_paths()
    return p.models if p.models is not None else p.model_gguf.parent


def init_dir(cfg: dict[str, Any]) -> Path | None:
    return models_dir() / cfg["init_from"] if cfg.get("init_from") else None


def start_state(cfg: dict[str, Any]) -> Path | None:
    """The adapter the run starts from (None: a new LoRA on the SFT model)."""
    d = init_dir(cfg)
    if d is None:
        return None
    f = d / "lora_state.safetensors"
    if not f.exists():
        raise FileNotFoundError(f"the start adapter is missing: {cfg['init_from']}/{f.name}")
    return f


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
    """The merged SFT model's folder: the copy on WSL's own disk when it is in WSL's file cache
    (the run's keep-alive holds it there), else the Windows path."""
    p = get_paths()
    if p.model_hf_wsl:
        local = Path(p.model_hf_wsl).expanduser()
        if (local / "config.json").exists() and cached_share(local) >= 0.9:
            return local
    return p.model_hf


def load_policy(cfg: dict[str, Any], lora_state: Path | None = None, grad_checkpointing=True):
    """(model, tokenizer, backbone, head weight). `lora_state` loads a saved LoRA; without one a
    new LoRA is seeded, which is the SFT model itself (its B matrices are zero)."""
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
    """The run's rollout GGUF."""
    return models_dir() / "run2" / run_name / "rollout-model.gguf"


def merged_step(out: Path) -> int | None:
    """The step whose LoRA the rollout GGUF `out` holds, from the marker `gguf_merge` writes
    after a finished merge (None: no file, or a merge that was cut). Step 0 is the start."""
    marker = out.with_suffix(".json")
    if not out.exists() or not marker.exists():
        return None
    try:
        return int(json.loads(marker.read_text(encoding="utf-8"))["step"])
    except (ValueError, KeyError):
        return None
