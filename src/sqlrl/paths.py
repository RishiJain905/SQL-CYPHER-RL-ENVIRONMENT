"""This machine's paths, from the git-ignored `configs/paths.local.toml` (standard library only:
it runs under Windows Python next to llama-server and inside WSL)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs" / "paths.local.toml"


def to_wsl(p: Path | str) -> str:
    """`D:\\a\\b` -> `/mnt/d/a/b`; anything else unchanged."""
    s = str(p).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if s[1:2] == ":" else s


def from_wsl(s: str) -> Path:
    """`/mnt/d/a/b` -> `D:/a/b` on Windows; unchanged elsewhere."""
    parts = PurePosixPath(s).parts
    if len(parts) > 2 and parts[1] == "mnt" and len(parts[2]) == 1 and Path("C:/").exists():
        return Path(f"{parts[2].upper()}:/", *parts[3:])
    return Path(s)


@dataclass(frozen=True)
class Paths:
    data: Path
    model: Path
    evalset: Path
    greedy_scores: Path
    trainer_repo: Path
    wsl_venv: str
    # step 2: the merged model in bf16 (the trainer's starting weights) and its Q8_0 (the
    # rollout server's), both on a fast disk; the llama.cpp source tree (convert_lora_to_gguf)
    model_hf: Path | None = None
    model_gguf: Path | None = None
    llama_cpp_src: Path | None = None

    @property
    def serve_script(self) -> Path:
        return self.trainer_repo / "scripts" / "llamacpp" / "serve.ps1"


def get_paths() -> Paths:
    if not LOCAL.exists():
        raise FileNotFoundError(
            "configs/paths.local.toml is missing: copy configs/paths.example.toml and fill it in"
        )
    raw = tomllib.loads(LOCAL.read_text(encoding="utf-8"))["paths"]
    local = Path("C:/").exists()  # Windows: the file's own paths; WSL: through /mnt/<drive>

    def p(key: str) -> Path:
        return Path(raw[key]) if local else Path(to_wsl(raw[key]))

    return Paths(
        data=p("data"),
        model=p("model"),
        evalset=p("evalset"),
        greedy_scores=p("greedy_scores"),
        trainer_repo=p("trainer_repo"),
        wsl_venv=raw["wsl_venv"],
        model_hf=p("model_hf") if "model_hf" in raw else None,
        model_gguf=p("model_gguf") if "model_gguf" in raw else None,
        llama_cpp_src=p("llama_cpp_src") if "llama_cpp_src" in raw else None,
    )
