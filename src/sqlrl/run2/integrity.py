"""Run 2's saved state, checked byte for byte (standard library only). Run 1's module,
copied (decisions R6).

Run 1b found its state files damaged on the way to the data drive: twice a 16 KB run of
another process's memory (a Java VM's counters, from a Neo4j container sharing WSL's kernel)
landed page-aligned inside `optim.pt`, written from WSL through the drive's 9p mount. The NaN
guard in `train_step` caught it one step later; this catches it at once.

- `save_verified` (the trainer, in WSL): writes a state file from bytes held in memory, reads it
  back and compares, tries again if they differ, and records the bytes' SHA-256 next to the file
  (`<name>.sha256`).
- `mismatched` (the orchestrator, on Windows): hashes the files as they are on the disk itself
  and names any that differ from what the trainer wrote; the orchestrator then trains the step
  again (its rollouts and scores are kept). Files written before the hashes existed have no
  `.sha256` and are taken as they are.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

STATE_FILES = ("lora_state.safetensors", "optim.pt", "ema_state.safetensors")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_cache(path: Path) -> None:
    """Ask the kernel to forget the file's cached pages, so the read-back comes from the disk
    where it can (Linux only; a no-op elsewhere or where the filesystem ignores it)."""
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_verified(data: bytes, path: Path, tries: int = 3) -> str:
    """Write `data` to `path` through a temporary file, read it back, and keep it only when the
    bytes match; then record their SHA-256 in `<name>.sha256`. Returns the hash."""
    want = hashlib.sha256(data).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    for _ in range(tries):
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _drop_cache(tmp)
        if sha256_file(tmp) == want:
            os.replace(tmp, path)
            path.with_name(path.name + ".sha256").write_text(want, encoding="utf-8")
            return want
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{path.name}: the file read back differs from what was written "
                       f"({tries} tries)")  # fmt: skip


def mismatched(step_dir: Path) -> list[str]:
    """The step's state files whose bytes on the disk differ from the hash recorded when they
    were written (a file without a recorded hash is taken as it is)."""
    bad = []
    for name in STATE_FILES:
        f, h = step_dir / name, step_dir / (name + ".sha256")
        if f.exists() and h.exists() and sha256_file(f) != h.read_text(encoding="utf-8").strip():
            bad.append(name)
    return bad
