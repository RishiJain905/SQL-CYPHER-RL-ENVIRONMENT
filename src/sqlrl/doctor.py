"""Is each secret set? Names and set / not set only, never a value (CLAUDE.md, Security).

python -m sqlrl.doctor
"""

from __future__ import annotations

import sys

from sqlrl import secrets

NAMES = {"openrouter_api_key": "OPENROUTER_API_KEY", "wandb_api_key": "WANDB_API_KEY",
         "hf_token": "HF_TOKEN"}  # fmt: skip


def report() -> dict[str, bool]:
    s = secrets.get()
    out = {}
    for field, name in NAMES.items():
        v = getattr(s, field)
        out[name] = bool(v is not None and v.get_secret_value().strip())
    return out


def main() -> int:
    for name, ok in report().items():
        print(f"{name}: {'set' if ok else 'not set'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
