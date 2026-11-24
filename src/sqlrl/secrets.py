"""The repo's secrets, loaded only here and only as `SecretStr` (CLAUDE.md, Security).

Read from the environment or the repo's git-ignored env file (names in `.env.example`). Nothing
here prints, logs or returns a secret's value; callers use `.get_secret_value()` at the moment
of use (an HTTP header) and nowhere else. `python -m sqlrl.doctor` reports set / not set.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO = Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO / ".env", env_file_encoding="utf-8",
                                      extra="ignore")  # fmt: skip

    openrouter_api_key: SecretStr | None = None
    wandb_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    def __repr__(self) -> str:  # never show values, even masked ones, in a traceback
        return "Secrets(...)"

    __str__ = __repr__


@lru_cache(maxsize=1)
def get() -> Secrets:
    return Secrets()
