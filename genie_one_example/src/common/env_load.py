"""Load Databricks connection settings from a parent .env (never commit secrets).

Prefers ``myprojects/_local/.env`` (DATABRICKS_HOST + U2M client id/secret).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# This file lives at src/common/env_load.py
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _EXAMPLE_ROOT.parent
_MYPROJECTS_ROOT = _REPO_ROOT.parent


def env_candidates() -> list[Path]:
    explicit = os.environ.get("GENIE_ONE_ENV_FILE", "").strip()
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.extend(
        [
            _MYPROJECTS_ROOT / "_local" / ".env",
            _EXAMPLE_ROOT / ".env",
            _REPO_ROOT / ".env",
        ]
    )
    return paths


def load_connection_env() -> Path:
    """Load the first existing .env from the search path. Does not override set vars."""
    for path in env_candidates():
        if path.is_file():
            load_dotenv(path, override=False)
            return path
    searched = "\n".join(f"  - {path}" for path in env_candidates())
    raise FileNotFoundError(
        "No .env found. Looked at:\n"
        f"{searched}\n"
        "Set GENIE_ONE_ENV_FILE or add DATABRICKS_HOST and U2M client credentials."
    )
