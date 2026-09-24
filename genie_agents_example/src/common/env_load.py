"""Load Genie Agents example settings from a local, uncommitted .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXAMPLE_ROOT.parent
MYPROJECTS_ROOT = REPO_ROOT.parent


def env_candidates() -> list[Path]:
    """Return env paths in precedence order."""
    explicit = os.environ.get("GENIE_AGENTS_ENV_FILE", "").strip()
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.extend(
        [
            MYPROJECTS_ROOT / "_local" / ".env",
            EXAMPLE_ROOT / ".env",
            REPO_ROOT / ".env",
        ]
    )
    return paths


def load_connection_env() -> Path:
    """Load the first existing env file without overriding exported values."""
    for path in env_candidates():
        if path.is_file():
            load_dotenv(path, override=False)
            return path
    searched = "\n".join(f"  - {path}" for path in env_candidates())
    raise FileNotFoundError(
        "No .env found. Looked at:\n"
        f"{searched}\n"
        "Set GENIE_AGENTS_ENV_FILE or create one of the files above."
    )


def required_env(name: str) -> str:
    """Read one required environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is required.")
    return value
