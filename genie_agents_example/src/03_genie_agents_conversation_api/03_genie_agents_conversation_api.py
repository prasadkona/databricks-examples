"""Example 03 — Genie Agent Conversation API over poll-based REST."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

MODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODE_DIR.parent / "common"))

from env_load import load_connection_env, required_env  # noqa: E402
from u2m_auth import acquire_access_token, normalize_host  # noqa: E402

from conversation_client import (  # noqa: E402
    ConversationApiError,
    GenieConversationClient,
    print_conversation_result,
)

DEFAULT_QUESTION = "get the total revenue for my bakehouse"
PREFIX = "[03_genie_agents_conversation_api]"


def main() -> None:
    try:
        env_path = load_connection_env()
        host = normalize_host(required_env("DATABRICKS_HOST"))
        agent_id = required_env("GENIE_AGENT_ID")
    except (FileNotFoundError, EnvironmentError) as exc:
        fail(f"configuration: {exc}")

    question = os.environ.get("GENIE_QUESTION", DEFAULT_QUESTION).strip()
    if not question:
        fail("GENIE_QUESTION cannot be blank.")

    print(f"{PREFIX} env file     : {env_path}")
    print(f"{PREFIX} host         : {host}")
    print(f"{PREFIX} agent id    : {agent_id}")
    print(f"{PREFIX} OAuth scope : genie")
    print(f"{PREFIX} question    : {question}")

    try:
        access_token = acquire_access_token(scope="genie")
        print(f"{PREFIX} token acquired (value not printed)\n")
        client = GenieConversationClient(
            host,
            access_token,
            agent_id,
            poll_interval_seconds=float(
                os.environ.get("GENIE_CONVERSATION_POLL_INTERVAL_SEC", "3")
            ),
            max_poll_attempts=int(
                os.environ.get("GENIE_CONVERSATION_MAX_POLL_ATTEMPTS", "200")
            ),
            timeout_seconds=float(
                os.environ.get("GENIE_CONVERSATION_HTTP_TIMEOUT_SEC", "120")
            ),
        )
        result = client.ask_and_wait(question)
        client.print_trace()
        print_conversation_result(result)
    except (
        EnvironmentError,
        ValueError,
        TimeoutError,
        RuntimeError,
        requests.HTTPError,
        ConversationApiError,
    ) as exc:
        fail(str(exc))


def fail(message: str) -> None:
    print(f"{PREFIX} ERROR — {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
