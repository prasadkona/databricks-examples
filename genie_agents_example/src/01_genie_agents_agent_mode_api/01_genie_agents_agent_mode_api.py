"""Example 01 — Genie Agent Mode API over REST + Server-Sent Events."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

MODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODE_DIR.parent / "common"))

from env_load import load_connection_env, required_env  # noqa: E402
from u2m_auth import acquire_access_token, normalize_host  # noqa: E402

from agent_mode_client import (  # noqa: E402
    AgentModeError,
    GenieAgentModeClient,
    print_final_response,
)

DEFAULT_QUESTION = "get the total revenue for my bakehouse"
PREFIX = "[01_genie_agents_agent_mode_api]"


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
    conversation_id = (
        os.environ.get("GENIE_AGENT_CONVERSATION_ID", "").strip() or None
    )
    enable_viz = False
    artifacts_dir = MODE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    events_path = artifacts_dir / f"01_agent_mode_events_{timestamp}.jsonl"
    summary_path = artifacts_dir / f"01_agent_mode_summary_{timestamp}.json"

    print(f"{PREFIX} env file        : {env_path}")
    print(f"{PREFIX} host            : {host}")
    print(f"{PREFIX} agent id       : {agent_id}")
    print(f"{PREFIX} OAuth scope    : genie")
    print(f"{PREFIX} question       : {question}")
    print(f"{PREFIX} enable viz     : {enable_viz}")
    print(f"{PREFIX} event artifact : {events_path}")

    try:
        access_token = acquire_access_token(scope="genie")
        print(f"{PREFIX} token acquired (value not printed)\n")
        client = GenieAgentModeClient(
            host,
            access_token,
            agent_id,
            timeout_seconds=float(
                os.environ.get("GENIE_AGENT_TIMEOUT_SEC", "1800")
            ),
        )
        result = client.create_response(
            question,
            conversation_id=conversation_id,
            enable_viz=enable_viz,
            artifact_path=events_path,
        )
        print_final_response(result)
        summary_path.write_text(
            json.dumps(
                {
                    "endpoint": client.endpoint,
                    "terminal_event": result.terminal_event.type,
                    "elapsed_seconds": result.elapsed_seconds,
                    "event_count": len(result.events),
                    "response": result.response,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\n{PREFIX} summary artifact: {summary_path}")
    except (
        EnvironmentError,
        ValueError,
        TimeoutError,
        RuntimeError,
        requests.HTTPError,
        AgentModeError,
    ) as exc:
        fail(str(exc))


def fail(message: str) -> None:
    print(f"{PREFIX} ERROR — {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
