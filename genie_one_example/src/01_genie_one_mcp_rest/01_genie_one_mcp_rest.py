"""Mode 1 — Genie One MCP over Unity Gateway (REST JSON-RPC).

This is Mode 1. Shared helpers live in ``src/common/``. Later modes land as
``src/0N_*``.

Docs:
    https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp

How this mode works:
    1. Load DATABRICKS_HOST and U2M client id/secret from myprojects/_local/.env
    2. Browser OAuth U2M (PKCE). Request scope ``ai-gateway``.
    3. MCP initialize + tools/list on
       ``/ai-gateway/mcp-services/system.ai.genie_one_mcp``
    4. Call ``genie_ask`` once. That only starts work; it does not wait for
       the answer.
    5. Poll ``genie_poll_response`` with the returned conversation_id and
       response_id until status is completed (or failed / incomplete).
    6. If Genie One flags a query result, call ``genie_get_query_result`` for
       the full table (poll payloads may be truncated).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import requests

from env_load import load_connection_env
from mcp_client import (
    GenieMcpError,
    GenieOneMcpClient,
    default_question,
    print_genie_response,
)
from u2m_auth import acquire_access_token, oauth_scope

_PREFIX = "[01_genie_one_mcp_rest]"


def main() -> None:
    # See https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp
    try:
        env_path = load_connection_env()
    except FileNotFoundError as exc:
        print(f"{_PREFIX} ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if not host:
        print(f"{_PREFIX} ERROR: DATABRICKS_HOST is required.", file=sys.stderr)
        sys.exit(1)

    question = default_question()
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "").strip() or None
    poll_interval = float(os.environ.get("GENIE_MCP_POLL_INTERVAL_SEC", "3"))
    max_polls = int(os.environ.get("GENIE_MCP_MAX_POLL_ATTEMPTS", "60"))

    print(f"{_PREFIX} env file        : {env_path}")
    print(f"{_PREFIX} host            : {host}")
    print(f"{_PREFIX} MCP endpoint    : {host.rstrip('/')}/ai-gateway/mcp-services/system.ai.genie_one_mcp")
    print(f"{_PREFIX} auth            : {os.environ.get('APP_AUTH_TYPE', 'oauth_u2m')}")
    print(f"{_PREFIX} oauth scope     : {oauth_scope()}")
    print(f"{_PREFIX} u2m client id   : {os.environ.get('DATABRICKS_U2M_CLIENT_ID', '')}")
    print(f"{_PREFIX} has u2m secret  : {bool(os.environ.get('DATABRICKS_U2M_CLIENT_SECRET', '').strip())}")
    print(f"{_PREFIX} question        : {question}")

    # Step 1: user-to-machine OAuth. Unity Gateway needs the ai-gateway scope.
    print(f"\n{_PREFIX} Step 1 — acquire U2M access token")
    try:
        access_token = acquire_access_token()
        print(f"{_PREFIX} token acquired (value not printed)")
    except (EnvironmentError, ValueError, RuntimeError, requests.HTTPError) as exc:
        print(f"{_PREFIX} ERROR — authentication: {exc}", file=sys.stderr)
        sys.exit(1)

    client = GenieOneMcpClient(
        host=host,
        access_token=access_token,
        poll_interval_sec=poll_interval,
        max_poll_attempts=max_polls,
    )

    # Step 2: MCP session, then discover tools (genie_ask, genie_poll_response, ...).
    print(f"\n{_PREFIX} Step 2 — MCP initialize + tools/list")
    try:
        client.initialize()
        client.list_tools()
    except (GenieMcpError, requests.HTTPError) as exc:
        print(f"{_PREFIX} ERROR — MCP handshake: {exc}", file=sys.stderr)
        sys.exit(1)

    # Step 3: genie_ask starts the turn; ask_and_wait polls until a result comes back.
    # Do not treat the first genie_ask payload as the final answer.
    print(f"\n{_PREFIX} Step 3 — genie_ask + genie_poll_response")
    try:
        response = client.ask_and_wait(question, warehouse_id=warehouse_id)
        # Request timings show the initialize/list/ask/poll lifecycle. The
        # final output also prints source attribution URLs independently from
        # the overall Genie One conversation deep link.
        client.print_trace()
        print_genie_response(response)
    except (GenieMcpError, TimeoutError, requests.HTTPError) as exc:
        print(f"{_PREFIX} ERROR — Genie One ask/poll: {exc}", file=sys.stderr)
        sys.exit(1)

    if response.query_result_available:
        # Optional: full SQL result when the polled answer only included a preview.
        print(f"\n{_PREFIX} Step 4 — genie_get_query_result")
        try:
            result = client.get_query_result(response)
            structured = result.get("structuredContent") or result
            print(json_preview(structured))
        except GenieMcpError as exc:
            print(f"{_PREFIX} WARN — query result fetch skipped: {exc}")

    print(f"\n{_PREFIX} Done.")


def json_preview(payload: object, limit: int = 4000) -> str:
    import json

    text = json.dumps(payload, indent=2, default=str)
    if len(text) > limit:
        return text[:limit] + "\n  ... (truncated)"
    return text


if __name__ == "__main__":
    main()
