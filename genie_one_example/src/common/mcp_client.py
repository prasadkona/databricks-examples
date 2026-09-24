"""JSON-RPC client for the Genie One MCP Service on Unity Gateway.

Docs:
    https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp

Genie One runs asynchronously. ``genie_ask`` only *starts* a response
(``status: in_progress`` plus ``conversation_id`` / ``response_id``). You
must keep calling ``genie_poll_response`` with those exact IDs until status
is terminal (``completed``, ``incomplete``, or ``failed``). Do not call
``genie_ask`` again to check status — that would start a new turn.

Poll cadence: wait for the prior ``genie_poll_response`` call to finish
before polling again. ``genie_ask`` / poll payloads may truncate SQL rows;
use ``genie_get_query_result`` for the full result.

Follow-ups: pass the previous ``conversation_id`` into a new ``genie_ask``.
That starts a new ``response_id`` that you poll separately.

OAuth: on-behalf-of user auth should include the ``ai-gateway`` scope.
The previous Beta URL ``/api/2.0/mcp/genie`` (scope ``genie``) is deprecated
and sunsets 2026-10-31.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from u2m_auth import normalize_host

USER_AGENT = "pkona_genie_one_mcp_src/1.0.0"
# GA Unity Gateway URL. See:
# https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp
MCP_SERVICE_PATH = "/ai-gateway/mcp-services/system.ai.genie_one_mcp"
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "failed", "unknown", "cancelled"})
_DEFAULT_QUESTION = "get the total revenue for my bakehouse"
MCP_APPS_EXTENSION = "io.modelcontextprotocol/ui"
MCP_APPS_MIME_TYPE = "text/html;profile=mcp-app"
_URL_PATTERN = re.compile(r"https?://[^\s<>\])\"']+")


class GenieMcpError(RuntimeError):
    """Raised when the Genie One MCP server returns an error."""


@dataclass
class GenieMcpResponse:
    """Parsed ``tools/call`` result from genie_ask or genie_poll_response."""
    status: str
    conversation_id: str
    response_id: str
    final_answer: Optional[str] = None
    deep_link: Optional[str] = None
    progress_steps: list[str] = field(default_factory=list)
    narration_instruction: Optional[str] = None
    raw_content_text: Optional[str] = None
    query_result_available: bool = False
    structured: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_result(cls, result: dict[str, Any]) -> "GenieMcpResponse":
        structured = dict(result.get("structuredContent") or {})
        content_items = result.get("content") or []
        content_text = None
        if content_items and isinstance(content_items[0], dict):
            content_text = content_items[0].get("text")

        extra_query_ids = any(
            structured.get(key)
            for key in ("query_id", "attachment_id", "statement_id", "query_result_id")
        )
        return cls(
            status=str(structured.get("status") or "unknown"),
            conversation_id=str(structured.get("conversation_id") or ""),
            response_id=str(structured.get("response_id") or ""),
            final_answer=structured.get("final_answer"),
            deep_link=structured.get("deep_link"),
            progress_steps=list(structured.get("progress_steps") or []),
            narration_instruction=structured.get("narration_instruction"),
            raw_content_text=content_text,
            query_result_available=bool(
                structured.get("has_query_result")
                or structured.get("query_result_available")
                or extra_query_ids
            ),
            structured=structured,
        )

    @property
    def is_terminal(self) -> bool:
        """True when polling should stop because Genie One finished this turn."""
        return self.status in _TERMINAL_STATUSES


@dataclass
class McpTraceEvent:
    """One HTTP/JSON-RPC exchange, safe to print because it excludes auth headers."""

    sequence: int
    method: str
    tool_name: Optional[str]
    http_status: int
    duration_ms: float


class GenieOneMcpClient:
    """Call Genie One MCP tools over POST JSON-RPC (no MCP SDK).

    Endpoint:
        ``POST https://<workspace>/ai-gateway/mcp-services/system.ai.genie_one_mcp``

    Typical flow (see docs): initialize → tools/list → genie_ask once →
    poll genie_poll_response until the result arrives → optional
    genie_get_query_result.
    """

    def __init__(
        self,
        host: str,
        access_token: str,
        *,
        timeout_sec: float = 120.0,
        poll_interval_sec: float = 3.0,
        max_poll_attempts: int = 60,
    ) -> None:
        self.endpoint = f"{normalize_host(host)}{MCP_SERVICE_PATH}"
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._max_poll_attempts = max_poll_attempts
        self._request_id = 0
        self._session_id: Optional[str] = None
        self.trace_events: list[McpTraceEvent] = []
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _jsonrpc(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """POST one JSON-RPC 2.0 message. Reuses Mcp-Session-Id when returned."""
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        print(f"POST {self.endpoint}")
        print(f"MCP JSON-RPC: {method}")

        started = time.perf_counter()
        response = requests.post(
            self.endpoint,
            headers=self._headers,
            json=payload,
            timeout=self._timeout_sec,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        tool_name = None
        if method == "tools/call" and params:
            tool_name = str(params.get("name") or "") or None
        self.trace_events.append(
            McpTraceEvent(
                sequence=len(self.trace_events) + 1,
                method=method,
                tool_name=tool_name,
                http_status=response.status_code,
                duration_ms=duration_ms,
            )
        )
        session_id = response.headers.get("mcp-session-id") or response.headers.get(
            "Mcp-Session-Id"
        )
        if session_id:
            self._session_id = session_id
            self._headers["Mcp-Session-Id"] = session_id

        print(f"MCP HTTP status: {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise GenieMcpError(
                f"MCP returned non-JSON body: HTTP {response.status_code}: {response.text[:2000]}"
            ) from exc

        if not response.ok:
            raise GenieMcpError(f"MCP HTTP {response.status_code}: {json.dumps(body)[:2000]}")
        if "error" in body:
            error = body["error"]
            raise GenieMcpError(f"MCP error {error.get('code')}: {error.get('message')}")
        return body.get("result") or {}

    def initialize(self, *, mcp_apps: bool = False) -> dict[str, Any]:
        """MCP handshake required before tools/list and tools/call.

        Set ``mcp_apps=True`` when the host implements the MCP Apps
        specification. The advertised extension causes Genie One to expose
        ``view_ask`` and UI resource metadata instead of only ``genie_ask``.

        MCP Apps specification:
        https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx
        """
        capabilities: dict[str, Any] = {}
        if mcp_apps:
            capabilities["extensions"] = {
                MCP_APPS_EXTENSION: {"mimeTypes": [MCP_APPS_MIME_TYPE]}
            }
        result = self._jsonrpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": capabilities,
                "clientInfo": {"name": USER_AGENT, "version": "1.0.0"},
            },
        )
        # Streamable HTTP clients send this notification after initialize.
        notify = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        requests.post(
            self.endpoint,
            headers=self._headers,
            json=notify,
            timeout=self._timeout_sec,
        )
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        """List tools (genie_ask, genie_poll_response, genie_get_query_result, ...)."""
        result = self._jsonrpc("tools/list", {})
        tools = result.get("tools") or []
        print(f"MCP tools/list returned {len(tools)} tool(s):")
        for tool in tools:
            print(f"  - {tool.get('name')}")
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke one MCP tool via JSON-RPC ``tools/call``."""
        return self._jsonrpc("tools/call", {"name": tool_name, "arguments": arguments})

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Fetch a UI resource advertised in a tool's ``_meta.ui.resourceUri``."""
        return self._jsonrpc("resources/read", {"uri": uri})

    def ask_and_wait(
        self,
        question: str,
        *,
        conversation_id: Optional[str] = None,
        warehouse_id: Optional[str] = None,
    ) -> GenieMcpResponse:
        """Start a Genie One turn with genie_ask, then poll until a result exists.

        ``genie_ask`` is not a complete answer. It returns IDs and usually
        ``in_progress``. This method polls ``genie_poll_response`` until
        Genie One finishes (or we hit max attempts). Never re-asks to poll.

        Pass ``conversation_id`` only for a follow-up in the same thread.
        Optional ``warehouse_id`` is sent as ``_meta`` so SQL runs on that
        warehouse (see Genie One MCP docs).
        """
        ask_args: dict[str, Any] = {"question": question}
        if conversation_id:
            ask_args["conversation_id"] = conversation_id
        if warehouse_id:
            ask_args["_meta"] = {"warehouse_id": warehouse_id}

        print(f"Genie One question: {question}")
        ask_result = self.call_tool("genie_ask", ask_args)
        ask_response = GenieMcpResponse.from_tool_result(ask_result)
        print(
            "genie_ask "
            f"status={ask_response.status} "
            f"conversation_id={ask_response.conversation_id} "
            f"response_id={ask_response.response_id}"
        )
        if not ask_response.conversation_id or not ask_response.response_id:
            raise GenieMcpError(
                "genie_ask did not return conversation_id and response_id; refusing to guess IDs"
            )
        if ask_response.is_terminal:
            return ask_response
        # Keep polling until status is completed / incomplete / failed.
        return self._poll_until_done(ask_response.conversation_id, ask_response.response_id)

    def _poll_until_done(self, conversation_id: str, response_id: str) -> GenieMcpResponse:
        """Call genie_poll_response until the turn finishes.

        Uses the conversation_id and response_id from genie_ask. Sleeps
        between polls; does not start a second genie_ask.
        """
        seen_steps: set[str] = set()
        for attempt in range(1, self._max_poll_attempts + 1):
            print(f"genie_poll_response attempt {attempt}/{self._max_poll_attempts}")
            poll_result = self.call_tool(
                "genie_poll_response",
                {"conversation_id": conversation_id, "response_id": response_id},
            )
            poll_response = GenieMcpResponse.from_tool_result(poll_result)
            for step in poll_response.progress_steps:
                if step not in seen_steps:
                    seen_steps.add(step)
                    print(f"  thinking: {step}")
            print(f"  status: {poll_response.status}")
            if poll_response.is_terminal:
                return poll_response
            # Wait before the next poll; docs: do not overlap poll calls.
            time.sleep(self._poll_interval_sec)
        raise TimeoutError(
            f"Genie One MCP did not complete after {self._max_poll_attempts} polls "
            f"(conversation_id={conversation_id}, response_id={response_id})"
        )

    def get_query_result(self, response: GenieMcpResponse) -> dict[str, Any]:
        """Fetch full SQL rows when poll/ask returned a truncated preview."""
        args: dict[str, Any] = {
            "conversation_id": response.conversation_id,
            "response_id": response.response_id,
        }
        for key in ("query_id", "attachment_id", "statement_id", "query_result_id"):
            if response.structured.get(key):
                args[key] = response.structured[key]
        return self.call_tool("genie_get_query_result", args)

    def print_trace(self) -> None:
        """Print request-level traces without tokens, headers, or secrets."""
        print("=== MCP request trace ===")
        for event in self.trace_events:
            operation = event.method
            if event.tool_name:
                operation = f"{operation} ({event.tool_name})"
            print(
                f"  [{event.sequence:02d}] {operation:<38} "
                f"HTTP {event.http_status}  {event.duration_ms:.0f} ms"
            )


def default_question() -> str:
    return os.environ.get("GENIE_MCP_QUESTION", _DEFAULT_QUESTION).strip() or _DEFAULT_QUESTION


def print_genie_response(response: GenieMcpResponse) -> None:
    print("=== Genie One MCP result ===")
    print(f"  status         : {response.status}")
    print(f"  conversation_id: {response.conversation_id}")
    print(f"  response_id    : {response.response_id}")
    if response.deep_link:
        print(f"  deep_link      : {response.deep_link}")
    answer = response.final_answer or response.raw_content_text
    if answer:
        print("  answer:")
        for line in str(answer).splitlines():
            print(f"    {line}")
    else:
        print("  (no final_answer returned)")

    # Genie One returns source attribution as deep links in the final answer and/or
    # structured payload. Print those URLs separately so a headless client can
    # present clickable provenance alongside the answer.
    source_urls = extract_attribution_urls(response)
    print("  source attribution URLs:")
    if source_urls:
        for url in source_urls:
            print(f"    - {url}")
    else:
        print("    (none returned)")

    if response.status not in {"completed"}:
        raise GenieMcpError(f"Genie One MCP finished with non-success status: {response.status}")


def extract_attribution_urls(response: GenieMcpResponse) -> list[str]:
    """Extract unique HTTP source/citation URLs from a Genie One response.

    ``deep_link`` points to the overall Genie One conversation, so it is printed
    separately and excluded here. These URLs are the answer's source
    attribution links, such as query results and routed Genie Agents.
    """
    values: list[str] = []
    _collect_strings(response.structured, values)
    if response.final_answer:
        values.append(response.final_answer)
    if response.raw_content_text:
        values.append(response.raw_content_text)

    urls: list[str] = []
    for value in values:
        for url in _URL_PATTERN.findall(value):
            if url != response.deep_link and url not in urls:
                urls.append(url)
    return urls


def _collect_strings(value: Any, output: list[str]) -> None:
    """Recursively collect strings from a JSON-compatible payload."""
    if isinstance(value, str):
        output.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            _collect_strings(nested, output)
    elif isinstance(value, list):
        for nested in value:
            _collect_strings(nested, output)
