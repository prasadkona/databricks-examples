"""Streaming SSE client for the Genie Agent Mode Responses API."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

TERMINAL_EVENTS = frozenset({"response.completed", "response.failed"})
URL_PATTERN = re.compile(r"https?://[^\s<>\])\"']+")


class AgentModeError(RuntimeError):
    """Raised for Agent Mode HTTP, SSE, or terminal response failures."""


@dataclass
class SseEvent:
    event: str
    data: Any
    raw_data: str

    @property
    def type(self) -> str:
        if isinstance(self.data, dict):
            return str(self.data.get("type") or self.event)
        return self.event

    @property
    def sequence_number(self) -> Optional[int]:
        if isinstance(self.data, dict):
            value = self.data.get("sequence_number")
            return value if isinstance(value, int) else None
        return None


@dataclass
class StreamResult:
    events: list[SseEvent]
    terminal_event: SseEvent
    elapsed_seconds: float

    @property
    def response(self) -> dict[str, Any]:
        data = self.terminal_event.data
        if not isinstance(data, dict):
            return {}
        nested = data.get("response")
        return nested if isinstance(nested, dict) else data


class GenieAgentModeClient:
    """Create one Agent Mode response and consume its SSE stream."""

    def __init__(
        self,
        host: str,
        access_token: str,
        agent_id: str,
        *,
        timeout_seconds: float = 1800,
    ) -> None:
        self.endpoint = (
            f"{host.rstrip('/')}/api/2.0/genie/agents/{agent_id}/responses"
        )
        self._timeout = (30, timeout_seconds)
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "genie-agents-agent-mode-api-example/1.0",
        }

    def create_response(
        self,
        question: str,
        *,
        conversation_id: Optional[str] = None,
        enable_viz: bool = False,
        artifact_path: Optional[Path] = None,
    ) -> StreamResult:
        payload: dict[str, Any] = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                }
            ],
            "enable_viz": enable_viz,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        print(f"POST {self.endpoint}")
        print("Accept: text/event-stream")
        print(f"Question: {question}")
        started = time.perf_counter()
        with requests.post(
            self.endpoint,
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=self._timeout,
        ) as response:
            if not response.ok:
                raise AgentModeError(
                    f"HTTP {response.status_code} from Agent Mode API\n"
                    f"{response.text[:8000]}"
                )
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise AgentModeError(
                    "Expected text/event-stream, received "
                    f"{content_type or '(missing Content-Type)'}\n"
                    f"{response.text[:8000]}"
                )

            events: list[SseEvent] = []
            terminal: Optional[SseEvent] = None
            artifact = (
                artifact_path.open("w", encoding="utf-8") if artifact_path else None
            )
            try:
                for event in iter_sse_events(response):
                    if event.raw_data == "[DONE]":
                        print("[DONE]")
                        break
                    validate_sequence(events, event)
                    events.append(event)
                    print_event(event)
                    if artifact:
                        artifact.write(json.dumps(event.data, default=str) + "\n")
                        artifact.flush()
                    if event.type in TERMINAL_EVENTS:
                        terminal = event
            finally:
                if artifact:
                    artifact.close()

        if terminal is None:
            raise AgentModeError("SSE stream ended without a terminal event.")
        result = StreamResult(
            events=events,
            terminal_event=terminal,
            elapsed_seconds=time.perf_counter() - started,
        )
        if terminal.type != "response.completed":
            raise AgentModeError(
                "Agent response failed: "
                + json.dumps(result.response.get("error") or terminal.data, default=str)
            )
        return result


def iter_sse_events(response: requests.Response) -> Iterator[SseEvent]:
    """Parse standard SSE fields, including multi-line data payloads."""
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        if line == "":
            if data_lines:
                yield decode_event(event_name, "\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield decode_event(event_name, "\n".join(data_lines))


def decode_event(event_name: str, raw_data: str) -> SseEvent:
    if raw_data == "[DONE]":
        return SseEvent(event_name, "[DONE]", raw_data)
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise AgentModeError(
            f"Invalid JSON in SSE event {event_name}: {raw_data[:1000]}"
        ) from exc
    return SseEvent(event_name, data, raw_data)


def validate_sequence(previous: list[SseEvent], current: SseEvent) -> None:
    sequence = current.sequence_number
    if sequence is None:
        return
    prior = [event.sequence_number for event in previous]
    prior = [value for value in prior if value is not None]
    if prior and sequence <= prior[-1]:
        raise AgentModeError(
            f"Non-monotonic sequence_number: {sequence} after {prior[-1]}"
        )


def print_event(event: SseEvent) -> None:
    sequence = event.sequence_number
    prefix = f"seq={sequence}" if sequence is not None else "seq=?"
    print(f"{prefix:<10} {event.type}")
    if isinstance(event.data, dict) and isinstance(event.data.get("item"), dict):
        item = event.data["item"]
        label = str(item.get("type") or "item")
        if item.get("name"):
            label += f" ({item['name']})"
        print(f"           {label}")


def print_final_response(result: StreamResult) -> None:
    """Print reasoning, SQL calls/results, final message, and citation URLs."""
    response = result.response
    print("\n=== Genie Agent Mode result ===")
    print(f"status          : {response.get('status')}")
    print(f"conversation_id : {response.get('conversation_id')}")
    print(f"response_id     : {response.get('id')}")
    print(f"elapsed         : {result.elapsed_seconds:.1f}s")
    print(f"events          : {len(result.events)}")

    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        print(f"\n[{item_type or 'output item'}]")
        if item_type in {"message", "reasoning"}:
            print_text_content(item.get("content") or [])
            for summary in item.get("summary") or []:
                if isinstance(summary, dict) and summary.get("text"):
                    print(summary["text"])
        elif item_type == "function_call":
            print(f"name: {item.get('name')}")
            print(item.get("arguments") or "")
        elif item_type == "function_call_output":
            output = item.get("output")
            print(output if isinstance(output, str) else json.dumps(output, indent=2))
        else:
            print(json.dumps(item, indent=2, default=str)[:12000])

    urls = extract_urls(response)
    print("\nSource/citation URLs:")
    for url in urls:
        print(f"  - {url}")
    if not urls:
        print("  (none returned)")


def print_text_content(content: list[Any]) -> None:
    for part in content:
        if isinstance(part, dict) and part.get("text"):
            print(part["text"])


def extract_urls(value: Any) -> list[str]:
    strings: list[str] = []
    collect_strings(value, strings)
    urls: list[str] = []
    for text in strings:
        for url in URL_PATTERN.findall(text):
            if url not in urls:
                urls.append(url)
    return urls


def collect_strings(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        output.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            collect_strings(nested, output)
    elif isinstance(value, list):
        for nested in value:
            collect_strings(nested, output)
