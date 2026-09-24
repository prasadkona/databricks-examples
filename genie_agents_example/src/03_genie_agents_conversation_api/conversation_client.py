"""Poll-based client for the Genie Agent Conversation REST API."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import requests

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
URL_PATTERN = re.compile(r"https?://[^\s<>\])\"']+")


class ConversationApiError(RuntimeError):
    """Raised for Conversation API HTTP or terminal message failures."""


@dataclass
class RequestTrace:
    method: str
    path: str
    status: int
    duration_ms: int


@dataclass
class ConversationResult:
    conversation_id: str
    message_id: str
    message: dict[str, Any]
    query_results: list[dict[str, Any]]
    conversation_url: str


class GenieConversationClient:
    """Start a conversation, poll its first message, and fetch query results."""

    def __init__(
        self,
        host: str,
        access_token: str,
        agent_id: str,
        *,
        poll_interval_seconds: float = 3,
        max_poll_attempts: int = 200,
        timeout_seconds: float = 120,
    ) -> None:
        self._host = host.rstrip("/")
        self._agent_id = quote(agent_id, safe="")
        self._poll_interval = poll_interval_seconds
        self._max_polls = max_poll_attempts
        self._timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "genie-agents-conversation-api-example/1.0",
            }
        )
        self.traces: list[RequestTrace] = []

    @property
    def base_path(self) -> str:
        return f"/api/2.0/genie/spaces/{self._agent_id}"

    def ask_and_wait(self, question: str) -> ConversationResult:
        """Start one conversation and poll its response to terminal state."""
        started = self._request(
            "POST",
            f"{self.base_path}/start-conversation",
            json_body={"content": question},
        )
        conversation = started.get("conversation") or {}
        message = started.get("message") or {}
        conversation_id = str(
            conversation.get("id")
            or conversation.get("conversation_id")
            or message.get("conversation_id")
            or ""
        )
        message_id = str(message.get("id") or message.get("message_id") or "")
        if not conversation_id or not message_id:
            raise ConversationApiError(
                "Start response did not include conversation_id and message_id:\n"
                + json.dumps(started, indent=2, default=str)[:8000]
            )

        print(f"conversation_id: {conversation_id}")
        print(f"message_id     : {message_id}")
        last_status = ""
        for attempt in range(1, self._max_polls + 1):
            message = self._request(
                "GET",
                self.message_path(conversation_id, message_id),
            )
            status = str(message.get("status") or "UNKNOWN").upper()
            if status != last_status:
                print(f"poll {attempt}: {status}")
                last_status = status
            if status in TERMINAL_STATUSES:
                if status != "COMPLETED":
                    raise ConversationApiError(
                        f"Message ended with {status}: "
                        f"{json.dumps(message.get('error'), default=str)}"
                    )
                query_results = self.fetch_query_results(
                    conversation_id, message_id, message
                )
                return ConversationResult(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    message=message,
                    query_results=query_results,
                    conversation_url=(
                        f"{self._host}/genie/rooms/{self._agent_id}"
                        f"/chats/{quote(conversation_id, safe='')}"
                    ),
                )
            if attempt < self._max_polls:
                time.sleep(self._poll_interval)
        raise TimeoutError(
            f"Message did not complete after {self._max_polls} polls "
            f"({self._max_polls * self._poll_interval:.0f}s of poll delay)."
        )

    def create_follow_up(
        self, conversation_id: str, content: str
    ) -> dict[str, Any]:
        """Start a follow-up turn; caller can poll its returned message ID."""
        return self._request(
            "POST",
            f"{self.base_path}/conversations/{quote(conversation_id, safe='')}/messages",
            json_body={"content": content},
        )

    def fetch_query_results(
        self,
        conversation_id: str,
        message_id: str,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch a result for every query attachment that advertises an ID."""
        results: list[dict[str, Any]] = []
        for attachment in message.get("attachments") or []:
            if not isinstance(attachment, dict) or "query" not in attachment:
                continue
            attachment_id = str(
                attachment.get("attachment_id") or attachment.get("id") or ""
            )
            if not attachment_id:
                continue
            path = (
                f"{self.message_path(conversation_id, message_id)}"
                f"/query-result/{quote(attachment_id, safe='')}"
            )
            try:
                payload = self._request("GET", path)
                results.append(
                    {"attachment_id": attachment_id, "result": payload}
                )
            except ConversationApiError as exc:
                results.append(
                    {"attachment_id": attachment_id, "error": str(exc)}
                )
        return results

    def message_path(self, conversation_id: str, message_id: str) -> str:
        return (
            f"{self.base_path}/conversations/{quote(conversation_id, safe='')}"
            f"/messages/{quote(message_id, safe='')}"
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._session.request(
            method,
            f"{self._host}{path}",
            json=json_body,
            timeout=self._timeout,
        )
        duration_ms = round((time.perf_counter() - started) * 1000)
        self.traces.append(
            RequestTrace(method, path, response.status_code, duration_ms)
        )
        if not response.ok:
            raise ConversationApiError(
                f"{method} {path} returned HTTP {response.status_code}\n"
                f"{response.text[:8000]}"
            )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise ConversationApiError(
                f"{method} {path} returned invalid JSON: {response.text[:2000]}"
            ) from exc
        if not isinstance(payload, dict):
            raise ConversationApiError(
                f"{method} {path} returned a non-object JSON response."
            )
        return payload

    def print_trace(self) -> None:
        print("\n=== HTTP request trace ===")
        for trace in self.traces:
            print(
                f"{trace.method:<4} {trace.status:<3} "
                f"{trace.duration_ms:>6} ms  {trace.path}"
            )


def print_conversation_result(result: ConversationResult) -> None:
    """Print text, generated SQL, suggested questions, and query previews."""
    print("\n=== Genie Agent Conversation API result ===")
    print(f"conversation_id : {result.conversation_id}")
    print(f"message_id      : {result.message_id}")
    print(f"status          : {result.message.get('status')}")
    print(f"Genie Agent deep link : {result.conversation_url}")

    attachments = result.message.get("attachments") or []
    if not attachments:
        print("\nNo attachments returned.")
    for index, attachment in enumerate(attachments, 1):
        if not isinstance(attachment, dict):
            continue
        print(f"\n--- attachment {index} ---")
        text = attachment.get("text")
        if isinstance(text, dict):
            print(text.get("content") or json.dumps(text, indent=2, default=str))
        query = attachment.get("query")
        if isinstance(query, dict):
            if query.get("title"):
                print(f"title: {query['title']}")
            if query.get("description"):
                print(f"description: {query['description']}")
            if query.get("query"):
                print("SQL:")
                print(query["query"])
        suggested = attachment.get("suggested_questions")
        if isinstance(suggested, dict):
            print("Suggested questions:")
            for question in suggested.get("questions") or []:
                print(f"  - {question}")
        known = {"attachment_id", "id", "text", "query", "suggested_questions"}
        extra = {key: value for key, value in attachment.items() if key not in known}
        if extra:
            print(json.dumps(extra, indent=2, default=str)[:4000])

    for item in result.query_results:
        print(f"\n=== Query result: {item['attachment_id']} ===")
        if item.get("error"):
            print(f"Could not fetch preview: {item['error']}")
        else:
            print(json.dumps(item.get("result"), indent=2, default=str)[:12000])

    urls = extract_urls(result.message)
    print("\nSource/attribution URLs returned by the API:")
    for url in urls:
        print(f"  - {url}")
    if not urls:
        print("  (none returned; use the Genie Agent deep link above for attribution)")


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
