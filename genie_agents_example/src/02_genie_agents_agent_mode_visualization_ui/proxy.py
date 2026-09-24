"""Local OAuth and API proxy for the Genie Agent Mode visualization UI."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import quote, urlparse

import httpx
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, SecretStr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from u2m_auth import normalize_host, run_pkce_browser_flow  # noqa: E402

DEFAULT_REDIRECT_URI = "http://localhost:8020/callback"
traces: deque[dict[str, object]] = deque(maxlen=200)


class ConnectionConfig(BaseModel):
    host: str
    agent_id: str
    client_id: str
    client_secret: SecretStr
    redirect_uri: str = DEFAULT_REDIRECT_URI


class QuestionRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(1800, connect=30),
        follow_redirects=True,
    )
    app.state.host = None
    app.state.agent_id = None
    app.state.token = None
    app.state.configure_lock = asyncio.Lock()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Genie Agent Mode visualization UI proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ready", "scope": "genie", "enable_viz": True}


@app.get("/api/connection")
async def connection_status(request: Request) -> dict[str, object]:
    return {
        "connected": bool(request.app.state.token),
        "workspace": (
            urlparse(request.app.state.host).hostname
            if request.app.state.host
            else None
        ),
        "redirect_uri": DEFAULT_REDIRECT_URI,
        "enable_viz": True,
    }


def validated_config(config: ConnectionConfig) -> tuple[str, str, str, str, str]:
    values = {
        "Workspace host": config.host.strip(),
        "Genie Agent ID": config.agent_id.strip(),
        "OAuth client ID": config.client_id.strip(),
        "OAuth client secret": config.client_secret.get_secret_value().strip(),
        "Redirect URL": config.redirect_uri.strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Fill in: {', '.join(missing)}.")

    host = normalize_host(values["Workspace host"])
    if urlparse(host).scheme != "https":
        raise ValueError("Workspace host must use https://.")
    if len(values["Genie Agent ID"]) != 32:
        raise ValueError("Genie Agent ID must be the 32-character agent identifier.")
    redirect = urlparse(values["Redirect URL"])
    if (
        redirect.scheme != "http"
        or redirect.hostname not in {"localhost", "127.0.0.1"}
        or redirect.path != "/callback"
    ):
        raise ValueError(
            "Redirect URL must be a localhost HTTP callback ending in /callback."
        )
    return (
        host,
        values["Genie Agent ID"],
        values["OAuth client ID"],
        values["OAuth client secret"],
        values["Redirect URL"],
    )


@app.post("/api/configure")
async def configure_connection(
    config: ConnectionConfig, request: Request
) -> JSONResponse:
    try:
        host, agent_id, client_id, client_secret, redirect_uri = validated_config(
            config
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    async with request.app.state.configure_lock:
        request.app.state.token = None
        request.app.state.host = None
        request.app.state.agent_id = None
        traces.clear()
        try:
            token = await asyncio.to_thread(
                run_pkce_browser_flow,
                host,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="genie",
            )
        except (
            ValueError,
            RuntimeError,
            TimeoutError,
            requests.RequestException,
        ) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"OAuth connection failed: {exc}"},
            )

        request.app.state.token = token
        request.app.state.host = host
        request.app.state.agent_id = agent_id
        return JSONResponse(
            content={
                "connected": True,
                "workspace": urlparse(host).hostname,
                "scope": "genie",
                "enable_viz": True,
            }
        )


@app.post("/api/disconnect")
async def disconnect(request: Request) -> dict[str, bool]:
    request.app.state.token = None
    request.app.state.host = None
    request.app.state.agent_id = None
    traces.clear()
    return {"connected": False}


@app.get("/api/traces")
async def get_traces() -> dict[str, list[dict[str, object]]]:
    return {"traces": list(traces)}


def connection_values(request: Request) -> tuple[str, str, str]:
    host = request.app.state.host
    agent_id = request.app.state.agent_id
    token = request.app.state.token
    if not host or not agent_id or not token:
        raise RuntimeError("Configure and authenticate the Databricks connection first.")
    return host, agent_id, token


@app.post("/api/responses")
async def create_response(payload: QuestionRequest, request: Request) -> Response:
    question = payload.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question is required."})
    try:
        host, agent_id, token = connection_values(request)
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})

    endpoint = (
        f"{host}/api/2.0/genie/agents/{quote(agent_id, safe='')}/responses"
    )
    started = time.perf_counter()
    upstream_request = request.app.state.client.build_request(
        "POST",
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "genie-agents-agent-mode-visualization-ui/1.0",
        },
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                }
            ],
            "enable_viz": True,
        },
    )
    upstream = await request.app.state.client.send(upstream_request, stream=True)
    duration_ms = round((time.perf_counter() - started) * 1000)
    traces.append(
        {
            "sequence": len(traces) + 1,
            "operation": "create response (enable_viz=true)",
            "status": upstream.status_code,
            "durationMs": duration_ms,
        }
    )
    if not upstream.is_success:
        body = (await upstream.aread()).decode(errors="replace")
        await upstream.aclose()
        return JSONResponse(
            status_code=upstream.status_code,
            content={"error": body[:8000]},
        )

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def visualization_attachment_ids(attachment_id: str) -> list[str]:
    """Try public IDs, then the stored attachment ID without the pub_ prefix."""
    candidates = [attachment_id]
    without_public_prefix = attachment_id.removeprefix("pub_")
    if without_public_prefix not in candidates:
        candidates.append(without_public_prefix)
    if without_public_prefix.endswith("_output"):
        base = without_public_prefix[: -len("_output")]
        if base and base not in candidates:
            candidates.append(base)
    return candidates


@app.get(
    "/api/visualizations/{conversation_id}/{message_id}/{attachment_id}"
)
async def download_visualization(
    conversation_id: str,
    message_id: str,
    attachment_id: str,
    request: Request,
) -> Response:
    try:
        host, agent_id, token = connection_values(request)
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})

    started = time.perf_counter()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
        "User-Agent": "genie-agents-agent-mode-visualization-ui/1.0",
    }
    upstream: httpx.Response | None = None
    for candidate_id in visualization_attachment_ids(attachment_id):
        endpoint = (
            f"{host}/api/2.0/genie/spaces/{quote(agent_id, safe='')}"
            f"/conversations/{quote(conversation_id, safe='')}"
            f"/messages/{quote(message_id, safe='')}"
            f"/attachments/{quote(candidate_id, safe='')}"
            "/download-visualization"
        )
        upstream = await request.app.state.client.get(endpoint, headers=headers)
        if upstream.status_code != 404:
            break
    assert upstream is not None
    duration_ms = round((time.perf_counter() - started) * 1000)
    traces.append(
        {
            "sequence": len(traces) + 1,
            "operation": "download visualization",
            "status": upstream.status_code,
            "durationMs": duration_ms,
        }
    )
    content_type = upstream.headers.get("content-type", "")
    if not upstream.is_success or not content_type.startswith("image/"):
        return JSONResponse(
            status_code=upstream.status_code if not upstream.is_success else 502,
            content={"error": upstream.text[:8000]},
        )
    return Response(
        content=upstream.content,
        media_type=content_type,
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(httpx.HTTPError)
async def upstream_error(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": "Databricks upstream request failed", "detail": str(exc)},
    )
