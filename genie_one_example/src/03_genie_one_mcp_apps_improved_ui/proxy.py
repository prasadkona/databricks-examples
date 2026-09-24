"""Configurable, server-side U2M proxy for the improved Genie One MCP Apps UI."""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from mcp_client import MCP_SERVICE_PATH  # noqa: E402
from u2m_auth import normalize_host, run_pkce_browser_flow  # noqa: E402

UPSTREAM_HEADERS = {
    "accept",
    "content-type",
    "last-event-id",
    "mcp-protocol-version",
    "mcp-session-id",
}
DOWNSTREAM_HEADERS = {
    "cache-control",
    "content-type",
    "mcp-session-id",
    "retry-after",
}
traces: deque[dict[str, object]] = deque(maxlen=200)
DEFAULT_REDIRECT_URI = "http://localhost:8020/callback"


class ConnectionConfig(BaseModel):
    host: str
    client_id: str
    client_secret: SecretStr
    redirect_uri: str = DEFAULT_REDIRECT_URI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=30))
    app.state.endpoint = None
    app.state.token = None
    app.state.workspace = None
    app.state.configure_lock = asyncio.Lock()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Genie One MCP local proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id", "mcp-session-id"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready", "scope": "ai-gateway"}


@app.get("/api/connection")
async def connection_status(request: Request) -> dict[str, object]:
    return {
        "connected": bool(request.app.state.token),
        "workspace": request.app.state.workspace,
        "redirect_uri": DEFAULT_REDIRECT_URI,
    }


def _validated_config(config: ConnectionConfig) -> tuple[str, str, str, str]:
    values = {
        "Workspace host": config.host.strip(),
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
    redirect = urlparse(values["Redirect URL"])
    if (
        redirect.scheme != "http"
        or redirect.hostname not in {"localhost", "127.0.0.1"}
        or redirect.path != "/callback"
    ):
        raise ValueError(
            "Redirect URL must be a localhost HTTP callback ending in /callback."
        )
    return host, values["OAuth client ID"], values["OAuth client secret"], values["Redirect URL"]


@app.post("/api/configure")
async def configure_connection(config: ConnectionConfig, request: Request) -> JSONResponse:
    try:
        host, client_id, client_secret, redirect_uri = _validated_config(config)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    async with request.app.state.configure_lock:
        request.app.state.token = None
        request.app.state.endpoint = None
        request.app.state.workspace = None
        traces.clear()
        try:
            token = await asyncio.to_thread(
                run_pkce_browser_flow,
                host,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="ai-gateway",
            )
        except Exception as exc:
            # Never echo request values: this response is safe for the browser UI.
            return JSONResponse(
                status_code=400,
                content={"error": f"OAuth connection failed: {exc}"},
            )

        request.app.state.token = token
        request.app.state.endpoint = f"{host}{MCP_SERVICE_PATH}"
        request.app.state.workspace = urlparse(host).hostname
        return JSONResponse(
            content={
                "connected": True,
                "workspace": request.app.state.workspace,
                "scope": "ai-gateway",
            }
        )


@app.post("/api/disconnect")
async def disconnect(request: Request) -> dict[str, bool]:
    request.app.state.token = None
    request.app.state.endpoint = None
    request.app.state.workspace = None
    traces.clear()
    return {"connected": False}


@app.get("/api/traces")
async def get_traces() -> dict[str, list[dict[str, object]]]:
    return {"traces": list(traces)}


def safe_request_summary(body: bytes) -> tuple[str, str | None]:
    """Extract operation names only; never retain arguments or credentials."""
    try:
        import json

        payload = json.loads(body) if body else {}
        method = str(payload.get("method") or "")
        params = payload.get("params") or {}
        tool = str(params.get("name") or "") or None
        return method, tool
    except (ValueError, TypeError, AttributeError):
        return "", None


@app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def proxy_mcp(request: Request) -> Response:
    if not request.app.state.token or not request.app.state.endpoint:
        return JSONResponse(
            status_code=409,
            content={"error": "Configure and authenticate the Databricks connection first."},
        )
    body = await request.body()
    method, tool = safe_request_summary(body)
    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in UPSTREAM_HEADERS
    }
    request_headers["Authorization"] = f"Bearer {request.app.state.token}"
    request_headers["User-Agent"] = "genie-one-mcp-apps-improved-local-host/1.0"
    request_headers["Accept-Encoding"] = "identity"

    started = time.perf_counter()
    upstream_request = request.app.state.client.build_request(
        request.method,
        request.app.state.endpoint,
        headers=request_headers,
        content=body,
    )
    upstream = await request.app.state.client.send(upstream_request, stream=True)
    duration_ms = round((time.perf_counter() - started) * 1000)
    traces.append(
        {
            "sequence": len(traces) + 1,
            "method": method or request.method,
            "tool": tool,
            "status": upstream.status_code,
            "durationMs": duration_ms,
        }
    )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() in DOWNSTREAM_HEADERS
    }

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            # Yield decoded bytes because hop-by-hop compression headers are
            # intentionally not forwarded to the browser.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=None,
    )


@app.exception_handler(httpx.HTTPError)
async def upstream_error(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
    # Do not include request headers because they contain the bearer token.
    return JSONResponse(
        status_code=502,
        content={"error": "Databricks MCP upstream request failed", "detail": str(exc)},
    )
