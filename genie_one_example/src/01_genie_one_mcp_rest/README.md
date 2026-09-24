# Mode 1: Genie One MCP over REST JSON-RPC

Headless CLI against the **Genie One MCP Service** on Unity Gateway. You own
rendering: this mode prints traces and the final answer to the terminal. It
does **not** advertise MCP Apps, so the server exposes text tools
(`genie_ask`, `genie_poll_response`, …), not `view_ask`.

Shared helpers: [`../common/`](../common/) (`env_load`, `u2m_auth`, `mcp_client`).
Product overview: [../../README.md](../../README.md).

## When to use this

Partner agents, services, and CLIs that need a governed Genie One answer
without hosting an iframe. Same MCP Service as Mode 2; different client
capabilities.

## How the API works in this mode

**Endpoint (pattern):** `https://<workspace-hostname>/ai-gateway/mcp-services/system.ai.genie_one_mcp`

**Transport:** JSON-RPC 2.0 over HTTP. OAuth U2M with scope **`ai-gateway`**.

**Lifecycle**

1. Load workspace host and OAuth app settings from a local `.env` (see
   [`.env.template`](../../.env.template)). Do not commit secrets.
2. Browser OAuth U2M (authorization code + PKCE). The access token is used
   on the MCP `POST`s; it is not printed.
3. `initialize` → `notifications/initialized` → `tools/list`.
4. **`genie_ask` once** with the question. That **starts** a turn and returns
   `conversation_id`, `response_id`, and usually `status: in_progress`.
5. **`genie_poll_response`** with those exact IDs until `completed`,
   `incomplete`, or `failed`. Do not call `genie_ask` again to check status
   (that starts a new turn). Wait for each poll HTTP call to finish before
   the next.
6. If the poll payload flags a truncated SQL result, call
   **`genie_get_query_result`** for the full table.

Follow-ups: pass the previous `conversation_id` into a **new** `genie_ask`,
then poll the new `response_id`. Optional `_meta.warehouse_id` on `genie_ask`
pins SQL to a warehouse.

The CLI prints HTTP traces (status, timing, method names — **not** bearer
tokens) and source attribution / Explore-in-Databricks URLs from the answer
text.

## Layout

```text
01_genie_one_mcp_rest.py   # entry: handshake, ask once, poll, optional table
run.sh                     # venv, PYTHONPATH=../common, run the CLI
```

## Run

From this directory:

```bash
./run.sh
```

Override the sample question with `GENIE_MCP_QUESTION`. Optional:
`DATABRICKS_WAREHOUSE_ID`, `GENIE_MCP_POLL_INTERVAL_SEC`,
`GENIE_MCP_MAX_POLL_ATTEMPTS`.

## Docs

- [Genie One MCP server](https://docs.databricks.com/aws/en/agents/mcp-tools/genie-mcp)
- [Managed MCP servers](https://docs.databricks.com/aws/en/agents/mcp-tools/managed-mcp) (`ai-gateway` scope)
- Deprecated Beta URL `/api/2.0/mcp/genie` (scope `genie`) sunsets **October 31, 2026**
