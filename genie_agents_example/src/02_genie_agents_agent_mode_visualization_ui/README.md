# Example 02: Genie Agent Mode visualization UI

A local React UI for the same Genie Agent Mode Responses API used by Example 01:

```text
POST /api/2.0/genie/agents/{agent_id}/responses
Accept: text/event-stream
```

The important difference is fixed in the implementation:

- **Example 01:** `enable_viz: false`, CLI output, no visualization request.
- **Example 02:** `enable_viz: true`, browser UI, generated visualization
  metadata and rendered chart download when the Genie Agent determines a chart
  is appropriate.

Setting `enable_viz` to true requests a visualization; it does not guarantee
one. Grouped, ranked, and time-series questions are more likely to produce a
chart than a single aggregate value. The default question asks for revenue by
franchise as a bar chart for that reason.

## Experience

1. Open the local UI.
2. Enter the workspace host, Genie Agent ID, OAuth client ID and secret.
3. Complete OAuth user-to-machine (U2M) sign-in with the `genie` scope.
4. Submit a question.
5. Watch Genie Agent Mode Server-Sent Events (SSE) arrive.
6. Review the narrative, SQL, query data, citations, and generated chart.

The browser calls only the local proxy. The OAuth token remains in server
memory and is never exposed to browser JavaScript. The proxy enforces
`enable_viz: true`; the browser cannot disable it.

When the Genie Agent emits `generate_visualization`, the final response includes a
visualization attachment ID and its source query attachment ID. The UI uses
those identifiers with the conversation and response IDs to retrieve and
display the rendered visualization from:

```text
GET /api/2.0/genie/spaces/{agent_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/download-visualization
```

The private visualization definition is not exposed by the public API.

## OAuth application

The OAuth app must:

- allow the `genie` scope;
- include `http://localhost:8020/callback` as an exact redirect URL, unless
  you change the value in the connection screen.

This is a single-user local example. A production service needs encrypted
per-user token storage, secure server sessions, token refresh/expiry handling,
logout, Cross-Site Request Forgery (CSRF) protection, and multi-instance state.

## Run

```bash
./run_web.sh
```

The launcher installs missing dependencies, builds the UI, starts the local
proxy on port 8000 and UI on port 8080, then opens:

```text
http://localhost:8080
```

Nothing runs until the user connects and selects **Execute**.

## Files

```text
proxy.py         # OAuth, forced enable_viz=true, SSE and chart proxy
src/index.tsx    # connection screen, query UI, event/result rendering
src/global.css   # advanced local UI styling
run_web.sh       # build and local process launcher
```

Docs:

- [Agent mode APIs in Genie Agents](https://docs.databricks.com/aws/en/genie-agents/api)
- [Create a response API](https://docs.databricks.com/api/genie/v1/agent-mode-create-response)
