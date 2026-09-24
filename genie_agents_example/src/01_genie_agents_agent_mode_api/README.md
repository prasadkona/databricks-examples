# Example 01: Genie Agent Mode API without visualization (REST/SSE)

Streams a multi-step answer from one curated Genie Agent:

```text
POST /api/2.0/genie/agents/{agent_id}/responses
Accept: text/event-stream
```

This is the recommended API when an integration needs the agent’s research
plan, SQL calls/results, final report, supporting tables, and citations as
typed streaming events. It is REST + SSE, not MCP.

This CLI intentionally sends **`enable_viz: false`**. It demonstrates the
text/table Agent Mode contract without requesting chart generation. Example
02 uses this same endpoint with **`enable_viz: true`** and displays the
generated visualization in a browser UI.

## Requirements

- Agent Mode APIs enabled from the workspace Previews page (currently Beta)
- Caller has `CAN QUERY` on the Genie Agent
- OAuth app permits the `genie` scope
- `GENIE_AGENT_ID` set in a local `.env`; the same ID is used by all examples

Connection values load using the shared order described in the top README.
Never commit workspace hosts, IDs, client credentials, or tokens.

## Lifecycle

The request contains exactly one user message:

```json
{
  "input": [{
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "<question>"}]
  }],
  "enable_viz": false
}
```

The client consumes:

```text
response.created
response.output_item.added / updated / done
response.completed | response.failed
```

It checks monotonic `sequence_number` values and prints reasoning, function
calls (including SQL), function outputs, the final message, and citation URLs.
Sanitized event and summary artifacts go under the gitignored `artifacts/`.

For a follow-up, set `GENIE_AGENT_CONVERSATION_ID` to the prior
`conversation_id`; the server retains context.

## Run

```bash
./run.sh
```

Optional env:

- `GENIE_QUESTION`
- `GENIE_AGENT_CONVERSATION_ID`
- `GENIE_AGENT_TIMEOUT_SEC` (default 1800)

## Known API behavior

- `404 FEATURE_DISABLED`: Agent Mode API preview is not enabled
- `403 PERMISSION_DENIED`: caller lacks `CAN QUERY`
- `409 RESOURCE_CONFLICT`: another response is in flight for the conversation
- `429 RATE_LIMIT_EXCEEDED`: workspace limit (documented as 5 requests/minute)
- Server timeout: 30 minutes

Docs:

- [Agent mode APIs in Genie Agents](https://docs.databricks.com/aws/en/genie-agents/api)
- [Create a response API](https://docs.databricks.com/api/genie/v1/agent-mode-create-response)
