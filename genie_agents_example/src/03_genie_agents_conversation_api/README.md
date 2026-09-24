# Example 03: Genie Agent Conversation REST API

Starts a conversation with a curated Genie Agent and polls its first response:

```text
POST /api/2.0/genie/spaces/{agent_id}/start-conversation
GET  /api/2.0/genie/spaces/{agent_id}/conversations/{conversation_id}/messages/{message_id}
```

The product is now called **Genie Agents**, but this established Conversation
API retains `/spaces/` in its URL. The `{agent_id}` is the same
`GENIE_AGENT_ID` used by Examples 01 and 02.

Use this API for a straightforward REST integration where polling is
acceptable and you do not need Agent Mode’s multi-step SSE stream.

## Requirements

- Caller can access/query the target Genie Agent and its underlying data
- OAuth app permits the `genie` scope
- `GENIE_AGENT_ID` set in a local `.env`

Connection values load using the shared order described in the top README.
Never commit workspace hosts, agent IDs, client credentials, or tokens.

## Lifecycle

1. `start-conversation` sends `{"content": "<question>"}`.
2. Read `conversation_id` and `message_id`.
3. Poll the message every 1–5 seconds.
4. Stop on `COMPLETED`, `FAILED`, or `CANCELLED`.
5. Print text, generated SQL, suggested questions, and other attachments.
6. For each query attachment, call its `query-result/{attachment_id}` endpoint.
7. Print the Genie Agent conversation deep link and source/attribution URLs.

Follow-ups use:

```text
POST /api/2.0/genie/spaces/{agent_id}/conversations/{conversation_id}/messages
```

## Run

```bash
./run.sh
```

Optional env:

- `GENIE_QUESTION`
- `GENIE_CONVERSATION_POLL_INTERVAL_SEC` (default 3)
- `GENIE_CONVERSATION_MAX_POLL_ATTEMPTS` (default 200)
- `GENIE_CONVERSATION_HTTP_TIMEOUT_SEC` (default 120 per request)

The CLI prints an HTTP trace without authorization headers or token values.

Docs:

- [Use the Genie Agents API](https://docs.databricks.com/aws/en/genie-agents/conversation-api)
- [Start conversation API](https://docs.databricks.com/api/genie/v1/genie-start-conversation)
- [Get conversation message API](https://docs.databricks.com/api/genie/v1/genie-get-conversation-message)
