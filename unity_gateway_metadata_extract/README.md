# Unity Gateway Metadata Extract

Discover Databricks **AI Gateway** inventory — built-in `system.ai` foundation models, Unity Catalog **model provider services**, and **AgentBricks** custom agents — then invoke models through a single gateway URL.

This is a workspace notebook (plus a matching `.py` Databricks source file). Auth comes from the notebook context (`dbutils`); you do not need a local `.env`.

It is **inventory + invoke**, not a full workspace agent/endpoint classifier. For reports that classify every serving endpoint and Knowledge Assistant, see [`../ai_agent_metadata_extract`](../ai_agent_metadata_extract/README.md).

## Files

| File | Description |
|------|-------------|
| `AI Gateway Discover Invoke system ai Models and Model Services.ipynb` | Databricks notebook — import this into a workspace and run cells in order |
| `AI Gateway Discover Invoke system ai Models and Model Services.py` | Same content as Databricks source (`.py` + `# MAGIC` markdown) for git review or `databricks workspace import` |

## What you get

| Asset | What it is | Discovery |
|-------|------------|-----------|
| **`system.ai` foundation models** | Databricks-managed chat, completion, and embedding models on every workspace. No endpoint create step. | `GET /api/2.0/serving-endpoints` where `endpoint_type == "FOUNDATION_MODEL_API"` |
| **Model provider services** | UC securables (`catalog.schema.service`) that route to external providers (OpenAI, Anthropic, Bedrock, Gemini, and others). Governed with UC privileges. | `GET /api/2.1/unity-catalog/model-provider-services` (paginated) |
| **App-backed custom agents** | Databricks Apps that look like agents in the Agents UI (apps with an MLflow `experiment` resource). | `GET /api/2.0/apps` filtered by `experiment` |
| **UC agent services** | Preferred governed path to call agents through the AI Gateway (same idea as model services). | `GET /api/2.1/unity-catalog/agent-services` (paginated) |

The notebook also fetches **one** foundation-model endpoint’s full metadata (`GET /api/2.0/serving-endpoints/<name>`) and shows **two** invoke paths: Responses API and OpenAI-compatible Chat Completions.

## Prerequisites

- A Databricks workspace with permission to list serving endpoints, model provider services, apps, and (if used) agent services
- Ability to call AI Gateway (`/ai-gateway/...`) for the invoke cells
- Cluster or serverless that can run Python with `requests` and `pandas` (standard on Databricks)

## Quick start

1. Import the `.ipynb` (or the `.py` source) into the workspace.
2. Attach a cluster (or serverless) and run **Setup and Imports**. Confirm the workspace URL prints and the token is present.
3. Run **Discover system.ai Foundation Models**. Use a row’s `endpoint_name` / `model_name` later.
4. Run **Discover Model Provider Services** if you have UC model services.
5. Optionally change `endpoint_name` in **Get Model Metadata** (default: `databricks-meta-llama-3-3-70b-instruct`).
6. Run the two **AI Gateway** invoke cells. Change `model` to `system.ai.<model-name>` or `<catalog>.<schema>.<service-name>`.
7. Run the AgentBricks discovery cells for apps and UC agent services.

## Invoke via the AI Gateway

One base URL; the **model** is in the JSON body.

| API | Path |
|-----|------|
| Responses | `POST https://<workspace-url>/ai-gateway/mlflow/v1/responses` |
| Chat completions (OpenAI-compatible) | `POST https://<workspace-url>/ai-gateway/mlflow/v1/chat/completions` |

**Model names in the body**

- Foundation models: `"model": "system.ai.<model-name>"`
- Model provider services: `"model": "<catalog>.<schema>.<service-name>"`

**Auth:** `Authorization: Bearer <token>` (notebook uses the session token).

### Provider-native managed paths (model provider services)

You can also call provider-shaped paths and set `Databricks-Model-Provider-Service` (see [Query model provider services](https://docs.databricks.com/aws/en/ai-gateway/query-model-provider-services/)):

| Provider API | Managed path |
|--------------|----------------|
| OpenAI chat completions | `POST /ai-gateway/openai/v1/chat/completions` |
| OpenAI responses | `POST /ai-gateway/openai/v1/responses` |
| OpenAI embeddings | `POST /ai-gateway/openai/v1/embeddings` |
| Anthropic messages | `POST /ai-gateway/anthropic/v1/messages` |
| Gemini generate content | `POST /ai-gateway/gemini/v1beta/models/<model>:generateContent` |

## Per-object metadata

| Object | Get |
|--------|-----|
| Foundation model endpoint | `GET /api/2.0/serving-endpoints/<endpoint-name>` — config, served entities, task, state, rate limits |
| Model provider service | `GET /api/2.1/unity-catalog/model-provider-services/<catalog>.<schema>.<service>` |

## Agent discovery (beyond this notebook’s tables)

The markdown cells also note:

| Source | Typical filter |
|--------|----------------|
| Serving endpoints | `task` starts with `agent/` |
| `GET /api/2.0/agents/deployments` | Agents deployed with the `databricks.agents` SDK |
| UC agent services | `GET /api/2.1/unity-catalog/agent-services` |

Calling agents through an **agent service** on the AI Gateway is the documented, governed path. See [Agent services](https://docs.databricks.com/aws/en/ai-gateway/agent-services).

## Tech stack

Python, `requests`, `pandas`, Databricks REST (serving endpoints, UC model provider services, apps, UC agent services), AI Gateway (`/ai-gateway/mlflow/v1/...`).

## References

**Foundation models**

- [Available models in Unity Gateway](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview/)
- [Query foundation and embedding models](https://docs.databricks.com/aws/en/agents/query-llms/)

**Model provider services**

- [Custom model endpoints (model services)](https://docs.databricks.com/aws/en/ai-gateway/model-services/)
- [Create and manage model provider services](https://docs.databricks.com/aws/en/ai-gateway/create-model-provider-services/)
- [Query model provider services](https://docs.databricks.com/aws/en/ai-gateway/query-model-provider-services/)
- [List model provider services API](https://docs.databricks.com/api/ai-gateway/v1/list-model-provider-services)

**Agents**

- [Agent services](https://docs.databricks.com/aws/en/ai-gateway/agent-services)
- [Author and deploy a custom agent on Databricks Apps](https://docs.databricks.com/aws/en/agents/custom-agents/author-agent/)
- [Query an agent deployed on Databricks](https://docs.databricks.com/aws/en/agents/custom-agents/query-agent/)

**AI Gateway**

- [AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-serving-endpoints/)
- [Govern model provider services](https://docs.databricks.com/aws/en/ai-gateway/govern-model-provider-services/)
