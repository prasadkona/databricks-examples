# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Databricks AI Gateway: Discover & Invoke system.ai Models and Model Services
# MAGIC
# MAGIC A reference notebook for discovering available foundation models (`system.ai`) and model provider services, and invoking them through the Databricks AI Gateway. Foundation models are available by default on every workspace; model provider services are user-created configurations that route to external providers (OpenAI, Anthropic, Bedrock, etc.).

# COMMAND ----------

# DBTITLE 1,Setup and Imports
import requests
import pandas as pd
import json

# Retrieve workspace URL and auth token from the notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = ctx.apiUrl().getOrElse(None)
token = ctx.apiToken().getOrElse(None)

print(f"Workspace URL: {workspace_url}")
print("Token retrieved successfully" if token else "WARNING: Token not found")

# COMMAND ----------

# DBTITLE 1,About system.ai Foundation Models
# MAGIC %md
# MAGIC ## Discovering system.ai Foundation Models
# MAGIC
# MAGIC `system.ai` models are Databricks-managed foundation models available **by default** on every workspace — no setup or endpoint creation required. They include LLMs (chat, completions), embedding models, and more from providers like OpenAI, Anthropic, Google, Meta, Mistral, and Databricks.
# MAGIC
# MAGIC **Discovery API:** `GET /api/2.0/serving-endpoints` — filter results where `endpoint_type == "FOUNDATION_MODEL_API"` to isolate the built-in foundation models.

# COMMAND ----------

# DBTITLE 1,Discover system.ai Foundation Models
# Discover system.ai foundation models
# API: GET /api/2.0/serving-endpoints (filter: endpoint_type == "FOUNDATION_MODEL_API")
# These are Databricks-managed models — available by default, no setup needed.
resp = requests.get(
    f"{workspace_url}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {token}"},
    timeout=30,
)
resp.raise_for_status()
endpoints = resp.json().get("endpoints", [])

fm_records = []
for ep in endpoints:
    if ep.get("endpoint_type") == "FOUNDATION_MODEL_API":
        entities = (ep.get("config") or {}).get("served_entities", [])
        fm_name = entities[0].get("foundation_model", {}).get("name", "") if entities else ""
        fm_records.append({
            "model_name": fm_name,
            "endpoint_name": ep.get("name"),
            "task": ep.get("task"),
            "state": (ep.get("state") or {}).get("ready"),
        })

df_models = pd.DataFrame(fm_records)
print(f"{len(df_models)} system.ai foundation models available")
display(df_models)

# COMMAND ----------

# DBTITLE 1,About Model Provider Services
# MAGIC %md
# MAGIC ## Discovering Model Provider Services
# MAGIC
# MAGIC Model provider services are **user-created** Unity Catalog securables that route requests to external model providers (OpenAI, Anthropic, Amazon Bedrock, Google Gemini, etc.) through the AI Gateway. Each service lives in a catalog and schema (`<catalog>.<schema>.<service>`) and is governed by UC privileges.
# MAGIC
# MAGIC **Discovery API:** `GET /api/2.1/unity-catalog/model-provider-services`
# MAGIC **Docs:** [List Model Provider Services](https://docs.databricks.com/api/ai-gateway/v1/list-model-provider-services)

# COMMAND ----------

# DBTITLE 1,Discover Model Provider Services
# Discover model provider services
# API: GET /api/2.1/unity-catalog/model-provider-services
all_services = []
params = {}

while True:
    response = requests.get(
        f"{workspace_url}/api/2.1/unity-catalog/model-provider-services",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    page = response.json()
    all_services.extend(page.get("model_provider_services", []))
    next_token = page.get("next_page_token")
    if not next_token:
        break
    params["page_token"] = next_token

records = []
for svc in all_services:
    config = svc.get("config", {})
    targets = config.get("targets", [])
    target_models = ", ".join(t.get("model", "") for t in targets if t.get("model")) if targets else None
    raw_name = svc.get("name", "")
    three_part_name = raw_name.replace("model-provider-services/", "") if raw_name else None
    records.append({
        "service_name": three_part_name,
        "provider_type": config.get("provider_type", "").replace("EXTERNAL_MODEL_PROVIDER_TYPE_", ""),
        "target_models": target_models,
        "allow_all_targets": config.get("allow_all_targets"),
        "owner": svc.get("effective_owner"),
    })

df_services = pd.DataFrame(records)
print(f"{len(df_services)} model provider services")
display(df_services)

# COMMAND ----------

# DBTITLE 1,Querying Model Metadata
# MAGIC %md
# MAGIC ## Querying Model Metadata
# MAGIC
# MAGIC Beyond listing, you can retrieve **detailed metadata** about a specific model or service — useful for configuration inspection, monitoring, or integration.
# MAGIC
# MAGIC ### system.ai Foundation Models
# MAGIC ```
# MAGIC GET /api/2.0/serving-endpoints/<endpoint-name>
# MAGIC ```
# MAGIC Returns endpoint configuration, served entities (with provider and model details), state, task type, rate limits, and traffic configuration.
# MAGIC
# MAGIC ### Model Provider Services
# MAGIC ```
# MAGIC GET /api/2.1/unity-catalog/model-provider-services/<catalog>.<schema>.<service-name>
# MAGIC ```
# MAGIC Returns provider type, target models, allowed API types, forwarding settings, owner, and creation metadata.

# COMMAND ----------

# DBTITLE 1,Example: Get Model Metadata
# Example: Get detailed metadata for a system.ai foundation model
endpoint_name = "databricks-meta-llama-3-3-70b-instruct"  # ← change to any endpoint from the list

resp = requests.get(
    f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}",
    headers={"Authorization": f"Bearer {token}"},
    timeout=30,
)
resp.raise_for_status()
metadata = resp.json()

print(f"Endpoint:  {metadata.get('name')}")
print(f"Type:      {metadata.get('endpoint_type')}")
print(f"Task:      {metadata.get('task')}")
print(f"State:     {metadata.get('state', {}).get('ready')}")
print(f"Creator:   {metadata.get('creator')}")

# Served entities — model details
entities = metadata.get("config", {}).get("served_entities", [])
print(f"\nServed Entities ({len(entities)}):")
for e in entities:
    fm = e.get("foundation_model", {})
    print(f"  Name:         {fm.get('name', e.get('name'))}")
    print(f"  Display Name: {fm.get('display_name', 'N/A')}")
    print(f"  Provider:     {fm.get('provider', 'N/A')}")
    desc = fm.get('description', 'N/A') or 'N/A'
    print(f"  Description:  {desc[:120]}{'...' if len(desc) > 120 else ''}")

# Rate limits
rate_limits = metadata.get("rate_limits", [])
if rate_limits:
    print(f"\nRate Limits:")
    for rl in rate_limits:
        print(f"  {rl.get('calls')} {rl.get('key')} per {rl.get('renewal_period')}")

# COMMAND ----------

# DBTITLE 1,How to Invoke Models via the AI Gateway
# MAGIC %md
# MAGIC ## How to Invoke Models via the AI Gateway
# MAGIC
# MAGIC Both **system.ai foundation models** and **model provider services** are invoked through the AI Gateway using the same patterns. The AI Gateway provides a single base URL for all models — no per-model endpoint URLs needed. The model is specified in the **request body** using its fully qualified name.
# MAGIC
# MAGIC ### Responses API
# MAGIC ```
# MAGIC POST https://<workspace-url>/ai-gateway/mlflow/v1/responses
# MAGIC ```
# MAGIC Best for multi-turn conversations with structured input/output types.
# MAGIC
# MAGIC ### Chat Completions API (OpenAI-compatible)
# MAGIC ```
# MAGIC POST https://<workspace-url>/ai-gateway/mlflow/v1/chat/completions
# MAGIC ```
# MAGIC Best for OpenAI-compatible clients and simple chat interactions.
# MAGIC
# MAGIC ### Model Naming in the Request Body
# MAGIC * **Foundation models:** `"model": "system.ai.<model-name>"`
# MAGIC * **Model provider services:** `"model": "<catalog>.<schema>.<service-name>"`
# MAGIC
# MAGIC ### Provider-Specific Managed Paths (Model Provider Services)
# MAGIC For model provider services, you can also use the `Databricks-Model-Provider-Service` header with provider-native managed paths:
# MAGIC
# MAGIC | Provider API | Managed Path |
# MAGIC | --- | --- |
# MAGIC | OpenAI (chat completions) | `POST /ai-gateway/openai/v1/chat/completions` |
# MAGIC | OpenAI (responses) | `POST /ai-gateway/openai/v1/responses` |
# MAGIC | OpenAI (embeddings) | `POST /ai-gateway/openai/v1/embeddings` |
# MAGIC | Anthropic (messages) | `POST /ai-gateway/anthropic/v1/messages` |
# MAGIC | Gemini (generate content) | `POST /ai-gateway/gemini/v1beta/models/<model>:generateContent` |
# MAGIC
# MAGIC **Authentication:** All paths require `Authorization: Bearer <DATABRICKS_TOKEN>`.

# COMMAND ----------

# DBTITLE 1,Example: AI Gateway — Responses API
# Example: Invoke a system.ai model via the AI Gateway Responses API
model = "system.ai.databricks-meta-llama-3-3-70b-instruct"  # ← change to any model from the discovery tables

resp = requests.post(
    f"{workspace_url}/ai-gateway/mlflow/v1/responses",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "max_output_tokens": 256,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is Databricks Unity Catalog in one sentence?"}],
            }
        ],
    },
    timeout=60,
)
resp.raise_for_status()
result = resp.json()

# Extract the assistant's text
try:
    output_text = result["output"][0]["content"][0]["text"]
except (KeyError, IndexError, TypeError):
    output_text = json.dumps(result, indent=2)

print(f"Model: {model}")
print(f"Usage: {result.get('usage', {})}")
print(f"\nResponse:\n{output_text}")

# COMMAND ----------

# DBTITLE 1,Example: AI Gateway — Chat Completions API
# Example: Invoke a system.ai model via the AI Gateway Chat Completions API
model = "system.ai.databricks-meta-llama-3-3-70b-instruct"  # ← change to any model from the discovery tables

resp = requests.post(
    f"{workspace_url}/ai-gateway/mlflow/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What is Databricks Unity Catalog in one sentence?"}
        ],
    },
    timeout=60,
)
resp.raise_for_status()
result = resp.json()

# Extract the assistant's text (OpenAI-compatible response format)
try:
    output_text = result["choices"][0]["message"]["content"]
except (KeyError, IndexError, TypeError):
    output_text = json.dumps(result, indent=2)

print(f"Model: {model}")
print(f"Usage: {result.get('usage', {})}")
print(f"\nResponse:\n{output_text}")

# COMMAND ----------

# DBTITLE 1,Discovering AgentBricks Custom Agents
# MAGIC %md
# MAGIC ## Discovering AgentBricks Custom Agents
# MAGIC
# MAGIC Custom agents appear in two forms on Databricks, both visible in the [Agents UI](/ml/agents):
# MAGIC
# MAGIC | Agent Type | Deployed As | How to Discover |
# MAGIC | --- | --- | --- |
# MAGIC | **Custom Agent** | Databricks App | `GET /api/2.0/apps` — filter for apps with an `experiment` resource (used for MLflow tracing of agent interactions) |
# MAGIC | **Custom Agent Endpoint** | Serving Endpoint | `GET /api/2.0/serving-endpoints` — filter where `task` starts with `agent/` |
# MAGIC
# MAGIC Additionally, `GET /api/2.0/agents/deployments` lists agents explicitly deployed via the `databricks.agents` SDK, and `GET /api/2.1/unity-catalog/agent-services` lists agents registered as UC agent services.
# MAGIC
# MAGIC **Invocation:** The **agent service** in the AI Gateway is the preferred way to call agents. Like model services, agent services provide a single governed entry point with the agent specified by its fully qualified UC name in the request body. See [Agent Services](https://docs.databricks.com/aws/en/ai-gateway/agent-services).

# COMMAND ----------

# DBTITLE 1,List App-Backed Custom Agents
# Discover app-backed custom agents
# API: GET /api/2.0/apps — filter for apps with an 'experiment' resource
# (MLflow tracing experiment is the distinguishing marker for agent apps)
all_apps = []
params = {}
while True:
    r = requests.get(
        f"{workspace_url}/api/2.0/apps",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    page = r.json()
    all_apps.extend(page.get("apps", []))
    if not page.get("next_page_token"):
        break
    params["page_token"] = page["next_page_token"]

agent_records = []
for app in all_apps:
    resources = app.get("resources") or []
    res_types = {k for r in resources for k in r if k != "name"}
    if "experiment" in res_types:
        # Extract MLflow experiment ID(s) from the experiment resources
        exp_ids = [r["experiment"]["experiment_id"] for r in resources if "experiment" in r and "experiment_id" in r["experiment"]]
        agent_records.append({
            "name": app.get("name"),
            "url": app.get("url"),
            "state": (app.get("compute_status") or {}).get("state"),
            "experiment_id": ", ".join(exp_ids) if exp_ids else None,
            "creator": app.get("creator"),
        })

df_agents = pd.DataFrame(agent_records)
print(f"{len(df_agents)} app-backed custom agents (out of {len(all_apps)} total apps)")
display(df_agents)

# COMMAND ----------

# DBTITLE 1,List UC Agent Services
# Discover UC agent services (the preferred governed path for calling agents)
# API: GET /api/2.1/unity-catalog/agent-services
# Agent services are registered in Unity Catalog and invoked through the AI Gateway,
# similar to model provider services. This is the recommended way to call agents.
all_agent_services = []
params = {}

while True:
    r = requests.get(
        f"{workspace_url}/api/2.1/unity-catalog/agent-services",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    page = r.json()
    all_agent_services.extend(page.get("agent_services", []))
    next_token = page.get("next_page_token")
    if not next_token:
        break
    params["page_token"] = next_token

if all_agent_services:
    as_records = []
    for svc in all_agent_services:
        as_records.append({
            "name": svc.get("name"),
            "owner": svc.get("effective_owner"),
            "config": json.dumps(svc.get("config", {}), indent=2)[:200],
        })
    df_agent_svc = pd.DataFrame(as_records)
    print(f"{len(df_agent_svc)} UC agent services")
    display(df_agent_svc)
else:
    print(f"0 UC agent services registered (via /api/2.1/unity-catalog/agent-services)")
    print("\nNo agent services have been registered in Unity Catalog on this workspace yet.")
    print("To register an agent as a UC agent service, see:")
    print("  https://docs.databricks.com/aws/en/ai-gateway/agent-services")

# COMMAND ----------

# DBTITLE 1,References
# MAGIC %md
# MAGIC ## References
# MAGIC
# MAGIC ### system.ai Foundation Models
# MAGIC * [Available Models in Unity Gateway](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview/) — full list of Databricks-served foundation models and how to use them
# MAGIC * [Query Foundation and Embedding Models](https://docs.databricks.com/aws/en/agents/query-llms/) — querying system.ai models via the OpenAI-compatible SDK, native provider APIs, or `ai_query`
# MAGIC
# MAGIC ### Model Provider Services
# MAGIC * [Custom Model Endpoints (Model Services)](https://docs.databricks.com/aws/en/ai-gateway/model-services/) — creating and managing model services in Unity Catalog, including system-provided vs user-created
# MAGIC * [Create and Manage Model Provider Services](https://docs.databricks.com/aws/en/ai-gateway/create-model-provider-services/) — setting up BYO-key external provider routing (OpenAI, Anthropic, Bedrock, Gemini, etc.)
# MAGIC * [Query Model Provider Services](https://docs.databricks.com/aws/en/ai-gateway/query-model-provider-services/) — invoking model provider services via managed paths and the `Databricks-Model-Provider-Service` header
# MAGIC * [List Model Provider Services API](https://docs.databricks.com/api/ai-gateway/v1/list-model-provider-services) — REST API reference for `GET /api/2.1/unity-catalog/model-provider-services`
# MAGIC
# MAGIC ### AgentBricks Custom Agents
# MAGIC * [Agent Services](https://docs.databricks.com/aws/en/ai-gateway/agent-services) — deploying and governing agents through the AI Gateway
# MAGIC * [Author and Deploy a Custom Agent on Databricks Apps](https://docs.databricks.com/aws/en/agents/custom-agents/author-agent/) — building, deploying, and querying agents as Databricks Apps
# MAGIC * [Query an Agent Deployed on Databricks](https://docs.databricks.com/aws/en/agents/custom-agents/query-agent/) — invoking agents via the Responses API with OAuth authentication
# MAGIC
# MAGIC ### AI Gateway (General)
# MAGIC * [AI Gateway Overview](https://docs.databricks.com/aws/en/ai-gateway/overview-serving-endpoints/) — centralized governance, monitoring, and production readiness for AI traffic
# MAGIC * [Govern Model Provider Services](https://docs.databricks.com/aws/en/ai-gateway/govern-model-provider-services/) — discovery, permissions, and access control via Unity Catalog