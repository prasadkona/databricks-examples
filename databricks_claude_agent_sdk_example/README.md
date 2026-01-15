# Claude Agent SDK with Databricks

Build autonomous AI agents using the Claude Agent SDK integrated with Databricks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
cat > .env << EOF
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-pat-token
ANTHROPIC_AUTH_TOKEN=your-databricks-pat-token
ANTHROPIC_BASE_URL=https://your-workspace.cloud.databricks.com/serving-endpoints/anthropic
ANTHROPIC_CUSTOM_HEADERS=x-databricks-disable-beta-headers: true
ANTHROPIC_MODEL=databricks-claude-haiku-4-5
MLFLOW_EXPERIMENT_NAME=/Shared/claude-agent-demo
EOF

# Run examples
python notebooks/01_basic_agent.py                          # ✅ Local execution
python notebooks/02_databricks_mlflow.py                    # 📊 Best on Databricks
python notebooks/03_databricks_mlflow_autologging.py        # 🚀 Best on Databricks
python notebooks/04_databricks_mlflow_genai_evaluation.py   # 🎯 Best on Databricks
python notebooks/05_localagent_databricks_mcp.py            # 🔧 Local with Databricks MCP
python notebooks/06_databricks_mcp_mlflow.py                # 🏢 Databricks MCP + MLflow
```

## What's Included

```
databricks_claude_agent_sdk_example/
├── README.md              # This file
├── SETUP.md              # Detailed setup guide
├── requirements.txt      # Dependencies
└── notebooks/            # Python examples
    ├── 01_basic_agent.py                          # Basic usage
    ├── 02_databricks_mlflow.py                    # Manual MLflow tracking
    ├── 03_databricks_mlflow_autologging.py        # MLflow autologging
    ├── 04_databricks_mlflow_genai_evaluation.py   # GenAI evaluation
    ├── 05_localagent_databricks_mcp.py            # Local agent with Databricks MCP
    └── 06_databricks_mcp_mlflow.py                # Databricks MCP + MLflow
```

## Examples Overview

| Notebook | Runs On | MLflow | Features |
|----------|---------|--------|----------|
| `01_basic_agent.py` | ✅ Local | ❌ | Basic agent, file analysis |
| `02_databricks_mlflow.py` | 📊 Databricks | ✅ | Manual logging, custom metrics |
| `03_databricks_mlflow_autologging.py` | 🚀 Databricks | ✅ | Automatic tracing |
| `04_databricks_mlflow_genai_evaluation.py` | 🎯 Databricks | ✅ | Quality evaluation, judges |
| `05_localagent_databricks_mcp.py` | 🔧 Local | ❌ | Databricks MCP servers, UC Functions |
| `06_databricks_mcp_mlflow.py` | 🏢 Databricks | ✅ | MCP + MLflow + autologging |

### 1. Basic Agent (`01_basic_agent.py`) - ✅ Runs Locally

Simple examples using Claude Agent SDK with built-in tools:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    async for message in query(
        prompt="List all Python files in this directory",
        options=ClaudeAgentOptions(allowed_tools=["Bash", "Glob"])
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

### 2. With MLflow Tracking (`02_databricks_mlflow.py`) - 📊 Best on Databricks

Agent with MLflow experiment tracking (recommended to run in Databricks workspace):

```python
import mlflow
from claude_agent_sdk import query, ClaudeAgentOptions

mlflow.set_experiment("/Users/your-email/claude-agents")

with mlflow.start_run():
    async for message in query(
        prompt="Analyze this directory",
        options=ClaudeAgentOptions(allowed_tools=["Bash", "Glob", "Read"])
    ):
        if hasattr(message, "result"):
            mlflow.log_text(message.result, "result.txt")
```

### 3. MLflow Autologging (`03_databricks_mlflow_autologging.py`) - 🚀 Best on Databricks

Automatic tracing of all Claude Agent SDK interactions using MLflow autologging:

```python
import asyncio
import mlflow.anthropic
from claude_agent_sdk import ClaudeSDKClient

# Enable autologging - traces all interactions automatically!
mlflow.anthropic.autolog()

mlflow.set_experiment("my_claude_app")

async def main():
    with mlflow.start_run():
        async with ClaudeSDKClient() as client:
            await client.query("What is the capital of France?")
            
            async for message in client.receive_response():
                print(message)

asyncio.run(main())
```

**Autologging captures:**
- All Claude API calls
- Request/response payloads
- Token usage and costs
- Latency and performance metrics
- Tool usage statistics

### 4. GenAI Evaluation (`04_databricks_mlflow_genai_evaluation.py`) - 🎯 Best on Databricks

Evaluate agent quality using MLflow's GenAI evaluation framework with custom judges:

```python
import asyncio
import pandas as pd
import mlflow.anthropic
from mlflow.genai import evaluate
from mlflow.genai.judges import make_judge
from claude_agent_sdk import ClaudeSDKClient

# Enable autologging
mlflow.anthropic.autolog()

# Create agent function
async def run_agent(query: str) -> str:
    async with ClaudeSDKClient() as client:
        await client.query(query)
        response = ""
        async for message in client.receive_response():
            response += str(message)
        return response

# Evaluation wrapper
def predict_fn(inputs: dict) -> dict:
    return {"outputs": asyncio.run(run_agent(inputs["query"]))}

# Create custom judge
relevance = make_judge(
    name="relevance",
    instructions="Evaluate if the response is relevant to the question.",
    model="openai:/gpt-4o",
)

# Create evaluation dataset
eval_data = pd.DataFrame([
    {"inputs": {"query": "What is machine learning?"}},
    {"inputs": {"query": "Explain neural networks"}},
])

# Run evaluation
mlflow.set_experiment("claude_evaluation")
results = evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[relevance]
)
```

**Evaluation features:**
- Custom judges for quality assessment
- Batch evaluation of multiple queries
- Automatic metrics calculation
- Performance and accuracy tracking
- Side-by-side response comparison

### 5. Local Agent with Databricks MCP (`05_localagent_databricks_mcp.py`) - 🔧 Runs Locally

Connect to Databricks MCP (Model Context Protocol) servers from your local machine to access enterprise data:

```python
import asyncio
import os
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    # Build MCP server configuration
    databricks_host = os.getenv("DATABRICKS_HOST").rstrip("/")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    
    mcp_servers_config = {
        "uc_functions": {
            "type": "http",
            "url": f"{databricks_host}/api/2.0/mcp/functions/system/ai",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {databricks_token}"}
        },
        "dbsql": {
            "type": "http",
            "url": f"{databricks_host}/api/2.0/mcp/sql",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {databricks_token}"}
        }
    }
    
    # Query with MCP servers enabled
    async for message in query(
        prompt="Show me the top NYC taxi trips by fare amount",
        options=ClaudeAgentOptions(
            mcp_servers=mcp_servers_config,
            allowed_tools=["Bash", "Read", "mcp__uc_functions__*", "mcp__dbsql__*"],
            permission_mode="acceptEdits"
        )
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

**MCP capabilities:**
- Unity Catalog Functions (`system.ai` schema)
- Genie natural language queries
- Direct SQL execution via DBSQL
- Enterprise data access with full security

**Configuration** (in `.env`):
```bash
# Required for MCP
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-pat-token

# Optional - for Genie natural language queries
MCP_GENIE_SPACE_ID=your-genie-space-id
```

**Note:** DBSQL and Unity Catalog Functions work automatically with your Databricks PAT token. No SQL warehouse ID is required.

### 6. Databricks MCP + MLflow (`06_databricks_mcp_mlflow.py`) - 🏢 Best on Databricks

Run on Databricks with MCP enterprise data access and MLflow tracking for complete observability:

```python
import asyncio
import os
import mlflow.anthropic
from claude_agent_sdk import query, ClaudeAgentOptions

# Enable autologging
mlflow.anthropic.autolog()
mlflow.set_experiment("/Shared/claude-agent-mcp-demo")

async def main():
    # Build MCP server configuration
    databricks_host = os.getenv("DATABRICKS_HOST").rstrip("/")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    
    mcp_servers_config = {
        "uc_functions": {
            "type": "http",
            "url": f"{databricks_host}/api/2.0/mcp/functions/system/ai",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {databricks_token}"}
        },
        "dbsql": {
            "type": "http",
            "url": f"{databricks_host}/api/2.0/mcp/sql",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {databricks_token}"}
        }
    }
    
    with mlflow.start_run():
        # Log MCP configuration
        mlflow.log_param("mcp_uc_functions", "system.ai")
        mlflow.log_param("mcp_dbsql_enabled", True)
        
        # Query with MCP + MLflow tracking
        async for message in query(
            prompt="Show me high-value NYC taxi trips using DBSQL",
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers_config,
                allowed_tools=["Bash", "Read", "mcp__uc_functions__*", "mcp__dbsql__*"],
                permission_mode="acceptEdits"
            )
        ):
            if hasattr(message, "result"):
                mlflow.log_text(message.result, "query_result.txt")
                print(message.result)

asyncio.run(main())
```

**Enterprise features:**
- Unity Catalog Functions integration
- Genie natural language queries
- DBSQL direct query execution
- MLflow autologging of all interactions
- MCP server configuration tracking
- Complete observability for enterprise agents

**What gets logged:**
- MCP server configurations and usage
- Enterprise data access patterns
- Query results and artifacts
- Token usage and costs (via autologging)
- Performance metrics and traces

## Built-in Tools

The Claude Agent SDK includes:
- **Read** - Read files
- **Write** - Create files
- **Edit** - Edit files
- **Bash** - Run commands
- **Glob** - Find files (`**/*.py`)
- **Grep** - Search contents
- **WebSearch** - Search web
- **WebFetch** - Fetch pages

## Configuration

Create a `.env` file with the following required variables:

```bash
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-pat-token

# Claude Agent SDK Configuration (Required)
ANTHROPIC_AUTH_TOKEN=your-databricks-pat-token
ANTHROPIC_BASE_URL=https://your-workspace.cloud.databricks.com/serving-endpoints/anthropic
ANTHROPIC_CUSTOM_HEADERS=x-databricks-disable-beta-headers: true
ANTHROPIC_MODEL=databricks-claude-haiku-4-5

# MLflow Configuration (Optional - for notebooks 2, 3, 4, 6)
MLFLOW_EXPERIMENT_NAME=/Shared/claude-agent-demo

# MCP Configuration (Optional - for notebooks 5, 6)
MCP_FUNCTIONS_CATALOG=system          # Unity Catalog catalog name
MCP_FUNCTIONS_SCHEMA=ai               # Unity Catalog schema name
MCP_GENIE_SPACE_ID=your-genie-space-id  # Optional: Genie space for natural language queries
```

**Important Notes:**
- `ANTHROPIC_AUTH_TOKEN` should be your Databricks PAT token
- `ANTHROPIC_BASE_URL` must point to `/serving-endpoints/anthropic` (not a specific model)
- `ANTHROPIC_CUSTOM_HEADERS` is required for Databricks compatibility: `x-databricks-disable-beta-headers: true`
- `ANTHROPIC_MODEL` specifies which Claude model to use (e.g., `databricks-claude-haiku-4-5` or `databricks-claude-sonnet-4-5`)
- MCP servers (Unity Catalog, DBSQL) work automatically with your Databricks PAT - no SQL warehouse ID required
- Genie is optional and only needed if you have a Genie space configured

## Prerequisites

1. **Claude Code Runtime**:
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

2. **Python 3.9+**

3. **Databricks workspace** with Claude models

## Resources

- [Claude Agent SDK Docs](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Agent SDK Examples](https://github.com/anthropics/claude-agent-sdk-demos)
- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/)

---

**Built with Claude Agent SDK**
