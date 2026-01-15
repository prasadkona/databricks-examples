# Setup Guide

## Prerequisites

### 1. Install Claude Code Runtime

```bash
# macOS/Linux/WSL
curl -fsSL https://claude.ai/install.sh | bash

# Or via Homebrew
brew install --cask claude-code

# Verify
claude-code --version
```

### 2. Python 3.9+

```bash
python3 --version
```

### 3. Databricks Access

- Workspace with Claude models enabled
- Personal Access Token (PAT)

## Installation

```bash
# Navigate to project
cd databricks_claude_agent_sdk_example

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Get Databricks Credentials

1. **Personal Access Token**:
   - Go to Databricks → User Settings → Access Tokens
   - Generate New Token
   - Copy the token

2. **Claude Model Access**:
   - Ensure your Databricks workspace has Claude models enabled
   - Models are accessed via the `/serving-endpoints/anthropic` endpoint
   - Specific model is selected via `ANTHROPIC_MODEL` environment variable
   - Available models: `databricks-claude-haiku-4-5`, `databricks-claude-sonnet-4-5`

### Create .env File

```bash
cat > .env << EOF
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef

# Claude Agent SDK Configuration (required)
ANTHROPIC_AUTH_TOKEN=dapi1234567890abcdef
ANTHROPIC_BASE_URL=https://your-workspace.cloud.databricks.com/serving-endpoints/anthropic
ANTHROPIC_CUSTOM_HEADERS=x-databricks-disable-beta-headers: true
ANTHROPIC_MODEL=databricks-claude-haiku-4-5

# MLflow Configuration (optional - for notebooks 2, 3, 4, 6)
MLFLOW_EXPERIMENT_NAME=/Shared/claude-agent-demo

# MCP Configuration (optional - for notebooks 5, 6)
MCP_FUNCTIONS_CATALOG=system
MCP_FUNCTIONS_SCHEMA=ai
MCP_GENIE_SPACE_ID=your-genie-space-id
EOF
```

**Important Configuration Notes:**

**Required Variables (for all notebooks):**
- `ANTHROPIC_AUTH_TOKEN` → Use your Databricks PAT (same as DATABRICKS_TOKEN)
- `ANTHROPIC_BASE_URL` → Must point to `/serving-endpoints/anthropic` (not a specific model endpoint)
- `ANTHROPIC_CUSTOM_HEADERS` → Required for Databricks compatibility: `x-databricks-disable-beta-headers: true`
- `ANTHROPIC_MODEL` → The Claude model to use (e.g., `databricks-claude-haiku-4-5` or `databricks-claude-sonnet-4-5`)

**Optional Variables (for MLflow notebooks 2, 3, 4, 6):**
- `MLFLOW_EXPERIMENT_NAME` → MLflow experiment path (use `/Shared/` for better permissions)

**Optional Variables (for MCP notebooks 5, 6):**
- `MCP_FUNCTIONS_CATALOG` → Unity Catalog catalog name (default: `system`)
- `MCP_FUNCTIONS_SCHEMA` → Unity Catalog schema name (default: `ai`)
- `MCP_GENIE_SPACE_ID` → Genie space ID for natural language queries (optional)

**Replace:**
- `your-workspace.cloud.databricks.com` → Your Databricks workspace URL
- `dapi1234567890abcdef` → Your Databricks Personal Access Token

## Test Setup

### Local Testing (Notebook 1)

```bash
# Test basic agent - runs locally
python notebooks/01_basic_agent.py
```

This should complete successfully with exit code 0 and display a comprehensive project analysis.

### MCP Testing (Notebook 5)

Test Databricks MCP integration locally:

```bash
# Test MCP integration - connects to Databricks MCP servers from local machine
python notebooks/05_localagent_databricks_mcp.py

# This will:
# - Connect to Unity Catalog Functions (system.ai)
# - Execute SQL queries via DBSQL MCP
# - Query Genie (if MCP_GENIE_SPACE_ID is set)
```

### Databricks Testing (Notebooks 2, 3, 4, 6)

The MLflow notebooks are designed to run in a Databricks workspace:

```bash
# These will fail locally with MLflow connection errors (expected behavior)
python notebooks/02_databricks_mlflow.py                    # Manual MLflow tracking
python notebooks/03_databricks_mlflow_autologging.py        # Autologging
python notebooks/04_databricks_mlflow_genai_evaluation.py   # Evaluation
python notebooks/06_databricks_mcp_mlflow.py                # MCP + MLflow

# To run these successfully:
# 1. Upload to your Databricks workspace
# 2. Run as Python scripts or Databricks notebooks
# 3. MLflow will work seamlessly in the Databricks environment
```

## MCP Server Configuration (Notebooks 5, 6)

Databricks MCP (Model Context Protocol) allows Claude agents to access enterprise data sources:

### Available MCP Servers

1. **Unity Catalog Functions** - Always available
   - Catalog: `system` (configurable via `MCP_FUNCTIONS_CATALOG`)
   - Schema: `ai` (configurable via `MCP_FUNCTIONS_SCHEMA`)
   - Provides: `python_exec`, vector search, and other AI functions
   - URL pattern: `{host}/api/2.0/mcp/functions/{catalog}/{schema}`

2. **DBSQL** - Always available
   - Direct SQL query execution
   - No SQL warehouse ID required
   - Works with any accessible tables
   - URL pattern: `{host}/api/2.0/mcp/sql`

3. **Genie** - Optional
   - Natural language to SQL translation
   - Requires a Genie space ID
   - Set `MCP_GENIE_SPACE_ID` in `.env`
   - URL pattern: `{host}/api/2.0/mcp/genie/{space_id}`

### How MCP Configuration Works

In your Python code, MCP servers are configured and passed to `ClaudeAgentOptions`:

```python
import os
from claude_agent_sdk import query, ClaudeAgentOptions

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

# Pass to agent options with wildcard tool permissions
options = ClaudeAgentOptions(
    mcp_servers=mcp_servers_config,
    allowed_tools=["Bash", "Read", "mcp__uc_functions__*", "mcp__dbsql__*"],
    permission_mode="acceptEdits"
)

async for message in query(prompt="Your query here", options=options):
    if hasattr(message, "result"):
        print(message.result)
```

**Key Configuration Points:**
- `"type": "http"` is required for all MCP servers
- `"transport": "sse"` enables Server-Sent Events streaming
- Wildcard tools like `mcp__uc_functions__*` allow all tools from that MCP server
- `permission_mode="acceptEdits"` auto-approves MCP tool usage

## Troubleshooting

### "Claude Code not found"
The Claude Code Runtime is required. Install it:
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### "Invalid API key" or "401 Credential not sent"
- Check `.env` file has correct `DATABRICKS_TOKEN`
- Verify `ANTHROPIC_AUTH_TOKEN` matches your PAT token
- Ensure token has not expired

### "404 Path must be of form /serving-endpoints/..."
- Check `ANTHROPIC_BASE_URL` points to `/serving-endpoints/anthropic`
- Do NOT include the specific model name in the URL
- The model is specified via `ANTHROPIC_MODEL` environment variable

### "403 Invalid access token" (MLflow notebooks)
This is **expected** when running locally. MLflow notebooks require:
- Databricks workspace environment
- Proper MLflow permissions
- Solution: Upload to Databricks and run there

### AsyncIO errors or "RuntimeError: Attempted to exit cancel scope"
- This is a known issue with multiple sequential queries in the same script
- Notebooks are designed to run single comprehensive queries
- For multiple queries, run separate scripts or use separate async contexts

### "Module not found: claude_agent_sdk"
```bash
pip install claude-agent-sdk>=0.1.19
```

### "Invalid MCP configuration: mcpServers.{name}: Does not adhere to MCP server configuration schema"
- Ensure `"type": "http"` is included in your MCP server configuration
- Verify `"transport": "sse"` is set correctly
- Check that the MCP server URL is valid and accessible
- Example correct configuration:
```python
{
    "type": "http",
    "url": "https://workspace.cloud.databricks.com/api/2.0/mcp/sql",
    "transport": "sse",
    "headers": {"Authorization": "Bearer YOUR_TOKEN"}
}
```

### MCP tools not being called / "I don't have access to MCP tools"
- Verify `mcp_servers` is passed to `ClaudeAgentOptions`
- Check `allowed_tools` includes MCP tool wildcards: `mcp__uc_functions__*`, `mcp__dbsql__*`
- Ensure `permission_mode="acceptEdits"` is set
- Verify your Databricks PAT token has permissions to access Unity Catalog and SQL

### Genie MCP not working
- Verify `MCP_GENIE_SPACE_ID` is set in `.env`
- Check that you have access to the Genie space
- Ensure the Genie space ID is correct
- Genie is optional - UC Functions and DBSQL should work without it

## Using Databricks Secrets (Production)

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
token = w.secrets.get_secret(
    scope="claude-secrets",
    key="databricks-token"
).value

import os
os.environ["DATABRICKS_TOKEN"] = token
```

## Notebooks Overview

| Notebook | Runs On | What It Does |
|----------|---------|--------------|
| `01_basic_agent.py` | ✅ Local | Basic Claude Agent SDK usage with file analysis |
| `02_databricks_mlflow.py` | 📊 Databricks | Manual MLflow logging with custom parameters |
| `03_databricks_mlflow_autologging.py` | 🚀 Databricks | Automatic tracing with mlflow.anthropic.autolog() |
| `04_databricks_mlflow_genai_evaluation.py` | 🎯 Databricks | Quality evaluation with GenAI judges |
| `05_localagent_databricks_mcp.py` | 🔧 Local | MCP integration: Unity Catalog, DBSQL, Genie |
| `06_databricks_mcp_mlflow.py` | 🏢 Databricks | MCP + MLflow for enterprise observability |

## Next Steps

### Local Development
1. ✅ Test basic agent: `python notebooks/01_basic_agent.py`
2. ✅ Test MCP integration: `python notebooks/05_localagent_databricks_mcp.py`
3. Verify Claude Agent SDK is working with Databricks
4. Customize prompts and tools for your use case

### Databricks Deployment
1. Upload notebooks 2, 3, 4, 6 to your Databricks workspace
2. Run them in Databricks for full MLflow integration
3. View results in MLflow experiments dashboard
4. Test MCP + MLflow integration with notebook 6
5. Set up scheduled jobs for evaluation pipelines

### MCP Integration
1. Verify Unity Catalog Functions access (system.ai schema)
2. Test DBSQL queries on sample data (samples.nyctaxi.trips)
3. Configure Genie space ID for natural language queries (optional)
4. Monitor MCP tool usage in MLflow (notebook 6)

### Production Deployment
1. Use Databricks Secrets for credential management
2. Create Databricks Jobs for automated agent runs
3. Monitor agent performance with MLflow metrics
4. Set up alerts for quality and performance thresholds
5. Track MCP server usage and enterprise data access patterns

---

**Ready to start?** 
- Local: `python notebooks/01_basic_agent.py` ✅
- MCP: `python notebooks/05_localagent_databricks_mcp.py` ✅
