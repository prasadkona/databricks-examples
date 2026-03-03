# Databricks Agent Bricks: Knowledge Assistant Examples

Create and manage Databricks Agent Bricks Knowledge Assistants programmatically using REST API and Python SDK.

## Databricks Agent Bricks

**Agent Bricks** are pre-built, configurable AI components in Databricks that accelerate the development of generative AI applications. They provide production-ready building blocks that can be deployed as serverless Model Serving endpoints without writing custom agent code.

Agent Bricks include:
- **Knowledge Assistant** - Document Q&A with RAG over Unity Catalog Volumes
- **Genie Space** - Natural language SQL exploration over structured data
- **Multi-Agent Supervisor** - Orchestration of multiple agents and tools

## Knowledge Assistant

A **Knowledge Assistant** is an Agent Brick that enables document-based question answering using Retrieval-Augmented Generation (RAG). It automatically indexes documents from Unity Catalog Volumes and provides accurate, citation-backed responses.

**Key capabilities:**
- Index documents from Unity Catalog Volumes (PDF, TXT, MD, DOC/DOCX, PPT/PPTX)
- Answer questions with citations from source documents
- Maintain conversation context across multiple turns
- Improve response quality through seeded example questions

**Documentation:**
- [Knowledge Assistant Overview](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/knowledge-assistant)
- [Create a Knowledge Assistant](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/create-knowledge-assistant)
- [Knowledge Assistant REST API](https://docs.databricks.com/api/workspace/knowledgeassistants)

---

## Scripts

| Script | Description | Execution |
|--------|-------------|-----------|
| `src/00_setup_sec_documents.py` | Download SEC annual reports and upload to UC volume | Run locally |
| `src/01_ka_using_rest_api.py` | Create KA using direct REST API calls | Run locally |
| `src/02_ka_using_agent_bricks_manager.py` | Create KA using AgentBricksManager wrapper | Run locally |
| `src/03_test_ka_conversation.py` | Test KA with multi-turn conversation | Run locally |
| `src/04_sync_ka_sources.py` | Force re-sync of KA knowledge sources | Run locally |
| `src/05_add_ka_examples.py` | Add sample questions with guidelines | Run locally |

## Prerequisites

1. **Unity Catalog enabled workspace**
2. **Serverless compute enabled**
3. **Access to Foundation Models** in `system.ai` schema
4. **Serverless budget policy** with nonzero budget
5. **Documents in a Unity Catalog Volume**
6. **Service Principal** with OAuth M2M credentials (recommended) or PAT

---

## Configuration

### Step 1: Create the _local Folder

Create a `_local` folder in the parent directory to store credentials and local data. This folder is gitignored and shared across projects.

```bash
# From the parent directory (e.g., databricks-examples)
mkdir -p _local
```

### Step 2: Create Environment File

Copy the template to `_local` with your workspace name:

```bash
# From agent_bricks_ka_example directory
cp .env.template ../_local/my-workspace.env

# Example for e2-demo-field-eng workspace:
cp .env.template ../_local/e2-demo-field-eng.env
```

### Step 3: Configure Environment Variables

Edit `_local/{workspace-name}.env` with your settings:

```bash
# =============================================================================
# Workspace Connection
# =============================================================================
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# =============================================================================
# Authentication - OAuth M2M (recommended)
# =============================================================================
DATABRICKS_CLIENT_ID=your-client-id
DATABRICKS_CLIENT_SECRET=your-client-secret

# =============================================================================
# Unity Catalog
# =============================================================================
UC_CATALOG=your_catalog
UC_SCHEMA=your_schema
UC_VOLUME=your_volume
UC_VOLUME_PATH=/Volumes/your_catalog/your_schema/your_volume

# =============================================================================
# Compute Resources
# =============================================================================
SQL_WAREHOUSE_ID=your-warehouse-id

# =============================================================================
# Knowledge Assistants (added after creation)
# =============================================================================
KA_NAME_01=SEC_Financial_Analyst
KA_TILE_ID_01=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Getting Service Principal Credentials

1. Go to your **Databricks Account Console**
2. Navigate to **User Management > Service Principals**
3. Create or select a Service Principal
4. Click **Generate secret** to create OAuth credentials
5. Copy the Client ID and Client Secret to your `.env` file

### Grant Volume Permissions

The Service Principal needs READ and WRITE access to the UC volume:

```sql
GRANT READ_VOLUME ON VOLUME catalog.schema.volume TO `service-principal-id`;
GRANT WRITE_VOLUME ON VOLUME catalog.schema.volume TO `service-principal-id`;
```

> **Note:** The `_local/` folder is gitignored. Your credentials stay local and are not committed to the repository.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Upload Documents to Volume

```bash
cd src
unset DATABRICKS_TOKEN DATABRICKS_CONFIG_PROFILE
python 00_setup_sec_documents.py
```

This will:
1. Download FY2024 annual reports (NVIDIA, Apple, Samsung) to `_local/datasets/sec_2024/`
2. Upload them to `/Volumes/{catalog}/{schema}/{volume}/sec_2024/`

### 3. Create a Knowledge Assistant

```bash
# Option 1: REST API approach
python 01_ka_using_rest_api.py

# Option 2: AgentBricksManager wrapper
python 02_ka_using_agent_bricks_manager.py
```

### 4. Wait for Endpoint

KA endpoints take 2-10 minutes to become READY. The scripts poll the status automatically.

### 5. Update Environment File

After creating KAs, add the `tile_id` to your env file:

```bash
KA_NAME_01=SEC_Financial_Analyst
KA_TILE_ID_01=9134de1a-9e66-445a-9f0b-1574dd2447a0
```

### 6. Test the Knowledge Assistant

```bash
# Test KA 01 (default)
python 03_test_ka_conversation.py 1

# Test KA 02
python 03_test_ka_conversation.py 2

# Test by name
python 03_test_ka_conversation.py SEC_Financial_Analyst
```

---

## Script Details

### 00_setup_sec_documents.py

**Purpose:** Download FY2024 annual reports and upload to Unity Catalog volume.

**What it does:**
1. Downloads SEC/IR annual reports for NVIDIA, Apple, Samsung
2. Creates the volume subfolder if it doesn't exist
3. Uploads files to the configured UC volume path

**Output:**
```
_local/datasets/sec_2024/                    # Local PDF cache
/Volumes/catalog/schema/volume/sec_2024/     # UC Volume
```

### 01_ka_using_rest_api.py

**Purpose:** Create a Knowledge Assistant using direct Databricks REST API calls.

**Key functions:**
- `create_knowledge_assistant()` - Create KA via REST API
- `wait_for_ka_online()` - Poll until endpoint is ONLINE
- `add_ka_example()` - Add sample questions
- `query_ka_endpoint()` - Invoke the serving endpoint

### 02_ka_using_agent_bricks_manager.py

**Purpose:** Create a Knowledge Assistant using the AgentBricksManager wrapper class.

**Key difference:** Higher-level abstraction with batch example creation and create-or-update semantics.

### 03_test_ka_conversation.py

**Purpose:** Test a KA endpoint with a multi-turn conversation.

**Usage:**
```bash
python 03_test_ka_conversation.py 1              # KA_NAME_01 from env
python 03_test_ka_conversation.py 2              # KA_NAME_02 from env
python 03_test_ka_conversation.py SEC_Financial_Analyst  # By name
```

### 04_sync_ka_sources.py

**Purpose:** Force re-sync of KA knowledge sources after adding/modifying files.

**What it does:**
1. Shows current KA status
2. Triggers sync via REST API
3. Monitors progress until complete

### 05_add_ka_examples.py

**Purpose:** Add sample questions with guidelines to improve KA response quality.

**Sample questions included:**
- What was NVIDIA's total revenue in FY2024?
- What are Apple's main business segments?
- Compare the revenue growth of NVIDIA, Apple, and Samsung
- What are the key risk factors mentioned in NVIDIA's annual report?

---

## REST API Reference

| Operation | Endpoint | Method |
|-----------|----------|--------|
| List KAs | `/api/2.0/tiles?filter=tile_type=KA` | GET |
| Create KA | `/api/2.0/knowledge-assistants` | POST |
| Get KA | `/api/2.0/knowledge-assistants/{tile_id}` | GET |
| Update KA | `/api/2.0/knowledge-assistants/{tile_id}` | PATCH |
| Delete KA | `/api/2.0/tiles/{tile_id}` | DELETE |
| Sync sources | `/api/2.0/knowledge-assistants/{tile_id}/sync-knowledge-sources` | POST |
| Add example | `/api/2.0/knowledge-assistants/{tile_id}/examples` | POST |
| Query | `/serving-endpoints/{endpoint_name}/invocations` | POST |

### Endpoint Naming Convention

KA endpoints follow: `ka-{first_8_chars_of_tile_id}-endpoint`

Example:
- tile_id: `9134de1a-9e66-445a-9f0b-1574dd2447a0`
- endpoint: `ka-9134de1a-endpoint`

### Query Payload Format

```python
# Single turn
payload = {"input": [{"role": "user", "content": "Your question"}]}

# Multi-turn (include full history)
payload = {
    "input": [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Follow-up question"}
    ]
}
```

---

## Project Structure

```
databricks-examples/
├── _local/                            # Gitignored, shared across projects
│   ├── {workspace}.env                # Your credentials
│   └── datasets/                      # Local data cache
│       └── sec_2024/                  # Downloaded PDFs
│
└── agent_bricks_ka_example/
    ├── README.md                      # This file
    ├── requirements.txt               # Python dependencies
    ├── .env.template                  # Template for credentials
    ├── .gitignore                     # Git ignore rules
    └── src/
        ├── config.py                  # Config loader utility
        ├── 00_setup_sec_documents.py  # Download & upload SEC docs
        ├── 01_ka_using_rest_api.py    # REST API approach
        ├── 02_ka_using_agent_bricks_manager.py  # Manager approach
        ├── 03_test_ka_conversation.py # Multi-turn conversation test
        ├── 04_sync_ka_sources.py      # Sync knowledge sources
        └── 05_add_ka_examples.py      # Add sample questions
```

---

## Troubleshooting

### Auth Conflict Error

```
ValueError: more than one authorization method configured: oauth and pat
```

**Fix:** Clear conflicting environment variables:
```bash
unset DATABRICKS_TOKEN DATABRICKS_CONFIG_PROFILE
```

### Permission Denied on Volume

```
403 Client Error: User does not have READ_VOLUME/WRITE_VOLUME privilege
```

**Fix:** Grant permissions to your Service Principal (see Configuration section).

### Endpoint Stays in PROVISIONING

- Wait up to 10 minutes before investigating
- Check workspace capacity and quotas
- Verify volume path is accessible
- Check serving endpoint logs in Databricks UI

### KA Says "No Documents Found"

- Knowledge source may still be indexing (`KNOWLEDGE_SOURCE_STATE_UPDATING`)
- Wait a few minutes after endpoint becomes READY
- Verify files exist in the volume path

### Poor Answer Quality

- Add more specific instructions in `KA_INSTRUCTIONS`
- Add example questions using `05_add_ka_examples.py`
- Consider using Markdown instead of PDF for better text extraction

---

## Example: Create a Knowledge Assistant Programmatically

```python
from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()

payload = {
    "name": "My_Document_Assistant",
    "knowledge_sources": [
        {
            "files_source": {
                "name": "company_docs",
                "type": "files",
                "files": {"path": "/Volumes/main/default/documents"}
            }
        }
    ],
    "description": "Answers questions about company documents",
    "instructions": "Be helpful, accurate, and cite sources."
}

headers = w.config.authenticate()
headers["Content-Type"] = "application/json"

response = requests.post(
    f"{w.config.host}/api/2.0/knowledge-assistants",
    headers=headers,
    json=payload
)

result = response.json()
tile_id = result["knowledge_assistant"]["tile"]["tile_id"]
print(f"Created KA with tile_id: {tile_id}")
```

---

## Resources

**Agent Bricks:**
- [Agent Bricks Overview](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/index.html)
- [Knowledge Assistant](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/knowledge-assistant)
- [Create a Knowledge Assistant](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/create-knowledge-assistant)
- [Knowledge Assistant REST API](https://docs.databricks.com/api/workspace/knowledgeassistants)

**Databricks Platform:**
- [Databricks SDK for Python](https://databricks-sdk-py.readthedocs.io/)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)
- [Model Serving Endpoints](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
