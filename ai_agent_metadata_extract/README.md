# AI Agent Metadata Extractor

Extract metadata about all deployed AI agents and serving endpoints in your Databricks workspace.

## Scripts

| Script | Description |
|--------|-------------|
| `01_extract_ai_endpoints_detailed.py` | Detailed extraction with per-endpoint API calls for full metadata including Tiles API enrichment |
| `02_generate_endpoint_analysis_report.py` | Generates markdown reports and exports subsets by `model_type` |
| `03_extract_ai_endpoints_fast.py` | Fast single API call extraction, classifies all endpoints by `model_type` |

## Configuration

Scripts load credentials from `../_local/{ENV_NAME}.env`. Set up as follows:

### Step 1: Create the _local folder (if it doesn't exist)

```bash
mkdir -p ../_local
```

### Step 2: Copy the template and configure

```bash
cp .env.template ../_local/my-workspace.env
```

Edit `../_local/my-workspace.env`:

```bash
# Workspace URL (required)
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Option 1: OAuth M2M Service Principal (recommended)
DATABRICKS_CLIENT_ID=your-client-id
DATABRICKS_CLIENT_SECRET=your-client-secret

# Option 2: Personal Access Token
DATABRICKS_TOKEN=dapi...
```

### Step 3: Update ENV_NAME in scripts

Change `ENV_NAME` at the top of each script to match your env file name (without `.env`):

```python
ENV_NAME = "my-workspace"  # loads ../_local/my-workspace.env
```

## Quick Start

```bash
cd ai_agent_metadata_extract
pip install databricks-sdk python-dotenv requests
python 03_extract_ai_endpoints_fast.py
```

All output files are saved to `../_local/reports/`.

## model_type Classification

Endpoints are classified using the `model_type` field based on API response data:

### Databricks Foundation Models

| model_type | Description |
|------------|-------------|
| `DATABRICKS_FM_PPT` | Pay Per Token - Official Databricks FM API |
| `DATABRICKS_FM_PT` | Provisioned Throughput - Reserved capacity |
| `DATABRICKS_FM_UC_SYSTEM_AI` | UC model backed by system.ai |
| `DATABRICKS_FM_UC_AGENTS` | User-built AI agents in UC |
| `DATABRICKS_CLASSIC_ML` | Classic ML models (sklearn, xgboost, pyfunc) |

### Agent Bricks

| model_type | Description |
|------------|-------------|
| `AGENT_BRICKS_KA` | Knowledge Assistants |
| `AGENT_BRICKS_MAS` | Multi-Agent Supervisors |
| `AGENT_BRICKS_KIE` | Key Information Extraction |
| `AGENT_BRICKS_MS` | Model Specialization (Playground Finetuning) |

### External Models (AI Gateway)

| model_type | Description |
|------------|-------------|
| `FM_EXTERNAL_MODEL` | External Provider (OpenAI, Anthropic, etc.) |
| `FM_EXTERNAL_MODEL_CUSTOM` | Custom Provider (user-specified URL) |

## API Notes

- **Script 01**: Makes per-endpoint API calls and enriches Agent Bricks with Tiles API metadata (`tile_name`, `tile_description`, `tile_instructions`)
- **Script 03**: Single list API call for speed; returns basic `tile_metadata` fields only
- Both use `tile_endpoint_metadata.problem_type` for Agent Bricks classification
- Unclassified endpoints are marked as `UNCLASSIFIED`

---

## Deep Dive

### JSON Output Structure

Each endpoint in the JSON output includes:

| Field | Description |
|-------|-------------|
| `endpoint_name` | Unique endpoint identifier |
| `endpoint_type` | Type of serving endpoint |
| `task` | Task type (e.g., `llm/v1/chat`, `agent/v1/responses`) |
| `state` | `ready` and `config_update` status |
| `entity` | `type` and `name` from served_entities |
| `model` | `name`, `display_name`, `class` (for Foundation Models) |
| `external_model` | `provider` and `name` (for AI Gateway) |
| `tile_metadata` | Agent Bricks metadata (see below) |
| `timestamps` | `created_ms`, `updated_ms` |
| `_metadata_derived` | Computed fields: `model_type`, `is_system_ai`, `is_provisioned_throughput`, ISO timestamps |

**Tile Metadata** (for Agent Bricks KA/MAS/KIE/MS):

| Field | Description | Source |
|-------|-------------|--------|
| `tile_id` | Unique tile identifier | Serving Endpoints API |
| `tile_model_name` | Internal model name | Serving Endpoints API |
| `problem_type` | `KNOWLEDGE_ASSISTANT`, `MULTI_AGENT_SUPERVISOR`, etc. | Serving Endpoints API |
| `tile_name` | User-defined name (e.g., "My Knowledge Assistant") | Tiles API (Script 01 only) |
| `tile_description` | What the tile does | Tiles API (Script 01 only) |
| `tile_instructions` | System instructions for the AI | Tiles API (Script 01 only) |

### Classification Criteria

#### Databricks Foundation Models

| model_type | Classification Criteria |
|------------|------------------------|
| `DATABRICKS_FM_PPT` | `entity_type="FOUNDATION_MODEL"` AND name starts with `databricks-` |
| `DATABRICKS_FM_PT` | `entity_type="PT_FOUNDATION_MODEL"` |
| `DATABRICKS_FM_UC_SYSTEM_AI` | `entity_type="UC_MODEL"` AND entity.name starts with `system.ai.` |
| `DATABRICKS_FM_UC_AGENTS` | `entity_type="UC_MODEL"` AND has task AND NOT system.ai |
| `DATABRICKS_CLASSIC_ML` | `entity_type="UC_MODEL"` AND no task |

#### Agent Bricks

| model_type | `problem_type` Value |
|------------|---------------------|
| `AGENT_BRICKS_KA` | `KNOWLEDGE_ASSISTANT` |
| `AGENT_BRICKS_MAS` | `MULTI_AGENT_SUPERVISOR` |
| `AGENT_BRICKS_KIE` | `INFORMATION_EXTRACTION` |
| `AGENT_BRICKS_MS` | `MODEL_SPECIALIZATION` |

#### External Models (AI Gateway)

| model_type | Classification Criteria |
|------------|------------------------|
| `FM_EXTERNAL_MODEL` | `entity_type="EXTERNAL_MODEL"` AND provider ≠ "custom" |
| `FM_EXTERNAL_MODEL_CUSTOM` | `entity_type="EXTERNAL_MODEL"` AND provider = "custom" |

### Tiles API for Agent Bricks

Script 01 enriches Agent Bricks endpoints with metadata from the Tiles API:

```
GET /api/2.0/tiles?filter=tile_type=KA   # Knowledge Assistants
GET /api/2.0/tiles?filter=tile_type=MAS  # Multi-Agent Supervisors
GET /api/2.0/tiles?filter=tile_type=KIE  # Key Information Extraction
GET /api/2.0/tiles?filter=tile_type=MS   # Model Specialization
```

This provides descriptive fields (`name`, `description`, `instructions`) that aren't available from the serving endpoints API alone.
