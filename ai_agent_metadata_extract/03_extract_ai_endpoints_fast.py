# Databricks notebook source
# MAGIC %md
# MAGIC # Serving Endpoints Classification
# MAGIC 
# MAGIC This notebook extracts metadata for serving endpoints and classifies them using the `model_type` tag.
# MAGIC 
# MAGIC ## model_type Classification
# MAGIC 
# MAGIC The `model_type` tag classifies endpoints by **model source** and **deployment/billing mode**:
# MAGIC 
# MAGIC ### Databricks-Hosted Foundation Models (system.ai.*)
# MAGIC 
# MAGIC | model_type | Description | Billing |
# MAGIC |---------|-------------|---------|
# MAGIC | `DATABRICKS_FM_PPT` | **Pay Per Token** - Official Databricks FM API | On-demand, per-token |
# MAGIC | `DATABRICKS_FM_PT` | **Provisioned Throughput** - Reserved capacity | Fixed, reserved |
# MAGIC | `DATABRICKS_FM_UC_SYSTEM_AI` | **UC + System AI** - UC model backed by system.ai | Varies |
# MAGIC | `DATABRICKS_FM_UC_AGENTS` | **UC + Custom** - UC model with task | Varies |
# MAGIC 
# MAGIC ### Classic ML Models
# MAGIC 
# MAGIC | model_type | Description | Billing |
# MAGIC |---------|-------------|---------|
# MAGIC | `DATABRICKS_CLASSIC_ML` | **Classic ML** - sklearn, xgboost, pyfunc (no task) | Compute |
# MAGIC 
# MAGIC ### Agent Bricks / Playground (via `tile_endpoint_metadata.problem_type`)
# MAGIC 
# MAGIC | model_type | Description | `problem_type` |
# MAGIC |---------|-------------|----------------|
# MAGIC | `AGENT_BRICKS_KA` | **Knowledge Assistants** | `KNOWLEDGE_ASSISTANT` |
# MAGIC | `AGENT_BRICKS_MAS` | **Multi-Agent Supervisors** | `MULTI_AGENT_SUPERVISOR` |
# MAGIC | `AGENT_BRICKS_KIE` | **Key Information Extraction** | `INFORMATION_EXTRACTION` |
# MAGIC | `AGENT_BRICKS_MS` | **Model Specialization** (Playground Finetuning) | `MODEL_SPECIALIZATION` |
# MAGIC 
# MAGIC ### External Models (via AI Gateway)
# MAGIC 
# MAGIC | model_type | Description | Billing |
# MAGIC |---------|-------------|---------|
# MAGIC | `FM_EXTERNAL_MODEL` | **External Provider** - OpenAI, Anthropic, Google, Bedrock | Pass-through |
# MAGIC | `FM_EXTERNAL_MODEL_CUSTOM` | **Custom Provider** - User-specified URL | User-managed |
# MAGIC 
# MAGIC ### Classification Logic
# MAGIC 
# MAGIC ```python
# MAGIC # First check tile_endpoint_metadata.problem_type (Agent Bricks / Playground)
# MAGIC if tile_problem_type == "KNOWLEDGE_ASSISTANT":
# MAGIC     model_type = "AGENT_BRICKS_KA"
# MAGIC elif tile_problem_type == "MULTI_AGENT_SUPERVISOR":
# MAGIC     model_type = "AGENT_BRICKS_MAS"
# MAGIC elif tile_problem_type == "INFORMATION_EXTRACTION":
# MAGIC     model_type = "AGENT_BRICKS_KIE"
# MAGIC elif tile_problem_type == "MODEL_SPECIALIZATION":
# MAGIC     model_type = "AGENT_BRICKS_MS"
# MAGIC # Then check entity_type based classifications
# MAGIC elif entity_type == "FOUNDATION_MODEL" and name.startswith("databricks-"):
# MAGIC     model_type = "DATABRICKS_FM_PPT"
# MAGIC elif entity_type == "PT_FOUNDATION_MODEL":
# MAGIC     model_type = "DATABRICKS_FM_PT"
# MAGIC elif entity_type == "UC_MODEL" and is_system_ai:
# MAGIC     model_type = "DATABRICKS_FM_UC_SYSTEM_AI"
# MAGIC elif entity_type == "UC_MODEL" and not task:
# MAGIC     model_type = "DATABRICKS_CLASSIC_ML"
# MAGIC elif entity_type == "UC_MODEL":
# MAGIC     model_type = "DATABRICKS_FM_UC_AGENTS"
# MAGIC elif entity_type == "EXTERNAL_MODEL" and provider == "custom":
# MAGIC     model_type = "FM_EXTERNAL_MODEL_CUSTOM"
# MAGIC elif entity_type == "EXTERNAL_MODEL":
# MAGIC     model_type = "FM_EXTERNAL_MODEL"
# MAGIC ```
# MAGIC 
# MAGIC ## Output Files
# MAGIC 
# MAGIC - `databricks_fm_api_latest.json` - Full endpoint data with `model_type` field
# MAGIC - `databricks_fm_api_summary.md` - Markdown summary grouped by `model_type`
# MAGIC 
# MAGIC ## Usage
# MAGIC 
# MAGIC ```bash
# MAGIC cd ai_agent_metadata_extract
# MAGIC python 03_extract_endpoints_fast.py
# MAGIC ```
# MAGIC 
# MAGIC Uses a **single API call** for fast extraction (~5 seconds for 800+ endpoints).

# COMMAND ----------

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Configuration
ENV_NAME = "my-workspace"  # Change to match your ../_local/{name}.env file
OUTPUT_DIR = Path("../_local/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR.absolute()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Authentication

# COMMAND ----------

def load_env_file(env_name: str) -> Dict[str, str]:
    """Load environment variables from .env file."""
    search_dirs = [Path("../_local"), Path("_local")]
    
    for d in search_dirs:
        env_path = d / f"{env_name}.env"
        if env_path.exists():
            print(f"Loading config from: {env_path}")
            config = {}
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            return config
    return {}


def setup_auth(config: Dict[str, str]) -> str:
    """Setup authentication, return host. Clears ALL conflicting auth vars."""
    host = os.environ.get('DATABRICKS_HOST') or config.get('DATABRICKS_HOST', '')
    
    # Clear profile to prevent SDK from reading ~/.databrickscfg
    os.environ.pop('DATABRICKS_CONFIG_PROFILE', None)
    
    # Check if PAT in env (takes precedence)
    if os.environ.get('DATABRICKS_TOKEN'):
        # Clear OAuth vars to avoid conflict
        for key in ['DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET']:
            os.environ.pop(key, None)
        os.environ['DATABRICKS_HOST'] = host
        print("Using PAT from environment")
        return host
    
    # Check config for OAuth or PAT
    if config.get('DATABRICKS_CLIENT_ID') and config.get('DATABRICKS_CLIENT_SECRET'):
        # Clear ALL conflicting vars
        os.environ.pop('DATABRICKS_TOKEN', None)
        os.environ['DATABRICKS_CLIENT_ID'] = config['DATABRICKS_CLIENT_ID']
        os.environ['DATABRICKS_CLIENT_SECRET'] = config['DATABRICKS_CLIENT_SECRET']
        os.environ['DATABRICKS_HOST'] = host
        print("Using OAuth M2M from config")
    elif config.get('DATABRICKS_TOKEN'):
        # Clear OAuth vars
        for key in ['DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET']:
            os.environ.pop(key, None)
        os.environ['DATABRICKS_TOKEN'] = config['DATABRICKS_TOKEN']
        os.environ['DATABRICKS_HOST'] = host
        print("Using PAT from config")
    
    return host


# Clear profile before loading to avoid conflicts
os.environ.pop('DATABRICKS_CONFIG_PROFILE', None)


config = load_env_file(ENV_NAME)
DATABRICKS_HOST = setup_auth(config)
print(f"Workspace: {DATABRICKS_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single API Call - List All Endpoints

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()

def get_headers() -> Dict[str, str]:
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers


def list_all_endpoints() -> List[Dict]:
    """
    Get all endpoints with single LIST API call.
    Returns full endpoint data including config.served_entities.
    """
    print("Fetching all endpoints (single API call)...")
    
    url = f"{DATABRICKS_HOST}/api/2.0/serving-endpoints"
    response = requests.get(url, headers=get_headers(), timeout=60)
    response.raise_for_status()
    
    endpoints = response.json().get("endpoints", [])
    print(f"Retrieved {len(endpoints)} endpoints")
    
    return endpoints

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Metadata from List Response

# COMMAND ----------

def extract_endpoint_data(endpoint: Dict) -> Dict:
    """
    Extract relevant metadata from a single endpoint (LIST API response).
    """
    # Basic fields
    name = endpoint.get("name", "")
    endpoint_type = endpoint.get("endpoint_type", "")
    task = endpoint.get("task", "")
    
    # State
    state = endpoint.get("state", {})
    ready_state = state.get("ready", "UNKNOWN")
    config_update = state.get("config_update", "")
    
    # Timestamps
    creation_timestamp = endpoint.get("creation_timestamp")
    last_updated_timestamp = endpoint.get("last_updated_timestamp")
    
    created_at = None
    updated_at = None
    if creation_timestamp:
        created_at = datetime.fromtimestamp(creation_timestamp / 1000).isoformat()
    if last_updated_timestamp:
        updated_at = datetime.fromtimestamp(last_updated_timestamp / 1000).isoformat()
    
    # Capabilities
    capabilities = endpoint.get("capabilities", {})
    
    # Extract tile_endpoint_metadata (Agent Bricks / Playground endpoints)
    tile_metadata = endpoint.get("tile_endpoint_metadata", {})
    tile_id = tile_metadata.get("tile_id", "")
    tile_model_name = tile_metadata.get("tile_model_name", "")
    tile_problem_type = tile_metadata.get("problem_type", "")
    
    # Extract from served_entities
    config = endpoint.get("config", {})
    served_entities = config.get("served_entities", [])
    
    entity_type = ""
    entity_name = ""
    model_name = ""
    model_display_name = ""
    model_description = ""
    model_price = ""
    model_class = ""
    external_model_provider = ""
    external_model_name = ""
    
    if served_entities:
        entity = served_entities[0]
        entity_type = entity.get("type", "")
        entity_name = entity.get("entity_name", "")
        
        # Foundation Model info
        fm = entity.get("foundation_model", {})
        if fm:
            model_name = fm.get("name", "")
            model_display_name = fm.get("display_name", "")
            model_description = fm.get("description", "")
            model_price = fm.get("price", "")
            model_class = fm.get("model_class", "")
        
        # External Model info
        em = entity.get("external_model", {})
        if em:
            external_model_provider = em.get("provider", "")
            external_model_name = em.get("name", "")
    
    # Determine if this is a Databricks system.ai endpoint (pay-per-token FM API)
    is_system_ai = (
        entity_name.startswith("system.ai.") or
        model_name.startswith("system.ai.")
    )
    
    # Provisioned Throughput endpoint
    is_provisioned_throughput = entity_type == "PT_FOUNDATION_MODEL"
    
    # ==========================================================================
    # model_type Classification
    # ==========================================================================
    # Classifies serving endpoints by their model source and deployment mode.
    #
    # DATABRICKS-HOSTED FOUNDATION MODELS (system.ai.*):
    #
    # DATABRICKS_FM_PPT (Pay Per Token):
    #   - Official Databricks FM API endpoints (e.g., databricks-claude-sonnet-4)
    #   - Pre-provisioned by Databricks, available to all workspaces
    #   - On-demand billing: pay only for tokens consumed
    #   - Criteria: entity_type="FOUNDATION_MODEL" AND name starts with "databricks-"
    #
    # DATABRICKS_FM_PT (Provisioned Throughput):
    #   - User-created endpoints with reserved capacity
    #   - Guaranteed throughput for high-volume production workloads
    #   - Fixed billing: pay for reserved capacity regardless of usage
    #   - Criteria: entity_type="PT_FOUNDATION_MODEL"
    #
    # DATABRICKS_FM_UC_SYSTEM_AI (Unity Catalog - System AI):
    #   - User-created endpoints registered in Unity Catalog
    #   - Backed by Databricks system.ai.* foundation models
    #   - Often includes guardrails, rate limits, or custom config
    #   - Criteria: entity_type="UC_MODEL" AND entity_name starts with "system.ai."
    #
    # DATABRICKS_FM_UC_AGENTS (Unity Catalog - Custom):
    #   - User-created endpoints registered in Unity Catalog
    #   - Custom models (fine-tuned, imported, or user-trained)
    #   - Not backed by system.ai.* models
    #   - Criteria: entity_type="UC_MODEL" AND entity_name does NOT start with "system.ai."
    #
    # EXTERNAL MODELS (routed via AI Gateway):
    #
    # FM_EXTERNAL_MODEL:
    #   - Routes to external providers (OpenAI, Anthropic, Google, Bedrock, etc.)
    #   - Pass-through billing to external provider
    #   - Criteria: entity_type="EXTERNAL_MODEL" AND provider != "custom"
    #
    # FM_EXTERNAL_MODEL_CUSTOM:
    #   - Routes to a custom user-specified URL/endpoint
    #   - User manages the external endpoint and billing
    #   - Criteria: entity_type="EXTERNAL_MODEL" AND provider == "custom"
    #
    # CLASSIC ML MODELS:
    #
    # DATABRICKS_CLASSIC_ML:
    #   - Traditional ML models (sklearn, xgboost, pyfunc, etc.)
    #   - No task type defined (not LLM/agent)
    #   - Criteria: entity_type="UC_MODEL" AND no task AND NOT system.ai
    #
    # AGENT BRICKS (identified by tile_endpoint_metadata.problem_type):
    #
    # AGENT_BRICKS_KA:
    #   - Knowledge Assistants
    #   - Criteria: tile_endpoint_metadata.problem_type == "KNOWLEDGE_ASSISTANT"
    #
    # AGENT_BRICKS_MAS:
    #   - Multi-Agent Supervisors
    #   - Criteria: tile_endpoint_metadata.problem_type == "MULTI_AGENT_SUPERVISOR"
    #
    # AGENT_BRICKS_KIE:
    #   - Key Information Extraction
    #   - Criteria: tile_endpoint_metadata.problem_type == "INFORMATION_EXTRACTION"
    #
    # AGENT_BRICKS_MS:
    #   - Model Specialization (finetuning via Playground)
    #   - Criteria: tile_endpoint_metadata.problem_type == "MODEL_SPECIALIZATION"
    # ==========================================================================
    model_type = None
    
    # First check tile_endpoint_metadata.problem_type (Agent Bricks / Playground)
    if tile_problem_type == "KNOWLEDGE_ASSISTANT":
        model_type = "AGENT_BRICKS_KA"
    elif tile_problem_type == "MULTI_AGENT_SUPERVISOR":
        model_type = "AGENT_BRICKS_MAS"
    elif tile_problem_type == "INFORMATION_EXTRACTION":
        model_type = "AGENT_BRICKS_KIE"
    elif tile_problem_type == "MODEL_SPECIALIZATION":
        model_type = "AGENT_BRICKS_MS"
    # Then check entity_type based classifications
    elif entity_type == "FOUNDATION_MODEL" and name.startswith("databricks-"):
        model_type = "DATABRICKS_FM_PPT"
    elif entity_type == "PT_FOUNDATION_MODEL":
        model_type = "DATABRICKS_FM_PT"
    elif entity_type == "UC_MODEL":
        if is_system_ai:
            model_type = "DATABRICKS_FM_UC_SYSTEM_AI"
        elif not task:
            model_type = "DATABRICKS_CLASSIC_ML"
        else:
            model_type = "DATABRICKS_FM_UC_AGENTS"
    elif entity_type == "EXTERNAL_MODEL":
        if external_model_provider == "custom":
            model_type = "FM_EXTERNAL_MODEL_CUSTOM"
        else:
            model_type = "FM_EXTERNAL_MODEL"
    
    return {
        # Core endpoint identifiers
        "endpoint_name": name,
        "endpoint_type": endpoint_type,
        "task": task,
        
        # State information
        "state": {
            "ready": ready_state,
            "config_update": config_update,
        },
        
        # Entity (served model) information
        "entity": {
            "type": entity_type,
            "name": entity_name,
        },
        
        # Foundation Model details (populated for FM endpoints)
        "model": {
            "name": model_name,
            "display_name": model_display_name,
            "description": model_description,
            "price": model_price,
            "class": model_class,
        },
        
        # External Model details (populated for external endpoints)
        "external_model": {
            "provider": external_model_provider,
            "name": external_model_name,
        },
        
        # Agent Bricks / Playground tile metadata (populated if tile_endpoint_metadata exists)
        "tile_metadata": {
            "tile_id": tile_id,
            "tile_model_name": tile_model_name,
            "problem_type": tile_problem_type,
        },
        
        # Raw timestamps from API (milliseconds)
        "timestamps": {
            "created_ms": creation_timestamp,
            "updated_ms": last_updated_timestamp,
        },
        
        # Capabilities from API
        "capabilities": capabilities,
        
        # Derived metadata - fields computed by this script for classification/analysis
        "_metadata_derived": {
            "model_type": model_type,
            "is_system_ai": is_system_ai,
            "is_provisioned_throughput": is_provisioned_throughput,
            "created_at": created_at,
            "updated_at": updated_at,
        },
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Extraction

# COMMAND ----------

extraction_start = datetime.now()
print(f"Starting extraction at {extraction_start.isoformat()}")

# Single API call
raw_endpoints = list_all_endpoints()

# Extract metadata for all endpoints first
all_endpoints = [extract_endpoint_data(ep) for ep in raw_endpoints]

# Filter to endpoints with a classified model_type:
# - DATABRICKS_FM_PPT: Official Databricks FM API (pay-per-token)
# - DATABRICKS_FM_PT: Provisioned Throughput
# - DATABRICKS_FM_UC_SYSTEM_AI: Unity Catalog backed by system.ai
# - DATABRICKS_FM_UC_AGENTS: Unity Catalog custom models (with task)
# - DATABRICKS_CLASSIC_ML: Classic ML models (sklearn, xgboost, pyfunc)
# - FM_EXTERNAL_MODEL: External provider (OpenAI, Anthropic, etc.)
# - FM_EXTERNAL_MODEL_CUSTOM: External custom provider
endpoints = [
    ep for ep in all_endpoints 
    if ep["_metadata_derived"]["model_type"] is not None
]

extraction_end = datetime.now()
duration = (extraction_end - extraction_start).total_seconds()

print(f"Extraction completed in {duration:.2f} seconds")
print(f"Total endpoints in workspace: {len(all_endpoints)}")
print(f"Classified endpoints (with model_type): {len(endpoints)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Summary Statistics

# COMMAND ----------

def generate_summary(endpoints: List[Dict]) -> Dict:
    """Generate summary statistics for classified serving endpoints."""
    
    # Helper to get model_type from nested structure
    def get_model_type(ep):
        return ep["_metadata_derived"]["model_type"]
    
    # Count by model_type
    by_model_type = Counter(get_model_type(ep) or "UNCLASSIFIED" for ep in endpoints)
    
    # Count by various fields (using nested structure)
    by_entity_type = Counter(ep["entity"]["type"] or "unknown" for ep in endpoints)
    by_task = Counter(ep["task"] or "unknown" for ep in endpoints)
    by_ready_state = Counter(ep["state"]["ready"] for ep in endpoints)
    by_model_class = Counter(ep["model"]["class"] or "unknown" for ep in endpoints if ep["model"]["class"])
    
    # Get endpoints by model_type
    ppt_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_PPT"]
    pt_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_PT"]
    uc_system_ai_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_UC_SYSTEM_AI"]
    uc_agents_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_UC_AGENTS"]
    classic_ml_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_CLASSIC_ML"]
    external_endpoints = [ep for ep in endpoints if get_model_type(ep) == "FM_EXTERNAL_MODEL"]
    external_custom_endpoints = [ep for ep in endpoints if get_model_type(ep) == "FM_EXTERNAL_MODEL_CUSTOM"]
    # Agent Bricks
    ka_endpoints = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_KA"]
    mas_endpoints = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_MAS"]
    kie_endpoints = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_KIE"]
    ms_endpoints = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_MS"]
    
    # External model provider breakdown
    by_external_provider = Counter(
        ep["external_model"]["provider"] for ep in endpoints 
        if ep["external_model"]["provider"]
    )
    
    return {
        "total_endpoints": len(endpoints),
        "by_model_type": dict(by_model_type.most_common()),
        "databricks_fm_ppt_count": len(ppt_endpoints),
        "databricks_fm_pt_count": len(pt_endpoints),
        "databricks_fm_uc_system_ai_count": len(uc_system_ai_endpoints),
        "databricks_fm_uc_agents_count": len(uc_agents_endpoints),
        "databricks_classic_ml_count": len(classic_ml_endpoints),
        "fm_external_model_count": len(external_endpoints),
        "fm_external_model_custom_count": len(external_custom_endpoints),
        "agent_bricks_ka_count": len(ka_endpoints),
        "agent_bricks_mas_count": len(mas_endpoints),
        "agent_bricks_kie_count": len(kie_endpoints),
        "agent_bricks_ms_count": len(ms_endpoints),
        "by_entity_type": dict(by_entity_type.most_common()),
        "by_task": dict(by_task.most_common()),
        "by_ready_state": dict(by_ready_state),
        "by_model_class": dict(by_model_class.most_common()),
        "by_external_provider": dict(by_external_provider.most_common()),
    }


summary = generate_summary(endpoints)

print("\n" + "="*60)
print(" SERVING ENDPOINTS SUMMARY")
print("="*60)
print(f"Total Classified Endpoints: {summary['total_endpoints']}")
print(f"\nBy model_type:")
for k, v in summary["by_model_type"].items():
    print(f"  {k}: {v}")

print("\nBy Entity Type:")
for k, v in summary["by_entity_type"].items():
    print(f"  {k}: {v}")

print("\nBy Task:")
for k, v in summary["by_task"].items():
    print(f"  {k}: {v}")

if summary["by_model_class"]:
    print("\nBy Model Class (Databricks FM):")
    for k, v in summary["by_model_class"].items():
        print(f"  {k}: {v}")

if summary["by_external_provider"]:
    print("\nBy External Provider:")
    for k, v in summary["by_external_provider"].items():
        print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save JSON Output

# COMMAND ----------

# Save JSON - just the endpoints array (pure data, no metadata/summary)
timestamp = extraction_start.strftime("%Y%m%d_%H%M%S")
json_timestamped = OUTPUT_DIR / f"databricks_fm_api_{timestamp}.json"
json_latest = OUTPUT_DIR / "databricks_fm_api_latest.json"

with open(json_timestamped, 'w') as f:
    json.dump(endpoints, f, indent=2, default=str)
print(f"Saved: {json_timestamped}")

with open(json_latest, 'w') as f:
    json.dump(endpoints, f, indent=2, default=str)
print(f"Saved: {json_latest}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Markdown Summary

# COMMAND ----------

def generate_markdown_summary(
    endpoints: List[Dict],
    summary: Dict,
    workspace: str,
    extraction_timestamp: str,
    duration_seconds: float
) -> str:
    """Generate a Markdown summary report for classified serving endpoints."""
    
    # Helper to get model_type from nested structure
    def get_model_type(ep):
        return ep["_metadata_derived"]["model_type"]
    
    # Split endpoints by model_type
    ppt_eps = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_PPT"]
    pt_eps = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_PT"]
    uc_system_ai_eps = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_UC_SYSTEM_AI"]
    uc_agents_eps = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_UC_AGENTS"]
    classic_ml_eps = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_CLASSIC_ML"]
    ext_eps = [ep for ep in endpoints if get_model_type(ep) == "FM_EXTERNAL_MODEL"]
    ext_custom_eps = [ep for ep in endpoints if get_model_type(ep) == "FM_EXTERNAL_MODEL_CUSTOM"]
    # Agent Bricks
    ka_eps = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_KA"]
    mas_eps = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_MAS"]
    kie_eps = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_KIE"]
    ms_eps = [ep for ep in endpoints if get_model_type(ep) == "AGENT_BRICKS_MS"]
    
    # Sort each
    ppt_sorted = sorted(ppt_eps, key=lambda x: x["endpoint_name"])
    pt_sorted = sorted(pt_eps, key=lambda x: x["endpoint_name"])
    uc_system_ai_sorted = sorted(uc_system_ai_eps, key=lambda x: x["endpoint_name"])
    uc_agents_sorted = sorted(uc_agents_eps, key=lambda x: x["endpoint_name"])
    classic_ml_sorted = sorted(classic_ml_eps, key=lambda x: x["endpoint_name"])
    ext_sorted = sorted(ext_eps, key=lambda x: x["endpoint_name"])
    ext_custom_sorted = sorted(ext_custom_eps, key=lambda x: x["endpoint_name"])
    ka_sorted = sorted(ka_eps, key=lambda x: x["endpoint_name"])
    mas_sorted = sorted(mas_eps, key=lambda x: x["endpoint_name"])
    kie_sorted = sorted(kie_eps, key=lambda x: x["endpoint_name"])
    ms_sorted = sorted(ms_eps, key=lambda x: x["endpoint_name"])
    
    md = f"""# Serving Endpoints Classification Report

**Workspace:** {workspace}  
**Extracted:** {extraction_timestamp}  
**Duration:** {duration_seconds:.2f}s (single API call)  

## model_type Classification

The `model_type` tag classifies serving endpoints by model source and deployment mode:

| model_type | Description | Billing | Count |
|---------|-------------|---------|-------|
| `DATABRICKS_FM_PPT` | **Pay Per Token** - Official Databricks FM API | On-demand, per-token | {len(ppt_eps)} |
| `DATABRICKS_FM_PT` | **Provisioned Throughput** - Reserved capacity | Fixed, reserved | {len(pt_eps)} |
| `DATABRICKS_FM_UC_SYSTEM_AI` | **UC + System AI** - UC model backed by system.ai | Varies | {len(uc_system_ai_eps)} |
| `DATABRICKS_FM_UC_AGENTS` | **UC + Agents** - User-built AI agents | Varies | {len(uc_agents_eps)} |
| `DATABRICKS_CLASSIC_ML` | **Classic ML** - sklearn, xgboost, pyfunc | Compute | {len(classic_ml_eps)} |
| `AGENT_BRICKS_KA` | **Knowledge Assistants** - Agent Bricks | Included | {len(ka_eps)} |
| `AGENT_BRICKS_MAS` | **Multi-Agent Supervisors** - Agent Bricks | Included | {len(mas_eps)} |
| `AGENT_BRICKS_KIE` | **Key Information Extraction** - Agent Bricks | Included | {len(kie_eps)} |
| `AGENT_BRICKS_MS` | **Model Specialization** - Playground Finetuning | Included | {len(ms_eps)} |
| `FM_EXTERNAL_MODEL` | **External Provider** - OpenAI, Anthropic, etc. | Pass-through | {len(ext_eps)} |
| `FM_EXTERNAL_MODEL_CUSTOM` | **External Custom** - User URL | User-managed | {len(ext_custom_eps)} |
| | **Total** | | **{summary['total_endpoints']}** |

### Classification Criteria

**Databricks-hosted Foundation Models (system.ai.*):**
- `DATABRICKS_FM_PPT`: `entity_type="FOUNDATION_MODEL"` AND `endpoint_name` starts with `databricks-`
- `DATABRICKS_FM_PT`: `entity_type="PT_FOUNDATION_MODEL"`
- `DATABRICKS_FM_UC_SYSTEM_AI`: `entity_type="UC_MODEL"` AND `entity.name` starts with `system.ai.`
- `DATABRICKS_FM_UC_AGENTS`: `entity_type="UC_MODEL"` AND has `task` AND NOT `system.ai.`

**Classic ML Models:**
- `DATABRICKS_CLASSIC_ML`: `entity_type="UC_MODEL"` AND no `task` (sklearn, xgboost, pyfunc, etc.)

**Agent Bricks / Playground (via `tile_endpoint_metadata.problem_type`):**
- `AGENT_BRICKS_KA`: `problem_type="KNOWLEDGE_ASSISTANT"` (Knowledge Assistants)
- `AGENT_BRICKS_MAS`: `problem_type="MULTI_AGENT_SUPERVISOR"` (Multi-Agent Supervisors)
- `AGENT_BRICKS_KIE`: `problem_type="INFORMATION_EXTRACTION"` (Key Information Extraction)
- `AGENT_BRICKS_MS`: `problem_type="MODEL_SPECIALIZATION"` (Model Finetuning via Playground)

**External Models:**
- `FM_EXTERNAL_MODEL`: `entity_type="EXTERNAL_MODEL"` AND `provider != "custom"`
- `FM_EXTERNAL_MODEL_CUSTOM`: `entity_type="EXTERNAL_MODEL"` AND `provider == "custom"`

## By Entity Type

| Entity Type | Count |
|-------------|-------|
"""
    for k, v in summary["by_entity_type"].items():
        md += f"| {k} | {v} |\n"
    
    md += """
## By Task

| Task | Count |
|------|-------|
"""
    for k, v in summary["by_task"].items():
        md += f"| {k} | {v} |\n"
    
    md += """
## By Model Class

| Model Class | Count |
|-------------|-------|
"""
    for k, v in summary["by_model_class"].items():
        md += f"| {k} | {v} |\n"
    
    md += """
## By Ready State

| State | Count |
|-------|-------|
"""
    for k, v in summary["by_ready_state"].items():
        md += f"| {k} | {v} |\n"
    
    # DATABRICKS_FM_PPT - Pay Per Token (official endpoints)
    md += f"""
## DATABRICKS_FM_PPT - Pay Per Token ({len(ppt_eps)})

Official Databricks FM API endpoints with on-demand pay-per-token billing.

| Endpoint Name | Model (entity_name) | Model Class | Task |
|---------------|---------------------|-------------|------|
"""
    for ep in ppt_sorted:
        name = ep["endpoint_name"]
        model = ep["entity"]["name"]
        model_class = ep["model"]["class"] or ""
        task = ep["task"]
        md += f"| {name} | {model} | {model_class} | {task} |\n"
    
    # DATABRICKS_FM_PT - Provisioned Throughput
    if pt_eps:
        md += f"""
## DATABRICKS_FM_PT - Provisioned Throughput ({len(pt_eps)})

User-created endpoints with reserved capacity.

| Endpoint Name | Model (entity.name) | Task |
|---------------|---------------------|------|
"""
        for ep in pt_sorted:
            name = ep["endpoint_name"]
            model = ep["entity"]["name"]
            task = ep["task"]
            md += f"| {name} | {model} | {task} |\n"
    
    # DATABRICKS_FM_UC_SYSTEM_AI - Unity Catalog backed by system.ai
    if uc_system_ai_eps:
        md += f"""
## DATABRICKS_FM_UC_SYSTEM_AI - UC + System AI ({len(uc_system_ai_eps)})

UC endpoints backed by Databricks system.ai.* foundation models.

| Endpoint Name | Model (entity.name) | Task |
|---------------|---------------------|------|
"""
        for ep in uc_system_ai_sorted:
            name = ep["endpoint_name"]
            model = ep["entity"]["name"]
            task = ep["task"]
            md += f"| {name} | {model} | {task} |\n"
    
    # DATABRICKS_FM_UC_AGENTS - Unity Catalog custom models
    if uc_agents_eps:
        md += f"""
## DATABRICKS_FM_UC_AGENTS - UC + Custom ({len(uc_agents_eps)})

UC endpoints with custom/user models (has task, not backed by system.ai).

| Endpoint Name | Model (entity.name) | Task |
|---------------|---------------------|------|
"""
        for ep in uc_agents_sorted:
            name = ep["endpoint_name"]
            model = ep["entity"]["name"]
            task = ep["task"]
            md += f"| {name} | {model} | {task} |\n"
    
    # DATABRICKS_CLASSIC_ML - Traditional ML models
    if classic_ml_eps:
        md += f"""
## DATABRICKS_CLASSIC_ML - Classic ML ({len(classic_ml_eps)})

Traditional ML models (sklearn, xgboost, pyfunc, etc.) without LLM/agent task.

| Endpoint Name | Model (entity.name) | State |
|---------------|---------------------|-------|
"""
        for ep in classic_ml_sorted:
            name = ep["endpoint_name"]
            model = ep["entity"]["name"]
            state = ep["state"]["ready"]
            md += f"| {name} | {model} | {state} |\n"
    
    # FM_EXTERNAL_MODEL - External Provider
    if ext_eps:
        md += f"""
## FM_EXTERNAL_MODEL - External Provider ({len(ext_eps)})

Routes to external providers (OpenAI, Anthropic, Google, Bedrock, etc.).

| Endpoint Name | Provider | Model Name | Task |
|---------------|----------|------------|------|
"""
        for ep in ext_sorted:
            name = ep["endpoint_name"]
            provider = ep["external_model"]["provider"]
            model = ep["external_model"]["name"]
            task = ep["task"]
            md += f"| {name} | {provider} | {model} | {task} |\n"
    
    # FM_EXTERNAL_MODEL_CUSTOM - External Custom Provider
    if ext_custom_eps:
        md += f"""
## FM_EXTERNAL_MODEL_CUSTOM - Custom Provider ({len(ext_custom_eps)})

Routes to user-specified custom URL/endpoint.

| Endpoint Name | Model Name | Task |
|---------------|------------|------|
"""
        for ep in ext_custom_sorted:
            name = ep["endpoint_name"]
            model = ep["external_model"]["name"]
            task = ep["task"]
            md += f"| {name} | {model} | {task} |\n"
    
    # AGENT_BRICKS_KA - Knowledge Assistants
    if ka_eps:
        md += f"""
## AGENT_BRICKS_KA - Knowledge Assistants ({len(ka_eps)})

Auto-generated endpoints for Knowledge Assistants created via Agent Bricks.

| Endpoint Name | Tile Model | Task | State |
|---------------|------------|------|-------|
"""
        for ep in ka_sorted[:20]:  # Limit to 20 examples
            name = ep["endpoint_name"]
            tile_model = ep["tile_metadata"]["tile_model_name"]
            task = ep["task"]
            state = ep["state"]["ready"]
            md += f"| {name} | {tile_model} | {task} | {state} |\n"
        if len(ka_eps) > 20:
            md += f"\n*...and {len(ka_eps) - 20} more*\n"
    
    # AGENT_BRICKS_MAS - Multi-Agent Supervisors
    if mas_eps:
        md += f"""
## AGENT_BRICKS_MAS - Multi-Agent Supervisors ({len(mas_eps)})

Auto-generated endpoints for Multi-Agent Supervisors created via Agent Bricks.

| Endpoint Name | Tile Model | Task | State |
|---------------|------------|------|-------|
"""
        for ep in mas_sorted[:20]:  # Limit to 20 examples
            name = ep["endpoint_name"]
            tile_model = ep["tile_metadata"]["tile_model_name"]
            task = ep["task"]
            state = ep["state"]["ready"]
            md += f"| {name} | {tile_model} | {task} | {state} |\n"
        if len(mas_eps) > 20:
            md += f"\n*...and {len(mas_eps) - 20} more*\n"
    
    # AGENT_BRICKS_KIE - Key Information Extraction
    if kie_eps:
        md += f"""
## AGENT_BRICKS_KIE - Key Information Extraction ({len(kie_eps)})

Auto-generated endpoints for Key Information Extraction created via Agent Bricks.

| Endpoint Name | Tile Model | Task | State |
|---------------|------------|------|-------|
"""
        for ep in kie_sorted[:20]:  # Limit to 20 examples
            name = ep["endpoint_name"]
            tile_model = ep["tile_metadata"]["tile_model_name"]
            task = ep["task"]
            state = ep["state"]["ready"]
            md += f"| {name} | {tile_model} | {task} | {state} |\n"
        if len(kie_eps) > 20:
            md += f"\n*...and {len(kie_eps) - 20} more*\n"
    
    # AGENT_BRICKS_MS - Model Specialization
    if ms_eps:
        md += f"""
## AGENT_BRICKS_MS - Model Specialization ({len(ms_eps)})

Endpoints for fine-tuned/specialized models created via Playground.

| Endpoint Name | Tile Model | Task | State |
|---------------|------------|------|-------|
"""
        for ep in ms_sorted[:20]:  # Limit to 20 examples
            name = ep["endpoint_name"]
            tile_model = ep["tile_metadata"]["tile_model_name"]
            task = ep["task"]
            state = ep["state"]["ready"]
            md += f"| {name} | {tile_model} | {task} | {state} |\n"
        if len(ms_eps) > 20:
            md += f"\n*...and {len(ms_eps) - 20} more*\n"
    
    md += """
---
*Generated by 03_extract_ai_endpoints_fast.py*
"""
    
    return md


# Generate and save Markdown
md_content = generate_markdown_summary(
    endpoints=endpoints,
    summary=summary,
    workspace=DATABRICKS_HOST,
    extraction_timestamp=extraction_start.isoformat(),
    duration_seconds=duration
)

md_timestamped = OUTPUT_DIR / f"databricks_fm_api_{timestamp}.md"
md_latest = OUTPUT_DIR / "databricks_fm_api_summary.md"

with open(md_timestamped, 'w') as f:
    f.write(md_content)
print(f"Saved: {md_timestamped}")

with open(md_latest, 'w') as f:
    f.write(md_content)
print(f"Saved: {md_latest}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done

# COMMAND ----------

print("\n" + "="*60)
print(" EXTRACTION COMPLETE")
print("="*60)
print(f"""
Extracted {len(endpoints)} classified endpoints in {duration:.2f} seconds (1 API call)

Output Files:
  JSON: {json_latest}
  Markdown: {md_latest}

By model_type:
  DATABRICKS_FM_PPT:          {summary['databricks_fm_ppt_count']}
  DATABRICKS_FM_PT:           {summary['databricks_fm_pt_count']}
  DATABRICKS_FM_UC_SYSTEM_AI: {summary['databricks_fm_uc_system_ai_count']}
  DATABRICKS_FM_UC_AGENTS:    {summary['databricks_fm_uc_agents_count']}
  DATABRICKS_CLASSIC_ML:      {summary['databricks_classic_ml_count']}
  AGENT_BRICKS_KA:            {summary['agent_bricks_ka_count']}
  AGENT_BRICKS_MAS:           {summary['agent_bricks_mas_count']}
  AGENT_BRICKS_KIE:           {summary['agent_bricks_kie_count']}
  AGENT_BRICKS_MS:            {summary['agent_bricks_ms_count']}
  FM_EXTERNAL_MODEL:          {summary['fm_external_model_count']}
  FM_EXTERNAL_MODEL_CUSTOM:   {summary['fm_external_model_custom_count']}
""")
