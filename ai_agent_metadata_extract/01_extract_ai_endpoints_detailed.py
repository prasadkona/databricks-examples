# Databricks notebook source
# MAGIC %md
# MAGIC # All Serving Endpoints Metadata Extractor
# MAGIC 
# MAGIC This notebook extracts comprehensive metadata about **ALL** serving endpoints
# MAGIC in your Databricks workspace, including:
# MAGIC 
# MAGIC | Endpoint Type | Description | Examples |
# MAGIC |---------------|-------------|----------|
# MAGIC | **FOUNDATION_MODEL_API** | Databricks-hosted LLMs | Llama, DBRX, GPT-OSS, Gemma |
# MAGIC | **EXTERNAL_MODEL** | External provider models via AI Gateway | OpenAI GPT-4, Claude, Gemini |
# MAGIC | **CUSTOM** | Custom MLflow models | PyFunc, Sklearn, Transformers |
# MAGIC | **AGENT** | Agent Framework endpoints | ChatAgent, Tool-calling agents |
# MAGIC | **FEATURE_SPEC** | Feature serving endpoints | Online features |
# MAGIC 
# MAGIC ## What This Notebook Extracts
# MAGIC 
# MAGIC For each endpoint:
# MAGIC 
# MAGIC | Category | Fields |
# MAGIC |----------|--------|
# MAGIC | **Identification** | name, id, endpoint_type, entity_type |
# MAGIC | **Model Info** | model_name, model_provider, model_version, task |
# MAGIC | **State** | ready_state, config_update_state |
# MAGIC | **Capacity** | scale_to_zero, workload_size, min/max throughput |
# MAGIC | **Timestamps** | created_at, updated_at |
# MAGIC | **AI Gateway** | rate_limits, usage_tracking, guardrails |
# MAGIC | **Tags** | All associated tags |
# MAGIC | **Tile Metadata** | tile_id, tile_name, tile_description, tile_instructions, problem_type (Agent Bricks KA/MAS/KIE) |
# MAGIC | **_metadata_derived** | model_type, is_system_ai, is_provisioned_throughput |
# MAGIC 
# MAGIC ## Usage
# MAGIC 
# MAGIC ```bash
# MAGIC # Run locally from the ai_agent_metadata_extract directory
# MAGIC cd ai_agent_metadata_extract
# MAGIC 
# MAGIC # Using PAT (recommended for testing)
# MAGIC DATABRICKS_TOKEN=dapi... python 01_extract_ai_endpoints_detailed.py
# MAGIC 
# MAGIC # Or uses config from ../_local/{ENV_NAME}.env
# MAGIC python 01_extract_ai_endpoints_detailed.py
# MAGIC ```
# MAGIC 
# MAGIC ## Output
# MAGIC 
# MAGIC JSON files are saved to `../_local/reports/`:
# MAGIC - `all_endpoints_YYYYMMDD_HHMMSS.json` - Timestamped (list of endpoints)
# MAGIC - `all_endpoints_latest.json` - Most recent (list of endpoints)
# MAGIC 
# MAGIC **Note:** JSON output is a flat list of endpoint objects (no summary/metadata wrapper).
# MAGIC Each endpoint includes `_metadata_derived.model_type` for classification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Ensure unbuffered output for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_NAME = "my-workspace"  # Change to match your ../_local/{name}.env file
OUTPUT_DIR = Path("../_local/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Environment: {ENV_NAME}")
print(f"Output directory: {OUTPUT_DIR.absolute()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Setup Authentication

# COMMAND ----------

def load_env_file(env_name: str = None) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    if env_name is None:
        env_name = ENV_NAME
    
    search_dirs = [Path("../_local"), Path("_local")]
    
    env_path = None
    for d in search_dirs:
        env_specific = d / f"{env_name}.env"
        if env_specific.exists():
            env_path = env_specific
            break
        generic = d / ".env"
        if generic.exists():
            env_path = generic
            break
    
    config = {}
    if env_path and env_path.exists():
        print(f"Loading config from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    else:
        print(f"Warning: Config file not found for environment '{env_name}'")
    
    return config


def setup_authentication(config: Dict[str, str]) -> Tuple[str, str]:
    """
    Set up Databricks authentication.
    Priority: PAT in env → PAT from config → OAuth from config
    
    Clears conflicting env vars to avoid SDK validation errors.
    """
    host = os.environ.get('DATABRICKS_HOST') or config.get('DATABRICKS_HOST', '')
    if host:
        os.environ['DATABRICKS_HOST'] = host
    
    # Clear profile to avoid conflicts (we set auth explicitly)
    os.environ.pop('DATABRICKS_CONFIG_PROFILE', None)
    
    def clear_all_auth_env():
        """Clear all auth-related env vars."""
        os.environ.pop('DATABRICKS_TOKEN', None)
        os.environ.pop('DATABRICKS_CLIENT_ID', None)
        os.environ.pop('DATABRICKS_CLIENT_SECRET', None)
    
    # Priority 1: PAT in environment (before we clear it)
    env_token = os.environ.get('DATABRICKS_TOKEN')
    if env_token:
        clear_all_auth_env()
        os.environ['DATABRICKS_TOKEN'] = env_token
        print("Using PAT from environment variable")
        return host, "pat"
    
    # Priority 2: PAT from config file
    token = config.get('DATABRICKS_TOKEN') or config.get('DATABRICKS_API_KEY')
    if token:
        clear_all_auth_env()
        os.environ['DATABRICKS_TOKEN'] = token
        print("Using PAT from config file")
        return host, "pat"
    
    # Priority 3: OAuth M2M from config file
    has_oauth = (
        config.get('DATABRICKS_CLIENT_ID') and 
        config.get('DATABRICKS_CLIENT_SECRET')
    )
    if has_oauth:
        clear_all_auth_env()
        os.environ['DATABRICKS_CLIENT_ID'] = config['DATABRICKS_CLIENT_ID']
        os.environ['DATABRICKS_CLIENT_SECRET'] = config['DATABRICKS_CLIENT_SECRET']
        print("Using OAuth M2M from config file")
        return host, "oauth"
    
    print("Warning: No authentication credentials found")
    return host, "none"


config = load_env_file()
DATABRICKS_HOST, AUTH_TYPE = setup_authentication(config)

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Auth type: {AUTH_TYPE.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Databricks Client

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## API Helper Functions

# COMMAND ----------

def get_api_headers() -> Dict[str, str]:
    """Get authenticated headers for REST API calls."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers


def api_get(path: str, params: Dict = None) -> Dict:
    """Make a GET request to the Databricks REST API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.get(url, headers=get_api_headers(), params=params or {}, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_all_tiles() -> Dict[str, Dict]:
    """
    Fetch all tiles (KA, MAS, KIE, etc.) from the /api/2.0/tiles API.
    
    Returns:
        Dictionary mapping serving_endpoint_name -> tile details
    """
    print("Fetching tiles from /api/2.0/tiles API...")
    
    tile_lookup = {}
    
    # Fetch each tile type
    for tile_type in ["KA", "MAS", "KIE", "MS"]:
        try:
            page_token = None
            while True:
                params = {"filter": f"tile_type={tile_type}", "page_size": 100}
                if page_token:
                    params["page_token"] = page_token
                
                response = api_get("/api/2.0/tiles", params=params)
                tiles = response.get("tiles", [])
                
                for tile in tiles:
                    endpoint_name = tile.get("serving_endpoint_name", "")
                    if endpoint_name:
                        tile_lookup[endpoint_name] = {
                            "tile_id": tile.get("tile_id", ""),
                            "tile_name": tile.get("name", ""),
                            "tile_type": tile.get("tile_type", ""),
                            "tile_description": tile.get("description", ""),
                            "tile_instructions": tile.get("instructions", ""),
                        }
                
                page_token = response.get("next_page_token")
                if not page_token:
                    break
                    
        except Exception as e:
            print(f"  Warning: Could not fetch {tile_type} tiles: {e}")
    
    print(f"  Found {len(tile_lookup)} tiles with endpoint mappings")
    return tile_lookup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Type Classification
# MAGIC 
# MAGIC This section classifies endpoints into categories based on their configuration.

# COMMAND ----------

def classify_endpoint(endpoint: Dict) -> Dict:
    """
    Classify an endpoint and extract its type information.
    
    Returns:
        Dictionary with:
        - endpoint_category: High-level category (foundation_model, external_model, custom, agent, etc.)
        - entity_type: The raw entity type from API
        - model_info: Extracted model details
    """
    config = endpoint.get("config", {})
    served_entities = config.get("served_entities", [])
    served_models = config.get("served_models", [])  # Legacy format
    
    # Default classification
    result = {
        "endpoint_category": "unknown",
        "entity_type": "unknown",
        "model_info": {
            "model_name": "",
            "model_provider": "",
            "model_version": "",
            "registered_model_name": "",
            "external_model_config": None,
        },
        "scale_to_zero": None,
        "workload_size": None,
    }
    
    # Check served_entities first (newer format)
    if served_entities:
        entity = served_entities[0]
        entity_type = entity.get("type", "")
        result["entity_type"] = entity_type
        
        # Extract scale and workload info
        result["scale_to_zero"] = entity.get("scale_to_zero_enabled")
        result["workload_size"] = entity.get("workload_size")
        
        # Classification based on entity type
        if entity_type == "EXTERNAL_MODEL":
            result["endpoint_category"] = "external_model"
            ext_model = entity.get("external_model", {})
            result["model_info"]["model_name"] = ext_model.get("name", "")
            result["model_info"]["model_provider"] = ext_model.get("provider", "")
            # Include raw config as-is from API
            result["model_info"]["external_model_config"] = ext_model
        
        elif entity_type in ["FOUNDATION_MODEL", "PT_FOUNDATION_MODEL"]:
            result["endpoint_category"] = "foundation_model"
            fm = entity.get("foundation_model", {})
            result["model_info"]["model_name"] = fm.get("name", "")
            result["model_info"]["model_provider"] = "databricks"
            result["model_info"]["model_version"] = fm.get("version", "")
            # Provisioned throughput info
            result["model_info"]["min_provisioned_throughput"] = fm.get("min_provisioned_throughput")
            result["model_info"]["max_provisioned_throughput"] = fm.get("max_provisioned_throughput")
        
        elif entity_type == "CUSTOM_MODEL":
            result["endpoint_category"] = "custom_model"
            result["model_info"]["model_name"] = entity.get("name", "")
            result["model_info"]["registered_model_name"] = entity.get("entity_name", "")
            result["model_info"]["model_version"] = entity.get("entity_version", "")
        
        elif entity_type == "FEATURE_SPEC":
            result["endpoint_category"] = "feature_serving"
            result["model_info"]["model_name"] = entity.get("name", "")
            result["model_info"]["registered_model_name"] = entity.get("entity_name", "")
        
        else:
            # Try to infer from other fields
            if entity.get("external_model"):
                result["endpoint_category"] = "external_model"
                ext_model = entity.get("external_model", {})
                result["model_info"]["model_name"] = ext_model.get("name", "")
                result["model_info"]["model_provider"] = ext_model.get("provider", "")
            elif entity.get("foundation_model"):
                result["endpoint_category"] = "foundation_model"
                fm = entity.get("foundation_model", {})
                result["model_info"]["model_name"] = fm.get("name", "")
                result["model_info"]["model_provider"] = "databricks"
            elif entity.get("entity_name"):
                result["endpoint_category"] = "custom_model"
                result["model_info"]["registered_model_name"] = entity.get("entity_name", "")
                result["model_info"]["model_version"] = entity.get("entity_version", "")
    
    # Check served_models (legacy format)
    elif served_models:
        model = served_models[0]
        result["endpoint_category"] = "custom_model"
        result["entity_type"] = "SERVED_MODEL"
        result["model_info"]["model_name"] = model.get("name", "")
        result["model_info"]["registered_model_name"] = model.get("model_name", "")
        result["model_info"]["model_version"] = model.get("model_version", "")
        result["scale_to_zero"] = model.get("scale_to_zero_enabled")
        result["workload_size"] = model.get("workload_size")
    
    # Check endpoint_type for additional classification
    endpoint_type = endpoint.get("endpoint_type", "")
    if endpoint_type == "FOUNDATION_MODEL_API" and result["endpoint_category"] == "unknown":
        result["endpoint_category"] = "foundation_model"
    
    # Check task for agent detection
    task = endpoint.get("task", "")
    if "agent" in task.lower():
        result["endpoint_category"] = "agent"
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract AI Gateway Configuration

# COMMAND ----------

def extract_ai_gateway_config(endpoint: Dict) -> Dict:
    """Extract AI Gateway configuration from an endpoint."""
    config = endpoint.get("config", {})
    ai_gateway = config.get("ai_gateway", {})
    
    inference_table = ai_gateway.get("inference_table_config", {})
    rate_limits = ai_gateway.get("rate_limits", [])
    guardrails = ai_gateway.get("guardrails", {})
    
    return {
        "usage_tracking_enabled": inference_table.get("enabled", False),
        "inference_table_catalog": inference_table.get("catalog_name"),
        "inference_table_schema": inference_table.get("schema_name"),
        "rate_limits_count": len(rate_limits),
        "rate_limits": rate_limits,
        "guardrails_enabled": bool(guardrails),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Extraction Function

# COMMAND ----------

def extract_endpoint_metadata(endpoint: Dict, tile_lookup: Dict[str, Dict] = None) -> Dict:
    """
    Extract comprehensive metadata from any serving endpoint.
    
    Args:
        endpoint: The endpoint data from the API
        tile_lookup: Optional dictionary mapping endpoint_name -> tile details
    """
    tile_lookup = tile_lookup or {}
    # Basic identification
    name = endpoint.get("name", "")
    endpoint_id = endpoint.get("id", "")
    endpoint_type = endpoint.get("endpoint_type", "")
    
    # State
    state = endpoint.get("state", {})
    ready_state = state.get("ready", "UNKNOWN")
    config_update_state = state.get("config_update", "UNKNOWN")
    
    # Timestamps
    creation_timestamp = endpoint.get("creation_timestamp")
    last_updated_timestamp = endpoint.get("last_updated_timestamp")
    
    created_at = None
    updated_at = None
    if creation_timestamp:
        created_at = datetime.fromtimestamp(creation_timestamp / 1000).isoformat()
    if last_updated_timestamp:
        updated_at = datetime.fromtimestamp(last_updated_timestamp / 1000).isoformat()
    
    # Task
    task = endpoint.get("task", "")
    
    # Creator
    creator = endpoint.get("creator", "")
    
    # Classification
    classification = classify_endpoint(endpoint)
    model_info = classification["model_info"]
    
    # AI Gateway
    ai_gateway = extract_ai_gateway_config(endpoint)
    
    # Tags
    tags = endpoint.get("tags", [])
    tags_dict = {tag.get("key"): tag.get("value") for tag in tags if tag.get("key")}
    
    # Extract tile_endpoint_metadata (Agent Bricks / Playground)
    tile_endpoint_meta = endpoint.get("tile_endpoint_metadata", {})
    tile_id = tile_endpoint_meta.get("tile_id", "")
    tile_model_name = tile_endpoint_meta.get("tile_model_name", "")
    tile_problem_type = tile_endpoint_meta.get("problem_type", "")
    
    # Enrich with tile details from /api/2.0/tiles API (has name, description, instructions)
    tile_details = tile_lookup.get(name, {})
    tile_name = tile_details.get("tile_name", "")
    tile_description = tile_details.get("tile_description", "")
    tile_instructions = tile_details.get("tile_instructions", "")
    
    # Use tile_id from tile_lookup if not in endpoint metadata
    if not tile_id and tile_details:
        tile_id = tile_details.get("tile_id", "")
    
    # Entity info from served_entities
    config = endpoint.get("config", {})
    served_entities = config.get("served_entities", [])
    entity_type = ""
    entity_name = ""
    if served_entities:
        entity = served_entities[0]
        entity_type = entity.get("type", "")
        entity_name = entity.get("entity_name", "")
    
    # Determine if system.ai backed
    is_system_ai = (
        entity_name.startswith("system.ai.") or
        model_info.get("model_name", "").startswith("system.ai.")
    )
    
    # Determine if provisioned throughput
    is_provisioned_throughput = entity_type == "PT_FOUNDATION_MODEL"
    
    # ==========================================================================
    # model_type Classification (matching 03_extract_ai_endpoints_fast.py logic)
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
        external_provider = model_info.get("model_provider", "")
        if external_provider == "custom":
            model_type = "FM_EXTERNAL_MODEL_CUSTOM"
        else:
            model_type = "FM_EXTERNAL_MODEL"
    elif entity_type == "FEATURE_SPEC":
        model_type = "FEATURE_SPEC"
    else:
        model_type = "UNCLASSIFIED"
    
    return {
        # Identification
        "endpoint_name": name,
        "endpoint_id": endpoint_id,
        "endpoint_type": endpoint_type,
        "endpoint_category": classification["endpoint_category"],
        "entity_type": classification["entity_type"],
        
        # Task
        "task": task,
        
        # State
        "ready_state": ready_state,
        "config_update_state": config_update_state,
        
        # Model information
        "model_name": model_info.get("model_name", ""),
        "model_provider": model_info.get("model_provider", ""),
        "model_version": model_info.get("model_version", ""),
        "registered_model_name": model_info.get("registered_model_name", ""),
        
        # External model details
        "external_model_config": model_info.get("external_model_config"),
        
        # Tile metadata (Agent Bricks / Playground) - enriched from /api/2.0/tiles
        "tile_metadata": {
            "tile_id": tile_id,
            "tile_name": tile_name,
            "tile_description": tile_description,
            "tile_instructions": tile_instructions,
            "tile_model_name": tile_model_name,
            "problem_type": tile_problem_type,
        } if (tile_problem_type or tile_name) else None,
        
        # Capacity
        "scale_to_zero": classification.get("scale_to_zero"),
        "workload_size": classification.get("workload_size"),
        "min_provisioned_throughput": model_info.get("min_provisioned_throughput"),
        "max_provisioned_throughput": model_info.get("max_provisioned_throughput"),
        
        # Ownership
        "creator": creator,
        
        # Timestamps
        "created_at": created_at,
        "updated_at": updated_at,
        "creation_timestamp": creation_timestamp,
        "last_updated_timestamp": last_updated_timestamp,
        
        # AI Gateway
        "ai_gateway": ai_gateway,
        
        # Tags
        "tags": tags_dict,
        
        # Derived metadata (computed by this script)
        "_metadata_derived": {
            "model_type": model_type,
            "is_system_ai": is_system_ai,
            "is_provisioned_throughput": is_provisioned_throughput,
        },
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## List and Extract All Endpoints

# COMMAND ----------

def list_all_endpoints() -> List[Dict]:
    """List all serving endpoints with full details."""
    print("Listing all serving endpoints...")
    
    response = api_get("/api/2.0/serving-endpoints")
    endpoints_list = response.get("endpoints", [])
    
    print(f"Found {len(endpoints_list)} total endpoints")
    
    detailed_endpoints = []
    
    for i, ep in enumerate(endpoints_list):
        name = ep.get("name", "")
        
        if (i + 1) % 50 == 0:
            print(f"  Fetching details: {i + 1}/{len(endpoints_list)}")
        
        try:
            details = api_get(f"/api/2.0/serving-endpoints/{name}")
            detailed_endpoints.append(details)
        except Exception as e:
            print(f"  Warning: Could not fetch details for {name}: {e}")
            detailed_endpoints.append(ep)
    
    return detailed_endpoints


def extract_all_endpoints() -> Tuple[List[Dict], Dict]:
    """
    Main extraction function for all endpoints.
    
    Returns:
        Tuple of (endpoints list, summary dict)
        The JSON output will only contain the endpoints list.
    """
    extraction_start = datetime.now()
    print(f"\nStarting extraction at {extraction_start.isoformat()}")
    print("=" * 60)
    
    # Fetch tiles first (for KA/MAS/KIE name, description, instructions)
    tile_lookup = fetch_all_tiles()
    
    # Get all endpoints
    all_endpoints = list_all_endpoints()
    
    # Extract metadata for each
    print("\nExtracting metadata...")
    extraction_timestamp = extraction_start.isoformat()
    extracted = []
    
    for endpoint in all_endpoints:
        metadata = extract_endpoint_metadata(endpoint, tile_lookup)
        # Add extraction timestamp to _metadata_derived
        metadata["_metadata_derived"]["extraction_timestamp"] = extraction_timestamp
        extracted.append(metadata)
    
    print(f"Extracted metadata for {len(extracted)} endpoints")
    
    # Generate summary (for display only, not saved to JSON)
    summary = generate_summary(extracted)
    
    extraction_end = datetime.now()
    duration = (extraction_end - extraction_start).total_seconds()
    
    # Add extraction metadata to summary for display
    summary["extraction_timestamp"] = extraction_start.isoformat()
    summary["workspace"] = DATABRICKS_HOST
    summary["auth_type"] = AUTH_TYPE
    summary["extraction_duration_seconds"] = duration
    
    print(f"\nExtraction completed in {duration:.1f} seconds")
    
    return extracted, summary


def generate_summary(endpoints: List[Dict]) -> Dict:
    """Generate comprehensive summary statistics."""
    
    # By model_type (from _metadata_derived)
    by_model_type = {}
    for ep in endpoints:
        mt = ep.get("_metadata_derived", {}).get("model_type", "UNCLASSIFIED")
        by_model_type[mt] = by_model_type.get(mt, 0) + 1
    
    # By endpoint category (legacy)
    by_category = {}
    for ep in endpoints:
        cat = ep.get("endpoint_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    # By endpoint type (raw API field)
    by_endpoint_type = {}
    for ep in endpoints:
        et = ep.get("endpoint_type", "unknown") or "unknown"
        by_endpoint_type[et] = by_endpoint_type.get(et, 0) + 1
    
    # By entity type
    by_entity_type = {}
    for ep in endpoints:
        et = ep.get("entity_type", "unknown")
        by_entity_type[et] = by_entity_type.get(et, 0) + 1
    
    # By ready state
    by_state = {}
    for ep in endpoints:
        state = ep.get("ready_state", "UNKNOWN")
        by_state[state] = by_state.get(state, 0) + 1
    
    # By task
    by_task = {}
    for ep in endpoints:
        task = ep.get("task", "unknown") or "unknown"
        by_task[task] = by_task.get(task, 0) + 1
    
    # By provider (for external models)
    by_provider = {}
    for ep in endpoints:
        provider = ep.get("model_provider", "") or "unknown"
        if provider:
            by_provider[provider] = by_provider.get(provider, 0) + 1
    
    # By model name (top models)
    by_model = {}
    for ep in endpoints:
        model = ep.get("model_name", "") or ""
        if model:
            # Normalize version suffixes
            if model and len(model) > 2 and model[-2] == "-" and model[-1].isdigit():
                model = model[:-2]
            by_model[model] = by_model.get(model, 0) + 1
    
    # Usage tracking stats
    usage_tracking_enabled = sum(
        1 for ep in endpoints 
        if ep.get("ai_gateway", {}).get("usage_tracking_enabled")
    )
    
    # Scale to zero stats
    scale_to_zero_enabled = sum(
        1 for ep in endpoints 
        if ep.get("scale_to_zero") is True
    )
    
    return {
        "total_count": len(endpoints),
        "by_model_type": dict(sorted(by_model_type.items(), key=lambda x: -x[1])),
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "by_endpoint_type": dict(sorted(by_endpoint_type.items(), key=lambda x: -x[1])),
        "by_entity_type": dict(sorted(by_entity_type.items(), key=lambda x: -x[1])),
        "by_ready_state": by_state,
        "by_task": dict(sorted(by_task.items(), key=lambda x: -x[1])),
        "by_provider": dict(sorted(by_provider.items(), key=lambda x: -x[1])),
        "by_model": dict(sorted(by_model.items(), key=lambda x: -x[1])[:20]),
        "usage_tracking_enabled": usage_tracking_enabled,
        "scale_to_zero_enabled": scale_to_zero_enabled,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Extraction

# COMMAND ----------

endpoints, summary = extract_all_endpoints()

# Display summary
print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

print(f"\nWorkspace: {summary['workspace']}")
print(f"Total endpoints: {summary['total_count']}")
print(f"Duration: {summary['extraction_duration_seconds']:.1f}s")

print(f"\n--- By model_type ---")
for mt, count in summary.get("by_model_type", {}).items():
    print(f"  {mt}: {count}")

print(f"\n--- By Entity Type ---")
for et, count in list(summary.get("by_entity_type", {}).items())[:10]:
    print(f"  {et}: {count}")

print(f"\n--- By Task ---")
for task, count in list(summary.get("by_task", {}).items())[:10]:
    print(f"  {task}: {count}")

print(f"\n--- By Provider ---")
for provider, count in summary.get("by_provider", {}).items():
    print(f"  {provider}: {count}")

print(f"\n--- Top Models ---")
for model, count in list(summary.get("by_model", {}).items())[:15]:
    print(f"  {model}: {count}")

print(f"\nUsage tracking enabled: {summary.get('usage_tracking_enabled', 0)}")
print(f"Scale to zero enabled: {summary.get('scale_to_zero_enabled', 0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

def save_endpoints(endpoints: List[Dict], output_dir: Path) -> Tuple[Path, Path]:
    """
    Save the endpoints list to JSON files.
    JSON contains only the endpoint data (no summary/metadata wrapper).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    timestamped_filename = f"all_endpoints_{timestamp}.json"
    timestamped_path = output_dir / timestamped_filename
    
    latest_filename = "all_endpoints_latest.json"
    latest_path = output_dir / latest_filename
    
    # Save only the endpoints list (no wrapper)
    with open(timestamped_path, 'w') as f:
        json.dump(endpoints, f, indent=2, default=str)
    print(f"\nSaved timestamped report: {timestamped_path}")
    
    with open(latest_path, 'w') as f:
        json.dump(endpoints, f, indent=2, default=str)
    print(f"Saved latest report: {latest_path}")
    
    return timestamped_path, latest_path


timestamped_path, latest_path = save_endpoints(endpoints, OUTPUT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Print Sample Endpoints by Category

# COMMAND ----------

print("\n" + "=" * 60)
print("SAMPLE ENDPOINTS BY model_type")
print("=" * 60)

by_model_type = {}
for ep in endpoints:
    mt = ep.get("_metadata_derived", {}).get("model_type", "UNCLASSIFIED")
    if mt not in by_model_type:
        by_model_type[mt] = []
    if len(by_model_type[mt]) < 3:
        by_model_type[mt].append(ep)

for mt, eps in sorted(by_model_type.items()):
    print(f"\n--- {mt} ---")
    for ep in eps:
        print(f"  Endpoint: {ep['endpoint_name']}")
        print(f"    Provider: {ep['model_provider']}")
        print(f"    Model: {ep['model_name']}")
        print(f"    Task: {ep['task']}")
        print(f"    State: {ep['ready_state']}")
        # Show tile metadata for Agent Bricks
        tile_meta = ep.get('tile_metadata')
        if tile_meta:
            print(f"    Tile Name: {tile_meta.get('tile_name', 'N/A')}")
            desc = tile_meta.get('tile_description', '')
            if desc:
                print(f"    Description: {desc[:80]}..." if len(desc) > 80 else f"    Description: {desc}")
            print(f"    Problem Type: {tile_meta.get('problem_type', 'N/A')}")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extraction Complete
# MAGIC 
# MAGIC Files saved to `../_local/reports/`:
# MAGIC - Timestamped JSON for historical tracking
# MAGIC - `all_endpoints_latest.json` for easy access
# MAGIC 
# MAGIC Use the JSON to filter and determine which endpoints to include in your final report.
