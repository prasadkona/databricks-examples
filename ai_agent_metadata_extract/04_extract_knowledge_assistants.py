# Databricks notebook source
# MAGIC %md
# MAGIC # Knowledge Assistants Metadata Extraction
# MAGIC 
# MAGIC This notebook extracts detailed metadata for all Knowledge Assistants using the 
# MAGIC [Knowledge Assistants API](https://docs.databricks.com/api/workspace/knowledgeassistants).
# MAGIC 
# MAGIC ## Output
# MAGIC 
# MAGIC - `knowledge_assistants_latest.json` - Full KA metadata including name, description, instructions, knowledge_sources
# MAGIC 
# MAGIC ## API Reference
# MAGIC 
# MAGIC | Step | API | Docs |
# MAGIC |------|-----|------|
# MAGIC | 1 | `GET /api/2.1/knowledge-assistants` | [ListKnowledgeAssistants](https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants) |
# MAGIC | 2 | `GET /api/2.1/knowledge-assistants/{tile_id}` | [GetKnowledgeAssistant](https://docs.databricks.com/api/workspace/knowledgeassistants/getknowledgeassistant) |
# MAGIC 
# MAGIC 
# MAGIC **Note:** The ListKnowledgeAssistants API may not be available on all workspaces (returns 404).
# MAGIC If any API call fails, the error code and message are recorded in `_extraction_metadata`.

# COMMAND ----------

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

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
        for key in ['DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET']:
            os.environ.pop(key, None)
        os.environ['DATABRICKS_HOST'] = host
        print("Using PAT from environment")
        return host
    
    # Check config for OAuth or PAT
    if config.get('DATABRICKS_CLIENT_ID') and config.get('DATABRICKS_CLIENT_SECRET'):
        os.environ.pop('DATABRICKS_TOKEN', None)
        os.environ['DATABRICKS_CLIENT_ID'] = config['DATABRICKS_CLIENT_ID']
        os.environ['DATABRICKS_CLIENT_SECRET'] = config['DATABRICKS_CLIENT_SECRET']
        os.environ['DATABRICKS_HOST'] = host
        print("Using OAuth M2M from config")
    elif config.get('DATABRICKS_TOKEN'):
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
# MAGIC ## Knowledge Assistants API
# MAGIC 
# MAGIC Uses the [Knowledge Assistants API](https://docs.databricks.com/api/workspace/knowledgeassistants):
# MAGIC 
# MAGIC 1. **List**: `GET /api/2.1/knowledge-assistants` ([ListKnowledgeAssistants](https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants))
# MAGIC 2. **Get Details**: `GET /api/2.1/knowledge-assistants/{tile_id}` ([GetKnowledgeAssistant](https://docs.databricks.com/api/workspace/knowledgeassistants/getknowledgeassistant))

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()

def get_headers() -> Dict[str, str]:
    """Get authenticated headers."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers


def api_get(endpoint: str, params: Dict = None) -> Dict:
    """Make authenticated GET request."""
    url = f"{DATABRICKS_HOST}{endpoint}"
    headers = get_headers()
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def list_knowledge_assistants() -> List[Dict[str, Any]]:
    """
    List all Knowledge Assistants using the Knowledge Assistants API.
    
    API: GET /api/2.1/knowledge-assistants
    Docs: https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants
    """
    print("Listing KAs from /api/2.1/knowledge-assistants ...")
    
    all_kas = []
    page_token = None
    
    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        
        data = api_get("/api/2.1/knowledge-assistants", params=params)
        kas = data.get("knowledge_assistants", [])
        all_kas.extend(kas)
        
        page_token = data.get("next_page_token")
        if not page_token:
            break
    
    print(f"  Found {len(all_kas)} KAs")
    return all_kas


def get_knowledge_assistant(ka_id: str) -> tuple:
    """
    Get detailed info for a Knowledge Assistant using the Knowledge Assistants API.
    
    API: GET /api/2.1/knowledge-assistants/{ka_id}
    Docs: https://docs.databricks.com/api/workspace/knowledgeassistants/getknowledgeassistant
    
    Returns:
        tuple: (success: bool, data: dict, error_code: int or None, error_message: str or None)
    """
    url = f"{DATABRICKS_HOST}/api/2.1/knowledge-assistants/{ka_id}"
    headers = get_headers()
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return (True, result.get("knowledge_assistant", result), None, None)
    else:
        error_msg = response.text[:200] if response.text else "Unknown error"
        return (False, {}, response.status_code, error_msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract All Knowledge Assistants

# COMMAND ----------

# Step 1: List all KAs using Knowledge Assistants API
print("Step 1: Listing KAs...")
ka_list = list_knowledge_assistants()
print(f"Found {len(ka_list)} Knowledge Assistants")

# COMMAND ----------

# Step 2: Get detailed info for each KA using Knowledge Assistants API (parallel)
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_ka_details(ka: Dict) -> Dict:
    """Fetch KA details and return formatted record."""
    ka_id = ka.get("id", "")
    display_name = ka.get("display_name", "Unnamed")
    
    if not ka_id:
        return None
    
    success, ka_details, error_code, error_message = get_knowledge_assistant(ka_id)
    
    if success:
        return {
            "id": ka_details.get("id", ka_id),
            "name": ka_details.get("display_name", display_name),
            "resource_name": ka_details.get("name", ""),
            "description": ka_details.get("description", ""),
            "instructions": ka_details.get("instructions", ""),
            "endpoint_name": ka_details.get("endpoint_name", ""),
            "model_name": ka_details.get("model_name", ""),
            "knowledge_bases": ka_details.get("knowledge_bases", []),
            "sample_questions": ka_details.get("sample_questions", []),
            "create_time": ka_details.get("create_time", ""),
            "update_time": ka_details.get("update_time", ""),
            "creator": ka_details.get("creator", ""),
            "experiment_id": ka_details.get("experiment_id", ""),
            "_extraction_metadata": {
                "extracted_at": datetime.now().isoformat(),
                "source_api": "knowledge-assistants",
                "api_status": "success",
                "workspace": DATABRICKS_HOST
            },
            "_raw_ka_response": ka_details
        }
    else:
        return {
            "id": ka_id,
            "name": display_name,
            "_extraction_metadata": {
                "extracted_at": datetime.now().isoformat(),
                "source_api": "knowledge-assistants",
                "api_status": "error",
                "error_code": error_code,
                "error_message": error_message,
                "workspace": DATABRICKS_HOST
            }
        }

print(f"\nStep 2: Fetching KA details (parallel, 10 threads)...")
knowledge_assistants = []
success_count = 0
error_count = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_ka_details, ka): ka for ka in ka_list}
    
    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        if result:
            knowledge_assistants.append(result)
            if result.get("_extraction_metadata", {}).get("api_status") == "success":
                success_count += 1
            else:
                error_count += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(ka_list)}...")

print(f"\nExtracted {len(knowledge_assistants)} Knowledge Assistants")
print(f"  Success: {success_count}")
print(f"  Errors: {error_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Output

# COMMAND ----------

# Save to JSON
output_file = OUTPUT_DIR / "knowledge_assistants_latest.json"
with open(output_file, 'w') as f:
    json.dump(knowledge_assistants, f, indent=2, default=str)

print(f"Saved {len(knowledge_assistants)} Knowledge Assistants to: {output_file}")

# Also save timestamped version
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_file = OUTPUT_DIR / f"knowledge_assistants_{timestamp}.json"
with open(timestamped_file, 'w') as f:
    json.dump(knowledge_assistants, f, indent=2, default=str)

print(f"Saved timestamped backup to: {timestamped_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

if knowledge_assistants:
    print("\n" + "="*60)
    print("KNOWLEDGE ASSISTANTS SUMMARY")
    print("="*60)
    
    for ka in knowledge_assistants:
        ka_id = ka.get("id", "N/A")
        name = ka.get("name", "Unnamed")
        description = ka.get("description", "")
        if len(description) > 80:
            description = description[:80] + "..."
        endpoint_name = ka.get("endpoint_name", "N/A")
        
        # Get knowledge bases info
        knowledge_bases = ka.get("knowledge_bases", [])
        kb_count = len(knowledge_bases) if isinstance(knowledge_bases, list) else 0
        
        # Get sample questions
        sample_questions = ka.get("sample_questions", [])
        sq_count = len(sample_questions) if isinstance(sample_questions, list) else 0
        
        print(f"\n{name}")
        print(f"  ID: {ka_id}")
        print(f"  Endpoint: {endpoint_name}")
        print(f"  Knowledge Bases: {kb_count}")
        print(f"  Sample Questions: {sq_count}")
        if description:
            print(f"  Description: {description}")
    
    print("\n" + "="*60)
    print(f"Total: {len(knowledge_assistants)} Knowledge Assistants")
    print("="*60)
else:
    print("No Knowledge Assistants found in this workspace.")
