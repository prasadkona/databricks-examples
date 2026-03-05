# Databricks notebook source
# MAGIC %md
# MAGIC # Fast Knowledge Assistant Metadata Extractor
# MAGIC 
# MAGIC **Purpose:** Extract Knowledge Assistant metadata using a single API call.
# MAGIC 
# MAGIC ## API Reference
# MAGIC - **List KAs:** `GET /api/2.1/knowledge-assistants` ([ListKnowledgeAssistants](https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants))
# MAGIC 
# MAGIC **Note:** This API may not be available on all workspaces (returns 404 if not enabled).
# MAGIC 
# MAGIC ## Comparison with 04_extract_knowledge_assistants.py
# MAGIC | Aspect | 04 (Detailed) | 05 (Fast) |
# MAGIC |--------|---------------|-----------|
# MAGIC | API Calls | ListKnowledgeAssistants + GetKnowledgeAssistant per KA | ListKnowledgeAssistants only |
# MAGIC | Speed | ~40 seconds | ~2 seconds |
# MAGIC | Fields | Full details from GetKnowledgeAssistant | Fields from list response |
# MAGIC | Use Case | Full audit/analysis | Quick inventory |

# COMMAND ----------

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Environment file name (change to match your ../_local/{name}.env file)
ENV_NAME = "my-workspace"  # Change to match your ../_local/{name}.env file

def load_env_file(env_name: str) -> Dict[str, str]:
    """Load environment variables from ../_local/{env_name}.env file."""
    script_dir = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    env_path = script_dir.parent / "_local" / f"{env_name}.env"
    
    env_vars = {}
    if env_path.exists():
        print(f"Loading config from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    else:
        print(f"Warning: {env_path} not found, using environment variables")
    
    return env_vars

env_vars = load_env_file(ENV_NAME)

DATABRICKS_HOST = env_vars.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST", ""))
DATABRICKS_TOKEN = env_vars.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN", ""))
CLIENT_ID = env_vars.get("DATABRICKS_CLIENT_ID", os.getenv("DATABRICKS_CLIENT_ID", ""))
CLIENT_SECRET = env_vars.get("DATABRICKS_CLIENT_SECRET", os.getenv("DATABRICKS_CLIENT_SECRET", ""))

if DATABRICKS_HOST:
    DATABRICKS_HOST = DATABRICKS_HOST.rstrip("/")

print(f"Workspace: {DATABRICKS_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Authentication

# COMMAND ----------

def get_oauth_token(host: str, client_id: str, client_secret: str) -> Optional[str]:
    """Get OAuth token using M2M authentication."""
    try:
        response = requests.post(
            f"{host}/oidc/v1/token",
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            auth=(client_id, client_secret),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception as e:
        print(f"OAuth error: {e}")
    return None

def setup_auth() -> Dict[str, str]:
    """Setup authentication headers."""
    if CLIENT_ID and CLIENT_SECRET:
        token = get_oauth_token(DATABRICKS_HOST, CLIENT_ID, CLIENT_SECRET)
        if token:
            print("Auth: OAuth M2M")
            return {"Authorization": f"Bearer {token}"}
    
    if DATABRICKS_TOKEN:
        print("Auth: PAT")
        return {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    
    raise ValueError("No valid authentication found")

headers = setup_auth()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Knowledge Assistants (Single API Call)

# COMMAND ----------

def list_knowledge_assistants() -> List[Dict]:
    """
    List all Knowledge Assistants using Knowledge Assistants API.
    
    API: GET /api/2.1/knowledge-assistants
    Docs: https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants
    """
    url = f"{DATABRICKS_HOST}/api/2.1/knowledge-assistants"
    
    all_kas = []
    page_token = None
    
    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        
        response = requests.get(url, headers=headers, params=params, timeout=60)
        
        if response.status_code != 200:
            print(f"ERROR: Knowledge Assistants API returned {response.status_code}")
            print(f"  {response.text[:200]}")
            return []
        
        data = response.json()
        kas = data.get("knowledge_assistants", [])
        all_kas.extend(kas)
        
        page_token = data.get("next_page_token")
        if not page_token:
            break
    
    return all_kas

# COMMAND ----------

print("Fetching Knowledge Assistants (single API call)...")
start_time = datetime.now()

ka_list = list_knowledge_assistants()

elapsed = (datetime.now() - start_time).total_seconds()
print(f"Found {len(ka_list)} Knowledge Assistants in {elapsed:.1f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Format Records

# COMMAND ----------

knowledge_assistants = []

for ka in ka_list:
    ka_record = {
        "id": ka.get("id", ""),
        "name": ka.get("display_name", ""),
        "resource_name": ka.get("name", ""),
        "description": ka.get("description", ""),
        "instructions": ka.get("instructions", ""),
        "endpoint_name": ka.get("endpoint_name", ""),
        "creator": ka.get("creator", ""),
        "create_time": ka.get("create_time", ""),
        "experiment_id": ka.get("experiment_id", ""),
        "_extraction_metadata": {
            "extracted_at": datetime.now().isoformat(),
            "source_api": "knowledge-assistants",
            "api_version": "fast (list only)",
            "workspace": DATABRICKS_HOST
        },
        "_raw_response": ka
    }
    knowledge_assistants.append(ka_record)

print(f"Formatted {len(knowledge_assistants)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Output

# COMMAND ----------

script_dir = Path(__file__).parent if "__file__" in dir() else Path.cwd()
output_dir = script_dir.parent / "_local" / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = output_dir / "knowledge_assistants_fast_latest.json"
backup_file = output_dir / f"knowledge_assistants_fast_{timestamp}.json"

output_data = {
    "extraction_info": {
        "workspace": DATABRICKS_HOST,
        "extracted_at": datetime.now().isoformat(),
        "total_count": len(knowledge_assistants),
        "api_version": "fast (list only)",
        "elapsed_seconds": elapsed
    },
    "knowledge_assistants": knowledge_assistants
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2, default=str)
print(f"Saved to: {output_file}")

with open(backup_file, "w") as f:
    json.dump(output_data, f, indent=2, default=str)
print(f"Backup: {backup_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("KNOWLEDGE ASSISTANT EXTRACTION COMPLETE (FAST)")
print("=" * 60)
print(f"Total KAs: {len(knowledge_assistants)}")
print(f"Elapsed: {elapsed:.1f} seconds")
print(f"Output: {output_file}")
print("=" * 60)

# Show sample
print("\nSample Records:")
for ka in knowledge_assistants[:5]:
    print(f"\n{ka['name']}")
    print(f"  ID: {ka['id']}")
    print(f"  Endpoint: {ka['endpoint_name']}")
    print(f"  Creator: {ka['creator']}")
    desc = ka.get('description', '')
    print(f"  Description: {desc[:70]}..." if len(desc) > 70 else f"  Description: {desc}")
