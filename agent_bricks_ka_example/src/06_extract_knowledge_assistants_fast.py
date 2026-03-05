# Databricks notebook source
# MAGIC %md
# MAGIC # Fast Knowledge Assistant Metadata Extractor
# MAGIC 
# MAGIC **Purpose:** Extract Knowledge Assistant metadata using a single API call.
# MAGIC 
# MAGIC ## API Reference
# MAGIC - **List KAs:** `GET /api/2.1/knowledge-assistants` ([ListKnowledgeAssistants](https://docs.databricks.com/api/workspace/knowledgeassistants/listknowledgeassistants))
# MAGIC 
# MAGIC ## Output
# MAGIC - `knowledge_assistants_fast_latest.json` - All KA metadata from list API

# COMMAND ----------

import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import get_workspace_client

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

w, config, auth_type = get_workspace_client()

DATABRICKS_HOST = config.get("DATABRICKS_HOST", "").rstrip("/")

def get_headers() -> Dict[str, str]:
    """Get authenticated headers."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers

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
    headers = get_headers()
    
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
