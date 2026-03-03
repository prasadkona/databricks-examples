# Databricks notebook source
# MAGIC %md
# MAGIC # Sync Knowledge Assistant Sources
# MAGIC 
# MAGIC This script triggers a re-sync of knowledge sources for a Knowledge Assistant.
# MAGIC Use this after adding, modifying, or deleting files in the UC volume.
# MAGIC 
# MAGIC ## Usage
# MAGIC ```bash
# MAGIC cd agent_bricks_ka_example/src
# MAGIC 
# MAGIC # Sync KA 01
# MAGIC python 04_sync_ka_sources.py 1
# MAGIC 
# MAGIC # Sync KA 02
# MAGIC python 04_sync_ka_sources.py 2
# MAGIC 
# MAGIC # Sync by KA name
# MAGIC python 04_sync_ka_sources.py SEC_Financial_Analyst
# MAGIC ```
# MAGIC 
# MAGIC ## Authentication
# MAGIC - **OAuth M2M** (default): Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
# MAGIC - **PAT** (fallback): Uses DATABRICKS_TOKEN if OAuth not configured

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import sys
import time
import requests

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from config import load_env_file, setup_databricks_auth

# Load configuration and set up authentication
config = load_env_file()
config, auth_type = setup_databricks_auth(config)

DATABRICKS_HOST = config.get('DATABRICKS_HOST')

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Auth: {auth_type.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## KA Selection

# COMMAND ----------

# KA configurations from env
KA_CONFIG = {
    "1": {
        "name": config.get("KA_NAME_01", "SEC_Financial_Analyst"),
        "tile_id": config.get("KA_TILE_ID_01", ""),
    },
    "2": {
        "name": config.get("KA_NAME_02", "SEC_Financial_Analyst_02"),
        "tile_id": config.get("KA_TILE_ID_02", ""),
    },
}

def get_ka_config(arg: str) -> dict:
    """Get KA config from argument."""
    if arg in KA_CONFIG:
        return KA_CONFIG[arg]
    for key, ka in KA_CONFIG.items():
        if ka['name'] == arg:
            return ka
    return {"name": arg, "tile_id": ""}

# Get argument or default to "1"
ka_arg = sys.argv[1] if len(sys.argv) > 1 else "1"

ka_config = get_ka_config(ka_arg)
KA_NAME = ka_config['name']
KA_TILE_ID = ka_config['tile_id']

if not KA_TILE_ID:
    print(f"Error: No tile_id found for KA '{KA_NAME}'")
    print("\nConfigured KAs:")
    for key, ka in KA_CONFIG.items():
        print(f"  [{key}] {ka['name']} (tile_id: {ka['tile_id'] or 'NOT SET'})")
    sys.exit(1)

print(f"\nKA Name: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_headers():
    """Get authenticated headers for REST API calls."""
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers

def api_get(path: str) -> dict:
    """Make a GET request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.get(url, headers=get_headers(), timeout=30)
    response.raise_for_status()
    return response.json()

def api_post(path: str, body: dict = None) -> dict:
    """Make a POST request to Databricks API."""
    url = f"{DATABRICKS_HOST}{path}"
    response = requests.post(url, headers=get_headers(), json=body or {}, timeout=60)
    if response.status_code >= 400:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()
    return response.json() if response.text else {}

def get_ka_status(tile_id: str) -> dict:
    """Get KA status including knowledge source state."""
    result = api_get(f"/api/2.0/knowledge-assistants/{tile_id}")
    return result.get('knowledge_assistant', {})

def sync_ka_sources(tile_id: str) -> None:
    """Trigger re-indexing of KA knowledge sources."""
    api_post(f"/api/2.0/knowledge-assistants/{tile_id}/sync-knowledge-sources")
    print("Sync triggered successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Current Status

# COMMAND ----------

print("=" * 60)
print("CURRENT KA STATUS")
print("=" * 60)

ka = get_ka_status(KA_TILE_ID)

# Endpoint status
endpoint_status = ka.get('status', {}).get('endpoint_status', 'UNKNOWN')
print(f"\nEndpoint Status: {endpoint_status}")

# Knowledge source status
sources = ka.get('knowledge_sources', [])
if sources:
    for i, source in enumerate(sources):
        state = source.get('state', 'UNKNOWN')
        file_info = source.get('file_source_index_info', {}).get('summary', {})
        print(f"\nKnowledge Source {i+1}:")
        print(f"  State: {state}")
        print(f"  Total files: {file_info.get('total_files', 0)}")
        print(f"  Success: {file_info.get('success_files', 0)}")
        print(f"  Skipped: {file_info.get('skipped_files', 0)}")
        print(f"  Failed: {file_info.get('failed_files', 0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trigger Sync

# COMMAND ----------

print("\n" + "=" * 60)
print("TRIGGERING SYNC")
print("=" * 60)

sync_ka_sources(KA_TILE_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Sync Progress

# COMMAND ----------

print("\n" + "=" * 60)
print("MONITORING SYNC PROGRESS")
print("=" * 60)

max_wait = 120  # 2 minutes
poll_interval = 10
elapsed = 0

while elapsed < max_wait:
    ka = get_ka_status(KA_TILE_ID)
    sources = ka.get('knowledge_sources', [])
    
    if sources:
        state = sources[0].get('state', 'UNKNOWN')
        print(f"\n[{elapsed}s] Knowledge Source State: {state}")
        
        if state == 'KNOWLEDGE_SOURCE_STATE_UPDATED':
            print("\nSync complete!")
            file_info = sources[0].get('file_source_index_info', {}).get('summary', {})
            print(f"  Total files: {file_info.get('total_files', 0)}")
            print(f"  Success: {file_info.get('success_files', 0)}")
            print(f"  Skipped: {file_info.get('skipped_files', 0)}")
            print(f"  Failed: {file_info.get('failed_files', 0)}")
            break
        elif 'ERROR' in state or 'FAILED' in state:
            print(f"\nSync failed with state: {state}")
            break
    
    time.sleep(poll_interval)
    elapsed += poll_interval

if elapsed >= max_wait:
    print(f"\nTimeout after {max_wait}s - sync may still be in progress")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("SYNC SUMMARY")
print("=" * 60)
print(f"\nKA Name: {KA_NAME}")
print(f"Tile ID: {KA_TILE_ID}")
print(f"\nSync triggered and monitored. Check status above.")
