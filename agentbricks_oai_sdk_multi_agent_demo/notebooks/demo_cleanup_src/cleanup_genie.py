# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup: Delete Genie Space
# MAGIC
# MAGIC Deletes the Genie Space used by the agent for structured data queries.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run demo-cleanup after-ka   # calls this module
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `GENIE_SPACE_ID`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from __future__ import annotations

import os
import sys
import urllib.request

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete Genie Space Function
# MAGIC
# MAGIC Standalone function that can be imported by other modules (e.g., run_sequence.py).

# COMMAND ----------

def delete_genie_space(host: str, token: str, space_id: str) -> bool:
    """Delete a Genie Space via REST API.
    
    Args:
        host: Databricks workspace host (e.g., https://xxx.cloud.databricks.com).
        token: PAT token for authentication.
        space_id: Genie Space ID to delete.
        
    Returns:
        True if deleted (or already gone), False on error.
    """
    url = f"{host.rstrip('/')}/api/2.0/genie/spaces/{space_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return True  # Already deleted
        raise
    except Exception:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Entry Point

# COMMAND ----------

def run() -> int:
    """Delete the Genie Space specified by GENIE_SPACE_ID env var."""
    host = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    space_id = os.environ.get("GENIE_SPACE_ID", "").strip()

    if not host or not token:
        print("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN required", file=sys.stderr)
        return 1
    if not space_id:
        print("GENIE_SPACE_ID not set -- nothing to delete")
        return 0

    print(f"Workspace: {host}")
    print(f"Deleting Genie Space: {space_id}")

    try:
        delete_genie_space(host, token, space_id)
        print(f"Deleted Genie Space: {space_id}")
    except urllib.error.HTTPError as e:
        print(f"ERROR deleting Genie Space: {e.code} {e.reason}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0
