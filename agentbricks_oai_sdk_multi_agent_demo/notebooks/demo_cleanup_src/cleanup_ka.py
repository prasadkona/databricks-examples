# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup: Delete Knowledge Assistant
# MAGIC
# MAGIC Deletes the Knowledge Assistant tile, either by `KA_TILE_ID` or
# MAGIC by searching for `KA_NAME`.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run demo-cleanup all   # calls this module
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `KA_TILE_ID` or `KA_NAME`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from __future__ import annotations
import os
import sys
from notebooks.agentbricks_ka_src.ka_manager import AgentBricksManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete KA

# COMMAND ----------

def run() -> int:
    tile_id = os.environ.get("KA_TILE_ID", "").strip()
    ka_name = os.environ.get("KA_NAME", "SEC_Financial_Analyst_KA")
    mgr = AgentBricksManager()
    print(f"Workspace: {mgr.w.config.host}")

    if not tile_id:
        print(f"KA_TILE_ID not set, searching by name: {ka_name}")
        found = mgr.find_by_name(AgentBricksManager.sanitize_name(ka_name))
        if found:
            tile_id = found["tile_id"]
            print(f"Found tile_id: {tile_id}")
        else:
            print(f"No KA named {ka_name!r} found, nothing to delete")
            return 0

    try:
        mgr.delete(tile_id)
        print(f"Deleted KA tile: {tile_id}")
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"KA tile {tile_id} already deleted or not found")
        else:
            print(f"ERROR deleting KA: {e}")
            return 1
    return 0
