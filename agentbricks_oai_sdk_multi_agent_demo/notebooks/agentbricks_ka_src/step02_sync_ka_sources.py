# Databricks notebook source
# MAGIC %md
# MAGIC # Step 02: Sync KA Knowledge Sources
# MAGIC
# MAGIC Triggers re-indexing of the KA's knowledge sources and monitors the sync
# MAGIC progress until completion or failure.
# MAGIC
# MAGIC Run this after adding, modifying, or deleting files in the UC Volume.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run sync-ka
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `KA_TILE_ID`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Bootstrap

# COMMAND ----------

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from notebooks.demo_shared import bootstrap

_project_root, _central_config = bootstrap(__file__)

from .ka_manager import AgentBricksManager
from .ka_config import KA_TILE_ID

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Sync Knowledge Sources
# MAGIC
# MAGIC 1. Retrieve current KA status and show knowledge source summary
# MAGIC 2. Trigger sync via REST API
# MAGIC 3. Poll for completion (up to 120s)

# COMMAND ----------

def main() -> int:
    tile_id = os.getenv("KA_TILE_ID", KA_TILE_ID)
    if not tile_id:
        print("ERROR: KA_TILE_ID not set. Run create-ka first.", file=sys.stderr)
        return 1

    mgr = AgentBricksManager()
    print(f"Workspace: {mgr.w.config.host}")
    print(f"KA Tile ID: {tile_id}")

    print(f"\n{'=' * 60}")
    print("Current KA Status")
    print(f"{'=' * 60}")
    ka = mgr.ka_get(tile_id)
    if not ka:
        print(f"ERROR: KA not found: {tile_id}")
        return 1

    ka_inner = ka.get("knowledge_assistant", {})
    endpoint_status = ka_inner.get("status", {}).get("endpoint_status", "UNKNOWN")
    print(f"Endpoint status: {endpoint_status}")

    sources = ka_inner.get("knowledge_sources", [])
    for i, src in enumerate(sources):
        state = src.get("state", "UNKNOWN")
        summary = src.get("file_source_index_info", {}).get("summary", {})
        print(f"\nKnowledge Source {i + 1}:")
        print(f"  State: {state}")
        print(f"  Total files: {summary.get('total_files', 0)}")
        print(f"  Success: {summary.get('success_files', 0)}")
        print(f"  Skipped: {summary.get('skipped_files', 0)}")
        print(f"  Failed: {summary.get('failed_files', 0)}")

    # Trigger sync
    print(f"\n{'=' * 60}")
    print("Triggering Sync")
    print(f"{'=' * 60}")
    mgr.ka_sync_sources(tile_id)

    # Monitor
    print(f"\n{'=' * 60}")
    print("Monitoring Sync Progress")
    print(f"{'=' * 60}")
    max_wait, poll = 120, 10
    elapsed = 0
    while elapsed < max_wait:
        ka = mgr.ka_get(tile_id)
        sources = ka.get("knowledge_assistant", {}).get("knowledge_sources", [])
        if sources:
            state = sources[0].get("state", "UNKNOWN")
            print(f"[{elapsed}s] State: {state}")
            if state == "KNOWLEDGE_SOURCE_STATE_UPDATED":
                summary = sources[0].get("file_source_index_info", {}).get("summary", {})
                print(f"\nSync complete!")
                print(f"  Total: {summary.get('total_files', 0)}")
                print(f"  Success: {summary.get('success_files', 0)}")
                print(f"  Failed: {summary.get('failed_files', 0)}")
                return 0
            if "ERROR" in state or "FAILED" in state:
                print(f"\nSync failed: {state}")
                return 1
        time.sleep(poll)
        elapsed += poll

    print(f"\nTimeout after {max_wait}s -- sync may still be in progress")
    return 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point

# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
