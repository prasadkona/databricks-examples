# Databricks notebook source
# MAGIC %md
# MAGIC # Step 01: Create Knowledge Assistant
# MAGIC
# MAGIC Creates a Knowledge Assistant, waits for the endpoint to come ONLINE,
# MAGIC adds example questions, and writes `KA_TILE_ID` / `KA_ENDPOINT` to the
# MAGIC central config file.
# MAGIC
# MAGIC **Re-run safe:** If a KA with the same name already exists and is ONLINE,
# MAGIC it skips creation and ensures examples are added. If still provisioning,
# MAGIC it resumes waiting. Only deletes and recreates if in an unexpected state.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run create-ka
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `UC_CATALOG`, `UC_SCHEMA`, `UC_VOLUME`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Bootstrap

# COMMAND ----------

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from notebooks.demo_shared import bootstrap, update_central_config

_project_root, _central_config = bootstrap(__file__)

from .ka_manager import AgentBricksManager
from .ka_config import (
    KA_NAME, KA_TILE_ID, KA_DESCRIPTION, KA_INSTRUCTIONS,
    EXAMPLE_QUESTIONS, VOLUME_DATASET_FOLDER,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Create KA Lifecycle
# MAGIC
# MAGIC 1. Search for existing KA by name
# MAGIC    - If ONLINE: reuse it (skip to examples)
# MAGIC    - If still provisioning: resume waiting
# MAGIC    - If not found: create from scratch
# MAGIC 2. Build knowledge sources from UC Volume (if creating new)
# MAGIC 3. Create KA via REST API (if creating new)
# MAGIC 4. Wait for endpoint ONLINE
# MAGIC 5. Add example questions
# MAGIC 6. Write IDs to central config

# COMMAND ----------

def main() -> int:
    catalog = os.getenv("UC_CATALOG", "your_catalog")
    schema = os.getenv("UC_SCHEMA", "your_schema")
    volume = os.getenv("UC_VOLUME", "ka_demo")
    volume_path = f"/Volumes/{catalog}/{schema}/{volume}/{VOLUME_DATASET_FOLDER}"

    mgr = AgentBricksManager()
    print(f"Workspace: {mgr.w.config.host}")
    print(f"Volume path: {volume_path}")

    sanitized_name = AgentBricksManager.sanitize_name(KA_NAME)
    tile_id = None
    endpoint_name = None
    need_create = True

    existing = mgr.find_by_name(sanitized_name)
    if existing:
        tile_id = existing["tile_id"]
        endpoint_name = f"ka-{tile_id.split('-')[0]}-endpoint"
        status = mgr.ka_get_endpoint_status(tile_id)
        print(f"\nExisting KA found: {tile_id}  (status: {status})")

        if status == "ONLINE":
            print("KA already ONLINE -- skipping creation, will ensure examples are added")
            need_create = False
        elif status in ("NOT_READY", "PROVISIONING", None):
            print("KA exists but endpoint is still provisioning -- resuming wait")
            need_create = False
        else:
            print(f"KA in unexpected state ({status}) -- deleting and recreating")
            mgr.delete(tile_id)
            tile_id = None
            endpoint_name = None
    else:
        print(f"\nNo existing KA named '{sanitized_name}' found")

    if need_create:
        knowledge_sources = AgentBricksManager.get_knowledge_sources_from_volumes(
            [(volume_path, None)]
        )

        print(f"\n{'=' * 60}")
        print(f"Creating Knowledge Assistant: {KA_NAME}")
        print(f"{'=' * 60}")
        result = mgr.ka_create(
            name=KA_NAME,
            knowledge_sources=knowledge_sources,
            description=KA_DESCRIPTION,
            instructions=KA_INSTRUCTIONS,
        )

        tile_id = result.get("knowledge_assistant", {}).get("tile", {}).get("tile_id")
        if not tile_id:
            print(f"ERROR: No tile_id in response:\n{json.dumps(result, indent=2)}")
            return 1

        endpoint_name = f"ka-{tile_id.split('-')[0]}-endpoint"
        print(f"\nTile ID:  {tile_id}")
        print(f"Endpoint: {endpoint_name}")

    update_central_config(_central_config, "KA_TILE_ID", tile_id)
    update_central_config(_central_config, "KA_ENDPOINT", endpoint_name)
    update_central_config(_central_config, "KA_NAME", sanitized_name)
    print(f"\nUpdated central config: {_central_config}")

    print(f"\nWaiting for endpoint to be ONLINE (timeout 600s)...")
    ka = mgr.ka_wait_until_endpoint_online(tile_id, timeout_s=600)
    status = ka.get("knowledge_assistant", {}).get("status", {}).get("endpoint_status")
    print(f"\nFinal endpoint status: {status}")

    examples_added = 0
    if status == "ONLINE":
        print(f"\nAdding {len(EXAMPLE_QUESTIONS)} example questions...")
        created = mgr.ka_add_examples_batch(tile_id, EXAMPLE_QUESTIONS)
        examples_added = len(created)
        print(f"Added {examples_added} examples")
    else:
        print(f"\nEndpoint not ONLINE yet (status: {status}).")
        print(f"Attempting to add examples anyway...")
        try:
            created = mgr.ka_add_examples_batch(tile_id, EXAMPLE_QUESTIONS)
            examples_added = len(created)
            print(f"Added {examples_added} examples")
        except Exception as e:
            print(f"Could not add examples while endpoint is provisioning: {e}")
            print(f"Examples can be added later once the endpoint is ONLINE.")
            print(f"Run:  uv run create-ka   (will re-run and add examples)")

    print(f"\n{'=' * 60}")
    if status == "ONLINE":
        print(f"KA Created Successfully")
    else:
        print(f"KA Created -- Endpoint Still Provisioning")
    print(f"  Name:     {KA_NAME}")
    print(f"  Tile ID:  {tile_id}")
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Status:   {status}")
    if examples_added:
        print(f"  Examples: {examples_added}")
    else:
        print(f"  Examples: pending (add once ONLINE)")
    print(f"{'=' * 60}")

    return 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point

# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
