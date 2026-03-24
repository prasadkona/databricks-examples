# Databricks notebook source
# MAGIC %md
# MAGIC # Step 03: Test KA Endpoint
# MAGIC
# MAGIC Runs a multi-turn conversation test against the deployed Knowledge Assistant
# MAGIC endpoint to verify functionality and context retention.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run test-ka
# MAGIC uv run test-ka --endpoint ka-abc123-endpoint   # override endpoint name
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `KA_TILE_ID` (or `--endpoint`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Bootstrap

# COMMAND ----------

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from notebooks.demo_shared import bootstrap

_project_root, _central_config = bootstrap(__file__)

from .ka_manager import AgentBricksManager
from .ka_config import KA_TILE_ID, KA_ENDPOINT

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Questions
# MAGIC
# MAGIC Multi-turn questions designed to test both factual recall and
# MAGIC context retention across turns.

# COMMAND ----------

QUESTIONS = [
    "What documents do you have access to? List the companies and fiscal years.",
    "For the companies you mentioned, compare their main business segments. What are the key differences?",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper

# COMMAND ----------

def _endpoint_from_tile(tile_id: str) -> str:
    return f"ka-{tile_id.split('-')[0]}-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Multi-Turn Conversation Test
# MAGIC
# MAGIC 1. Resolve the KA endpoint from config or CLI arg
# MAGIC 2. Verify endpoint status (if `KA_TILE_ID` is available)
# MAGIC 3. Send questions sequentially, accumulating conversation context
# MAGIC 4. Print a summary of passed/failed turns

# COMMAND ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="Test KA with multi-turn conversation")
    parser.add_argument("--endpoint", default=None, help="Override KA endpoint name")
    args = parser.parse_args()

    tile_id = os.getenv("KA_TILE_ID", KA_TILE_ID)
    endpoint = args.endpoint or os.getenv("KA_ENDPOINT", KA_ENDPOINT)
    if not endpoint and tile_id:
        endpoint = _endpoint_from_tile(tile_id)
    if not endpoint:
        print("ERROR: KA_ENDPOINT not set and no KA_TILE_ID. Run create-ka first.", file=sys.stderr)
        return 1

    mgr = AgentBricksManager()
    print(f"Workspace: {mgr.w.config.host}")
    print(f"Endpoint:  {endpoint}")

    if tile_id:
        status = mgr.ka_get_endpoint_status(tile_id)
        print(f"Status:    {status}")
        if status != "ONLINE":
            print("WARNING: Endpoint is not ONLINE -- queries may fail")

    conversation: list[dict[str, str]] = []
    passed = 0

    print(f"\n{'=' * 70}")
    print("MULTI-TURN CONVERSATION TEST")
    print(f"{'=' * 70}")

    for turn, question in enumerate(QUESTIONS, 1):
        print(f"\n--- Turn {turn} ---")
        print(f"USER: {question}")
        print("-" * 60)

        conversation.append({"role": "user", "content": question})

        try:
            response = mgr.ka_query(endpoint, conversation)
        except Exception as e:
            print(f"ERROR: {e}")
            conversation.pop()
            continue

        answer = AgentBricksManager.extract_response_text(response)
        if not answer:
            print("WARNING: Empty response from KA")
            conversation.pop()
            continue

        print(f"ASSISTANT:\n{answer}\n")
        conversation.append({"role": "assistant", "content": answer})
        passed += 1

    print(f"\n{'=' * 70}")
    print("CONVERSATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Turns:    {passed}/{len(QUESTIONS)} succeeded")
    print(f"  Messages: {len(conversation)}")
    print(f"{'=' * 70}")

    return 0 if passed == len(QUESTIONS) else 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point

# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
