# Databricks notebook source
# MAGIC %md
# MAGIC # Run KA Sequence
# MAGIC
# MAGIC Orchestrates the full KA lifecycle: **create** -> **test**.
# MAGIC
# MAGIC Each step runs as a subprocess via `uv` entry points for environment isolation.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run run-ka-sequence
# MAGIC uv run run-ka-sequence --skip-test      # skip conversation test
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Bootstrap

# COMMAND ----------

from __future__ import annotations

import os
import sys
from pathlib import Path

from notebooks.demo_shared import bootstrap, run_step
from notebooks.demo_shared.subprocess_runner import print_summary

_project_root, _central_config = bootstrap(__file__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Orchestrate Steps
# MAGIC
# MAGIC Steps:
# MAGIC 1. `create-ka` -- create KA, wait ONLINE, add examples
# MAGIC 2. `test-ka` -- multi-turn conversation test (optional)

# COMMAND ----------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run KA lifecycle: create -> test")
    parser.add_argument("--skip-test", action="store_true", help="Skip conversation test")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("KA SEQUENCE: create-ka -> test-ka")
    print(f"{'=' * 60}")

    steps: list[tuple[str, list[str]]] = []

    steps.append(("Create Knowledge Assistant", ["uv", "run", "create-ka"]))

    if not args.skip_test:
        steps.append(("Test KA Conversation", ["uv", "run", "test-ka"]))

    results: list[tuple[str, int, float]] = []
    for label, cmd in steps:
        rc, elapsed = run_step(cmd, label, _project_root)
        results.append((label, rc, elapsed))
        if rc != 0:
            print(f"\nABORTED: '{label}' failed")
            break

    all_ok = all(rc == 0 for _, rc, _ in results)
    
    extra_info = {}
    if all_ok:
        ka_endpoint = os.getenv("KA_ENDPOINT", "")
        ka_tile_id = os.getenv("KA_TILE_ID", "")
        if ka_tile_id:
            extra_info["KA Tile ID"] = ka_tile_id
        if ka_endpoint:
            extra_info["KA Endpoint"] = ka_endpoint
        extra_info["Central config"] = str(_central_config)
    
    print_summary(results, extra_info)
    print(f"{'=' * 60}")

    return 0 if all_ok else 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point

# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
