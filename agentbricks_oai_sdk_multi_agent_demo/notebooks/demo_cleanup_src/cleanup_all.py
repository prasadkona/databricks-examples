# Databricks notebook source
# MAGIC %md
# MAGIC # Master Cleanup Orchestrator
# MAGIC
# MAGIC Provides phased teardown levels for the demo environment.
# MAGIC
# MAGIC | Level | What is removed |
# MAGIC |---|---|
# MAGIC | `after-genie` | App only (keep KA, Genie, tables, pipeline) |
# MAGIC | `after-ka` | App + Genie + tables + pipeline + views (keep KA) |
# MAGIC | `all` | Everything: App + Genie + tables + pipeline + KA + views |
# MAGIC | `tables` | Tables + pipeline only (backward compat) |
# MAGIC
# MAGIC **Usage:**
# MAGIC ```
# MAGIC uv run demo-cleanup all
# MAGIC uv run demo-cleanup after-genie
# MAGIC uv run demo-cleanup after-ka
# MAGIC uv run demo-cleanup tables
# MAGIC uv run demo-cleanup tables --include-views
# MAGIC ```
# MAGIC
# MAGIC **Requires:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (loaded from central config)

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

from . import cleanup_app
from . import cleanup_genie
from . import cleanup_ka
from . import cleanup_tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup Levels

# COMMAND ----------

LEVELS = {
    "after-genie": ["app"],
    "after-ka":    ["app", "genie", "tables"],
    "all":         ["app", "genie", "tables", "ka"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper

# COMMAND ----------

def _header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main: Phased Cleanup
# MAGIC
# MAGIC Delegates to the individual cleanup modules based on the selected level.

# COMMAND ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo cleanup -- phased teardown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
levels:
  after-genie   Remove App only (keep KA + Genie + data)
  after-ka      Remove App + Genie + tables/pipeline/views (keep KA)
  all           Remove everything
  tables        Tables + pipeline only (backward compat)
""",
    )
    parser.add_argument("level", choices=["all", "after-ka", "after-genie", "tables"],
                        help="Cleanup level")
    parser.add_argument("--include-views", action="store_true",
                        help="Also drop views (for 'tables' and 'after-ka' levels)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    level = args.level

    if level == "tables":
        _header("Cleanup: Tables + Pipeline")
        return cleanup_tables.run(include_views=args.include_views, dry_run=args.dry_run)

    targets = LEVELS[level]

    _header(f"Cleanup Level: {level}")
    print(f"  Targets: {', '.join(targets)}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    errors = 0

    if "app" in targets:
        _header("Deleting Databricks App")
        if not args.dry_run:
            errors += cleanup_app.run()
        else:
            print(f"  Would delete app: {os.environ.get('APP_NAME', '(not set)')}")

    if "genie" in targets:
        _header("Deleting Genie Space")
        if not args.dry_run:
            errors += cleanup_genie.run()
        else:
            print(f"  Would delete Genie: {os.environ.get('GENIE_SPACE_ID', '(not set)')}")

    if "tables" in targets:
        _header("Dropping Tables + Pipeline")
        include_views = args.include_views or level in ("after-ka", "all")
        if not args.dry_run:
            errors += cleanup_tables.run(include_views=include_views)
        else:
            cleanup_tables.run(include_views=include_views, dry_run=True)

    if "ka" in targets:
        _header("Deleting Knowledge Assistant")
        if not args.dry_run:
            errors += cleanup_ka.run()
        else:
            print(f"  Would delete KA: {os.environ.get('KA_TILE_ID', '(not set)')}")

    _header(f"Cleanup Complete -- Level: {level}")
    if errors:
        print(f"  {errors} step(s) had errors")
    else:
        print("  All steps succeeded")

    return min(errors, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backward-Compatible Entry Point

# COMMAND ----------

def main_tables_compat() -> int:
    """Entry point for backward-compatible 'cleanup-tables' command."""
    sys.argv = [sys.argv[0], "tables"] + sys.argv[1:]
    return main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point

# COMMAND ----------

if __name__ == "__main__":
    sys.exit(main())
