#!/usr/bin/env python3
"""
Run Databricks workspace notebooks from the local machine: upload (if missing) and execute via Jobs API.

Usage:
  uv run run-workspace-notebooks 03                  # run step 03
  uv run run-workspace-notebooks 05 06 07           # run steps 05, 06, 07
  uv run run-workspace-notebooks --sync-only        # upload ALL notebooks + sdp_pipeline_src to workspace (no run)

With --sync-only, syncs to WORKSPACE_PROJECT_ROOT/agentbricks_custom_agent/notebooks/:
  - config, 00_setup_sec_documents, 05–07 (runnable notebooks)
  - sdp_pipeline_src/*.sql (SDP pipeline SQL files)

Requires: DATABRICKS_HOST, DATABRICKS_TOKEN; CLUSTER_ID only when running (not for --sync-only).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from notebooks.demo_shared import bootstrap

_project_root, _central_config = bootstrap(__file__)

# Step number -> (workspace name, local file path relative to notebooks/) for runnable notebooks
WORKSPACE_NOTEBOOKS = {
    "03": ("03_load_stock_data", "data_engg_src/ingest/load_stock_data.py"),
    "05": ("05_create_stock_views", "data_engg_src/serve/create_stock_views.py"),
    "06": ("06_create_uc_functions", "data_engg_src/serve/create_uc_functions.py"),
    "07": ("07_create_genie_space", "data_engg_src/serve/create_genie_space.py"),
}

# All notebooks to sync to workspace
SYNC_NOTEBOOKS = [
    ("config", "config.py"),
    ("00_setup_sec_documents", "data_engg_src/setup/setup_sec_documents.py"),
    ("05_create_stock_views", "data_engg_src/serve/create_stock_views.py"),
    ("06_create_uc_functions", "data_engg_src/serve/create_uc_functions.py"),
    ("07_create_genie_space", "data_engg_src/serve/create_genie_space.py"),
]


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def run_notebook_and_wait(
    host: str,
    token: str,
    cluster_id: str,
    notebook_path: str,
    run_name: str,
    timeout_minutes: int = 30,
) -> bool:
    from datetime import timedelta
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.jobs import Task, NotebookTask, RunResultState

    w = WorkspaceClient(host=host, token=token)
    run = w.jobs.submit_and_wait(
        run_name=run_name,
        tasks=[
            Task(
                task_key="main",
                existing_cluster_id=cluster_id,
                notebook_task=NotebookTask(notebook_path=notebook_path),
            )
        ],
        timeout=timedelta(minutes=timeout_minutes),
    )
    if run.state and run.state.result_state != RunResultState.SUCCESS:
        raise RuntimeError(f"Run {run_name} failed: {run.state.result_state}")
    print(f"  OK: {run_name}")
    return True


def ensure_notebook_on_workspace(
    host: str,
    token: str,
    workspace_path: str,
    local_path: Path,
    language: str = "PYTHON",
) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language

    lang = Language.SQL if language.upper() == "SQL" else Language.PYTHON
    w = WorkspaceClient(host=host, token=token)
    try:
        w.workspace.get_status(path=workspace_path)
        return
    except Exception:
        pass
    parent = str(Path(workspace_path).parent)
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    content = local_path.read_text()
    print(f"  Importing {local_path.name} -> {workspace_path}")
    w.workspace.import_(
        path=workspace_path,
        content=base64.standard_b64encode(content.encode("utf-8")).decode("ascii"),
        format=ImportFormat.SOURCE,
        language=lang,
    )


def sync_all_to_workspace(
    host: str,
    token: str,
    base: str,
    notebooks_dir: Path,
) -> None:
    """Upload all notebooks and sdp_pipeline_src to workspace so they appear under .../notebooks/."""
    # All Python notebooks
    for name, filename in SYNC_NOTEBOOKS:
        local_path = notebooks_dir / filename
        if not local_path.exists():
            continue
        ensure_notebook_on_workspace(host, token, f"{base}/{name}", local_path, language="PYTHON")
    # SDP pipeline SQL files (notebooks/data_engg_src/transform/sdp_pipeline_src/*.sql)
    sdp_src = notebooks_dir / "data_engg_src" / "transform" / "sdp_pipeline_src"
    if sdp_src.is_dir():
        for f in sorted(sdp_src.glob("*.sql")):
            workspace_path = f"{base}/sdp_pipeline_src/{f.stem}"
            ensure_notebook_on_workspace(host, token, workspace_path, f, language="SQL")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload and run workspace notebooks on Databricks.",
    )
    parser.add_argument(
        "steps",
        nargs="*",
        default=None,
        help="Step numbers to run, e.g. 03 or 05 06 07 (default: 03 05 06 07)",
    )
    parser.add_argument(
        "--steps",
        dest="steps_flag",
        nargs="+",
        metavar="N",
        help="Same as positional steps",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync all notebooks + sdp_pipeline_src to workspace; do not run any job",
    )
    args = parser.parse_args()
    steps = args.steps_flag or args.steps
    if args.sync_only:
        steps = []
    elif steps is None:
        steps = list(WORKSPACE_NOTEBOOKS.keys())
    steps = [s.strip() for s in steps if str(s).strip() in WORKSPACE_NOTEBOOKS]

    host = get_env("DATABRICKS_HOST")
    token = get_env("DATABRICKS_TOKEN")
    cluster_id = get_env("CLUSTER_ID")
    workspace_root = get_env("WORKSPACE_PROJECT_ROOT", "/Workspace/Users/your-user@databricks.com/my_projects")
    notebooks_dir = _project_root / "notebooks"

    if not host or not token:
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN", file=sys.stderr)
        return 1
    if not args.sync_only and not cluster_id:
        print("Set CLUSTER_ID (existing all-purpose cluster)", file=sys.stderr)
        return 1

    base = f"{workspace_root}/agentbricks_custom_agent/notebooks"

    if args.sync_only:
        print("Syncing all notebooks and sdp_pipeline_src to workspace...")
        sync_all_to_workspace(host, token, base, notebooks_dir)
        print(f"Done. Check: {base}")
        return 0

    if not steps:
        print("Specify at least one step: 03 05 06 07 (or use --sync-only to only upload)", file=sys.stderr)
        return 1
    # Config is required by %run ./config in 03, 05, 06, 07
    config_path = notebooks_dir / "config.py"
    if config_path.exists():
        ensure_notebook_on_workspace(host, token, f"{base}/config", config_path)

    for step in steps:
        name, filename = WORKSPACE_NOTEBOOKS[step]
        local_path = notebooks_dir / filename
        if not local_path.exists():
            print(f"  Skip: {local_path} not found", file=sys.stderr)
            continue
        workspace_path = f"{base}/{name}"
        ensure_notebook_on_workspace(host, token, workspace_path, local_path)
        run_notebook_and_wait(
            host=host,
            token=token,
            cluster_id=cluster_id,
            notebook_path=workspace_path,
            run_name=f"sec-fin-{name}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
