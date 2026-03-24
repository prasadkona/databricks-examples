#!/usr/bin/env python3
"""
deploy_sdp_pipeline.py - Deploy SDP pipeline only (no app), run it, verify tables

This script ONLY deploys and runs the SDP pipeline. It never deploys the Databricks app.
It uses data_engg_src/transform/sdp_pipeline_src/ (DAB + SQL in one folder) for the DLT pipeline.

When running run_sequence, the pipeline is created in the workspace defined in
data_engg_src/transform/sdp_pipeline_src/databricks.yml (targets.<target>.workspace.host).
The same host is used for deploy and for API calls so the new pipeline is found and run there.

  1. Validates the pipeline bundle (sdp_pipeline_src/)
  2. Deploys the pipeline bundle (databricks bundle deploy --auto-approve)
  3. Starts full-refresh pipeline run and polls until COMPLETED/FAILED
  4. Verifies pipeline tables have data (SELECT COUNT(*) checks)

  uv run deploy-sdp-pipeline
  uv run deploy-sdp-pipeline -t prod
  uv run deploy-sdp-pipeline --no-verify
  uv run deploy-sdp-pipeline --skip-app-deploy   # Skip bundle deploy; only run existing pipeline + verify

Requires .env: DATABRICKS_TOKEN (PAT only; no client id/secret). Optionally DATABRICKS_HOST; host is read from bundle when present. All API calls use PAT.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from notebooks.demo_shared import bootstrap, api_request

_project_root, _central_config = bootstrap(__file__, sync_profile=True)

PIPELINE_NAME = "sec_financial_analyst_pipeline"
BUNDLE_TARGET_DEFAULT = "dev"

# Pipeline bundle variable overrides driven by env vars.
# These map env var names → databricks.yml variable names.
BUNDLE_VAR_MAP = {
    "UC_CATALOG":        "catalog",
    "UC_SCHEMA":         "schema",
    "UC_VOLUME":         "volume",
    "SEC_DOCS_SUBFOLDER": "docs_subfolder",
}

# SDP pipeline run can take up to ~10 min; we allow up to 45 min for completion
PIPELINE_WAIT_TIMEOUT_MINUTES = 45
# Wait this long before first poll so pipeline can progress past WAITING_FOR_RESOURCES (often > 30s)
PIPELINE_INITIAL_WAIT_SECONDS = 90
# If run fails within this many seconds, retry once (transient resource issues)
PIPELINE_EARLY_FAIL_RETRY_THRESHOLD_SECONDS = 180
# Subprocess timeout for bundle validate/deploy (each); allow 15 min per command
BUNDLE_CMD_TIMEOUT_SECONDS = 900

# Tables that must have at least one row after pipeline run
VERIFY_TABLES = [
    "gold_company_financials",
    "gold_revenue_by_segment",
    "gold_revenue_by_geography",
    "silver_stock_daily_prices",
    "gold_stock_summary",
]


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def get_bundle_workspace_host(bundle_dir: Path) -> str | None:
    """Read workspace host from bundle databricks.yml so deploy and API use the same workspace."""
    yml = bundle_dir / "databricks.yml"
    if not yml.exists():
        return None
    text = yml.read_text(encoding="utf-8")
    m = re.search(r"workspace:\s*\n\s*host:\s*(\S+)", text)
    return m.group(1).strip() if m else None


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    cwd = cwd or _project_root
    env = env or os.environ
    return subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **env} if env else os.environ,
        capture_output=True,
        text=True,
        timeout=BUNDLE_CMD_TIMEOUT_SECONDS,
    )


def _pipeline_matches(name: str) -> bool:
    """True if this pipeline is our demo pipeline (exact name sec_financial_analyst_pipeline or legacy name)."""
    return name == PIPELINE_NAME or name.endswith("sec_financial_analyst_pipeline")


def list_all_pipelines(host: str, token: str) -> list[dict]:
    """List all pipelines in the workspace with pagination (max_results=100, follow next_page_token)."""
    all_statuses: list[dict] = []
    params: dict = {"max_results": 100}
    while True:
        result = api_request("GET", host, token, "/api/2.0/pipelines", params)
        all_statuses.extend(result.get("statuses", []))
        next_token = result.get("next_page_token")
        if not next_token:
            break
        params = {"page_token": next_token, "max_results": 100}
    return all_statuses


def get_pipeline_id(host: str, token: str) -> str:
    """Find pipeline by consistent demo name sec_financial_analyst_pipeline (uses paginated list)."""
    for p in list_all_pipelines(host, token):
        if _pipeline_matches(p.get("name", "")):
            return p["pipeline_id"]
    raise RuntimeError(f"Pipeline '{PIPELINE_NAME}' not found. Run 'databricks bundle deploy -t <target>' first.")


def delete_pipeline_if_exists(host: str, token: str) -> bool:
    """Delete the SDP pipeline if it exists (name sec_financial_analyst_pipeline). Returns True if deleted."""
    for p in list_all_pipelines(host, token):
        if _pipeline_matches(p.get("name", "")):
            pid = p.get("pipeline_id")
            if pid:
                api_request("DELETE", host, token, f"/api/2.0/pipelines/{pid}")
                return True
            break
    return False


def list_pipeline_names(host: str, token: str) -> list[str]:
    """List all pipeline names in the workspace (for debugging; uses pagination)."""
    return [p.get("name", "") for p in list_all_pipelines(host, token)]


def create_pipeline_via_api(
    host: str,
    token: str,
    catalog: str,
    schema: str,
    workspace_root: str,
) -> str:
    """Create the SDP pipeline via REST API using SQL under workspace notebooks/sdp_pipeline_src (synced by sync-workspace)."""
    base = (workspace_root or "").rstrip("/")
    if not base:
        raise ValueError("WORKSPACE_PROJECT_ROOT is required to create pipeline via API")
    pipeline_src = f"{base}/agentbricks_custom_agent/notebooks/sdp_pipeline_src"
    spec = {
        "name": PIPELINE_NAME,
        "catalog": catalog,
        "target": schema,
        "serverless": True,
        "continuous": False,
        "development": False,
        "channel": "CURRENT",
        "libraries": [
            {"notebook": {"path": f"{pipeline_src}/01_bronze_parsed_documents"}},
            {"notebook": {"path": f"{pipeline_src}/02_silver_financial_metrics"}},
            {"notebook": {"path": f"{pipeline_src}/03_gold_company_financials"}},
            {"notebook": {"path": f"{pipeline_src}/04_gold_revenue_segments"}},
            {"notebook": {"path": f"{pipeline_src}/05_gold_revenue_geography"}},
            {"notebook": {"path": f"{pipeline_src}/06_silver_stock_daily"}},
            {"notebook": {"path": f"{pipeline_src}/07_gold_stock_summary"}},
        ],
        "configuration": {"pipelines.enableTrackHistory": "true"},
    }
    result = api_request("POST", host, token, "/api/2.0/pipelines", spec)
    pid = result.get("pipeline_id")
    if not pid:
        raise RuntimeError("Create pipeline API did not return pipeline_id")
    return pid


def _wait_for_pipeline_update(
    host: str,
    token: str,
    pipeline_id: str,
    update_id: str,
    timeout_minutes: int,
    poll_interval: int,
    initial_wait_seconds: int,
) -> str:
    """Poll pipeline update until terminal state. Returns final state."""
    terminal = {"COMPLETED", "FAILED", "CANCELED", "TERMINATED"}
    start = time.time()
    final_state = "UNKNOWN"

    # Give pipeline time to progress past WAITING_FOR_RESOURCES (often > 30s)
    if initial_wait_seconds > 0:
        print(f"  Waiting {initial_wait_seconds}s before first poll (pipeline may take >30s to acquire resources)...")
        time.sleep(initial_wait_seconds)

    while True:
        if time.time() - start > timeout_minutes * 60:
            print("  Timeout waiting for pipeline run.")
            break
        try:
            r = api_request("GET", host, token, f"/api/2.0/pipelines/{pipeline_id}/updates/{update_id}")
            upd = r.get("update") or {}
            state = upd.get("state", "UNKNOWN")
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] state: {state}")
            if state in terminal:
                final_state = state
                if state == "FAILED":
                    cause = upd.get("cause") or upd.get("failure") or ""
                    if cause:
                        print(f"  Update cause/failure: {str(cause)[:300]}")
                    try:
                        ev_params = {"max_results": 20}
                        events = api_request(
                            "GET", host, token, f"/api/2.0/pipelines/{pipeline_id}/events",
                            ev_params,
                        )
                        for ev in (events.get("events") or [])[:5]:
                            print(f"  ERROR: {(ev.get('message') or '')[:200]}")
                    except Exception:
                        pass
                break
        except Exception as e:
            print(f"  Poll error: {e}")
        time.sleep(poll_interval)

    return final_state


def _wait_for_no_active_update(host: str, token: str, pipeline_id: str, max_wait: int = 120) -> None:
    """Wait until there is no active update running on the pipeline."""
    print(f"  Checking for active updates (max wait {max_wait}s)...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = api_request("GET", host, token, f"/api/2.0/pipelines/{pipeline_id}")
            latest_updates = r.get("latest_updates") or []
            # Check if any update is not in a terminal state
            active = [u for u in latest_updates if u.get("state") not in {"COMPLETED", "FAILED", "CANCELED", "TERMINATED"}]
            if not active:
                print("  No active updates found.")
                return
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Active update(s) still running, waiting...")
        except Exception as e:
            print(f"  Check error: {e}")
        time.sleep(10)
    print("  Warning: Max wait reached, proceeding anyway...")


def run_pipeline_and_wait(
    host: str,
    token: str,
    pipeline_id: str,
    timeout_minutes: int = PIPELINE_WAIT_TIMEOUT_MINUTES,
    poll_interval: int = 15,
) -> str:
    """Start full refresh, poll until terminal state. Retries once if run fails early (e.g. resource timeout)."""
    # Ensure no active update is running before starting
    _wait_for_no_active_update(host, token, pipeline_id)
    result = api_request("POST", host, token, f"/api/2.0/pipelines/{pipeline_id}/updates", {"full_refresh": True})
    update_id = result.get("update_id")
    if not update_id:
        raise RuntimeError("Pipeline run did not return update_id")
    print(f"  Pipeline run started: update_id={update_id}")
    print(f"  Waiting for completion (timeout: {timeout_minutes} min, poll every {poll_interval}s)...")

    final_state = _wait_for_pipeline_update(
        host, token, pipeline_id, update_id,
        timeout_minutes, poll_interval,
        PIPELINE_INITIAL_WAIT_SECONDS,
    )

    # NOTE: DLT has built-in RETRY_ON_FAILURE, so we don't retry manually here
    # to avoid race conditions. If pipeline fails, user should check UI for details.
    if final_state == "FAILED":
        print("  Pipeline run FAILED. Check the Databricks UI for details.")
        print(f"  Pipeline URL: https://{host.replace('https://', '')}/pipelines/{pipeline_id}")

    return final_state


def get_table_count(host: str, token: str, warehouse_id: str, catalog: str, schema: str, table: str) -> int:
    """Run SELECT COUNT(*) and return the count. Uses Statement Execution API."""
    sql = f"SELECT COUNT(*) AS cnt FROM {catalog}.{schema}.{table}"
    payload = {
        "warehouse_id": warehouse_id,
        "catalog": catalog,
        "schema": schema,
        "statement": sql,
        "wait_timeout": "50s",
    }
    result = api_request("POST", host, token, "/api/2.0/sql/statements", json_data=payload)
    state = (result.get("status") or {}).get("state", "")
    stmt_id = result.get("statement_id")

    def parse_count_from_response(r: dict) -> int:
        # First chunk can be in result (execute) or in result (get statement)
        res = r.get("result") or {}
        data = res.get("data_array") if isinstance(res, dict) else None
        if data and len(data) > 0 and len(data[0]) > 0:
            try:
                return int(data[0][0])
            except (TypeError, ValueError):
                pass
        return -1

    if state == "SUCCEEDED":
        cnt = parse_count_from_response(result)
        if cnt >= 0:
            return cnt

    if stmt_id and state in ("PENDING", "RUNNING"):
        for _ in range(60):
            time.sleep(1)
            r = api_request("GET", host, token, f"/api/2.0/sql/statements/{stmt_id}")
            state = (r.get("status") or {}).get("state", "")
            if state == "SUCCEEDED":
                return parse_count_from_response(r)
            if state in ("FAILED", "CANCELED", "CLOSED"):
                break

    return -1


def verify_tables(
    host: str,
    token: str,
    warehouse_id: str,
    catalog: str,
    schema: str,
) -> tuple[bool, list[str]]:
    """Check that each VERIFY_TABLES table has at least one row. Returns (all_ok, list of failures)."""
    failures = []
    for table in VERIFY_TABLES:
        try:
            cnt = get_table_count(host, token, warehouse_id, catalog, schema, table)
            if cnt < 0:
                failures.append(f"{table}: could not get count")
            elif cnt == 0:
                failures.append(f"{table}: 0 rows")
            else:
                print(f"  OK {table}: {cnt} rows")
        except Exception as e:
            failures.append(f"{table}: {e}")
    return (len(failures) == 0, failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy and run SDP pipeline only (no app), verify tables.")
    parser.add_argument("-t", "--target", default=BUNDLE_TARGET_DEFAULT, help=f"Bundle target (default: {BUNDLE_TARGET_DEFAULT})")
    parser.add_argument("--no-verify", action="store_true", help="Skip table count verification")
    parser.add_argument(
        "--skip-app-deploy",
        action="store_true",
        help="Skip bundle deploy step; only run existing pipeline + verify",
    )
    args = parser.parse_args()

    host = get_env("DATABRICKS_HOST")
    token = get_env("DATABRICKS_TOKEN")
    warehouse_id = get_env("SQL_WAREHOUSE_ID")
    catalog = get_env("UC_CATALOG", "your_catalog")
    schema = get_env("UC_SCHEMA", "your_schema")

    if not host or not token:
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN (PAT; e.g. in .env). Do not use client id/secret.", file=sys.stderr)
        return 1
    if not args.no_verify and not warehouse_id:
        print("Set SQL_WAREHOUSE_ID for verification (or use --no-verify)", file=sys.stderr)
        return 1

    # Use data_engg_src/transform/sdp_pipeline_src (DAB + SQL in one folder; never deploy the app)
    pipeline_only_dir = _project_root / "notebooks" / "data_engg_src" / "transform" / "sdp_pipeline_src"
    if not pipeline_only_dir.is_dir() or not (pipeline_only_dir / "databricks.yml").exists():
        print("data_engg_src/transform/sdp_pipeline_src/databricks.yml not found.", file=sys.stderr)
        return 1
    deploy_cwd = pipeline_only_dir

    # Use workspace host from bundle so deploy and API target the same workspace (create/find pipeline there)
    bundle_host = get_bundle_workspace_host(pipeline_only_dir)
    if bundle_host:
        host = bundle_host
        print(f"Using workspace from bundle: {host}")
    else:
        print(f"Using DATABRICKS_HOST from env: {host}")

    target = args.target
    print("=" * 60)
    print("Deploy SDP pipeline only (no app): validate → deploy → run → verify")
    print("=" * 60)
    if args.skip_app_deploy:
        print("(Bundle deploy skipped via --skip-app-deploy)")
    print(f"Target: {target}")
    print(f"Catalog.Schema: {catalog}.{schema}")
    print()

    # Use PAT only for CLI and API (no client id/secret)
    deploy_env = {**os.environ, "DATABRICKS_HOST": host, "DATABRICKS_TOKEN": token}
    for k in ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        deploy_env.pop(k, None)

    if not args.skip_app_deploy:
        print("Step 1: Validating bundle...")
        r = run_cmd(["databricks", "bundle", "validate", "-t", target], cwd=deploy_cwd, env=deploy_env)
        if r.returncode != 0:
            print(r.stdout or "")
            print(r.stderr or "", file=sys.stderr)
            print("Bundle validate failed.", file=sys.stderr)
            return 1
        print("  Bundle valid.")
        print()

        # 2. Bundle deploy (--auto-approve: delete/recreate DLT pipeline without prompts)
        print("Step 2: Deploying bundle...")
        r = run_cmd(["databricks", "bundle", "deploy", "-t", target, "--auto-approve"], cwd=deploy_cwd, env=deploy_env)
        _stdout = (r.stdout or "").strip()
        _stderr = (r.stderr or "").strip()
        if _stdout:
            print(_stdout)
        if _stderr:
            print(_stderr, file=sys.stderr)
        if r.returncode != 0:
            stderr = _stderr
            stdout = _stdout
            if "already used by another pipeline" in (stderr + stdout):
                print("  Pipeline already exists; deleting and retrying deploy...")
                if delete_pipeline_if_exists(host, token):
                    print("  Deleted existing pipeline.")
                    r = run_cmd(["databricks", "bundle", "deploy", "-t", target, "--auto-approve"], cwd=deploy_cwd, env=deploy_env)
                    _stdout = (r.stdout or "").strip()
                    _stderr = (r.stderr or "").strip()
                    if _stdout:
                        print(_stdout)
                    if _stderr:
                        print(_stderr, file=sys.stderr)
                if r.returncode != 0:
                    print("Bundle deploy failed.", file=sys.stderr)
                    return 1
            else:
                print("Bundle deploy failed.", file=sys.stderr)
                return 1
        else:
            print("  Deploy done.")
        print()
    else:
        print("Step 1 & 2: Skipped (--skip-app-deploy). Assuming pipeline already exists.")
        print()

    # 3. Get pipeline ID and run (deploy pipeline if not found)
    print("Step 3: Running pipeline (full refresh)...")

    def do_deploy() -> bool:
        """Run bundle validate + deploy. Returns True on success."""
        # Build --var overrides from env vars defined in BUNDLE_VAR_MAP
        var_overrides: list[str] = []
        for env_key, var_name in BUNDLE_VAR_MAP.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                var_overrides += ["--var", f"{var_name}={val}"]
        if var_overrides:
            print(f"  Pipeline variable overrides: {[f'{v}' for v in var_overrides if not v.startswith('--')]}")

        r = run_cmd(["databricks", "bundle", "validate", "-t", target] + var_overrides, cwd=deploy_cwd, env=deploy_env)
        if r.returncode != 0:
            print(r.stdout or "")
            print(r.stderr or "", file=sys.stderr)
            return False
        r = run_cmd(["databricks", "bundle", "deploy", "-t", target, "--auto-approve"] + var_overrides, cwd=deploy_cwd, env=deploy_env)
        if r.returncode != 0:
            print(r.stdout or "")
            print(r.stderr or "", file=sys.stderr)
            return False
        return True

    pipeline_id = None
    try:
        pipeline_id = get_pipeline_id(host, token)
    except RuntimeError:
        print(f"  Pipeline '{PIPELINE_NAME}' not found. Deploying pipeline...")
        if not do_deploy():
            print("Deploy failed.", file=sys.stderr)
            return 1
        print("  Deploy done. Waiting for pipeline to be visible...")
        for attempt in range(6):  # up to ~60s
            time.sleep(10)
            try:
                pipeline_id = get_pipeline_id(host, token)
                break
            except RuntimeError:
                if attempt < 5:
                    print(f"  Retry {attempt + 2}/6...")
                else:
                    # Fallback: create pipeline via API using SQL under workspace (sync-workspace uploads these)
                    workspace_root = get_env("WORKSPACE_PROJECT_ROOT")
                    if workspace_root:
                        try:
                            print(f"  Creating pipeline via API (using {workspace_root}/.../sdp_pipeline_src)...")
                            pipeline_id = create_pipeline_via_api(host, token, catalog, schema, workspace_root)
                            print(f"  Pipeline created: {pipeline_id}")
                            break  # success, exit retry loop
                        except Exception as e:
                            print(f"  Create via API failed: {e}", file=sys.stderr)
                            try:
                                names = list_pipeline_names(host, token)
                                print(f"  Pipelines in workspace: {names or '(none)'}", file=sys.stderr)
                            except Exception:
                                pass
                            print(f"  Pipeline still not found after deploy.", file=sys.stderr)
                            return 1
                    else:
                        try:
                            names = list_pipeline_names(host, token)
                            print(f"  Pipelines in workspace: {names or '(none)'}", file=sys.stderr)
                        except Exception as e:
                            print(f"  (Could not list pipelines: {e})", file=sys.stderr)
                        print(f"  Set WORKSPACE_PROJECT_ROOT (e.g. /Workspace/Users/<you>/my_projects) to create pipeline via API.", file=sys.stderr)
                        print(f"  Pipeline still not found after deploy.", file=sys.stderr)
                        return 1
    if pipeline_id is None:
        print("  Pipeline not found.", file=sys.stderr)
        return 1

    base = host.rstrip("/")
    pipeline_url = f"{base}/pipelines/{pipeline_id}"
    print(f"  Pipeline ID: {pipeline_id}")
    print(f"  View pipeline & runs: {pipeline_url}")
    # Write URL for run_sequence to print (easier debugging)
    _local_dir = _project_root / "_local"
    _local_dir.mkdir(exist_ok=True)
    (_local_dir / "last_pipeline_url.txt").write_text(pipeline_url, encoding="utf-8")
    final_state = run_pipeline_and_wait(host, token, pipeline_id)
    print(f"  Final state: {final_state}")
    if final_state != "COMPLETED":
        print("Pipeline did not complete successfully.", file=sys.stderr)
        return 1
    print()

    # 4. Verify tables
    if args.no_verify:
        print("Step 4: Verification skipped (--no-verify).")
        return 0

    print("Step 4: Verifying pipeline tables have data...")
    ok, failures = verify_tables(host, token, warehouse_id, catalog, schema)
    if not ok:
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        print("Table verification failed.", file=sys.stderr)
        return 1
    print("  All tables have data.")
    print()
    print("=" * 60)
    print("Deploy and pipeline run completed successfully.")
    print("=" * 60)
    print(f"View pipeline and runs in the UI: {pipeline_url}")
    print("(In the workspace: Workflows → Pipelines, or use the link above.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
