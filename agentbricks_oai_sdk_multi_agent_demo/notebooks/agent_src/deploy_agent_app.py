#!/usr/bin/env python3
"""
deploy_agent_app.py - Deploy SEC Financial Analyst Agent to Databricks Apps

Execution order: run last, after test-agent passes.
Deploys via Asset Bundles, grants SP permissions, saves endpoint, runs 1 verification query.

  uv run deploy-agent-app                   # deploy + verify (1 query with verbose output)
  uv run deploy-agent-app --skip-validation # skip local tests before deployment
  uv run deploy-agent-app --skip-test       # deploy only, no verification query
  uv run deploy-agent-app -t dev

For comprehensive deployed app testing:
  uv run test-agent-app                     # smoke test (Genie+KA+UC)
  uv run test-agent-app --full              # all 8 test scenarios
  uv run test-agent-app --test-services     # test UC functions, Genie, KA individually
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

from notebooks.demo_shared import bootstrap, update_central_config, get_app_dir as shared_get_app_dir
from notebooks.demo_shared.paths import get_project_root

_project_root, _central_config = bootstrap(__file__, override=False, sync_profile=True)

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_step(msg: str):
    print(f"\n{BLUE}{BOLD}▶ {msg}{RESET}")


def print_success(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")


def print_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")


def print_warning(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def print_info(msg: str):
    print(f"  {msg}")


def wait_for_app_healthy(app_url: str, token: str, max_retries: int = 5) -> bool:
    """Wait for app to be healthy with exponential backoff.
    
    Args:
        app_url: The app's base URL
        token: OAuth/PAT token for authentication
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if app becomes healthy, False otherwise
    """
    delays = [10, 20, 30, 45, 60]  # Backoff delays in seconds
    health_url = f"{app_url.rstrip('/')}/health"
    headers = {"Authorization": f"Bearer {token}"}
    
    for attempt in range(max_retries):
        try:
            r = requests.get(health_url, headers=headers, timeout=30)
            if r.status_code == 200:
                print_success("App is healthy")
                return True
            elif r.status_code == 302:
                print_info(f"Health check returned 302 (redirect) - app may still be starting")
            else:
                print_info(f"Health check returned {r.status_code}")
        except requests.exceptions.RequestException as e:
            print_info(f"Health check failed: {e}")
        
        if attempt < max_retries - 1:
            delay = delays[min(attempt, len(delays) - 1)]
            print_info(f"Retrying in {delay}s... (attempt {attempt + 2}/{max_retries})")
            time.sleep(delay)
    
    print_error(f"App did not become healthy after {max_retries} attempts")
    return False


def get_app_status(host: str, token: str, app_name: str) -> dict:
    """Get app status from Databricks API.
    
    Returns:
        Dict with app_status and compute_status, or empty dict on error
    """
    url = f"{host}/api/2.0/apps/{app_name}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print_warning(f"Could not get app status: {e}")
    
    return {}


def deploy_app_code(host: str, token: str, app_name: str, workspace_path: str) -> bool:
    """Deploy app source code from workspace path.
    
    Args:
        host: Databricks host URL
        token: PAT token
        app_name: Name of the app
        workspace_path: Workspace path containing the app source code
        
    Returns:
        True if deployment succeeded, False otherwise
    """
    import subprocess
    
    print_step(f"Deploying app source code from {workspace_path}...")
    
    try:
        result = subprocess.run(
            ["databricks", "apps", "deploy", app_name, "--source-code-path", workspace_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            print_success("App source code deployed successfully")
            return True
        else:
            print_error(f"App source code deployment failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error("App source code deployment timed out")
        return False
    except Exception as e:
        print_error(f"App source code deployment failed: {e}")
        return False


def sync_app_config(app_dir: Path) -> list[str]:
    """Sync app/databricks.yml with current env vars (KA_ENDPOINT, GENIE_SPACE_ID, etc.).
    
    Returns list of updated fields.
    """
    import re
    
    yml_path = app_dir / "databricks.yml"
    content = yml_path.read_text()
    updated = []
    
    # Config values to sync from env
    sync_values = {
        "KA_ENDPOINT": os.environ.get("KA_ENDPOINT", ""),
        "GENIE_SPACE_ID": os.environ.get("GENIE_SPACE_ID", ""),
        "SQL_WAREHOUSE_ID": os.environ.get("SQL_WAREHOUSE_ID", ""),
        "UC_CATALOG": os.environ.get("UC_CATALOG", ""),
        "UC_SCHEMA": os.environ.get("UC_SCHEMA", ""),
        "TABLE_PREFIX": os.environ.get("TABLE_PREFIX", ""),
        "MLFLOW_EXPERIMENT_ID": os.environ.get("MLFLOW_EXPERIMENT_ID", ""),
    }
    
    for key, new_value in sync_values.items():
        if not new_value:
            continue
        
        # Update env section: - name: KEY\n            value: "..."
        pattern = rf'(- name: {key}\s+value: ")[^"]*(")'
        match = re.search(pattern, content)
        if match:
            old_value = content[match.start(1) + len(match.group(1)):match.start(2)]
            if old_value != new_value:
                content = re.sub(pattern, rf'\g<1>{new_value}\g<2>', content)
                updated.append(f"{key}: {old_value[:20]}... -> {new_value[:20]}...")
    
    # Special handling for KA_ENDPOINT in resources section
    ka_endpoint = sync_values.get("KA_ENDPOINT", "")
    if ka_endpoint:
        # Update serving_endpoint name in resources
        pattern = r"(serving_endpoint:\s+name: ')[^']*(')"
        match = re.search(pattern, content)
        if match:
            old_value = content[match.start(1) + len(match.group(1)):match.start(2)]
            if old_value != ka_endpoint:
                content = re.sub(pattern, rf"\g<1>{ka_endpoint}\g<2>", content)
                if f"KA_ENDPOINT" not in [u.split(":")[0] for u in updated]:
                    updated.append(f"serving_endpoint.name: {old_value[:20]}... -> {ka_endpoint[:20]}...")
    
    # Special handling for SQL_WAREHOUSE_ID in resources section
    warehouse_id = sync_values.get("SQL_WAREHOUSE_ID", "")
    if warehouse_id:
        # Update sql_warehouse id in resources
        pattern = r"(sql_warehouse:\s+id: ')[^']*(')"
        match = re.search(pattern, content)
        if match:
            old_value = content[match.start(1) + len(match.group(1)):match.start(2)]
            if old_value != warehouse_id:
                content = re.sub(pattern, rf"\g<1>{warehouse_id}\g<2>", content)
    
    if updated:
        yml_path.write_text(content)
    
    return updated


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print_info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print_error(f"Command failed: {result.stderr}")
        if result.stdout:
            print_info(f"stdout: {result.stdout}")
    return result


def check_prerequisites(app_dir: Path) -> bool:
    """Check that all prerequisites are met."""
    print_step("Checking prerequisites...")

    # Show config source
    if _central_config.exists():
        print_success(f"Config loaded from {_central_config}")
    else:
        print_warning(f"Central config not found at {_central_config}")

    # Verify app/ directory
    if not (app_dir / "agent_server" / "agent.py").exists():
        print_error(f"app/agent_server/agent.py not found at {app_dir}")
        return False
    print_success(f"App directory OK: {app_dir}")

    # Check databricks CLI
    result = run_command(["databricks", "--version"], check=False)
    if result.returncode != 0:
        print_error("Databricks CLI not found. Install with: brew install databricks")
        return False
    print_success(f"Databricks CLI: {result.stdout.strip()}")

    # Check authentication (PAT via env vars)
    result = run_command(["databricks", "auth", "describe"], check=False)
    if result.returncode != 0:
        print_error("Not authenticated to Databricks. Run: databricks auth login")
        return False
    print_success("Databricks authentication OK")

    # Check databricks.yml exists in app/
    if not (app_dir / "databricks.yml").exists():
        print_error("app/databricks.yml not found")
        return False
    print_success("app/databricks.yml found")

    return True


def run_local_tests() -> bool:
    """Run local agent tests before deployment."""
    print_step("Running local agent tests...")
    
    result = subprocess.run(
        ["uv", "run", "test-agent", "--quick"],
        capture_output=False,
        text=True,
    )
    
    if result.returncode != 0:
        print_error("Local tests failed. Fix issues before deploying.")
        return False
    
    print_success("Local tests passed")
    return True


def validate_bundle(target: str, app_dir: Path) -> bool:
    """Validate the Databricks Asset Bundle from app/ directory."""
    print_step(f"Validating bundle for target: {target} (in {app_dir})...")

    result = subprocess.run(
        ["databricks", "bundle", "validate", "-t", target],
        capture_output=True, text=True, cwd=str(app_dir),
    )

    if result.returncode != 0:
        print_error("Bundle validation failed")
        print_info(result.stderr)
        return False

    print_success("Bundle validation passed")
    if result.stdout:
        print_info(result.stdout)
    return True


def deploy_bundle(target: str, app_dir: Path) -> bool:
    """Deploy the Databricks Asset Bundle from app/ directory."""
    print_step(f"Deploying bundle from {app_dir} to target: {target}...")

    result = subprocess.run(
        ["databricks", "bundle", "deploy", "-t", target],
        capture_output=True, text=True, cwd=str(app_dir),
    )

    if result.returncode != 0:
        print_error("Bundle deployment failed")
        print_info(result.stderr)

        if "maximum number of apps" in result.stderr.lower():
            print_warning("Workspace has reached the maximum number of apps (300)")
            print_info("Options:")
            print_info("  1. Delete unused apps: databricks apps list | grep -i unused")
            print_info("  2. Use a different workspace")
            print_info("  3. Contact Databricks support to increase limit")

        return False

    print_success("Bundle deployment successful")
    if result.stdout:
        print_info(result.stdout)
    return True


def get_app_url(target: str, app_dir: Path) -> str | None:
    """Get the deployed app URL."""
    print_step("Getting app URL...")

    result = subprocess.run(
        ["databricks", "bundle", "describe", "-t", target, "--output", "json"],
        capture_output=True, text=True, cwd=str(app_dir),
    )

    if result.returncode != 0:
        print_warning("Could not get app URL from bundle describe")
        return None

    try:
        import json
        data = json.loads(result.stdout)
        resources = data.get("resources", {})
        apps = resources.get("apps", {})
        if apps:
            app_name = list(apps.keys())[0]
            app_info = apps[app_name]
            url = app_info.get("url")
            if url:
                return url
    except (json.JSONDecodeError, KeyError):
        pass

    host = os.getenv("DATABRICKS_HOST", "")
    if host:
        app_name_fallback = os.getenv("APP_NAME", "sec-financial-analyst")
        return f"{host.rstrip('/')}/apps/{app_name_fallback}"

    return None


def get_oauth_token(host: str) -> str | None:
    """Get an OAuth token for the workspace using the Databricks CLI."""
    result = subprocess.run(
        ["databricks", "auth", "token", "--host", host],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("access_token")
    except (json.JSONDecodeError, KeyError):
        return result.stdout.strip() if result.stdout.strip() else None


def deploy_source_code(app_name: str, source_path: str) -> bool:
    """Deploy source code to the app and wait for it to start."""
    print_step(f"Deploying source code to app {app_name}...")

    result = subprocess.run(
        ["databricks", "apps", "deploy", app_name, "--source-code-path", source_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print_error(f"Source code deployment failed: {result.stderr}")
        return False

    try:
        data = json.loads(result.stdout)
        state = data.get("status", {}).get("state", "?")
        msg = data.get("status", {}).get("message", "")
        if state == "SUCCEEDED":
            print_success(f"App deployed: {msg}")
            return True
        else:
            print_error(f"Deployment state: {state} - {msg}")
            return False
    except json.JSONDecodeError:
        print_info(result.stdout)
        return result.returncode == 0


def get_app_info(app_name: str) -> dict | None:
    """Get app details including URL."""
    result = subprocess.run(
        ["databricks", "apps", "get", app_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def grant_sp_permissions(app_info: dict) -> bool:
    """Grant the app's Service Principal SELECT/EXECUTE on the UC schema."""
    print_step("Granting UC permissions to app Service Principal...")

    sp_client_id = app_info.get("service_principal_client_id")
    sp_name = app_info.get("service_principal_name", "")
    if not sp_client_id:
        print_error("Could not find service_principal_client_id in app info")
        return False

    print_info(f"SP: {sp_name} ({sp_client_id})")

    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")
    uc_catalog = os.getenv("UC_CATALOG", "your_catalog")
    uc_schema = os.getenv("UC_SCHEMA", "your_schema")
    full_schema = f"{uc_catalog}.{uc_schema}"

    if not host or not token:
        print_error("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        return False

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{host}/api/2.1/unity-catalog/permissions/schema/{full_schema}"

    grants_needed = ["SELECT", "EXECUTE"]
    payload = {"changes": [{"principal": sp_client_id, "add": grants_needed}]}

    try:
        resp = requests.patch(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            print_success(f"Granted {grants_needed} on {full_schema} to SP {sp_client_id}")
            return True
        else:
            print_error(f"Grant failed: HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Grant request failed: {e}")
        return False


def save_app_endpoint(app_info: dict) -> None:
    """Save the app URL and SP details to the central config file."""
    print_step("Saving app endpoint to central config...")

    app_url = app_info.get("url", "")
    sp_client_id = app_info.get("service_principal_client_id", "")
    sp_id = app_info.get("service_principal_id", "")

    updates = {
        "APP_URL": app_url,
        "APP_SP_CLIENT_ID": sp_client_id,
        "APP_SP_ID": str(sp_id) if sp_id else "",
    }

    for key, value in updates.items():
        if value:
            if update_central_config(_central_config, key, value):
                print_success(f"  {key}={value}")
            else:
                print_warning(f"  Could not save {key}")

    os.environ["APP_URL"] = app_url


def verify_deployment(app_url: str, host: str) -> bool:
    """Send one test query to the deployed app and print full request/response for debugging."""
    print_step("Post-deploy verification (1 query)...")

    token = get_oauth_token(host)
    if not token:
        print_error("Could not get OAuth token. Run: databricks auth login")
        return False
    print_success("OAuth token acquired")

    # Health check with retry/backoff
    print_step("Checking app health...")
    if not wait_for_app_healthy(app_url, token, max_retries=5):
        return False

    query = "What is NVIDIA's valuation score and total FY2024 revenue?"
    invocation_url = f"{app_url.rstrip('/')}/invocations"
    payload = {"input": [{"role": "user", "content": query}]}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    print(f"\n{'='*60}")
    print(f"  Post-Deploy Verification")
    print(f"{'='*60}")
    print(f"\n  {BOLD}Request:{RESET}")
    print(f"    POST {invocation_url}")
    print(f"    Query: \"{query}\"")

    try:
        start_time = time.time()
        resp = requests.post(invocation_url, json=payload, headers=headers, timeout=180)
        elapsed = time.time() - start_time

        print(f"\n  {BOLD}Response:{RESET}  (HTTP {resp.status_code}, {elapsed:.1f}s)")

        if resp.status_code != 200:
            print_error(f"HTTP {resp.status_code}")
            print(f"\n  {RED}Body:{RESET}")
            print(f"    {resp.text[:500]}")
            return False

        data = resp.json()
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                print(f"    {BLUE}[TOOL]{RESET} {item['name']}({item.get('arguments', '')[:80]})")
            elif item.get("type") == "function_call_output":
                output = item.get("output", "")
                preview = output if isinstance(output, str) else json.dumps(output)
                print(f"    {YELLOW}[TOOL OUTPUT]{RESET} {preview[:120]}...")
            elif item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and len(c.get("text", "")) > 50:
                        print(f"\n    {GREEN}[AGENT RESPONSE]{RESET}")
                        for line in c["text"][:600].split("\n"):
                            print(f"      {line}")
                        if len(c["text"]) > 600:
                            print(f"      ... ({len(c['text'])} chars total)")

        print_success(f"Verification passed in {elapsed:.1f}s")
        print_info(f"For comprehensive testing run: uv run test-agent-app")
        return True

    except requests.exceptions.Timeout:
        print_error("Request timed out (>180s)")
        return False
    except Exception as e:
        print_error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy SEC Financial Analyst Agent to Databricks Apps"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local tests before deployment",
    )
    parser.add_argument(
        "--target", "-t",
        default="dev",
        help="Deployment target (default: dev)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deployment even if validation fails",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip post-deploy verification query",
    )
    args = parser.parse_args()
    
    # Use shared path helpers
    project_root = get_project_root()
    os.chdir(project_root)

    # Resolve app/ directory using shared helper
    app_dir = shared_get_app_dir()
    print_info(f"Project root: {project_root}")
    print_info(f"App directory: {app_dir}")

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  SEC Financial Analyst Agent - Deployment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Step 1: Check prerequisites
    if not check_prerequisites(app_dir):
        sys.exit(1)

    # Step 1.5: Sync app config with current env values
    print_step("Syncing app/databricks.yml with current config...")
    updated = sync_app_config(app_dir)
    if updated:
        for change in updated:
            print_info(f"Updated: {change}")
        print_success(f"Synced {len(updated)} config value(s)")
    else:
        print_success("App config already in sync")

    # Step 2: Run local tests (unless skipped)
    if not args.skip_validation:
        if not run_local_tests():
            if not args.force:
                print_error("\nDeployment aborted. Use --force to deploy anyway.")
                sys.exit(1)
            print_warning("Continuing despite test failures (--force)")
    else:
        print_warning("Skipping local tests (--skip-validation)")

    # Step 3: Validate bundle from app/ directory
    if not validate_bundle(args.target, app_dir):
        if not args.force:
            print_error("\nDeployment aborted. Fix bundle issues or use --force.")
            sys.exit(1)
        print_warning("Continuing despite validation failures (--force)")

    # Step 4: Deploy from app/ directory
    if not deploy_bundle(args.target, app_dir):
        sys.exit(1)

    # Step 5: Deploy source code to the app
    app_name = os.getenv("APP_NAME", "agent-sec-financial-analyst-v3")
    bundle_name = os.getenv("BUNDLE_NAME", "sec_financial_analyst_agent")
    workspace_user = os.getenv("WORKSPACE_PROJECT_ROOT", "").split("/Users/")[-1].split("/")[0] or "your-user@databricks.com"
    bundle_files_path = f"/Workspace/Users/{workspace_user}/.bundle/{bundle_name}/{args.target}/files"
    if not deploy_source_code(app_name, bundle_files_path):
        sys.exit(1)

    # Step 6: Get app info and save endpoint to central config
    app_info = get_app_info(app_name)
    app_url = app_info.get("url") if app_info else None
    if not app_url:
        app_url = get_app_url(args.target, app_dir)

    if app_info:
        save_app_endpoint(app_info)

    # Step 7: Grant UC permissions to app Service Principal
    if app_info:
        if not grant_sp_permissions(app_info):
            print_warning("SP permission grants failed - deployed app may not have full access")

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}  Deployment Complete!{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if app_url:
        print(f"\n  App URL: {BLUE}{app_url}{RESET}")

    # Step 8: Post-deploy verification
    if not args.skip_test and app_url:
        host = os.getenv("DATABRICKS_HOST", "")
        if not verify_deployment(app_url, host):
            print_warning("\nVerification failed. The app is deployed but may need debugging.")
            return 1
    elif args.skip_test:
        print_warning("Skipping post-deploy verification (--skip-test)")

    print(f"\n  Next steps:")
    print(f"    1. Open the app URL in your browser")
    print(f"    2. Run comprehensive tests: uv run test-agent-app")
    print(f"    3. Run comprehensive tests: uv run test-agent-app --full")
    print(f"    4. Test backend services:   uv run test-agent-app --test-services")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
