#!/usr/bin/env python3
"""
test_deployed_agent_app.py - Test the deployed SEC Financial Analyst App

Tests the deployed Databricks App endpoints. Reads APP_URL from central config.

  uv run test-agent-app                     # default: 1 smoke test (Genie+KA+UC)
  uv run test-agent-app --full              # all 8 test scenarios
  uv run test-agent-app --test-services     # test backend services (UC functions, Genie, KA)
  uv run test-agent-app --test-services --full  # services + all agent tests
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from notebooks.demo_shared import bootstrap

_project_root, _central_config = bootstrap(__file__, override=False)

# Colors
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


# ---------------------------------------------------------------------------
# OAuth helper
# ---------------------------------------------------------------------------

def get_oauth_token(host: str) -> str | None:
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


def wait_for_app_healthy(app_url: str, token: str, max_retries: int = 5) -> bool:
    """Wait for app to be healthy with exponential backoff.
    
    Args:
        app_url: The app's base URL
        token: OAuth/PAT token for authentication
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if app becomes healthy, False otherwise
    """
    delays = [10, 20, 30, 45, 60]
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
            elif r.status_code == 502:
                print_info(f"Health check returned 502 (Bad Gateway) - app is starting")
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


# ---------------------------------------------------------------------------
# Service test result tracking
# ---------------------------------------------------------------------------

@dataclass
class ServiceResult:
    name: str
    success: bool
    elapsed_seconds: float
    detail: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Service tests: verify UC functions, Genie, and KA independently
# ---------------------------------------------------------------------------

def test_services(host: str, token: str) -> list[ServiceResult]:
    """Test the backend services that the agent depends on. Returns results list."""
    print(f"\n{'='*60}")
    print(f"  Backend Service Tests")
    print(f"{'='*60}")

    pat = os.getenv("DATABRICKS_TOKEN", "")
    uc_catalog = os.getenv("UC_CATALOG", "your_catalog")
    uc_schema = os.getenv("UC_SCHEMA", "your_schema")
    table_prefix = os.getenv("TABLE_PREFIX", "sec_fin_")
    warehouse_id = os.getenv("SQL_WAREHOUSE_ID", "your-warehouse-id")
    genie_space_id = os.getenv("GENIE_SPACE_ID", "")
    ka_endpoint = os.getenv("KA_ENDPOINT", "")
    headers = {"Authorization": f"Bearer {pat}", "Content-Type": "application/json"}
    results: list[ServiceResult] = []

    # --- UC Function: valuation_score ---
    print_step("Testing UC Function: valuation_score...")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{host}/api/2.0/sql/statements",
            json={
                "warehouse_id": warehouse_id,
                "statement": f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}valuation_score('NVDA')",
                "wait_timeout": "30s",
            },
            headers=headers, timeout=60,
        )
        elapsed = time.time() - t0
        data = resp.json()
        status = data.get("status", {}).get("state", "")
        rows = data.get("result", {}).get("data_array", [])
        if status == "SUCCEEDED" and rows:
            detail = f"Score: {rows[0][2]}/100" if len(rows[0]) > 2 else f"{len(rows)} row(s)"
            print_success(f"valuation_score('NVDA') → {detail}")
            results.append(ServiceResult("UC: valuation_score", True, elapsed, detail))
        else:
            err = data.get("status", {}).get("error", {}).get("message", f"status={status}")
            print_error(f"valuation_score failed: {err}")
            results.append(ServiceResult("UC: valuation_score", False, elapsed, error=err))
    except Exception as e:
        results.append(ServiceResult("UC: valuation_score", False, time.time() - t0, error=str(e)))
        print_error(f"UC Function test failed: {e}")

    # --- UC Function: compare_peers ---
    print_step("Testing UC Function: compare_peers...")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{host}/api/2.0/sql/statements",
            json={
                "warehouse_id": warehouse_id,
                "statement": f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}compare_peers('NVDA')",
                "wait_timeout": "30s",
            },
            headers=headers, timeout=60,
        )
        elapsed = time.time() - t0
        data = resp.json()
        status = data.get("status", {}).get("state", "")
        rows = data.get("result", {}).get("data_array", [])
        if status == "SUCCEEDED" and rows:
            print_success(f"compare_peers('NVDA') → {len(rows)} row(s)")
            results.append(ServiceResult("UC: compare_peers", True, elapsed, f"{len(rows)} rows"))
        else:
            print_error(f"compare_peers failed: status={status}")
            results.append(ServiceResult("UC: compare_peers", False, elapsed, error=f"status={status}"))
    except Exception as e:
        results.append(ServiceResult("UC: compare_peers", False, time.time() - t0, error=str(e)))
        print_error(f"UC Function test failed: {e}")

    # --- Genie Space ---
    if genie_space_id:
        print_step(f"Testing Genie Space ({genie_space_id})...")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{host}/api/2.0/genie/spaces/{genie_space_id}/start-conversation",
                json={"content": "What companies are in the dataset?"},
                headers=headers, timeout=60,
            )
            elapsed = time.time() - t0
            if resp.status_code == 200:
                conv_id = resp.json().get("conversation_id", "")
                print_success(f"Genie Space accessible (conv {conv_id[:12]}...)")
                results.append(ServiceResult("Genie Space", True, elapsed, f"conv {conv_id[:12]}"))
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:120]}"
                print_error(f"Genie Space: {err}")
                results.append(ServiceResult("Genie Space", False, elapsed, error=err))
        except Exception as e:
            results.append(ServiceResult("Genie Space", False, time.time() - t0, error=str(e)))
            print_error(f"Genie Space test failed: {e}")
    else:
        print_warning("GENIE_SPACE_ID not set - skipping Genie test")

    # --- Knowledge Assistant ---
    if ka_endpoint:
        print_step(f"Testing Knowledge Assistant ({ka_endpoint})...")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{host}/serving-endpoints/{ka_endpoint}/invocations",
                json={"input": [{"role": "user", "content": "What is NVIDIA's fiscal year end date?"}]},
                headers=headers, timeout=60,
            )
            elapsed = time.time() - t0
            if resp.status_code == 200:
                text_len = len(json.dumps(resp.json()))
                print_success(f"KA endpoint responded ({text_len} chars)")
                results.append(ServiceResult("Knowledge Assistant", True, elapsed, f"{text_len} chars"))
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:120]}"
                print_error(f"KA endpoint: {err}")
                results.append(ServiceResult("Knowledge Assistant", False, elapsed, error=err))
        except Exception as e:
            results.append(ServiceResult("Knowledge Assistant", False, time.time() - t0, error=str(e)))
            print_error(f"KA test failed: {e}")
    else:
        print_warning("KA_ENDPOINT not set - skipping KA test")

    return results


# ---------------------------------------------------------------------------
# Agent endpoint tests
# ---------------------------------------------------------------------------

def run_agent_tests(app_url: str, token: str, full: bool = False) -> list:
    """Run agent tests against the deployed app endpoint. Returns list of TestResult."""
    from notebooks.agent_src.test_agent import (
        SMOKE_TEST,
        FULL_TEST_CASES,
        TestResult,
        extract_response_text,
        validate_response_content,
    )
    test_cases = FULL_TEST_CASES if full else [SMOKE_TEST]

    invocation_url = f"{app_url.rstrip('/')}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    label = "Full Test Suite" if full else "Smoke Test (Genie + KA + UC)"
    print(f"\n{'='*60}")
    print(f"  Deployed App: {label}")
    print(f"{'='*60}")

    results: list[TestResult] = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}]")
        print(f"--- {test_case.name} ---")
        query_preview = test_case.query[:60] + "..." if len(test_case.query) > 60 else test_case.query
        print(f"    Query: \"{query_preview}\"")
        print(f"    Expected tools: {test_case.expected_tools}")

        payload = {"input": [{"role": "user", "content": test_case.query}]}
        try:
            start_time = time.time()
            resp = requests.post(
                invocation_url, json=payload, headers=headers,
                timeout=max(test_case.timeout, 240),
            )
            elapsed = time.time() - start_time

            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print_error(f"Failed in {elapsed:.1f}s: {error_msg}")
                results.append(TestResult(
                    test_case=test_case, success=False,
                    response_text="", elapsed_seconds=elapsed, error=error_msg,
                ))
                continue

            response_data = {"status_code": 200, "response": resp.json()}
            response_text = extract_response_text(response_data)

            if "max turns" in response_text.lower():
                print_warning(f"Max turns exceeded in {elapsed:.1f}s")
                results.append(TestResult(
                    test_case=test_case, success=False,
                    response_text=response_text[:300], elapsed_seconds=elapsed,
                    error="Max turns exceeded",
                ))
                continue

            content_valid, missing = validate_response_content(
                response_text, test_case.response_contains or [],
            )
            if not content_valid:
                print_info(f"Response missing expected content: {missing}")

            print_success(f"Response in {elapsed:.1f}s ({len(response_text)} chars)")
            results.append(TestResult(
                test_case=test_case, success=True,
                response_text=response_text[:500], elapsed_seconds=elapsed,
            ))

        except requests.exceptions.Timeout:
            t = max(test_case.timeout, 240)
            print_error(f"Timed out (>{t}s)")
            results.append(TestResult(
                test_case=test_case, success=False,
                response_text="", elapsed_seconds=t, error=f"Timeout (>{t}s)",
            ))
        except Exception as e:
            print_error(f"Error: {e}")
            results.append(TestResult(
                test_case=test_case, success=False,
                response_text="", elapsed_seconds=0, error=str(e),
            ))

        if i < len(test_cases):
            time.sleep(2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the deployed SEC Financial Analyst App"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run all 8 test scenarios (default: 1 smoke test hitting Genie+KA+UC)",
    )
    parser.add_argument(
        "--test-services", action="store_true",
        help="Test backend services (UC functions, Genie Space, KA endpoint) individually",
    )
    parser.add_argument(
        "--app-url", type=str, default=None,
        help="Override app URL (default: reads APP_URL from central config)",
    )
    args = parser.parse_args()

    app_url = args.app_url or os.getenv("APP_URL", "")
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")

    if not app_url:
        print_error("APP_URL not set. Run deploy-agent-app first or pass --app-url")
        sys.exit(1)
    if not host:
        print_error("DATABRICKS_HOST not set")
        sys.exit(1)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  SEC Financial Analyst - Deployed App Tests{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print_info(f"App URL: {app_url}")
    print_info(f"Host:    {host}")

    # Get OAuth token for the app
    print_step("Acquiring OAuth token...")
    token = get_oauth_token(host)
    if not token:
        print_error("Could not get OAuth token. Run: databricks auth login")
        sys.exit(1)
    print_success("OAuth token acquired")

    # Health check with retry/backoff
    print_step("Checking app health...")
    if not wait_for_app_healthy(app_url, token, max_retries=5):
        print_error("App health check failed after all retries")
        sys.exit(1)

    service_results: list[ServiceResult] = []
    agent_results = []

    # Backend service tests
    if args.test_services:
        service_results = test_services(host, token)

    # Agent endpoint tests (always run)
    agent_results = run_agent_tests(app_url, token, full=args.full)

    # ---------------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------------
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Final Report{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"\n  App: {app_url}")
    print(f"  Mode: {'Full' if args.full else 'Smoke'}"
          f"{' + Service Tests' if args.test_services else ''}")

    if service_results:
        svc_passed = sum(1 for r in service_results if r.success)
        svc_total = len(service_results)
        svc_color = GREEN if svc_passed == svc_total else RED
        print(f"\n  {BOLD}Backend Services:{RESET}  {svc_color}{svc_passed}/{svc_total} passed{RESET}")
        print(f"  {'Service':<28} {'Status':<12} {'Time':<8} {'Detail'}")
        print(f"  {'-'*28} {'-'*12} {'-'*8} {'-'*20}")
        for r in service_results:
            status = f"{GREEN}✓ PASS{RESET}" if r.success else f"{RED}✗ FAIL{RESET}"
            detail = r.detail if r.success else r.error[:30]
            print(f"  {r.name:<28} {status:<22} {r.elapsed_seconds:>6.1f}s {detail}")

    if agent_results:
        agt_passed = sum(1 for r in agent_results if r.success)
        agt_total = len(agent_results)
        agt_color = GREEN if agt_passed == agt_total else RED
        print(f"\n  {BOLD}Agent Endpoint:{RESET}    {agt_color}{agt_passed}/{agt_total} passed{RESET}")
        print(f"  {'Test Name':<28} {'Status':<12} {'Time':<8}")
        print(f"  {'-'*28} {'-'*12} {'-'*8}")
        for r in agent_results:
            status = f"{GREEN}✓ PASS{RESET}" if r.success else f"{RED}✗ FAIL{RESET}"
            print(f"  {r.test_case.name:<28} {status:<22} {r.elapsed_seconds:>6.1f}s")

    # Totals
    all_results_pass = (
        all(r.success for r in service_results) if service_results else True
    ) and (
        all(r.success for r in agent_results) if agent_results else True
    )
    total_time = (
        sum(r.elapsed_seconds for r in service_results)
        + sum(r.elapsed_seconds for r in agent_results)
    )

    if not all_results_pass:
        print(f"\n  {BOLD}Failures:{RESET}")
        for r in service_results:
            if not r.success:
                print(f"    {RED}✗{RESET} [Service] {r.name}: {r.error}")
        for r in agent_results:
            if not r.success:
                print(f"    {RED}✗{RESET} [Agent]   {r.test_case.name}: {r.error}")

    overall = f"{GREEN}{BOLD}ALL PASSED{RESET}" if all_results_pass else f"{RED}{BOLD}FAILURES DETECTED{RESET}"
    print(f"\n  Overall: {overall}   Total time: {total_time:.1f}s")
    print()
    sys.exit(0 if all_results_pass else 1)


if __name__ == "__main__":
    main()
