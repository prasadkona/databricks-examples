#!/usr/bin/env python3
"""
test_services.py - Test backend services (KA, Genie, UC functions) independently.

Run BEFORE test-agent to verify all services are working without needing the agent.

Usage:
  uv run test-services           # Test all services (KA + Genie + UC)
  uv run test-services --ka      # Test KA only
  uv run test-services --genie   # Test Genie only
  uv run test-services --uc      # Test UC functions only
  uv run test-services --verbose # Show detailed output
"""

# COMMAND ----------
# MAGIC %md
# MAGIC # Service Tests
# MAGIC
# MAGIC Tests Knowledge Assistant, Genie Space, and UC functions independently.

# COMMAND ----------

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import requests

from notebooks.demo_shared import bootstrap

_project_root, _central_config = bootstrap(__file__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Colors and Helpers

# COMMAND ----------

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


# COMMAND ----------
# MAGIC %md
# MAGIC ## Service Result Tracking

# COMMAND ----------

@dataclass
class ServiceResult:
    """Result of a single service test."""
    name: str
    success: bool
    elapsed_seconds: float
    detail: str = ""
    error: str = ""


# COMMAND ----------
# MAGIC %md
# MAGIC ## UC Function Tests

# COMMAND ----------

def test_uc_functions(host: str, token: str, verbose: bool = False) -> list[ServiceResult]:
    """Test UC functions: valuation_score, compare_peers, growth_trajectory, risk_summary."""
    results: list[ServiceResult] = []
    
    uc_catalog = os.getenv("UC_CATALOG", "your_catalog")
    uc_schema = os.getenv("UC_SCHEMA", "your_schema")
    table_prefix = os.getenv("TABLE_PREFIX", "sec_fin_")
    warehouse_id = os.getenv("SQL_WAREHOUSE_ID", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    if not warehouse_id:
        print_warning("SQL_WAREHOUSE_ID not set - skipping UC function tests")
        return results
    
    functions = [
        ("valuation_score", f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}valuation_score('NVDA')"),
        ("compare_peers", f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}compare_peers('NVDA')"),
        ("growth_trajectory", f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}growth_trajectory('NVDA')"),
        ("risk_summary", f"SELECT * FROM {uc_catalog}.{uc_schema}.{table_prefix}risk_summary('NVDA')"),
    ]
    
    for func_name, sql in functions:
        print_step(f"Testing UC Function: {func_name}...")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{host}/api/2.0/sql/statements",
                json={
                    "warehouse_id": warehouse_id,
                    "statement": sql,
                    "wait_timeout": "30s",
                },
                headers=headers, timeout=60,
            )
            elapsed = time.time() - t0
            data = resp.json()
            status = data.get("status", {}).get("state", "")
            rows = data.get("result", {}).get("data_array", [])
            
            if status == "SUCCEEDED" and rows:
                detail = f"{len(rows)} row(s)"
                if func_name == "valuation_score" and len(rows[0]) > 2:
                    detail = f"Score: {rows[0][2]}/100"
                print_success(f"{func_name}('NVDA') → {detail} ({elapsed:.1f}s)")
                if verbose:
                    print_info(f"  First row: {rows[0][:3]}...")
                results.append(ServiceResult(f"UC: {func_name}", True, elapsed, detail))
            else:
                err = data.get("status", {}).get("error", {}).get("message", f"status={status}")
                print_error(f"{func_name} failed: {err}")
                results.append(ServiceResult(f"UC: {func_name}", False, elapsed, error=err))
        except Exception as e:
            results.append(ServiceResult(f"UC: {func_name}", False, time.time() - t0, error=str(e)))
            print_error(f"UC Function {func_name} test failed: {e}")
    
    return results


# COMMAND ----------
# MAGIC %md
# MAGIC ## Genie Space Test

# COMMAND ----------

def test_genie_space(host: str, token: str, verbose: bool = False) -> list[ServiceResult]:
    """Test Genie Space with a sample query."""
    results: list[ServiceResult] = []
    
    genie_space_id = os.getenv("GENIE_SPACE_ID", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    if not genie_space_id:
        print_warning("GENIE_SPACE_ID not set - skipping Genie test")
        return results
    
    print_step(f"Testing Genie Space ({genie_space_id})...")
    test_query = "What is NVIDIA's total revenue for FY2024?"
    print_info(f"Query: {test_query}")
    
    t0 = time.time()
    try:
        # Start conversation
        resp = requests.post(
            f"{host}/api/2.0/genie/spaces/{genie_space_id}/start-conversation",
            json={"content": test_query},
            headers=headers, timeout=60,
        )
        
        if resp.status_code != 200:
            err = f"HTTP {resp.status_code}: {resp.text[:120]}"
            print_error(f"Failed to start conversation: {err}")
            results.append(ServiceResult("Genie Space", False, time.time() - t0, error=err))
            return results
        
        data = resp.json()
        conv_id = data.get("conversation_id", "")
        msg_id = data.get("message_id", "")
        print_info(f"Conversation: {conv_id[:20]}...")
        
        # Poll for completion
        for i in range(12):
            time.sleep(5)
            poll = requests.get(
                f"{host}/api/2.0/genie/spaces/{genie_space_id}/conversations/{conv_id}/messages/{msg_id}",
                headers=headers, timeout=30,
            )
            if poll.status_code == 200:
                status = poll.json().get("status", "")
                if verbose:
                    print_info(f"  Poll {i+1}/12: {status}")
                if status == "COMPLETED":
                    elapsed = time.time() - t0
                    result_data = poll.json()
                    # Extract SQL or answer if available
                    attachments = result_data.get("attachments", [])
                    detail = f"completed in {elapsed:.1f}s"
                    if attachments:
                        detail = f"{len(attachments)} attachment(s), {elapsed:.1f}s"
                    print_success(f"Genie Space query {detail}")
                    results.append(ServiceResult("Genie Space", True, elapsed, detail))
                    return results
                elif status in ("FAILED", "CANCELLED"):
                    elapsed = time.time() - t0
                    print_error(f"Genie query failed: {status}")
                    results.append(ServiceResult("Genie Space", False, elapsed, error=status))
                    return results
        
        # Timeout
        elapsed = time.time() - t0
        print_warning(f"Genie query timed out after {elapsed:.1f}s")
        results.append(ServiceResult("Genie Space", False, elapsed, error="Timeout"))
        
    except Exception as e:
        results.append(ServiceResult("Genie Space", False, time.time() - t0, error=str(e)))
        print_error(f"Genie Space test failed: {e}")
    
    return results


# COMMAND ----------
# MAGIC %md
# MAGIC ## Knowledge Assistant Test

# COMMAND ----------

def test_knowledge_assistant(host: str, token: str, verbose: bool = False) -> list[ServiceResult]:
    """Test Knowledge Assistant endpoint with a sample question."""
    results: list[ServiceResult] = []
    
    ka_endpoint = os.getenv("KA_ENDPOINT", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    if not ka_endpoint:
        print_warning("KA_ENDPOINT not set - skipping KA test")
        return results
    
    print_step(f"Testing Knowledge Assistant ({ka_endpoint})...")
    test_query = "What is NVIDIA's fiscal year end date according to their 10-K?"
    print_info(f"Query: {test_query}")
    
    t0 = time.time()
    try:
        resp = requests.post(
            f"{host}/serving-endpoints/{ka_endpoint}/invocations",
            json={"input": [{"role": "user", "content": test_query}]},
            headers=headers, timeout=120,
        )
        elapsed = time.time() - t0
        
        if resp.status_code == 200:
            response_data = resp.json()
            # Try to extract response text
            output_text = ""
            if isinstance(response_data, dict):
                output_text = response_data.get("output_text", "")
                if not output_text:
                    output_text = str(response_data)[:200]
            text_len = len(output_text)
            
            print_success(f"KA endpoint responded ({text_len} chars, {elapsed:.1f}s)")
            if verbose and output_text:
                print_info(f"  Response preview: {output_text[:150]}...")
            results.append(ServiceResult("Knowledge Assistant", True, elapsed, f"{text_len} chars"))
        else:
            err = f"HTTP {resp.status_code}: {resp.text[:120]}"
            print_error(f"KA endpoint: {err}")
            results.append(ServiceResult("Knowledge Assistant", False, elapsed, error=err))
            
    except Exception as e:
        results.append(ServiceResult("Knowledge Assistant", False, time.time() - t0, error=str(e)))
        print_error(f"KA test failed: {e}")
    
    return results


# COMMAND ----------
# MAGIC %md
# MAGIC ## Report Generator

# COMMAND ----------

def print_report(results: list[ServiceResult]):
    """Print a summary report of all service test results."""
    print(f"\n{'='*60}")
    print(f"  SERVICE TEST REPORT")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_time = sum(r.elapsed_seconds for r in results)
    
    print(f"\n  Total: {len(results)} tests | Passed: {passed} | Failed: {failed}")
    print(f"  Total time: {total_time:.1f}s")
    print()
    
    for r in results:
        status = f"{GREEN}PASS{RESET}" if r.success else f"{RED}FAIL{RESET}"
        time_str = f"{r.elapsed_seconds:.1f}s"
        detail = r.detail if r.success else r.error
        detail_str = f" - {detail}" if detail else ""
        print(f"  [{status}] {r.name}: {time_str}{detail_str}")
    
    print(f"\n{'='*60}")
    
    if failed > 0:
        print(f"\n{RED}Some services failed. Fix issues before running test-agent.{RESET}")
        return 1
    else:
        print(f"\n{GREEN}All services passed! Ready for: uv run test-agent{RESET}")
        return 0


# COMMAND ----------
# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test backend services (KA, Genie, UC functions) independently."
    )
    parser.add_argument("--ka", action="store_true", help="Test Knowledge Assistant only")
    parser.add_argument("--genie", action="store_true", help="Test Genie Space only")
    parser.add_argument("--uc", action="store_true", help="Test UC functions only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # If no specific flags, test all
    test_all = not (args.ka or args.genie or args.uc)
    
    # Get credentials
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    
    if not host or not token:
        print_error("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        print_info(f"Check your config: {_central_config}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"  BACKEND SERVICE TESTS")
    print(f"{'='*60}")
    print(f"  Host: {host[:50]}...")
    print(f"  Config: {_central_config}")
    
    results: list[ServiceResult] = []
    
    # Run requested tests
    if test_all or args.uc:
        results.extend(test_uc_functions(host, token, args.verbose))
    
    if test_all or args.genie:
        results.extend(test_genie_space(host, token, args.verbose))
    
    if test_all or args.ka:
        results.extend(test_knowledge_assistant(host, token, args.verbose))
    
    if not results:
        print_warning("No services to test. Check that IDs are configured.")
        return 1
    
    return print_report(results)


if __name__ == "__main__":
    sys.exit(main())
