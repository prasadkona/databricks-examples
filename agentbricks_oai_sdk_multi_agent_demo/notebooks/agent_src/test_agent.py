#!/usr/bin/env python3
"""
test_agent.py - Local testing for SEC Financial Analyst Multi-Agent

Execution order: run after pipeline and notebooks 05, 06, 07 are done.
Starts the agent server, sends test queries, validates traces.

  uv run test-agent                        # default: one smoke test (Genie+KA+UC)
  uv run test-agent --full                 # run all 8 test cases
  uv run test-agent --skip-trace-validation
  uv run test-agent --no-start             # assume server already running
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from notebooks.demo_shared import bootstrap, get_app_dir
from notebooks.agent_src.trace_validator import TraceValidator

_project_root, _central_config = bootstrap(__file__, override=False)


# Configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
INVOCATION_URL = f"{SERVER_URL}/invocations"
HEALTH_URL = f"{SERVER_URL}/health"
EXPERIMENT_ID = None  # Set via --experiment-id or MLFLOW_EXPERIMENT_ID env var


@dataclass
class TestCase:
    """Defines a test case with expected tool usage."""
    name: str
    query: str
    description: str
    expected_tools: list[str]  # Tools that should be called
    timeout: int = 120
    response_contains: list[str] = None  # Optional: strings that should appear in response


# Smoke test: one comprehensive query that exercises all three tool types
SMOKE_TEST = TestCase(
    name="Smoke: Genie + KA + UC",
    query=(
        "Give me a quick investment overview for NVIDIA: "
        "What was their total FY2024 revenue, "
        "what does their 10-K SEC filing say about key risk factors, "
        "and what is their current valuation score?"
    ),
    description="Comprehensive smoke test hitting Genie (revenue), KA (10-K risk factors), and UC function (valuation score)",
    expected_tools=["genie", "knowledge_assistant", "valuation_score"],
    response_contains=["NVIDIA", "revenue", "risk", "valuation"],
    timeout=180,
)

# Full test suite - individual and multi-tool test cases
FULL_TEST_CASES = [
    TestCase(
        name="Genie: Revenue Query",
        query="What is NVIDIA's total revenue for FY2024?",
        description="Tests Genie Space for structured financial data",
        expected_tools=["genie"],
        response_contains=["NVIDIA", "revenue", "60"],  # ~$60B
    ),
    TestCase(
        name="UC Function: Valuation Score",
        query="What is NVIDIA's valuation score?",
        description="Tests UC valuation_score function",
        expected_tools=["valuation_score"],
        response_contains=["valuation", "score", "NVIDIA"],
    ),
    TestCase(
        name="UC Function: Peer Comparison",
        query="How does NVIDIA compare to its peers on revenue growth and margins?",
        description="Tests UC compare_peers function",
        expected_tools=["compare_peers"],
        response_contains=["NVIDIA", "peer", "margin"],
    ),
    TestCase(
        name="UC Function: Risk Summary",
        query="What are the investment risks for NVIDIA stock?",
        description="Tests UC risk_summary function",
        expected_tools=["risk_summary"],
        response_contains=["risk", "NVIDIA"],
    ),
    TestCase(
        name="Knowledge Assistant: Risk Factors",
        query="What are the key risk factors mentioned in Apple's 10-K SEC filing?",
        description="Tests Knowledge Assistant for document analysis",
        expected_tools=["knowledge_assistant"],
        response_contains=["Apple", "risk"],
    ),
    # Multi-tool test cases
    TestCase(
        name="Multi: Genie + UC Function",
        query="What is NVIDIA's revenue for FY2024 and how does its valuation compare to Apple and Samsung?",
        description="Tests multi-tool: Genie for revenue data + compare_peers for valuation comparison",
        expected_tools=["genie", "compare_peers"],
        response_contains=["NVIDIA", "revenue"],
    ),
    TestCase(
        name="Multi: Genie + KA",
        query="What was Apple's total revenue last year and what risk factors from their SEC 10-K filing could impact future revenue?",
        description="Tests multi-tool: Genie for financials + Knowledge Assistant for filing analysis",
        expected_tools=["genie", "knowledge_assistant"],
        response_contains=["Apple", "revenue", "risk"],
        timeout=150,
    ),
    TestCase(
        name="Multi: UC Function + KA",
        query="Give me NVIDIA's investment risk summary and also summarize what their 10-K SEC filing says about competition.",
        description="Tests multi-tool: risk_summary UC function + Knowledge Assistant for document analysis",
        expected_tools=["risk_summary", "knowledge_assistant"],
        response_contains=["NVIDIA", "risk", "compet"],
        timeout=150,
    ),
]

# TEST_CASES is what local_09 imports; defaults to smoke only, set by main()
TEST_CASES = [SMOKE_TEST]


@dataclass 
class TestResult:
    """Result of a single test execution."""
    test_case: TestCase
    success: bool
    response_text: str
    elapsed_seconds: float
    trace_validated: bool = False
    tools_found: list[str] = None
    missing_tools: list[str] = None
    error: Optional[str] = None


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_step(text: str) -> None:
    print(f"\n→ {text}")


def print_success(text: str) -> None:
    print(f"  ✓ {text}")


def print_error(text: str) -> None:
    print(f"  ✗ {text}")


def print_info(text: str) -> None:
    print(f"  ℹ {text}")


def check_env_vars() -> bool:
    print_step("Checking environment variables...")
    
    if _central_config.exists():
        print_success(f"Loaded config from {_central_config}")
    else:
        print_info(f"Central config not found at {_central_config}")
    
    required_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
    missing = []
    
    for var in required_vars:
        if os.environ.get(var):
            print_success(f"{var} is set")
        else:
            missing.append(var)
            print_error(f"{var} is not set")
    
    if missing:
        print(f"\nPlease set the missing environment variables in {_central_config}")
        return False
    
    return True


def wait_for_server(timeout: int = 90) -> bool:
    print_step(f"Waiting for server to be ready (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                print_success(f"Server is ready at {SERVER_URL}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        elapsed = int(time.time() - start_time)
        print(f"    Waiting... ({elapsed}s)", end="\r")
        time.sleep(2)
    
    print_error(f"Server did not become ready within {timeout} seconds")
    return False


def start_server() -> subprocess.Popen:
    print_step("Starting agent server from app/ directory...")

    app_dir = get_app_dir()

    excluded_vars = {"VIRTUAL_ENV", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET", "DATABRICKS_CONFIG_PROFILE"}
    server_env = {k: v for k, v in os.environ.items() if k not in excluded_vars}
    server_env["PORT"] = str(SERVER_PORT)

    process = subprocess.Popen(
        ["uv", "run", "start-server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(app_dir),
        env=server_env,
    )

    print_success(f"Server process started (PID: {process.pid}) in {app_dir}")
    return process


def refresh_experiment_id_from_env():
    """Get experiment ID from environment (loaded from central config)."""
    global EXPERIMENT_ID
    
    if EXPERIMENT_ID:
        return  # Already set via CLI arg
    
    EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if EXPERIMENT_ID:
        print_success(f"Using MLflow Experiment ID: {EXPERIMENT_ID}")
    else:
        print_info("No MLFLOW_EXPERIMENT_ID found - trace validation will be skipped")


def stop_server(process: subprocess.Popen) -> None:
    if process:
        print_step("Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print_success("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            process.kill()
            print_success("Server killed")


def send_query(query: str, timeout: int = 120) -> dict:
    """Send a query to the agent and return the response."""
    payload = {
        "input": [
            {"role": "user", "content": query}
        ]
    }
    
    response = requests.post(
        INVOCATION_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    
    return {
        "status_code": response.status_code,
        "response": response.json() if response.ok else response.text,
    }


def extract_response_text(response: dict) -> str:
    """Extract the text content from the agent response."""
    try:
        output = response.get("response", {}).get("output", [])
        texts = []
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
        return "\n".join(texts) if texts else str(output)
    except Exception:
        return str(response)


def validate_response_content(response_text: str, expected_strings: list[str]) -> tuple[bool, list[str]]:
    """Check if response contains expected strings."""
    if not expected_strings:
        return True, []
    
    response_lower = response_text.lower()
    missing = []
    
    for expected in expected_strings:
        if expected.lower() not in response_lower:
            missing.append(expected)
    
    return len(missing) == 0, missing


def run_test(test_case: TestCase, validate_traces: bool = True) -> TestResult:
    """Execute a single test case and optionally validate traces."""
    print(f"\n--- {test_case.name} ---")
    print(f"    Query: \"{test_case.query[:60]}...\"" if len(test_case.query) > 60 else f"    Query: \"{test_case.query}\"")
    print(f"    Expected tools: {test_case.expected_tools}")
    
    try:
        start_time = time.time()
        result = send_query(test_case.query, timeout=test_case.timeout)
        elapsed = time.time() - start_time
        
        if result["status_code"] != 200:
            return TestResult(
                test_case=test_case,
                success=False,
                response_text="",
                elapsed_seconds=elapsed,
                error=f"HTTP {result['status_code']}: {str(result['response'])[:200]}",
            )
        
        response_text = extract_response_text(result)
        
        # Check for errors in response
        if "error" in response_text.lower() and "max turns" in response_text.lower():
            return TestResult(
                test_case=test_case,
                success=False,
                response_text=response_text[:300],
                elapsed_seconds=elapsed,
                error="Max turns exceeded - agent got stuck",
            )
        
        # Validate response content
        content_valid, missing_content = validate_response_content(
            response_text, 
            test_case.response_contains or []
        )
        
        if not content_valid:
            print_info(f"Response missing expected content: {missing_content}")
        
        print_success(f"Response received in {elapsed:.1f}s ({len(response_text)} chars)")
        
        # Validate traces if enabled
        trace_validated = False
        tools_found = []
        missing_tools = []
        
        if validate_traces:
            try:
                # Get experiment ID from global variable (set from CLI arg or env var)
                experiment_id = EXPERIMENT_ID
                if not experiment_id:
                    print_info("Trace validation skipped: use --experiment-id or set MLFLOW_EXPERIMENT_ID")
                else:
                    validator = TraceValidator(experiment_id=experiment_id)
                    trace_result = validator.validate_latest_trace(
                        expected_tools=test_case.expected_tools,
                        wait_seconds=3,
                        max_attempts=2,
                    )
                    
                    trace_validated = trace_result.success
                    tools_found = [t.name for t in trace_result.tools_called]
                    missing_tools = trace_result.missing_tools
                    
                    if trace_validated:
                        print_success(f"Trace validated: {tools_found}")
                    elif trace_result.error:
                        print_info(f"Trace validation: {trace_result.error}")
                    else:
                        print_info(f"Trace validation: tools found {tools_found}, missing {missing_tools}")
                    
            except Exception as e:
                print_info(f"Trace validation error: {e}")
        
        return TestResult(
            test_case=test_case,
            success=True,
            response_text=response_text[:500],
            elapsed_seconds=elapsed,
            trace_validated=trace_validated,
            tools_found=tools_found,
            missing_tools=missing_tools,
        )
        
    except requests.exceptions.Timeout:
        return TestResult(
            test_case=test_case,
            success=False,
            response_text="",
            elapsed_seconds=test_case.timeout,
            error=f"Request timed out (>{test_case.timeout}s)",
        )
    except Exception as e:
        return TestResult(
            test_case=test_case,
            success=False,
            response_text="",
            elapsed_seconds=0,
            error=str(e),
        )


def run_tests(test_cases: list[TestCase], validate_traces: bool = True) -> list[TestResult]:
    """Run the given test cases and return results."""
    label = "Full Test Suite" if len(test_cases) > 1 else "Smoke Test (Genie + KA + UC)"
    print_header(f"Running {label}")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}]")
        result = run_test(test_case, validate_traces=validate_traces)
        results.append(result)
        
        if i < len(test_cases):
            time.sleep(2)
    
    return results


def print_summary(results: list[TestResult]) -> None:
    """Print a summary of all test results."""
    print_header("Test Results Summary")
    
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    trace_validated = sum(1 for r in results if r.trace_validated)
    
    if EXPERIMENT_ID:
        print(f"\n  MLflow Experiment ID: {EXPERIMENT_ID}")
    
    print(f"\n  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Traces Validated: {trace_validated}/{len(results)}")
    
    print(f"\n  {'Test Name':<35} {'Status':<10} {'Time':<8} {'Trace':<8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")
    
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        trace = "✓" if result.trace_validated else "-"
        print(f"  {result.test_case.name:<35} {status:<10} {result.elapsed_seconds:>6.1f}s {trace:^8}")
    
    if failed > 0:
        print(f"\n  Failed Tests:")
        for result in results:
            if not result.success:
                print(f"    - {result.test_case.name}: {result.error}")
    
    total_time = sum(r.elapsed_seconds for r in results)
    print(f"\n  Total execution time: {total_time:.1f}s")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SEC Financial Analyst Agent locally")
    parser.add_argument("--full", action="store_true", help="Run all test cases (default: one smoke test hitting Genie+KA+UC)")
    parser.add_argument("--no-start", action="store_true", help="Don't start server (assume already running)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--skip-trace-validation", action="store_true", help="Skip MLflow trace validation")
    parser.add_argument("--experiment-id", type=str, help="MLflow experiment ID for trace validation")
    # backward compat: --quick is now the default, accepted but ignored
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    # Store experiment ID globally for trace validation
    global EXPERIMENT_ID
    EXPERIMENT_ID = args.experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    
    global SERVER_PORT, SERVER_URL, INVOCATION_URL, HEALTH_URL
    SERVER_PORT = args.port
    SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
    INVOCATION_URL = f"{SERVER_URL}/invocations"
    HEALTH_URL = f"{SERVER_URL}/health"

    # Select test cases: --full runs everything, default is smoke only
    global TEST_CASES
    if args.full:
        TEST_CASES = FULL_TEST_CASES
    else:
        TEST_CASES = [SMOKE_TEST]
    
    print_header("SEC Financial Analyst Agent - Local Test")
    
    if not check_env_vars():
        sys.exit(1)
    
    server_process = None
    
    try:
        if not args.no_start:
            server_process = start_server()
        
        if not wait_for_server(timeout=90):
            if server_process:
                print("\nServer output (last 20 lines):")
                try:
                    for i, line in enumerate(server_process.stdout):
                        print(f"  {line.rstrip()}")
                        if i >= 20 or server_process.poll() is not None:
                            break
                except:
                    pass
            sys.exit(1)
        
        if not args.skip_trace_validation:
            refresh_experiment_id_from_env()
        
        results = run_tests(TEST_CASES, validate_traces=not args.skip_trace_validation)
        print_summary(results)
        
        failed = sum(1 for r in results if not r.success)
        sys.exit(0 if failed == 0 else 1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    
    finally:
        if server_process:
            stop_server(server_process)


if __name__ == "__main__":
    main()
