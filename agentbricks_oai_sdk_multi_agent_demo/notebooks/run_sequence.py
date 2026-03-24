#!/usr/bin/env python3
"""
Run the demo sequence with configurable phases.

Phase Flags:
  --ka            KA lifecycle: build, deploy, add examples, test endpoint
  --data-eng      Data engineering: ingest (03), transform (04), views (05), functions (06), Genie (07)
  --test-services Run test-services to validate KA + Genie + UC functions
  --deploy-agent  Agent lifecycle: test-agent, deploy-agent-app, test-agent-app (implies --test-services)
  --all           Full lifecycle: --ka + --data-eng + --deploy-agent

Resume/Skip:
  --from PHASE    Resume from specific phase (setup|ka|data-eng|ingest|transform|views|services|agent|deploy)
  --dry-run       Show execution plan without running

Examples:
  uv run run-sequence                      # Default: --data-eng only (backward compatible)
  uv run run-sequence --ka                 # KA lifecycle only
  uv run run-sequence --ka --data-eng      # KA + data engineering
  uv run run-sequence --data-eng --deploy-agent  # Data + agent deployment
  uv run run-sequence --all                # Full lifecycle (KA + data + agent)
  uv run run-sequence --all --quick        # Full lifecycle with quick agent tests
  uv run run-sequence --all --from services  # Resume from services phase
  uv run run-sequence --all --dry-run      # Show what would run

Uses PAT-only auth; loads config from _local/config/<workspace>.env
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from notebooks.demo_shared import bootstrap, update_central_config, run_step
from notebooks.demo_shared.subprocess_runner import print_summary
from notebooks.demo_cleanup_src.cleanup_genie import delete_genie_space

_project_root, _central_config = bootstrap(__file__)

GENIE_SPACE_NAME = os.environ.get("GENIE_SPACE_NAME", "SEC_Financial_Data_Explorer")

# Phase ordering for --from flag
PHASE_ORDER = [
    "setup",
    "ka",
    "data-eng",
    "transform",
    "views",
    "services",
    "agent",
    "deploy",
]


@dataclass
class Step:
    """Represents a single execution step."""
    label: str
    cmd: list[str]
    phase: str
    post_action: Callable[[], None] | None = None


def preflight_check() -> bool:
    """Validate essential config vars are set before running."""
    required = {
        "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", "").strip(),
        "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", "").strip(),
        "SQL_WAREHOUSE_ID": os.environ.get("SQL_WAREHOUSE_ID", "").strip(),
        "CLUSTER_ID": os.environ.get("CLUSTER_ID", "").strip(),
    }
    
    missing = [k for k, v in required.items() if not v]
    
    if missing:
        print("\n" + "=" * 60)
        print("PRE-FLIGHT CHECK FAILED")
        print("=" * 60)
        print(f"Missing required config: {', '.join(missing)}")
        print(f"\nCheck your central config: {_central_config}")
        print("=" * 60)
        return False
    
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK PASSED")
    print("=" * 60)
    print(f"  Host:      {required['DATABRICKS_HOST'][:50]}...")
    print(f"  Warehouse: {required['SQL_WAREHOUSE_ID']}")
    print(f"  Cluster:   {required['CLUSTER_ID']}")
    print("=" * 60)
    return True


def _fetch_latest_genie_space_id() -> str | None:
    """Use the Databricks SDK to list all Genie spaces matching GENIE_SPACE_NAME,
    pick the most recently created one, and delete any older duplicates.
    """
    from databricks.sdk import WorkspaceClient

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        print("  WARNING: DATABRICKS_HOST or DATABRICKS_TOKEN not set; cannot fetch Genie Space ID.")
        return None

    try:
        w = WorkspaceClient(host=host, token=token)
        all_spaces = []
        page_token = None
        while True:
            kwargs = {}
            if page_token:
                kwargs["page_token"] = page_token
            resp = w.genie.list_spaces(**kwargs)
            all_spaces.extend(resp.spaces or [])
            page_token = resp.next_page_token
            if not page_token:
                break
    except Exception as e:
        print(f"  WARNING: Failed to list Genie spaces: {e}")
        return None

    def _is_match(s) -> bool:
        t = s.title or ""
        return t == GENIE_SPACE_NAME or t.startswith(GENIE_SPACE_NAME + " (")

    matches = [s for s in all_spaces if _is_match(s)]

    if not matches:
        print(f"  WARNING: No Genie Space found with title '{GENIE_SPACE_NAME}' (or suffixed variant).")
        print(f"  Available spaces ({len(all_spaces)}): {[s.title for s in all_spaces[:10]]}")
        return None

    matches_sorted = sorted(matches, key=lambda s: getattr(s, "create_time", "") or "", reverse=True)
    best = matches_sorted[0]
    best_id = best.space_id

    if len(matches) == 1:
        print(f"  Found 1 Genie Space matching '{GENIE_SPACE_NAME}': {best_id} (title: '{best.title}')")
    else:
        print(f"  Found {len(matches)} Genie Spaces matching '{GENIE_SPACE_NAME}':")
        for s in matches_sorted:
            print(f"    - {s.space_id}  title='{s.title}'  created={getattr(s, 'create_time', '?')}")
        print(f"  Using most recently created: {best_id} (title: '{best.title}')")
        stale = matches_sorted[1:]
        print(f"  Deleting {len(stale)} older duplicate(s)...")
        for s in stale:
            try:
                delete_genie_space(host, token, s.space_id)
                print(f"    ✓ Deleted: {s.space_id} (title: '{s.title}')")
            except Exception as e:
                print(f"    WARNING: Could not delete stale space {s.space_id}: {e}")

    return best_id


def update_genie_space_id() -> None:
    """Fetch the latest Genie Space ID and update the central config file."""
    print("\n" + "=" * 60)
    print("Updating Genie Space ID in central config...")
    print("=" * 60)

    new_id = _fetch_latest_genie_space_id()
    if not new_id:
        print("  Could not determine new Genie Space ID. Skipping config update.")
        return

    print(f"  New Genie Space ID: {new_id}")

    if update_central_config(_central_config, "GENIE_SPACE_ID", new_id):
        print(f"  ✓ Updated GENIE_SPACE_ID in {_central_config}")
    
    os.environ["GENIE_SPACE_ID"] = new_id

    print(f"\n  Genie Space URL: {os.environ.get('DATABRICKS_HOST', '').rstrip('/')}/genie/spaces/{new_id}")
    print("=" * 60)


def print_phase_header(phase_name: str, flag: str = "") -> None:
    """Print a phase header."""
    print("\n" + "=" * 60)
    if flag:
        print(f"PHASE: {phase_name} ({flag})")
    else:
        print(f"PHASE: {phase_name}")
    print("=" * 60)


def should_skip_phase(phase: str, from_phase: str | None) -> bool:
    """Check if a phase should be skipped based on --from flag."""
    if not from_phase:
        return False
    
    try:
        from_idx = PHASE_ORDER.index(from_phase)
        phase_idx = PHASE_ORDER.index(phase)
        return phase_idx < from_idx
    except ValueError:
        return False


def build_steps(args) -> list[Step]:
    """Build the list of steps based on flags."""
    steps: list[Step] = []
    
    # Setup phase (always included)
    steps.append(Step(
        label="sync-workspace",
        cmd=["uv", "run", "sync-workspace"],
        phase="setup",
    ))
    
    # KA phase
    if args.ka:
        steps.append(Step(
            label="KA: build + deploy + test",
            cmd=["uv", "run", "run-ka-sequence"],
            phase="ka",
        ))
    
    # Data Engineering phase
    if args.data_eng:
        steps.append(Step(
            label="cleanup tables + pipeline",
            cmd=["uv", "run", "demo-cleanup", "tables"],
            phase="data-eng",
        ))
        # Stock loading is now handled inside the SDP pipeline (00_bronze_stock_initial.py).
        # The pipeline discovers companies via ai_extract → company_tickers_registry,
        # then fetches yfinance history automatically. No separate ingest step needed.
        steps.append(Step(
            label="deploy SDP pipeline (bronze→silver→registry→stock→gold)",
            cmd=["uv", "run", "deploy-sdp-pipeline"],
            phase="transform",
        ))
        steps.append(Step(
            label="05-07: Views + Functions + Genie",
            cmd=["uv", "run", "run-workspace-notebooks", "05", "06", "07"],
            phase="views",
            post_action=update_genie_space_id,
        ))

    # Optional: incremental stock price refresh
    if args.refresh_stocks:
        steps.append(Step(
            label="refresh-stocks (incremental yfinance update)",
            cmd=["uv", "run", "refresh-stocks"],
            phase="data-eng",
        ))
    
    # Test Services phase
    if args.test_services:
        steps.append(Step(
            label="test-services (KA + Genie + UC)",
            cmd=["uv", "run", "test-services"],
            phase="services",
        ))
    
    # Agent phase
    if args.deploy_agent:
        test_agent_cmd = ["uv", "run", "test-agent"]
        if args.quick:
            test_agent_cmd.append("--quick")
        
        steps.append(Step(
            label="test-agent (local)",
            cmd=test_agent_cmd,
            phase="agent",
        ))
        steps.append(Step(
            label="deploy-agent-app",
            cmd=["uv", "run", "deploy-agent-app", "--skip-validation"],
            phase="deploy",
        ))
        steps.append(Step(
            label="test-agent-app",
            cmd=["uv", "run", "test-agent-app"],
            phase="deploy",
        ))
    
    return steps


def print_dry_run(steps: list[Step], from_phase: str | None) -> None:
    """Print what would be executed without running."""
    print("\n" + "=" * 60)
    print("DRY RUN - Execution Plan")
    if from_phase:
        print(f"  (Resuming from: {from_phase})")
    print("=" * 60)
    
    current_phase = None
    step_num = 0
    
    for step in steps:
        skip = should_skip_phase(step.phase, from_phase)
        
        if step.phase != current_phase:
            current_phase = step.phase
            print(f"\n  Phase: {current_phase}")
        
        step_num += 1
        status = "[SKIP]" if skip else f"[{step_num:2d}]"
        cmd_str = " ".join(step.cmd)
        print(f"    {status} {step.label}")
        print(f"         {cmd_str}")
    
    print("\n" + "=" * 60)
    print("  Use without --dry-run to execute")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the demo sequence with configurable phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Phase Flags:
  --ka              KA lifecycle (build, deploy, test)
  --data-eng        Data engineering (SDP pipeline: SEC docs → AI extraction → registry → stock history → gold)
  --refresh-stocks  Incremental stock price refresh (append new rows to bronze_stock_daily_refresh)
  --test-services   Validate KA + Genie + UC functions
  --deploy-agent    Agent lifecycle (implies --test-services)
  --all             Full lifecycle (--ka + --data-eng + --deploy-agent)

Resume/Skip:
  --from PHASE    Resume from: setup|ka|data-eng|transform|views|services|agent|deploy
  --dry-run       Show execution plan without running

Examples:
  uv run run-sequence                            # Default: data-eng only
  uv run run-sequence --ka --data-eng            # KA + data engineering
  uv run run-sequence --all                      # Full lifecycle
  uv run run-sequence --all --quick              # Full lifecycle, quick tests
  uv run run-sequence --all --from services      # Resume from services
  uv run run-sequence --all --dry-run            # Show what would run
  uv run run-sequence --data-eng --refresh-stocks  # Data pipeline + incremental stock refresh
""",
    )
    
    # Phase flags
    parser.add_argument("--ka", action="store_true",
                        help="KA lifecycle: build, deploy, add examples, test endpoint")
    parser.add_argument("--data-eng", action="store_true",
                        help="Data engineering: SDP pipeline (discovers companies + loads stock history), views (05), functions (06), Genie (07)")
    parser.add_argument("--refresh-stocks", action="store_true",
                        help="Run incremental stock price refresh (refresh_stock_prices.py) after data pipeline")
    parser.add_argument("--test-services", action="store_true",
                        help="Run test-services to validate KA + Genie + UC functions")
    parser.add_argument("--deploy-agent", action="store_true",
                        help="Agent lifecycle: test-agent, deploy-agent-app, test-agent-app (implies --test-services)")
    parser.add_argument("--all", action="store_true",
                        help="Full lifecycle: --ka + --data-eng + --deploy-agent")
    
    # Resume/skip flags
    parser.add_argument("--from", dest="from_phase", metavar="PHASE",
                        choices=PHASE_ORDER,
                        help="Resume from phase: " + "|".join(PHASE_ORDER))
    parser.add_argument("--dry-run", action="store_true",
                        help="Show execution plan without running")
    
    # Convenience flags
    parser.add_argument("--quick", action="store_true",
                        help="Use quick/smoke tests for agent testing")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-flight config validation")
    
    # Deprecated flag (backward compatibility)
    parser.add_argument("--include-ka", action="store_true",
                        help=argparse.SUPPRESS)  # Hidden, deprecated
    
    args = parser.parse_args()
    
    # Handle deprecated --include-ka
    if args.include_ka:
        warnings.warn(
            "--include-ka is deprecated. Use --ka instead. "
            "Note: --ka now includes KA testing; --include-ka did not.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.ka = True
    
    # Handle --all flag
    if args.all:
        args.ka = True
        args.data_eng = True
        args.deploy_agent = True
    
    # Handle flag dependencies
    if args.deploy_agent:
        args.test_services = True
    
    # Default behavior: if no phase flags, run data-eng
    if not args.ka and not args.data_eng and not args.test_services and not args.deploy_agent:
        args.data_eng = True
    
    # Build steps
    steps = build_steps(args)
    
    # Dry run mode
    if args.dry_run:
        print_dry_run(steps, args.from_phase)
        return 0
    
    # Preflight check (skip in dry-run mode)
    if not args.skip_preflight and not preflight_check():
        return 1

    # Build phase description
    phases = []
    if args.ka:
        phases.append("KA")
    if args.data_eng:
        phases.append("Data-Eng")
    if args.test_services:
        phases.append("Test-Services")
    if args.deploy_agent:
        phases.append("Deploy-Agent")
    
    print("\n" + "=" * 60)
    print(f"RUN SEQUENCE: {' → '.join(phases)}")
    if args.from_phase:
        print(f"  (Resuming from: {args.from_phase})")
    print("=" * 60)

    results: list[tuple[str, int, float]] = []
    current_phase = None
    
    for step in steps:
        # Check if we should skip this step
        if should_skip_phase(step.phase, args.from_phase):
            if step.phase != current_phase:
                current_phase = step.phase
                print_phase_header(f"{current_phase} [SKIPPED]")
            print(f"  [SKIP] {step.label}")
            continue
        
        # Print phase header on phase change
        if step.phase != current_phase:
            current_phase = step.phase
            flag = ""
            if current_phase == "ka":
                flag = "--ka"
            elif current_phase in ("data-eng", "ingest", "transform", "views"):
                flag = "--data-eng"
            elif current_phase == "services":
                flag = "--test-services"
            elif current_phase in ("agent", "deploy"):
                flag = "--deploy-agent"
            print_phase_header(current_phase.replace("-", " ").title(), flag)
        
        # Execute step
        rc, elapsed = run_step(step.cmd, step.label, _project_root)
        results.append((step.label, rc, elapsed))
        
        if rc != 0:
            print(f"\nABORTED: '{step.label}' failed")
            print_summary(results, {"Central config": str(_central_config)})
            return 1
        
        # Run post-action if any
        if step.post_action:
            step.post_action()

    # Summary
    extra_info = {}
    
    # Pipeline URL
    _url_file = _project_root / "_local" / "last_pipeline_url.txt"
    if _url_file.exists():
        extra_info["Pipeline URL"] = _url_file.read_text(encoding="utf-8").strip()
    
    # Genie Space ID
    genie_id = os.environ.get("GENIE_SPACE_ID")
    if genie_id:
        extra_info["Genie Space ID"] = genie_id
    
    # KA Endpoint
    ka_endpoint = os.environ.get("KA_ENDPOINT", "")
    if ka_endpoint:
        extra_info["KA Endpoint"] = ka_endpoint
    
    # App URL
    app_url = os.environ.get("APP_URL", "")
    if app_url:
        extra_info["App URL"] = app_url
    
    extra_info["Central config"] = str(_central_config)

    print_summary(results, extra_info)

    return 0


if __name__ == "__main__":
    sys.exit(main())
