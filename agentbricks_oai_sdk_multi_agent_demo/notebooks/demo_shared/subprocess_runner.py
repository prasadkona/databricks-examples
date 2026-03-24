# Databricks notebook source
# MAGIC %md
# MAGIC # Subprocess Runner: Step Execution with Timing
# MAGIC
# MAGIC Consolidates the subprocess-with-banner pattern from `run_sequence.py` and `run_ka_sequence.py`.
# MAGIC Provides standardized subprocess execution with timing and status output.
# MAGIC
# MAGIC **Usage:**
# MAGIC ```python
# MAGIC from notebooks.demo_shared.subprocess_runner import run_step
# MAGIC rc, elapsed = run_step(["uv", "run", "create-ka"], "Create KA", cwd)
# MAGIC ```

# COMMAND ----------

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Step with Banner and Timing

# COMMAND ----------

def run_step(
    cmd: list[str],
    label: str,
    cwd: Path | str,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    """Run a subprocess with banner output and timing.
    
    Args:
        cmd: Command and arguments to execute.
        label: Human-readable step description.
        cwd: Working directory for the subprocess.
        env: Environment variables (defaults to os.environ).
        
    Returns:
        Tuple of (returncode, elapsed_seconds).
    """
    print(f"\n{'=' * 60}")
    print(f">>> {label}")
    print(f"    {' '.join(cmd)}")
    print(f"{'=' * 60}")
    
    t0 = time.time()
    process_env = env if env is not None else os.environ.copy()
    result = subprocess.run(cmd, cwd=cwd, env=process_env)
    elapsed = time.time() - t0
    
    status = "COMPLETED" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"<<< {status}: {label}  [{elapsed:.1f}s]")
    print(f"{'=' * 60}")
    
    return result.returncode, elapsed


def print_summary(
    results: list[tuple[str, int, float]],
    extra_info: dict[str, str] | None = None,
) -> None:
    """Print a summary table of step results.
    
    Args:
        results: List of (label, returncode, elapsed) tuples.
        extra_info: Optional dict of additional info to display (e.g., IDs, URLs).
    """
    total_time = sum(elapsed for _, _, elapsed in results)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    
    all_ok = True
    for label, rc, elapsed in results:
        icon = "OK" if rc == 0 else "FAIL"
        print(f"  [{icon:4s}] {label:40s} {elapsed:6.1f}s")
        if rc != 0:
            all_ok = False
    
    print(f"  {'-' * 52}")
    print(f"  Total: {total_time:.1f}s")
    
    if extra_info:
        print()
        for key, value in extra_info.items():
            if value:
                print(f"  {key}: {value}")
    
    print(f"{'=' * 60}")
