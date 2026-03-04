# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Endpoint Analysis Report
# MAGIC 
# MAGIC This notebook reads the extracted endpoint metadata from `all_endpoints_latest.json`
# MAGIC and performs analysis grouped by `_metadata_derived.model_type`.
# MAGIC 
# MAGIC ## model_type Categories
# MAGIC 
# MAGIC | Category | Description |
# MAGIC |----------|-------------|
# MAGIC | `DATABRICKS_FM_PPT` | Pay-per-token Foundation Models |
# MAGIC | `DATABRICKS_FM_PT` | Provisioned Throughput Foundation Models |
# MAGIC | `DATABRICKS_FM_UC_SYSTEM_AI` | UC Models backed by system.ai.* |
# MAGIC | `DATABRICKS_FM_UC_AGENTS` | UC Models with agent tasks |
# MAGIC | `DATABRICKS_CLASSIC_ML` | UC Models without task (classic ML) |
# MAGIC | `AGENT_BRICKS_KA` | Knowledge Assistant (Agent Bricks) |
# MAGIC | `AGENT_BRICKS_MAS` | Multi-Agent Supervisor (Agent Bricks) |
# MAGIC | `AGENT_BRICKS_KIE` | Key Info Extraction (Agent Bricks) |
# MAGIC | `AGENT_BRICKS_MS` | Model Specialization (Agent Bricks) |
# MAGIC | `FM_EXTERNAL_MODEL` | External provider models (OpenAI, etc.) |
# MAGIC | `FM_EXTERNAL_MODEL_CUSTOM` | Custom external models |
# MAGIC | `FEATURE_SPEC` | Feature serving endpoints |
# MAGIC | `UNCLASSIFIED` | Endpoints not matching any category |
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC Run `01_extract_ai_endpoints_detailed.py` first to generate the data file.
# MAGIC 
# MAGIC ## Usage
# MAGIC 
# MAGIC ```bash
# MAGIC cd ai_agent_metadata_extract
# MAGIC python 02_generate_endpoint_analysis_report.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any

# Ensure unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the latest extracted data
DATA_FILE = Path("../_local/reports/all_endpoints_latest.json")

# Export detailed JSON files per model_type to _detailed subfolder
EXPORT_DETAILED_BY_MODEL_TYPE = True  # Set to False to skip detailed exports

if not DATA_FILE.exists():
    print(f"Error: Data file not found: {DATA_FILE}")
    print("Run 01_extract_ai_endpoints_detailed.py first to extract endpoint data.")
    sys.exit(1)

print(f"Loading data from: {DATA_FILE}")

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

# JSON is now a list of endpoints (no wrapper)
if isinstance(data, list):
    endpoints = data
else:
    # Legacy format support
    endpoints = data.get("endpoints", [])

print(f"Loaded {len(endpoints)} endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis Functions

# COMMAND ----------

def count_by_field(endpoints: List[Dict], field: str) -> Dict[str, int]:
    """Count endpoints by a specific field value."""
    counts = Counter()
    for ep in endpoints:
        value = ep.get(field, "unknown") or "unknown"
        counts[value] += 1
    return dict(counts.most_common())


def filter_endpoints(endpoints: List[Dict], **filters) -> List[Dict]:
    """
    Filter endpoints by field values.
    
    Example:
        filter_endpoints(endpoints, entity_type="EXTERNAL_MODEL", ready_state="READY")
    """
    result = endpoints
    for field, value in filters.items():
        if isinstance(value, list):
            result = [ep for ep in result if ep.get(field) in value]
        else:
            result = [ep for ep in result if ep.get(field) == value]
    return result


def print_table(title: str, data: Dict[str, Any], max_rows: int = 20):
    """Print a formatted table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    items = list(data.items())[:max_rows]
    if not items:
        print("  (no data)")
        return
    
    max_key_len = max(len(str(k)) for k, _ in items)
    for key, value in items:
        print(f"  {str(key):<{max_key_len}}  {value}")
    
    if len(data) > max_rows:
        print(f"  ... and {len(data) - max_rows} more")


def print_endpoints(title: str, endpoints: List[Dict], max_rows: int = 10, show_tile_info: bool = False):
    """Print a list of endpoints."""
    print(f"\n{'='*60}")
    print(f" {title} ({len(endpoints)} total)")
    print(f"{'='*60}")
    
    for ep in endpoints[:max_rows]:
        name = ep.get("endpoint_name", ep.get("name", "unknown"))
        model = ep.get("model_name", "")
        provider = ep.get("model_provider", "")
        task = ep.get("task", "")
        state = ep.get("ready_state", "")
        print(f"  - {name}")
        if provider:
            print(f"      Provider: {provider}")
        if model:
            print(f"      Model: {model}")
        if task:
            print(f"      Task: {task}")
        print(f"      State: {state}")
        
        # Show tile metadata for Agent Bricks
        tile = ep.get("tile_metadata")
        if tile and (show_tile_info or tile.get("tile_name")):
            tile_name = tile.get("tile_name", "")
            tile_desc = tile.get("tile_description", "")
            if tile_name:
                print(f"      Tile Name: {tile_name}")
            if tile_desc:
                desc_truncated = tile_desc[:70] + "..." if len(tile_desc) > 70 else tile_desc
                print(f"      Description: {desc_truncated}")
    
    if len(endpoints) > max_rows:
        print(f"\n  ... and {len(endpoints) - max_rows} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall Summary

# COMMAND ----------

def get_model_type(ep: Dict) -> str:
    """Get model_type from _metadata_derived."""
    return ep.get("_metadata_derived", {}).get("model_type", "UNCLASSIFIED")


def count_by_model_type(endpoints: List[Dict]) -> Dict[str, int]:
    """Count endpoints by model_type."""
    counts = Counter(get_model_type(ep) for ep in endpoints)
    return dict(counts.most_common())


print("\n" + "="*60)
print(" ENDPOINT METADATA ANALYSIS")
print("="*60)

print(f"\nTotal Endpoints: {len(endpoints)}")

# By model_type (primary classification)
print_table("By model_type", count_by_model_type(endpoints))

# By entity type (raw API values)
print_table("By Entity Type", count_by_field(endpoints, "entity_type"))

# By task
print_table("By Task", count_by_field(endpoints, "task"))

# By ready state
print_table("By Ready State", count_by_field(endpoints, "ready_state"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation Model API Analysis
# MAGIC 
# MAGIC Databricks-hosted LLMs (Llama, DBRX, GPT-OSS, Gemma)

# COMMAND ----------

print("\n" + "#"*60)
print(" DATABRICKS FOUNDATION MODELS")
print("#"*60)

# Filter by model_type
fm_endpoints = [ep for ep in endpoints if get_model_type(ep) in [
    "DATABRICKS_FM_PPT", "DATABRICKS_FM_PT", "DATABRICKS_FM_UC_SYSTEM_AI"
]]

print(f"\nTotal Databricks FM endpoints: {len(fm_endpoints)}")

# By model_type
print_table("By model_type", count_by_model_type(fm_endpoints))

# By model name
print_table("Top Models", count_by_field(fm_endpoints, "model_name"), max_rows=15)

# By task
print_table("By Task", count_by_field(fm_endpoints, "task"))

# Sample endpoints
print_endpoints("Sample FM Endpoints", fm_endpoints, max_rows=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## External Model Analysis
# MAGIC 
# MAGIC External provider models via AI Gateway (OpenAI, Anthropic, Google, etc.)

# COMMAND ----------

print("\n" + "#"*60)
print(" EXTERNAL MODELS (OpenAI, Anthropic, Google, etc.)")
print("#"*60)

# Filter by model_type
ext_endpoints = [ep for ep in endpoints if get_model_type(ep) in [
    "FM_EXTERNAL_MODEL", "FM_EXTERNAL_MODEL_CUSTOM"
]]

print(f"\nTotal External Model endpoints: {len(ext_endpoints)}")

# By model_type
print_table("By model_type", count_by_model_type(ext_endpoints))

# By provider
print_table("By Provider", count_by_field(ext_endpoints, "model_provider"))

# By model name
print_table("By Model", count_by_field(ext_endpoints, "model_name"), max_rows=15)

# Sample endpoints with details
print_endpoints("Sample External Model Endpoints", ext_endpoints, max_rows=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Endpoints Analysis

# COMMAND ----------

print("\n" + "#"*60)
print(" AGENT BRICKS ENDPOINTS")
print("#"*60)

# Filter by model_type (Agent Bricks)
agent_bricks_endpoints = [ep for ep in endpoints if get_model_type(ep) in [
    "AGENT_BRICKS_KA", "AGENT_BRICKS_MAS", "AGENT_BRICKS_KIE", "AGENT_BRICKS_MS"
]]

print(f"\nTotal Agent Bricks endpoints: {len(agent_bricks_endpoints)}")

# By model_type
print_table("By model_type", count_by_model_type(agent_bricks_endpoints))

# By task
print_table("By Task", count_by_field(agent_bricks_endpoints, "task"))

# By tile_metadata.problem_type
def get_tile_problem_type(ep: Dict) -> str:
    tile = ep.get("tile_metadata") or {}
    return tile.get("problem_type", "unknown")

tile_problem_counts = Counter(get_tile_problem_type(ep) for ep in agent_bricks_endpoints)
print_table("By problem_type", dict(tile_problem_counts.most_common()))

# Show tile names and descriptions
def print_agent_bricks_details(endpoints: List[Dict], max_rows: int = 10):
    """Print Agent Bricks tile details."""
    print(f"\n{'='*60}")
    print(f" Agent Bricks Tile Details ({len(endpoints)} total)")
    print(f"{'='*60}")
    
    for ep in endpoints[:max_rows]:
        tile = ep.get("tile_metadata") or {}
        endpoint_name = ep.get("endpoint_name", "")
        tile_name = tile.get("tile_name", "N/A")
        tile_desc = tile.get("tile_description", "")
        tile_type = tile.get("problem_type", "")
        
        print(f"\n  [{tile_type}] {tile_name}")
        print(f"    Endpoint: {endpoint_name}")
        if tile_desc:
            desc_truncated = tile_desc[:100] + "..." if len(tile_desc) > 100 else tile_desc
            print(f"    Description: {desc_truncated}")
    
    if len(endpoints) > max_rows:
        print(f"\n  ... and {len(endpoints) - max_rows} more")

print_agent_bricks_details(agent_bricks_endpoints, max_rows=8)

# Sample endpoints (with tile info)
print_endpoints("Sample Agent Bricks Endpoints", agent_bricks_endpoints, max_rows=5, show_tile_info=True)

# UC Agents (user-built)
print("\n" + "#"*60)
print(" UC AGENTS (User-built)")
print("#"*60)

uc_agent_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_FM_UC_AGENTS"]

print(f"\nTotal UC Agent endpoints: {len(uc_agent_endpoints)}")

# By task
print_table("By Task", count_by_field(uc_agent_endpoints, "task"))

# Sample endpoints
print_endpoints("Sample UC Agent Endpoints", uc_agent_endpoints, max_rows=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Model Endpoints

# COMMAND ----------

print("\n" + "#"*60)
print(" CLASSIC ML ENDPOINTS")
print("#"*60)

# Filter by model_type
classic_ml_endpoints = [ep for ep in endpoints if get_model_type(ep) == "DATABRICKS_CLASSIC_ML"]

print(f"\nTotal Classic ML endpoints: {len(classic_ml_endpoints)}")

# By entity type
print_table("By Entity Type", count_by_field(classic_ml_endpoints, "entity_type"))

# Sample endpoints
print_endpoints("Sample Classic ML Endpoints", classic_ml_endpoints, max_rows=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Serving Endpoints

# COMMAND ----------

print("\n" + "#"*60)
print(" FEATURE SERVING ENDPOINTS")
print("#"*60)

# Filter by model_type
feature_endpoints = [ep for ep in endpoints if get_model_type(ep) == "FEATURE_SPEC"]

print(f"\nTotal Feature Serving endpoints: {len(feature_endpoints)}")

# Sample endpoints
print_endpoints("Feature Serving Endpoints", feature_endpoints, max_rows=10)

# UNCLASSIFIED
print("\n" + "#"*60)
print(" UNCLASSIFIED ENDPOINTS")
print("#"*60)

unclassified_endpoints = [ep for ep in endpoints if get_model_type(ep) == "UNCLASSIFIED"]

print(f"\nTotal Unclassified endpoints: {len(unclassified_endpoints)}")

# By entity type
print_table("By Entity Type", count_by_field(unclassified_endpoints, "entity_type"))

# By task
print_table("By Task", count_by_field(unclassified_endpoints, "task"))

# Sample endpoints
print_endpoints("Sample Unclassified Endpoints", unclassified_endpoints, max_rows=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Health Analysis

# COMMAND ----------

print("\n" + "#"*60)
print(" ENDPOINT HEALTH")
print("#"*60)

# Not ready endpoints
not_ready = filter_endpoints(endpoints, ready_state="NOT_READY")
print(f"\nNOT_READY endpoints: {len(not_ready)}")

if not_ready:
    print_table("NOT_READY by Entity Type", count_by_field(not_ready, "entity_type"))
    print_endpoints("NOT_READY Endpoints", not_ready, max_rows=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Provider Breakdown (All Models)

# COMMAND ----------

print("\n" + "#"*60)
print(" PROVIDER BREAKDOWN")
print("#"*60)

# All endpoints with a model_provider
with_provider = [ep for ep in endpoints if ep.get("model_provider")]

print_table("All Providers", count_by_field(with_provider, "model_provider"))

# External providers only (excluding databricks)
external_providers = [ep for ep in with_provider if ep.get("model_provider") != "databricks"]
print(f"\nExternal Provider Endpoints: {len(external_providers)}")
print_table("External Providers", count_by_field(external_providers, "model_provider"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Filtered Data
# MAGIC 
# MAGIC Export specific subsets for further analysis.

# COMMAND ----------

def export_subset(endpoints: List[Dict], filename: str, output_dir: Path = Path("../_local/reports")):
    """Export a subset of endpoints to a JSON file (list only, no wrapper)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(endpoints, f, indent=2, default=str)
    print(f"  Exported {len(endpoints):>4} endpoints to: {output_path.name}")
    return output_path


# All model_types used in classification
ALL_MODEL_TYPES = [
    "DATABRICKS_FM_PPT",
    "DATABRICKS_FM_PT",
    "DATABRICKS_FM_UC_SYSTEM_AI",
    "DATABRICKS_FM_UC_AGENTS",
    "DATABRICKS_CLASSIC_ML",
    "AGENT_BRICKS_KA",
    "AGENT_BRICKS_MAS",
    "AGENT_BRICKS_KIE",
    "AGENT_BRICKS_MS",
    "FM_EXTERNAL_MODEL",
    "FM_EXTERNAL_MODEL_CUSTOM",
    "FEATURE_SPEC",
    "UNCLASSIFIED",
]


def export_by_model_type(endpoints: List[Dict], output_dir: Path = Path("../_local/reports/_detailed")):
    """Export endpoints grouped by model_type to the _detailed subfolder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f" Exporting by model_type to: {output_dir}")
    print(f"{'='*60}")
    
    exported_count = 0
    for model_type in ALL_MODEL_TYPES:
        filtered = [ep for ep in endpoints if get_model_type(ep) == model_type]
        if filtered:
            # Convert model_type to filename (lowercase, replace underscores)
            filename = f"{model_type.lower()}.json"
            export_subset(filtered, filename, output_dir)
            exported_count += 1
    
    # Also export NOT_READY endpoints
    not_ready_eps = [ep for ep in endpoints if ep.get("ready_state") == "NOT_READY"]
    if not_ready_eps:
        export_subset(not_ready_eps, "not_ready.json", output_dir)
        exported_count += 1
    
    print(f"\nExported {exported_count} category files to {output_dir}")


# Export detailed JSON files per model_type if enabled
if EXPORT_DETAILED_BY_MODEL_TYPE:
    export_by_model_type(endpoints)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Report

# COMMAND ----------

print("\n" + "="*60)
print(" SUMMARY REPORT")
print("="*60)

model_type_counts = count_by_model_type(endpoints)

summary_text = f"""
TOTAL ENDPOINTS: {len(endpoints)}

BY model_type:
  Databricks Foundation Models:
    - DATABRICKS_FM_PPT (pay-per-token): {model_type_counts.get('DATABRICKS_FM_PPT', 0)}
    - DATABRICKS_FM_PT (provisioned): {model_type_counts.get('DATABRICKS_FM_PT', 0)}
    - DATABRICKS_FM_UC_SYSTEM_AI: {model_type_counts.get('DATABRICKS_FM_UC_SYSTEM_AI', 0)}
  
  Agent Bricks:
    - AGENT_BRICKS_KA: {model_type_counts.get('AGENT_BRICKS_KA', 0)}
    - AGENT_BRICKS_MAS: {model_type_counts.get('AGENT_BRICKS_MAS', 0)}
    - AGENT_BRICKS_KIE: {model_type_counts.get('AGENT_BRICKS_KIE', 0)}
    - AGENT_BRICKS_MS: {model_type_counts.get('AGENT_BRICKS_MS', 0)}
  
  UC Models:
    - DATABRICKS_FM_UC_AGENTS: {model_type_counts.get('DATABRICKS_FM_UC_AGENTS', 0)}
    - DATABRICKS_CLASSIC_ML: {model_type_counts.get('DATABRICKS_CLASSIC_ML', 0)}
  
  External Models:
    - FM_EXTERNAL_MODEL: {model_type_counts.get('FM_EXTERNAL_MODEL', 0)}
    - FM_EXTERNAL_MODEL_CUSTOM: {model_type_counts.get('FM_EXTERNAL_MODEL_CUSTOM', 0)}
  
  Other:
    - FEATURE_SPEC: {model_type_counts.get('FEATURE_SPEC', 0)}
    - UNCLASSIFIED: {model_type_counts.get('UNCLASSIFIED', 0)}

HEALTH:
  READY: {len(filter_endpoints(endpoints, ready_state="READY"))}
  NOT_READY: {len(not_ready)}
"""

print(summary_text)

# Generate markdown summary report
def generate_markdown_summary() -> str:
    """Generate a markdown summary report."""
    # Get extraction timestamp from first endpoint
    extraction_ts = "unknown"
    if endpoints and endpoints[0].get("_metadata_derived"):
        extraction_ts = endpoints[0]["_metadata_derived"].get("extraction_timestamp", "unknown")
    
    md = []
    md.append("# AI Endpoint Analysis Report")
    md.append("")
    md.append(f"**Generated:** {datetime.now().isoformat()}")
    md.append(f"**Data Extracted:** {extraction_ts}")
    md.append(f"**Total Endpoints:** {len(endpoints)}")
    md.append("")
    
    # Summary table
    md.append("## Summary by model_type")
    md.append("")
    md.append("| Category | model_type | Count |")
    md.append("|----------|------------|-------|")
    md.append(f"| Databricks FM | DATABRICKS_FM_PPT | {model_type_counts.get('DATABRICKS_FM_PPT', 0)} |")
    md.append(f"| Databricks FM | DATABRICKS_FM_PT | {model_type_counts.get('DATABRICKS_FM_PT', 0)} |")
    md.append(f"| Databricks FM | DATABRICKS_FM_UC_SYSTEM_AI | {model_type_counts.get('DATABRICKS_FM_UC_SYSTEM_AI', 0)} |")
    md.append(f"| Agent Bricks | AGENT_BRICKS_KA | {model_type_counts.get('AGENT_BRICKS_KA', 0)} |")
    md.append(f"| Agent Bricks | AGENT_BRICKS_MAS | {model_type_counts.get('AGENT_BRICKS_MAS', 0)} |")
    md.append(f"| Agent Bricks | AGENT_BRICKS_KIE | {model_type_counts.get('AGENT_BRICKS_KIE', 0)} |")
    md.append(f"| Agent Bricks | AGENT_BRICKS_MS | {model_type_counts.get('AGENT_BRICKS_MS', 0)} |")
    md.append(f"| UC Models | DATABRICKS_FM_UC_AGENTS | {model_type_counts.get('DATABRICKS_FM_UC_AGENTS', 0)} |")
    md.append(f"| UC Models | DATABRICKS_CLASSIC_ML | {model_type_counts.get('DATABRICKS_CLASSIC_ML', 0)} |")
    md.append(f"| External | FM_EXTERNAL_MODEL | {model_type_counts.get('FM_EXTERNAL_MODEL', 0)} |")
    md.append(f"| External | FM_EXTERNAL_MODEL_CUSTOM | {model_type_counts.get('FM_EXTERNAL_MODEL_CUSTOM', 0)} |")
    md.append(f"| Other | FEATURE_SPEC | {model_type_counts.get('FEATURE_SPEC', 0)} |")
    md.append(f"| Other | UNCLASSIFIED | {model_type_counts.get('UNCLASSIFIED', 0)} |")
    md.append(f"| **Total** | | **{len(endpoints)}** |")
    md.append("")
    
    # Health
    md.append("## Endpoint Health")
    md.append("")
    md.append("| State | Count |")
    md.append("|-------|-------|")
    md.append(f"| READY | {len(filter_endpoints(endpoints, ready_state='READY'))} |")
    md.append(f"| NOT_READY | {len(not_ready)} |")
    md.append("")
    
    # Agent Bricks with names and descriptions
    if agent_bricks_endpoints:
        md.append("## Agent Bricks Details (Sample)")
        md.append("")
        md.append("| Type | Tile Name | Description |")
        md.append("|------|-----------|-------------|")
        for ep in agent_bricks_endpoints[:10]:
            tile = ep.get("tile_metadata") or {}
            tile_type = tile.get("problem_type", "N/A")
            tile_name = tile.get("tile_name", ep.get("endpoint_name", "N/A"))
            tile_desc = tile.get("tile_description", "")
            # Truncate description for table
            desc_short = tile_desc[:50] + "..." if len(tile_desc) > 50 else tile_desc
            md.append(f"| {tile_type} | {tile_name} | {desc_short} |")
        if len(agent_bricks_endpoints) > 10:
            md.append(f"| ... | *{len(agent_bricks_endpoints) - 10} more tiles* | |")
        md.append("")
    
    # External providers
    md.append("## External Model Providers")
    md.append("")
    md.append("| Provider | Count |")
    md.append("|----------|-------|")
    for provider, count in count_by_field(ext_endpoints, "model_provider").items():
        md.append(f"| {provider} | {count} |")
    md.append("")
    
    # Top tasks
    md.append("## Tasks Distribution")
    md.append("")
    md.append("| Task | Count |")
    md.append("|------|-------|")
    for task, count in list(count_by_field(endpoints, "task").items())[:10]:
        md.append(f"| {task} | {count} |")
    md.append("")
    
    md.append("---")
    md.append(f"*Generated by 02_generate_endpoint_analysis_report.py*")
    
    return "\n".join(md)


def save_markdown_summary(output_dir: Path = Path("../_local/reports")):
    """Save markdown summary with timestamped and latest versions."""
    md_content = generate_markdown_summary()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Timestamped file
    ts_path = output_dir / f"all_endpoints_{timestamp}.md"
    with open(ts_path, 'w') as f:
        f.write(md_content)
    print(f"Saved markdown report: {ts_path}")
    
    # Latest file
    latest_path = output_dir / "all_endpoints_latest.md"
    with open(latest_path, 'w') as f:
        f.write(md_content)
    print(f"Saved markdown report: {latest_path}")


save_markdown_summary()

print("="*60)
print(" Analysis complete. Check exported JSON and MD files for details.")
print("="*60)
