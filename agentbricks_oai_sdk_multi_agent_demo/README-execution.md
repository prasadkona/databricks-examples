# Execution & Testing Workflow

This guide covers all execution options for the SEC Financial Analyst demo, from individual steps to full lifecycle automation.

## Quick Reference

```bash
# Full lifecycle (recommended for fresh setup)
uv run run-sequence --all

# Data engineering only (default)
uv run run-sequence

# KA + data engineering
uv run run-sequence --ka --data-eng

# Deploy agent after data is ready
uv run run-sequence --data-eng --deploy-agent

# Resume from a specific phase (skip earlier phases)
uv run run-sequence --all --from services

# Preview execution plan without running
uv run run-sequence --all --dry-run
```

---

## run-sequence Flags

The `run-sequence` command orchestrates demo setup with configurable phases:

### Phase Flags

| Flag | Phase | Description |
|------|-------|-------------|
| `--ka` | Knowledge Assistant | Build KA endpoint, add examples, test endpoint |
| `--data-eng` | Data Engineering | SDP pipeline (company discovery + stock load), views, UC functions, Genie |
| `--refresh-stocks` | Stock Refresh | Incremental stock price update (append new rows) |
| `--test-services` | Validation | Test KA + Genie + UC functions independently |
| `--deploy-agent` | Agent Lifecycle | Test local agent, deploy app, test deployed app |
| `--all` | Full Lifecycle | All phases combined |

### Resume/Skip Flags

| Flag | Description |
|------|-------------|
| `--from PHASE` | Resume from specific phase, skipping earlier phases |
| `--dry-run` | Show execution plan without running |

**Valid phases for `--from`:** `setup`, `ka`, `data-eng`, `transform`, `views`, `services`, `agent`, `deploy`

### Convenience Flags

| Flag | Description |
|------|-------------|
| `--quick` | Use smoke tests instead of full test suites |
| `--skip-preflight` | Skip environment variable checks |

### Flag Dependencies

- `--deploy-agent` automatically enables `--test-services`
- `--all` enables `--ka`, `--data-eng`, and `--deploy-agent`
- Default (no flags) = `--data-eng` only

### Resume Examples

```bash
# Resume from services (skip KA + data-eng)
uv run run-sequence --all --from services

# Resume from agent deployment
uv run run-sequence --all --from agent

# Resume from transform (skip cleanup)
uv run run-sequence --data-eng --from transform

# Show what would run without executing
uv run run-sequence --all --dry-run

# Data pipeline + incremental stock refresh
uv run run-sequence --data-eng --refresh-stocks
```

---

## Execution Phases

### Phase 1: Setup (Always Runs)

```bash
uv run sync-workspace    # Upload notebooks to Databricks workspace
```

### Phase 2: Knowledge Assistant (`--ka`)

```bash
uv run run-ka-sequence   # Build KA, deploy endpoint, add examples, test
```

**Steps:**
1. Create KA tile with document sources
2. Deploy serving endpoint
3. Add example questions
4. Wait for endpoint to be ONLINE
5. Test with sample query

**Duration:** ~10-12 minutes

### Phase 3: Data Engineering (`--data-eng`)

```bash
uv run demo-cleanup tables           # Drop existing tables/pipeline
uv run deploy-sdp-pipeline           # Deploy SDP pipeline (discovers companies + loads stock)
uv run run-workspace-notebooks 05 06 07  # 05-07: Views, functions, Genie
```

**Steps:**

| Step | Name | Description | Duration |
|------|------|-------------|----------|
| cleanup | Tables | Drop existing tables and pipeline | ~30 sec |
| transform | SDP Pipeline | Bronze → Silver → Registry → Stock → Gold | ~5-7 min |
| 05 | Views | Create analytical views for Genie | ~15 sec |
| 06 | Functions | Create UC functions (valuation_score, etc.) | ~45 sec |
| 07 | Genie | Create Genie Space + validate | ~1 min |

Stock loading is now fully integrated into the SDP pipeline — no separate ingest step required.

**Total Duration:** ~7-9 minutes

### Phase 4: Validation (`--test-services`)

```bash
uv run test-services    # Test KA, Genie, and UC functions
```

**Tests:**
- UC Functions: `valuation_score`, `compare_peers`, `growth_trajectory`, `risk_summary`
- Genie Space: Sample revenue query
- Knowledge Assistant: Sample 10-K question

**Duration:** ~30-60 seconds

### Phase 5: Agent (`--deploy-agent`)

```bash
uv run test-agent           # Test agent locally
uv run deploy-agent-app     # Deploy to Databricks Apps
uv run test-agent-app       # Test deployed agent
```

**Steps:**
1. Start local agent server
2. Run smoke test (query hitting Genie + KA + UC)
3. Deploy app via Asset Bundles
4. Grant SP permissions
5. Test deployed endpoint

**Duration:** ~3-5 minutes

---

## Phase Map for `--from` Flag

When using `--from PHASE`, all phases before the specified phase are skipped:

```
setup → ka → data-eng → transform → views → services → agent → deploy
```

Examples:
- `--from ka`: Skip setup, start from KA
- `--from transform`: Skip cleanup, start from SDP pipeline deployment
- `--from views`: Skip up through transform, start from views (05-07)
- `--from services`: Skip data engineering, start from test-services
- `--from agent`: Skip services, start from local agent test
- `--from deploy`: Skip local test, start from deploy-agent-app

---

## Common Workflows

### Fresh Setup (First Time)

```bash
# Full lifecycle - creates everything from scratch
uv run run-sequence --all
```

### Update Data Only (KA Already Exists)

```bash
# Skip KA, rebuild data pipeline
uv run run-sequence --data-eng
```

### Rebuild KA Only

```bash
# Just rebuild Knowledge Assistant
uv run run-sequence --ka
```

### Test Everything Before Demo

```bash
# Validate all services work
uv run test-services

# Quick local agent test
uv run test-agent --quick

# Full agent test (all scenarios)
uv run test-agent --full
```

### Redeploy Agent After Code Changes

```bash
# Deploy and test agent
uv run deploy-agent-app
uv run test-agent-app
```

### Resume After Failure

```bash
# If pipeline failed at transform, resume from there
uv run run-sequence --data-eng --from transform

# If services failed, resume from services
uv run run-sequence --all --from services
```

### Preview Before Running

```bash
# See what would run
uv run run-sequence --all --dry-run
```

### Full Reset and Rebuild

```bash
# Clean everything
uv run demo-cleanup all

# Rebuild everything
uv run run-sequence --all
```

---

## Individual Commands

### Setup Commands

```bash
uv run sync-workspace        # Upload notebooks to workspace
uv run run-workspace-notebooks 05 06 07   # Run specific notebooks
```

### KA Commands

```bash
uv run run-ka-sequence       # Full KA lifecycle (create + test)
uv run run-ka-sequence --skip-test   # Create only, no testing
uv run create-ka             # Create KA endpoint
uv run sync-ka               # Sync KA sources
uv run test-ka               # Test KA endpoint
```

### Data Pipeline Commands

```bash
uv run deploy-sdp-pipeline   # Deploy and run SDP pipeline
uv run refresh-stocks        # Incremental stock price update
uv run refresh-stocks --dry-run   # Preview what would be fetched
uv run refresh-stocks --ticker NVDA  # Refresh a single ticker
```

### Agent Commands

```bash
uv run test-agent            # Local agent test (smoke)
uv run test-agent --quick    # Quick smoke test
uv run test-agent --full     # All 8 test scenarios
uv run deploy-agent-app      # Deploy to Databricks Apps
uv run test-agent-app        # Test deployed agent
uv run test-agent-app --full # Full deployed tests
```

### Service Validation

```bash
uv run test-services         # Test all services (KA + Genie + UC)
uv run test-services --ka    # Test KA only
uv run test-services --genie # Test Genie only
uv run test-services --uc    # Test UC functions only
```

### Cleanup Commands

```bash
uv run demo-cleanup all        # Delete everything (app, Genie, tables, KA)
uv run demo-cleanup after-ka   # Keep KA, delete Genie + tables + app
uv run demo-cleanup after-genie # Keep KA + Genie, delete tables + app
uv run demo-cleanup tables     # Delete tables and pipeline only
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           run-sequence --all                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Setup                                                           │
│  ├── sync-workspace                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Knowledge Assistant (--ka)                                      │
│  ├── create KA tile                                                     │
│  ├── deploy serving endpoint                                            │
│  ├── add example questions                                              │
│  ├── wait for ONLINE status                                             │
│  └── test KA endpoint                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Data Engineering (--data-eng)                                   │
│  ├── cleanup tables + pipeline                                          │
│  ├── deploy SDP pipeline (bronze → silver → registry → stock → gold)   │
│  ├── 05: Views - create_stock_views                                     │
│  ├── 06: Functions - create_uc_functions                                │
│  └── 07: Genie - create + validate                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (optional)
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Stock Refresh (--refresh-stocks)                                │
│  └── refresh-stocks (incremental yfinance update)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Validation (--test-services)                                    │
│  ├── test UC functions (valuation_score, compare_peers, etc.)           │
│  ├── test Genie Space                                                   │
│  └── test KA endpoint                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE: Agent (--deploy-agent)                                          │
│  ├── test-agent (local)                                                 │
│  ├── deploy-agent-app                                                   │
│  └── test-agent-app                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Timing Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup | ~10 sec | Sync notebooks |
| KA | ~10-12 min | Endpoint provisioning takes time |
| Data Engineering | ~8-10 min | SDP pipeline is the longest |
| Validation | ~30-60 sec | Quick API tests |
| Agent | ~3-5 min | Deploy + test |
| **Full Lifecycle** | **~25-30 min** | All phases |

---

## Troubleshooting

### Pre-flight Check Fails

```
PRE-FLIGHT CHECK FAILED
Missing required config: DATABRICKS_HOST, DATABRICKS_TOKEN
```

**Fix:** Ensure your config file exists and has required variables:
```bash
cat _local/config/databricks.env
```

### KA Endpoint Not Coming ONLINE

If KA creation times out, check the endpoint status in the Databricks UI:
1. Go to **Serving** in your workspace
2. Find endpoint starting with `ka-`
3. Check provisioning status

### SDP Pipeline Fails

```bash
# Check pipeline status
uv run deploy-sdp-pipeline

# View pipeline in UI (link printed in output)
```

### Agent Tests Fail

```bash
# Test services independently first
uv run test-services

# Then test agent locally
uv run test-agent --quick
```

### 502 Error on Deployed App

The app may need time to start. Wait 30 seconds and retry:
```bash
sleep 30 && uv run test-agent-app
```

---

## Environment Variables

Key variables loaded from `_local/config/<workspace>.env`:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | PAT token |
| `SQL_WAREHOUSE_ID` | SQL warehouse for queries |
| `CLUSTER_ID` | Cluster for notebook execution |
| `UC_CATALOG` | Unity Catalog name |
| `UC_SCHEMA` | Schema name |
| `UC_VOLUME` | UC Volume name containing SEC PDFs |
| `SEC_DOCS_SUBFOLDER` | Subfolder within volume (e.g., `sec_2024`) |
| `KA_ENDPOINT` | Knowledge Assistant endpoint (auto-populated) |
| `GENIE_SPACE_ID` | Genie Space ID (auto-populated) |
| `APP_URL` | Deployed app URL (auto-populated) |
