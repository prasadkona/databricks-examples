# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Genie Space Setup
# MAGIC 
# MAGIC Creates a Genie Space for natural language queries over SEC financial data.
# MAGIC Uses w.genie.create_space() with a serialized_space v2 JSON payload.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tables and Views for Genie Space

# COMMAND ----------

# Tables and views for Genie Space: pipeline gold/silver tables + analytical views (04)
genie_tables = [
    # Pipeline tables (used directly - no copy step)
    TABLES["company_financials"],
    TABLES["revenue_by_segment"],
    TABLES["revenue_by_geography"],
    TABLES["stock_daily_prices"],
    TABLES["stock_summary"],
    # Views created by 05_create_stock_views
    f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}company_overview",
    f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}peer_comparison",
    f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}stock_performance",
    f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}revenue_breakdown",
]

print("Tables/Views for Genie Space:")
print("-" * 50)
for table in genie_tables:
    print(f"  {table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify All Tables Exist

# COMMAND ----------

print("Verifying tables and views...")
print("-" * 50)
all_exist = True
for table in genie_tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) FROM {table}").collect()[0][0]
        print(f"✅ {table.split('.')[-1]}: {count} rows")
    except Exception as e:
        print(f"❌ {table}: {str(e)[:60]}")
        all_exist = False

if all_exist:
    print("\n✅ All tables verified!")
else:
    print("\n⚠️ Some tables missing - run previous notebooks first")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sample Questions for Genie

# COMMAND ----------

sample_questions = [
    "What is NVIDIA's total revenue and growth rate for FY2024?",
    "Compare gross margins across NVIDIA, Apple, and Samsung",
    "Which company has the highest P/E ratio?",
    "Show me NVIDIA's Data Center segment as a percentage of total revenue",
    "What percentage of Apple's revenue comes from Greater China?",
    "Rank all companies by market cap",
    "Which stock has the best YTD return?",
    "Compare R&D spending as percentage of revenue across all companies",
    "Show me Samsung's revenue breakdown by business segment",
    "What is the trend signal for each stock?",
]

print("Sample Questions:")
print("-" * 50)
for i, q in enumerate(sample_questions, 1):
    print(f"{i}. {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Genie Space Instructions

# COMMAND ----------

GENIE_INSTRUCTIONS = """
# SEC Financial Analyst Genie Space

You are a financial analyst assistant helping users explore SEC filing data for NVIDIA, Apple, and Samsung.

## Available Data
- **Company Financials**: FY2024 revenue, profits, margins, balance sheet metrics
- **Business Segments**: Revenue breakdown by product/service line
- **Geographic Revenue**: Revenue by region (Americas, Europe, APAC, China)
- **Stock Prices**: 2 years of daily stock data with technical indicators
- **Stock Summary**: Current valuation metrics (P/E, market cap, etc.)

## Key Metrics to Know
- All dollar values in BILLIONS USD unless otherwise specified
- Growth percentages are year-over-year (YoY)
- Stock metrics include SMA (Simple Moving Average) at 20, 50, 200 days
- Trend signals: BULLISH, BEARISH, or NEUTRAL

## Company Tickers
- NVIDIA: NVDA
- Apple: AAPL  
- Samsung: 005930.KS
"""

print("Genie Instructions:")
print("-" * 50)
print(GENIE_INSTRUCTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Genie Space

# COMMAND ----------

import json
import uuid
from databricks.sdk import WorkspaceClient

w = get_workspace_client()
current_user = w.current_user.me()
print(f"Authenticated as: {current_user.user_name}")

# Build instruction lines in the format required by the API
instruction_lines = [line + "\n" for line in GENIE_INSTRUCTIONS.strip().split("\n")]

# Build serialized_space JSON payload (v2 format required by create_space)
serialized_space = json.dumps({
    "version": 2,
    "config": {
        "sample_questions": [
            {"id": uuid.uuid4().hex, "question": [q]}
            for q in sample_questions
        ]
    },
    "data_sources": {
        "tables": [
            {"identifier": t} for t in sorted(genie_tables)
        ]
    },
    "instructions": {
        "text_instructions": [
            {
                "id": uuid.uuid4().hex,
                "content": instruction_lines
            }
        ]
    }
}, indent=2)

# Create Genie Space
genie_space = w.genie.create_space(
    warehouse_id=SQL_WAREHOUSE_ID,
    serialized_space=serialized_space,
    title=GENIE_SPACE_NAME,
    description="SEC financial data for NVIDIA, Apple, Samsung",
    parent_path=f"/Users/{current_user.user_name}"
)

GENIE_SPACE_ID = genie_space.space_id
print(f"\n✅ Created Genie Space: {GENIE_SPACE_ID}")
print(f"   Name: {GENIE_SPACE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate Genie Space
# MAGIC 
# MAGIC Run a test query against the newly created Genie Space to verify it works.

# COMMAND ----------

import time
import requests

print("Validating Genie Space with test query...")
print("-" * 50)

# Get auth token from workspace client
host = w.config.host.rstrip("/")
token = w.config.token

# Test question
test_question = "What is NVIDIA's total revenue for FY2024?"
print(f"Test query: {test_question}")

# Start conversation
try:
    resp = requests.post(
        f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": test_question},
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        conv_id = data.get("conversation_id")
        msg_id = data.get("message_id")
        print(f"  Conversation started: {conv_id[:20]}...")
        
        # Poll for result (max 60s)
        validation_passed = False
        for i in range(12):
            time.sleep(5)
            poll = requests.get(
                f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv_id}/messages/{msg_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
            if poll.status_code == 200:
                status = poll.json().get("status")
                print(f"  Poll {i+1}/12: {status}")
                if status == "COMPLETED":
                    print(f"\n✅ Genie Space validation PASSED")
                    validation_passed = True
                    break
                elif status in ("FAILED", "CANCELLED"):
                    print(f"\n❌ Genie Space validation FAILED: {status}")
                    break
        
        if not validation_passed and status not in ("FAILED", "CANCELLED"):
            print("\n⚠️ Genie validation timed out - check manually in UI")
    else:
        print(f"❌ Failed to start conversation: {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        
except Exception as e:
    print(f"⚠️ Validation error: {str(e)[:100]}")
    print("   Genie Space created but validation skipped - test manually")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary

# COMMAND ----------

print("\n" + "=" * 70)
print("GENIE SPACE SETUP COMPLETE")
print("=" * 70)
print(f"\n✅ Genie Space created: {GENIE_SPACE_ID}")
print(f"""
Tables configured: {len(genie_tables)}
Sample questions:  {len(sample_questions)}

Access: Databricks SQL > Genie > {GENIE_SPACE_NAME}

IMPORTANT: Update the following with the new Genie Space ID:
  - notebooks/config.py        → GENIE_SPACE_ID = "{GENIE_SPACE_ID}"
  - agent_server/agent.py      → "space_id": "{GENIE_SPACE_ID}"
  - .env / .env.example        → GENIE_SPACE_ID={GENIE_SPACE_ID}
""")
print("=" * 70)
