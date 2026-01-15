# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install claude-agent-sdk>=0.1.19 python-dotenv>=1.0.0 databricks-sdk>=0.30.0 mlflow>=2.16.0 anthropic>=0.39.0

# COMMAND ----------

# Restart Python to ensure new packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Get workspace hostname using spark.conf (the correct method for DBR 18.0)
DATABRICKS_HOSTNAME = spark.conf.get("spark.databricks.workspaceUrl")

# Get Databricks access token from notebook context using dbutils (recommended for notebooks)
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Set environment variables for Claude Agent SDK
import os
os.environ["DATABRICKS_HOST"] = f"https://{DATABRICKS_HOSTNAME}"
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["ANTHROPIC_AUTH_TOKEN"] = DATABRICKS_TOKEN
os.environ["ANTHROPIC_BASE_URL"] = f"https://{DATABRICKS_HOSTNAME}/serving-endpoints/anthropic"
os.environ["ANTHROPIC_CUSTOM_HEADERS"] = "x-databricks-disable-beta-headers: true"
os.environ["ANTHROPIC_MODEL"] = "databricks-claude-haiku-4-5"

# Set MCP server configuration (optional - customize as needed)
os.environ["MCP_FUNCTIONS_CATALOG"] = "system"
os.environ["MCP_FUNCTIONS_SCHEMA"] = "ai"
os.environ["MCP_GENIE_SPACE_ID"] = os.getenv("MCP_GENIE_SPACE_ID", "01f0ed98945116be9f373479af170485")

# Note: DBSQL MCP server does NOT require a SQL warehouse ID
# It's always available and ready to use

# COMMAND ----------

"""
Claude Agent SDK with Databricks MCP + MLflow Integration

This example demonstrates:
- Using Claude Agent SDK with Databricks MCP servers
- MLflow experiment tracking for MCP-enhanced agents
- Unity Catalog Functions, Genie, and DBSQL integration
- Enterprise data access with full observability
- Tracking agent interactions with enterprise data

NOTE: This notebook is designed to run in a Databricks workspace for best
      results. It combines MCP server access with MLflow tracking.
"""

import asyncio
import sys
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions

# Load environment variables from .env (for local execution only)
# In Databricks, credentials are already set above
load_dotenv()


async def run_mcp_agent_with_mlflow():
    """Run Claude Agent with MCP servers and MLflow tracking."""
    print("="*60)
    print("Claude Agent SDK: MCP + MLflow on Databricks")
    print("="*60)
    
    # Setup MLflow
    try:
        import mlflow
        import mlflow.anthropic
        
        print("\n🔧 Configuring MLflow...")
        
        # Clear cluster ID if set (can cause auth issues for local execution)
        if "DATABRICKS_CLUSTER_ID" in os.environ:
            print("   Clearing DATABRICKS_CLUSTER_ID...")
            del os.environ["DATABRICKS_CLUSTER_ID"]
        
        mlflow.set_tracking_uri("databricks")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/claude-agent-mcp-demo")
        
        print(f"   Connecting to: {os.getenv('DATABRICKS_HOST', 'default')}")
        print(f"   Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ MLflow configured successfully")
        
        # Enable autologging for Claude Agent SDK
        print("\n🔍 Enabling MLflow autologging...")
        mlflow.anthropic.autolog()
        print("✅ Autologging enabled")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package: {e}")
        print("   Install with: pip install mlflow")
        return
    except Exception as e:
        print(f"\n❌ ERROR: MLflow configuration failed")
        print(f"   Error: {e}")
        print("\n   💡 Recommended Solution:")
        print("   This notebook works best when run inside a Databricks workspace.")
        return
    
    # Display MCP configuration
    databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    databricks_token = os.getenv("DATABRICKS_TOKEN", "")
    
    if not databricks_host or not databricks_token:
        print("\n❌ ERROR: Missing Databricks credentials")
        print("   Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env")
        return
    
    print(f"\n🔧 Databricks Configuration:")
    print(f"   Host: {databricks_host}")
    print(f"   Token: {'SET' if databricks_token else 'NOT SET'}")
    
    # Configure MCP Servers - pass directly to ClaudeAgentOptions
    print(f"\n📡 Configuring MCP Servers...")
    
    # Unity Catalog Functions MCP Server
    functions_catalog = os.getenv("MCP_FUNCTIONS_CATALOG", "system")
    functions_schema = os.getenv("MCP_FUNCTIONS_SCHEMA", "ai")
    uc_functions_url = f"{databricks_host}/api/2.0/mcp/functions/{functions_catalog}/{functions_schema}"
    print(f"   ✓ Unity Catalog Functions: {functions_catalog}.{functions_schema}")
    print(f"     URL: {uc_functions_url}")
    
    # DBSQL MCP Server (always available - no warehouse ID required)
    dbsql_url = f"{databricks_host}/api/2.0/mcp/sql"
    print(f"   ✓ DBSQL: {dbsql_url}")
    
    # Build MCP servers configuration to pass to SDK
    mcp_servers_config = {
        "uc_functions": {
            "type": "http",
            "url": uc_functions_url,
            "transport": "sse",
            "headers": {
                "Authorization": f"Bearer {databricks_token}"
            }
        },
        "dbsql": {
            "type": "http",
            "url": dbsql_url,
            "transport": "sse",
            "headers": {
                "Authorization": f"Bearer {databricks_token}"
            }
        }
    }
    
    mcp_servers_list = ["uc_functions", "dbsql"]
    
    # Genie MCP Server (optional - if space ID provided)
    genie_space_id = os.getenv("MCP_GENIE_SPACE_ID", "")
    if genie_space_id:
        genie_url = f"{databricks_host}/api/2.0/mcp/genie/{genie_space_id}"
        print(f"   ✓ Genie Space: {genie_space_id}")
        print(f"     URL: {genie_url}")
        mcp_servers_config["genie"] = {
            "type": "http",
            "url": genie_url,
            "transport": "sse",
            "headers": {
                "Authorization": f"Bearer {databricks_token}"
            }
        }
        mcp_servers_list.append("genie")
    else:
        print(f"   ℹ️  Genie: Not configured (set MCP_GENIE_SPACE_ID to enable)")
    
    print(f"\n✅ MCP Configuration Complete")
    print(f"   Configured {len(mcp_servers_config)} MCP server(s): {', '.join(mcp_servers_config.keys())}")
    
    # Start MLflow run
    print("\n" + "="*60)
    print("🚀 Starting MLflow Run with MCP Integration")
    print("="*60)
    
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            
            print(f"\n📊 MLflow Run Started: {run_id}")
            print(f"   Experiment ID: {experiment_id}")
            
            # Log MCP configuration as parameters
            mlflow.log_param("mcp_uc_functions", f"{functions_catalog}.{functions_schema}")
            mlflow.log_param("mcp_genie_enabled", bool(genie_space_id))
            mlflow.log_param("mcp_dbsql_enabled", True)  # DBSQL always available
            mlflow.log_param("mcp_servers_count", len(mcp_servers_config))
            mlflow.log_param("agent_type", "mcp_enhanced")
            mlflow.log_param("model", os.getenv("ANTHROPIC_MODEL", "databricks-claude-haiku-4-5"))
            
            # Example 1: Unity Catalog Functions Query
            print("\n" + "-"*60)
            print("📝 Example 1: Unity Catalog Functions Analysis")
            print("-"*60)
            
            uc_prompt = """
            Analyze the Unity Catalog system.ai functions available:
            1. What built-in AI functions are available?
            2. What are their primary use cases?
            3. Provide a practical example of when to use them
            
            Be concise and technical.
            """
            
            result_1 = None
            async for message in query(
                prompt=uc_prompt,
                options=ClaudeAgentOptions(
                    mcp_servers=mcp_servers_config,
                    allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                    permission_mode="acceptEdits"
                )
            ):
                if hasattr(message, "result"):
                    result_1 = message.result
            
            if result_1:
                print("\n📊 Result:")
                print("-"*60)
                print(result_1)
                print("-"*60)
                
                # Log result to MLflow
                mlflow.log_text(result_1, "uc_functions_analysis.txt")
                mlflow.log_metric("example_1_complete", 1)
            
            # Example 2: Genie Natural Language Query (if available)
            if "genie" in mcp_servers_list:
                print("\n" + "-"*60)
                print("📝 Example 2: Genie Natural Language Query on NYC Taxi Data")
                print("-"*60)
                
                genie_prompt = """
                Use the Genie MCP tool to answer this question:
                
                "Show me the top 5 NYC taxi trips by fare amount from the samples.nyctaxi.trips table"
                
                Execute this query using Genie and show me the results with trip details.
                """
                
                result_2 = None
                async for message in query(
                    prompt=genie_prompt,
                    options=ClaudeAgentOptions(
                        mcp_servers=mcp_servers_config,
                        allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                        permission_mode="acceptEdits"
                    )
                ):
                    if hasattr(message, "result"):
                        result_2 = message.result
                
                if result_2:
                    print("\n📊 Genie Result:")
                    print("-"*60)
                    print(result_2)
                    print("-"*60)
                    
                    # Log result to MLflow
                    mlflow.log_text(result_2, "genie_query_result.txt")
                    mlflow.log_metric("example_2_complete", 1)
            else:
                print("\n" + "-"*60)
                print("📝 Example 2: Genie (Skipped - Not Configured)")
                print("-"*60)
                mlflow.log_param("genie_skipped", True)
            
            # Example 3: Direct SQL Query (always available)
            print("\n" + "-"*60)
            print("📝 Example 3: DBSQL Direct SQL Query on NYC Taxi Trips")
            print("-"*60)
            
            sql_prompt = """
            Use the DBSQL MCP tool to execute this SQL query:
            
            SELECT 
                trip_distance,
                fare_amount,
                passenger_count,
                pickup_zip,
                dropoff_zip
            FROM samples.nyctaxi.trips
            WHERE fare_amount > 50
            ORDER BY fare_amount DESC
            LIMIT 10
            
            Execute this query and show me the results in a clear table format.
            """
            
            result_3 = None
            async for message in query(
                prompt=sql_prompt,
                options=ClaudeAgentOptions(
                    mcp_servers=mcp_servers_config,
                    allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                    permission_mode="acceptEdits"
                )
            ):
                if hasattr(message, "result"):
                    result_3 = message.result
            
            if result_3:
                print("\n📊 SQL Query Result:")
                print("-"*60)
                print(result_3)
                print("-"*60)
                
                # Log result to MLflow
                mlflow.log_text(result_3, "dbsql_query_result.txt")
                mlflow.log_metric("example_3_complete", 1)
            
            # Log summary metrics
            total_examples = sum([
                1,  # UC Functions always runs
                1 if "genie" in mcp_servers_list else 0,
                1   # DBSQL always runs
            ])
            mlflow.log_metric("total_examples", total_examples)
            mlflow.log_metric("mcp_servers_used", len(mcp_servers_config))
            
            # After run completes, print MLflow details
            mlflow_url = f"{databricks_host}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
            
            print("\n" + "="*60)
            print("✅ SUCCESS: MCP + MLflow Integration Complete!")
            print("="*60)
            print(f"\n📈 MLflow Run Details:")
            print(f"   Run ID: {run_id}")
            print(f"   Experiment: {experiment_name}")
            print(f"   View in Databricks: {mlflow_url}")
            print("\n🔍 Logged to MLflow:")
            print("   • MCP server configurations")
            print("   • All agent query results")
            print("   • Enterprise data access patterns")
            print("   • Token usage and costs (via autologging)")
            print("   • Performance metrics")
            print("\n💡 What This Demonstrates:")
            print("   • MCP integration for enterprise data access")
            print("   • MLflow tracking of MCP-enhanced agents")
            print("   • Unity Catalog, Genie, and DBSQL usage")
            print("   • Complete observability for enterprise agents")
            print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR during agent execution: {e}")
        import traceback
        traceback.print_exc()


# COMMAND ----------

# Run the async function directly in the notebook
await run_mcp_agent_with_mlflow()
