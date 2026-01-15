"""
Claude Agent SDK with Databricks MCP Integration

This example demonstrates:
- Connecting to Databricks MCP (Model Context Protocol) servers
- Using Unity Catalog Functions as agent tools
- Querying Genie spaces for natural language SQL
- Running SQL queries via DBSQL MCP server
- Extending agent capabilities with enterprise data

NOTE: This notebook runs locally but requires:
      - Databricks workspace with MCP servers enabled
      - Unity Catalog access
      - Genie space (optional)
      - SQL warehouse (optional)
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions

# Load environment variables from .env
load_dotenv()


async def run_agent_with_mcp():
    """Run Claude Agent with Databricks MCP servers."""
    print("="*60)
    print("Claude Agent SDK with Databricks MCP Integration")
    print("="*60)
    
    # Display MCP configuration
    databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    databricks_token = os.getenv("DATABRICKS_TOKEN", "")
    
    if not databricks_host or not databricks_token:
        print("\n❌ ERROR: Missing Databricks credentials")
        print("   Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env")
        return 1
    
    print(f"\n🔧 Configuration:")
    print(f"   Databricks Host: {databricks_host}")
    print(f"   Token: {'SET' if databricks_token else 'NOT SET'}")
    
    print("\n📋 MCP Configuration Guide:")
    print("-" * 60)
    print("MCP Servers Available:")
    print("")
    print("1. ✓ Unity Catalog Functions (Always Available)")
    print("   • Catalog: system")
    print("   • Schema: ai")
    print("   • Provides: python_exec, vector search, etc.")
    print("")
    print("2. ✓ DBSQL (Always Available)")
    print("   • Direct SQL query execution")
    print("   • No warehouse ID required")
    print("   • Tests with: samples.nyctaxi.trips")
    print("")
    print("3. Genie (Optional - Natural Language Queries)")
    print(f"   • Status: {'✓ CONFIGURED' if os.getenv('MCP_GENIE_SPACE_ID') else '✗ NOT CONFIGURED'}")
    print("   • To enable: Set MCP_GENIE_SPACE_ID in .env")
    print("-" * 60)
    
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
    
    # Example 1: Query with Unity Catalog Functions
    print("\n" + "="*60)
    print("Example 1: Unity Catalog Functions")
    print("="*60)
    print("\nAsking agent to use system.ai functions...")
    print("(This may use python_exec or other built-in UC functions)")
    
    prompt = """
    Use the available Unity Catalog functions to help me understand:
    1. What functions are available in system.ai?
    2. Can you demonstrate using one of them?
    
    Be concise and practical.
    """
    
    try:
        result = None
        
        # Pass MCP servers directly to ClaudeAgentOptions
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers_config,
                allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                permission_mode="acceptEdits"
            )
        ):
            if hasattr(message, "result"):
                result = message.result
        
        if result:
            print("\n📊 Result:")
            print("-"*60)
            print(result)
            print("-"*60)
        else:
            print("\n⚠️  No result received")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Genie Query (if configured)
    if "genie" in mcp_servers_list:
        print("\n" + "="*60)
        print("Example 2: Genie Natural Language Query")
        print("="*60)
        print("\nAsking Genie to query NYC taxi data...")
        
        genie_prompt = """
        Use the Genie MCP tool to answer this question:
        
        "Show me the top 5 NYC taxi trips by fare amount from the samples.nyctaxi.trips table"
        
        Execute this query using Genie and show me the results.
        """
        
        try:
            result = None
            async for message in query(
                prompt=genie_prompt,
                options=ClaudeAgentOptions(
                    mcp_servers=mcp_servers_config,
                    allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                    permission_mode="acceptEdits"
                )
            ):
                if hasattr(message, "result"):
                    result = message.result
            
            if result:
                print("\n📊 Genie Result:")
                print("-"*60)
                print(result)
                print("-"*60)
        
        except Exception as e:
            print(f"\n⚠️  Genie query error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*60)
        print("Example 2: Genie (Skipped - Not Configured)")
        print("="*60)
        print("💡 Set MCP_GENIE_SPACE_ID in .env to enable Genie natural language queries")
    
    # Example 3: Direct SQL Query (always available)
    print("\n" + "="*60)
    print("Example 3: DBSQL Direct SQL Query")
    print("="*60)
    print("\nExecuting SQL query on NYC taxi trips...")
    
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
    
    try:
        result = None
        async for message in query(
            prompt=sql_prompt,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers_config,
                allowed_tools=["Bash", "Read", "Glob", "mcp__uc_functions__*", "mcp__dbsql__*", "mcp__genie__*"],
                permission_mode="acceptEdits"
            )
        ):
            if hasattr(message, "result"):
                result = message.result
        
        if result:
            print("\n📊 SQL Query Result:")
            print("-"*60)
            print(result)
            print("-"*60)
    
    except Exception as e:
        print(f"\n⚠️  SQL query error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ MCP Integration Examples Complete")
    print("="*60)
    print("\n💡 What This Demonstrates:")
    print("   • Connecting to Databricks MCP servers")
    print("   • Using Unity Catalog functions as agent tools")
    print("   • Natural language queries via Genie")
    print("   • Direct SQL execution via DBSQL")
    print("   • Extending agent with enterprise data access")
    
    return 0


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(run_agent_with_mcp())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
