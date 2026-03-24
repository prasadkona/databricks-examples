"""
SEC Financial Analyst Multi-Agent Orchestrator

This orchestrator routes queries to three specialized sources:
1. Knowledge Assistant (KA) - Deep document analysis from SEC filings
2. Genie Space - Structured financial data queries
3. UC Functions - Complex analytical computations

Uses Claude on Databricks as the LLM and OpenAI Agents SDK.
"""

import logging
import os
from contextlib import nullcontext
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    get_user_workspace_client,
    process_agent_stream_events,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_env(name: str, default: str) -> str:
    """Get env var with default. Logs warning if using default."""
    value = os.environ.get(name)
    if not value:
        logging.warning(f"Env var {name} not set, using default: {default}")
        return default
    return value

# Config with defaults that match current databricks.yml values
# These defaults are used if env var injection fails
UC_CATALOG = _get_env("UC_CATALOG", "your_catalog")
UC_SCHEMA = _get_env("UC_SCHEMA", "your_schema")
TABLE_PREFIX = _get_env("TABLE_PREFIX", "sec_fin_")
SQL_WAREHOUSE_ID = _get_env("SQL_WAREHOUSE_ID", "your-warehouse-id")
GENIE_SPACE_ID = _get_env("GENIE_SPACE_ID", "")
KA_ENDPOINT = _get_env("KA_ENDPOINT", "")

SUBAGENTS = [
    {
        "name": "genie",
        "type": "genie",
        "space_id": GENIE_SPACE_ID,
        "description": (
            "Query structured financial data including company financials, revenue segments, "
            "geographic breakdown, stock prices, and market metrics. Use this for questions about "
            "numbers, comparisons, rankings, and data exploration. "
            "Covers NVIDIA (NVDA), Apple (AAPL), and Samsung (005930.KS) FY2024 data."
        ),
    },
    {
        "name": "knowledge_assistant",
        "type": "serving_endpoint",
        "endpoint": KA_ENDPOINT,
        "description": (
            "Query the SEC Financial Analyst knowledge assistant for deep document analysis. "
            "Use this for questions about risk factors, business descriptions, management discussion, "
            "accounting policies, legal proceedings, and detailed content from 10-K filings. "
            "Best for qualitative insights and document-based research."
        ),
    },
]

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

try:
    set_default_openai_client(AsyncDatabricksOpenAI())
    set_default_openai_api("chat_completions")
    set_trace_processors([])
    mlflow.openai.autolog()
    logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
except Exception as _init_err:
    logging.warning(f"Non-fatal init error (will retry at request time): {_init_err}")

_tool_client = AsyncDatabricksOpenAI()

# ---------------------------------------------------------------------------
# UC Function Tools
# ---------------------------------------------------------------------------


@function_tool
async def get_valuation_score(ticker: str) -> str:
    """
    Get a composite valuation score (1-100) for a stock based on PE ratio, growth,
    and profitability. Returns score and component breakdown.
    Use for: "What's NVIDIA's valuation score?", "Is Apple overvalued?"

    Args:
        ticker: Stock ticker symbol (NVDA, AAPL, or 005930.KS)
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=f"SELECT * FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}valuation_score('{ticker}')"
        ).result

        if result.data_array:
            row = result.data_array[0]
            score = int(row[2]) if row[2] else 0
            recommendation = "BUY" if score >= 70 else "HOLD" if score >= 40 else "SELL"
            return f"""
Valuation Analysis for {row[1]} ({row[0]}):
- Overall Score: {score}/100
- Recommendation: {recommendation}
- P/E Score: {row[3]}/100
- Growth Score: {row[4]}/100
- Profitability Score: {row[5]}/100
"""
        return f"No valuation data found for {ticker}"
    except Exception as e:
        return f"Error fetching valuation score: {str(e)}"


@function_tool
async def compare_peers(ticker: str) -> str:
    """
    Compare a company against its peers on key metrics like revenue growth,
    gross margin, and net margin. Shows company value vs peer average.
    Use for: "How does NVIDIA compare to peers?", "Compare Apple's margins to competitors"

    Args:
        ticker: Stock ticker symbol (NVDA, AAPL, or 005930.KS)
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=f"SELECT * FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}compare_peers('{ticker}')"
        ).result

        if result.data_array:
            output = [f"Peer Comparison for {ticker}:\n"]
            for row in result.data_array:
                metric = row[0]
                company_val = row[1]
                peer_avg = row[2]
                try:
                    diff = float(company_val) - float(peer_avg)
                    status = "ABOVE" if diff > 0 else "BELOW"
                    output.append(f"- {metric}: {company_val} (Peer Avg: {peer_avg}) → {status} avg by {abs(diff):.1f}")
                except:
                    output.append(f"- {metric}: {company_val} (Peer Avg: {peer_avg})")
            return "\n".join(output)
        return f"No peer comparison data found for {ticker}"
    except Exception as e:
        return f"Error comparing peers: {str(e)}"


@function_tool
async def get_growth_trajectory(ticker: str) -> str:
    """
    Get revenue growth trajectory and projections based on historical trends.
    Returns year-over-year growth analysis.
    Use for: "What's NVIDIA's growth outlook?", "Show Apple's revenue trend"

    Args:
        ticker: Stock ticker symbol (NVDA, AAPL, or 005930.KS)
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=f"SELECT * FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}growth_trajectory('{ticker}')"
        ).result

        if result.data_array:
            output = [f"Growth Trajectory for {ticker}:\n"]
            for row in result.data_array:
                output.append(f"- {row[0]}: {row[1]}")
            return "\n".join(output)
        return f"No growth trajectory data found for {ticker}"
    except Exception as e:
        return f"Error projecting growth: {str(e)}"


@function_tool
async def get_risk_summary(ticker: str) -> str:
    """
    Summarize key risk factors including valuation risk, growth sustainability,
    and debt levels with severity assessments.
    Use for: "What are NVIDIA's risks?", "Risk analysis for Apple"

    Args:
        ticker: Stock ticker symbol (NVDA, AAPL, or 005930.KS)
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=f"SELECT * FROM {UC_CATALOG}.{UC_SCHEMA}.{TABLE_PREFIX}risk_summary('{ticker}')"
        ).result

        if result.data_array:
            output = [f"Risk Summary for {ticker}:\n"]
            for row in result.data_array:
                risk_type = row[0]
                severity = row[1]
                description = row[2]
                emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
                output.append(f"{emoji} [{severity}] {risk_type}: {description}")
            return "\n".join(output)
        return f"No risk summary found for {ticker}"
    except Exception as e:
        return f"Error fetching risk summary: {str(e)}"


# ---------------------------------------------------------------------------
# Subagent Tools
# ---------------------------------------------------------------------------

def _make_subagent_tool(subagent: dict):
    """Create a function_tool for a single subagent definition."""
    endpoint = subagent["endpoint"]
    model = f"apps/{endpoint}" if subagent["type"] == "app" else endpoint

    async def _call(question: str) -> str:
        response = await _tool_client.responses.create(
            model=model,
            input=[{"role": "user", "content": question}],
        )
        return response.output_text

    _call.__name__ = f"query_{subagent['name']}"
    _call.__doc__ = subagent["description"]
    return function_tool(_call)


subagent_tools = [_make_subagent_tool(sa) for sa in SUBAGENTS if sa["type"] != "genie"]

uc_function_tools = [
    get_valuation_score,
    compare_peers,
    get_growth_trajectory,
    get_risk_summary,
]

all_tools = subagent_tools + uc_function_tools

# ---------------------------------------------------------------------------
# MCP Server + Orchestrator Agent
# ---------------------------------------------------------------------------

async def init_mcp_server():
    """Create a Genie MCP server if a genie subagent is configured."""
    genie = next((sa for sa in SUBAGENTS if sa["type"] == "genie"), None)
    if genie is None:
        return nullcontext()
    return McpServer(
        url=build_mcp_url(f"/api/2.0/mcp/genie/{genie['space_id']}"),
        name=genie["description"],
    )


def create_orchestrator_agent(mcp_server: McpServer) -> Agent:
    """Build the orchestrator agent with all tools and MCP servers."""

    instructions = """You are the SEC Financial Analyst, a multi-agent orchestrator that helps users analyze SEC filings and financial data for NVIDIA, Apple, and Samsung.

## Your Capabilities

You have access to three specialized data sources:

### 1. Genie Space (Structured Data)
Use the Genie MCP tools for:
- Financial metrics (revenue, profit, margins, etc.)
- Stock price data and technical indicators
- Business segment breakdowns
- Geographic revenue distribution
- Peer comparisons and rankings
- Historical trends and YTD performance

### 2. Knowledge Assistant (Document Analysis)
Use query_knowledge_assistant for:
- Risk factors from 10-K filings
- Management discussion and analysis (MD&A)
- Business descriptions and strategy
- Accounting policies and procedures
- Legal proceedings and contingencies
- Qualitative insights from SEC documents

### 3. UC Functions (Complex Analytics)
Use these specialized functions:
- get_valuation_score: Composite valuation scoring (1-100)
- compare_peers: Cross-company metric comparisons
- get_growth_trajectory: Revenue growth trends
- get_risk_summary: Risk factor assessment

## Routing Guidelines

**For quantitative questions** → Use Genie Space first
Examples: "What is NVIDIA's revenue?", "Compare margins", "Stock performance"

**For qualitative questions** → Use Knowledge Assistant
Examples: "What are Apple's risk factors?", "Describe Samsung's business strategy"

**For analytical questions** → Use UC Functions
Examples: "Is NVIDIA overvalued?", "Compare NVIDIA to peers", "Investment thesis"

**For comprehensive questions** → Combine multiple sources
Example: "Should I invest in NVIDIA?" → Use valuation_score + compare_peers + risk_summary + genie data

## Response Guidelines

1. Always cite your data source (Genie, KA, or UC Function)
2. For multi-part questions, break them down and use appropriate tools
3. Provide specific numbers with context (YoY change, peer comparison)
4. When uncertain, ask clarifying questions
5. If data is unavailable, explain what alternatives exist

## Available Companies
- NVIDIA Corporation (NVDA)
- Apple Inc. (AAPL)
- Samsung Electronics (005930.KS)

Data covers FY2024 financials and 2 years of stock price history.
"""

    return Agent(
        name="SEC_Financial_Analyst_Orchestrator",
        instructions=instructions,
        model="databricks-claude-3-7-sonnet",
        mcp_servers=[mcp_server] if mcp_server else [],
        tools=all_tools,
    )


# ---------------------------------------------------------------------------
# MLflow Responses API Handlers
# ---------------------------------------------------------------------------

@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with await init_mcp_server() as mcp_server:
        agent = create_orchestrator_agent(mcp_server)
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages, max_turns=25)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with await init_mcp_server() as mcp_server:
        agent = create_orchestrator_agent(mcp_server)
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages, max_turns=25)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
