#!/usr/bin/env python3
"""
MLflow Trace Validator for SEC Financial Analyst Agent.

This module provides functionality to:
1. Query MLflow traces after test execution
2. Validate which tools/components were called
3. Generate validation reports

Usage:
    from scripts.trace_validator import TraceValidator
    
    validator = TraceValidator(experiment_id="...")
    result = validator.validate_trace(
        trace_id="...",
        expected_tools=["get_valuation_score", "genie"]
    )
"""

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import mlflow
from mlflow.entities import SpanType


@dataclass
class ToolCall:
    """Represents a tool call extracted from a trace."""
    name: str
    span_type: str
    inputs: dict = field(default_factory=dict)
    outputs: str = ""
    duration_ms: float = 0
    status: str = "OK"


@dataclass
class TraceValidationResult:
    """Result of validating a trace against expected tools."""
    trace_id: str
    success: bool
    tools_called: list[ToolCall]
    expected_tools: list[str]
    missing_tools: list[str]
    unexpected_tools: list[str]
    total_duration_ms: float
    llm_calls: int
    error: Optional[str] = None


class TraceValidator:
    """
    Validates MLflow traces to verify correct tool usage.
    
    Example:
        validator = TraceValidator()
        
        # After making a request, validate the trace
        result = validator.validate_latest_trace(
            expected_tools=["get_valuation_score"],
            wait_seconds=5
        )
        
        if result.success:
            print("All expected tools were called!")
        else:
            print(f"Missing tools: {result.missing_tools}")
    """
    
    # Tool name patterns to identify different components
    TOOL_PATTERNS = {
        "genie": [r"genie", r"mcp.*genie", r"execute_genie", r"query_space_", r"poll_response_"],
        "knowledge_assistant": [r"query_knowledge_assistant", r"ka", r"knowledge"],
        "valuation_score": [r"get_valuation_score", r"valuation"],
        "compare_peers": [r"compare_peers", r"peer"],
        "growth_trajectory": [r"get_growth_trajectory", r"growth"],
        "risk_summary": [r"get_risk_summary", r"risk"],
    }
    
    def __init__(self, experiment_id: Optional[str] = None, tracking_uri: str = "databricks"):
        """
        Initialize the trace validator.
        
        Args:
            experiment_id: MLflow experiment ID. If not provided, uses MLFLOW_EXPERIMENT_ID env var.
            tracking_uri: MLflow tracking URI (default: "databricks")
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
        
    def search_traces(
        self, 
        max_results: int = 10, 
        since_minutes: int = 30
    ) -> list:
        """
        Search for recent traces.
        
        Args:
            max_results: Maximum number of traces to return
            since_minutes: Only return traces from the last N minutes
            
        Returns:
            List of Trace objects
        """
        # Use locations parameter (new API) instead of deprecated experiment_ids
        locations = None
        if self.experiment_id:
            locations = [f"experiments/{self.experiment_id}"]
        
        try:
            traces = mlflow.search_traces(
                locations=locations,
                max_results=max_results,
                order_by=["timestamp_ms DESC"],
                return_type="list",
                include_spans=True,
            )
            return traces
        except Exception as e:
            # If locations param not supported, try without it
            if "locations" in str(e).lower() or "experiment" in str(e).lower():
                # Try to get experiment from active experiment
                try:
                    active_experiment = mlflow.get_experiment(mlflow.tracking.fluent._get_experiment_id())
                    if active_experiment:
                        return mlflow.search_traces(
                            experiment_ids=[active_experiment.experiment_id],
                            max_results=max_results,
                            order_by=["timestamp_ms DESC"],
                            return_type="list",
                            include_spans=True,
                        )
                except:
                    pass
            raise
    
    def get_trace(self, trace_id: str) -> Optional[object]:
        """Get a specific trace by ID."""
        return mlflow.get_trace(trace_id)
    
    def extract_tool_calls(self, trace) -> list[ToolCall]:
        """
        Extract all tool calls from a trace.
        
        Args:
            trace: MLflow Trace object
            
        Returns:
            List of ToolCall objects
        """
        tool_calls = []
        
        # Search for TOOL spans
        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
        
        for span in tool_spans:
            tool_call = ToolCall(
                name=span.name,
                span_type="TOOL",
                inputs=span.inputs or {},
                outputs=str(span.outputs)[:500] if span.outputs else "",
                duration_ms=self._calculate_duration(span),
                status=span.status.status_code.name if span.status else "UNKNOWN",
            )
            tool_calls.append(tool_call)
        
        # Also search for AGENT spans (might include MCP tools)
        agent_spans = trace.search_spans(span_type=SpanType.AGENT)
        for span in agent_spans:
            if self._is_tool_span(span.name):
                tool_call = ToolCall(
                    name=span.name,
                    span_type="AGENT",
                    inputs=span.inputs or {},
                    outputs=str(span.outputs)[:500] if span.outputs else "",
                    duration_ms=self._calculate_duration(span),
                    status=span.status.status_code.name if span.status else "UNKNOWN",
                )
                tool_calls.append(tool_call)
        
        return tool_calls
    
    def _calculate_duration(self, span) -> float:
        """Calculate span duration in milliseconds."""
        if hasattr(span, 'start_time_ns') and hasattr(span, 'end_time_ns'):
            if span.start_time_ns and span.end_time_ns:
                return (span.end_time_ns - span.start_time_ns) / 1_000_000
        return 0
    
    def _is_tool_span(self, name: str) -> bool:
        """Check if a span name represents a tool call."""
        name_lower = name.lower()
        tool_keywords = ["genie", "query_", "get_", "compare_", "mcp", "tool"]
        return any(kw in name_lower for kw in tool_keywords)
    
    def _categorize_tool(self, tool_name: str) -> str:
        """
        Categorize a tool name into a known component.
        
        Args:
            tool_name: The raw tool name from the span
            
        Returns:
            Categorized tool name (e.g., "genie", "valuation_score")
        """
        name_lower = tool_name.lower()
        
        for category, patterns in self.TOOL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return category
        
        return tool_name  # Return original if no match
    
    def count_llm_calls(self, trace) -> int:
        """Count the number of LLM calls in a trace."""
        llm_spans = trace.search_spans(span_type=SpanType.LLM)
        chat_spans = trace.search_spans(span_type=SpanType.CHAT_MODEL)
        return len(llm_spans) + len(chat_spans)
    
    def validate_trace(
        self,
        trace_id: str,
        expected_tools: list[str],
    ) -> TraceValidationResult:
        """
        Validate a specific trace against expected tools.
        
        Args:
            trace_id: The trace ID to validate
            expected_tools: List of expected tool names/categories
            
        Returns:
            TraceValidationResult with validation details
        """
        try:
            trace = self.get_trace(trace_id)
            if not trace:
                return TraceValidationResult(
                    trace_id=trace_id,
                    success=False,
                    tools_called=[],
                    expected_tools=expected_tools,
                    missing_tools=expected_tools,
                    unexpected_tools=[],
                    total_duration_ms=0,
                    llm_calls=0,
                    error=f"Trace {trace_id} not found",
                )
            
            tool_calls = self.extract_tool_calls(trace)
            tools_found = set()
            
            # Categorize each tool call
            for tool in tool_calls:
                category = self._categorize_tool(tool.name)
                tools_found.add(category)
            
            # Compare with expected
            expected_set = set(expected_tools)
            missing = list(expected_set - tools_found)
            unexpected = list(tools_found - expected_set)
            
            # Calculate total duration
            total_duration = sum(t.duration_ms for t in tool_calls)
            
            return TraceValidationResult(
                trace_id=trace_id,
                success=len(missing) == 0,
                tools_called=tool_calls,
                expected_tools=expected_tools,
                missing_tools=missing,
                unexpected_tools=unexpected,
                total_duration_ms=total_duration,
                llm_calls=self.count_llm_calls(trace),
            )
            
        except Exception as e:
            return TraceValidationResult(
                trace_id=trace_id,
                success=False,
                tools_called=[],
                expected_tools=expected_tools,
                missing_tools=expected_tools,
                unexpected_tools=[],
                total_duration_ms=0,
                llm_calls=0,
                error=str(e),
            )
    
    def validate_latest_trace(
        self,
        expected_tools: list[str],
        wait_seconds: int = 5,
        max_attempts: int = 3,
    ) -> TraceValidationResult:
        """
        Validate the most recent trace against expected tools.
        
        Args:
            expected_tools: List of expected tool names/categories
            wait_seconds: Seconds to wait for trace to be written
            max_attempts: Number of retry attempts
            
        Returns:
            TraceValidationResult with validation details
        """
        for attempt in range(max_attempts):
            time.sleep(wait_seconds)
            
            traces = self.search_traces(max_results=1, since_minutes=5)
            
            if traces:
                return self.validate_trace(
                    trace_id=traces[0].info.trace_id,
                    expected_tools=expected_tools,
                )
            
            if attempt < max_attempts - 1:
                print(f"  No traces found, retrying ({attempt + 1}/{max_attempts})...")
        
        return TraceValidationResult(
            trace_id="",
            success=False,
            tools_called=[],
            expected_tools=expected_tools,
            missing_tools=expected_tools,
            unexpected_tools=[],
            total_duration_ms=0,
            llm_calls=0,
            error="No traces found within the time window",
        )


def format_validation_result(result: TraceValidationResult) -> str:
    """Format a validation result for display."""
    lines = []
    
    status = "✓ PASSED" if result.success else "✗ FAILED"
    lines.append(f"\n{'='*50}")
    lines.append(f"Trace Validation: {status}")
    lines.append(f"{'='*50}")
    
    if result.trace_id:
        lines.append(f"Trace ID: {result.trace_id}")
    
    lines.append(f"\nTools Called ({len(result.tools_called)}):")
    for tool in result.tools_called:
        status_emoji = "✓" if tool.status == "OK" else "✗"
        lines.append(f"  {status_emoji} {tool.name} ({tool.duration_ms:.0f}ms)")
    
    if result.missing_tools:
        lines.append(f"\nMissing Expected Tools: {result.missing_tools}")
    
    if result.unexpected_tools:
        lines.append(f"Additional Tools: {result.unexpected_tools}")
    
    lines.append(f"\nLLM Calls: {result.llm_calls}")
    lines.append(f"Total Tool Duration: {result.total_duration_ms:.0f}ms")
    
    if result.error:
        lines.append(f"\nError: {result.error}")
    
    return "\n".join(lines)


# CLI for standalone validation
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MLflow traces")
    parser.add_argument("--trace-id", help="Specific trace ID to validate")
    parser.add_argument("--experiment-id", help="MLflow experiment ID")
    parser.add_argument("--list", action="store_true", help="List recent traces")
    parser.add_argument("--expected", nargs="+", help="Expected tools (e.g., genie valuation_score)")
    args = parser.parse_args()
    
    validator = TraceValidator(experiment_id=args.experiment_id)
    
    if args.list:
        print("Recent traces:")
        traces = validator.search_traces(max_results=10)
        for trace in traces:
            print(f"  - {trace.info.trace_id} ({trace.info.timestamp_ms})")
        return
    
    if args.trace_id:
        expected = args.expected or []
        result = validator.validate_trace(args.trace_id, expected)
        print(format_validation_result(result))
    else:
        print("Searching for latest trace...")
        expected = args.expected or []
        result = validator.validate_latest_trace(expected, wait_seconds=2)
        print(format_validation_result(result))


if __name__ == "__main__":
    main()
