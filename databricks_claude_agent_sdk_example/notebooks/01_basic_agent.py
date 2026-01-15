"""
Basic Claude Agent Example with Databricks

This example demonstrates:
- Using the Claude Agent SDK with built-in tools
- Connecting to Claude via Databricks serving endpoint
- Autonomous agent execution with file analysis
"""

import asyncio
import sys
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions

# Load environment variables from .env
# This will load ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, and ANTHROPIC_CUSTOM_HEADERS
load_dotenv()


async def run_agent_analysis():
    """Run a comprehensive agent analysis of the project."""
    
    prompt = """
    Please analyze this project and provide a comprehensive summary:
    
    1. List all files in the current directory (excluding hidden files)
    2. Read and summarize the README.md file
    3. Check the requirements.txt and list the key dependencies
    4. Identify what type of project this is and what it demonstrates
    5. Suggest what a user might want to try next
    
    Be thorough but concise.
    """
    
    print("="*60)
    print("Claude Agent SDK with Databricks")
    print("="*60)
    print("\nRunning comprehensive project analysis...")
    print("(Claude will use built-in tools: Bash, Glob, Read)\n")
    print("-"*60)
    
    try:
        result = None
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                allowed_tools=["Bash", "Glob", "Read"]
            )
        ):
            if hasattr(message, "result"):
                result = message.result
        
        if result:
            print("\n📊 Analysis Result:")
            print("="*60)
            print(result)
            print("="*60)
            print("\n✅ Analysis completed successfully!")
            return 0
        else:
            print("\n⚠️  No result received from agent")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(run_agent_analysis())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
