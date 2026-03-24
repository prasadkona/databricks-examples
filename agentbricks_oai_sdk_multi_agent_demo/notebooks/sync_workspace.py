"""Entry point for uv run sync-workspace: same as run-workspace-notebooks --sync-only."""
import sys


def main() -> int:
    sys.argv = [sys.argv[0], "--sync-only"] + sys.argv[1:]
    from .run_workspace_notebooks import main as run_main
    return run_main()


if __name__ == "__main__":
    sys.exit(main())
