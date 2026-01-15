# Databricks Examples

A collection of practical examples and demonstrations for building AI agents and applications using Databricks.

## 📚 Projects

### [Claude Agent SDK with Databricks](./databricks_claude_agent_sdk_example)

**Access Claude AI models through Databricks** and build autonomous AI agents using the Claude Agent SDK. This comprehensive example demonstrates how to leverage Claude models (Haiku, Sonnet, Opus) served via Databricks Model Serving endpoints:

- **Basic Agent Usage** - File analysis, code generation, autonomous task execution
- **MLflow Integration** - Manual logging, autologging, and GenAI evaluation
- **MCP Integration** - Unity Catalog Functions, DBSQL, and Genie natural language queries
- **Enterprise Observability** - Complete tracking and monitoring of agent interactions

**Key Features:**
- 🤖 **Claude via Databricks** - Access Claude models (Haiku, Sonnet, Opus) through Databricks Model Serving
- ✅ **Runs locally and on Databricks** - Develop locally, deploy on Databricks infrastructure
- 📊 **6 progressive examples** - From basic to enterprise with MLflow and MCP integration
- 🔧 **Enterprise data access** - Unity Catalog Functions, DBSQL, and Genie MCP servers
- 🚀 **Complete observability** - MLflow autologging and GenAI evaluation framework
- 📈 **Production-ready** - Databricks authentication, security, and governance

**Tech Stack:** Claude Agent SDK, Databricks Model Serving, MLflow, Model Context Protocol (MCP)

**Why This Matters:** All Claude interactions route through your Databricks workspace, enabling enterprise security controls, data governance, cost tracking, and seamless integration with your lakehouse data.

👉 [View detailed documentation](./databricks_claude_agent_sdk_example/README.md)

---

## 🚀 Getting Started

Each project contains:
- **README.md** - Project overview, examples, and usage
- **SETUP.md** - Detailed setup instructions and configuration
- **requirements.txt** - Python dependencies
- **Example notebooks/scripts** - Runnable code examples

### Quick Start

1. Choose a project from the list above
2. Follow the project's SETUP.md for installation
3. Configure your environment (`.env` file)
4. Run the examples

## 📖 Prerequisites

- **Python 3.9+**
- **Databricks workspace** with appropriate access
- **Claude Code Runtime** (for Claude Agent SDK examples)
- **Git** for cloning this repository

## 🤝 Contributing

This repository contains working examples and demonstrations. Each project is self-contained with its own documentation and dependencies.

## 📄 License

See individual project folders for license information.

---

**Repository maintained by:** Prasad Kona (prasad.kona@gmail.com)
