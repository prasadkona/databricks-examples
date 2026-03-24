# Databricks Examples

**Author**: Prasad Kona  
**Last Updated**: March 19, 2026

A collection of practical, production-ready examples demonstrating how to build AI agents, deploy ML models, and create intelligent applications on the Databricks platform. Each project includes complete code, detailed documentation, and best practices for enterprise deployment.

---

## 📚 Projects

### 1. [Python UDFs with Custom Dependencies](./python_udfs_custom_dependencies)

Deploy ML models as Unity Catalog UDFs with custom dependencies for real-time SQL inference at scale. This project demonstrates the complete lifecycle of training ML models, packaging them as Python wheels, and deploying them as UDFs that can be called directly from SQL queries.

**Highlights:**
- Train scikit-learn models on NYC Taxi data (regression & classification)
- Package models as Python wheel files stored in Unity Catalog volumes
- Create UDFs with custom dependencies for real-time inference
- 40+ SQL query examples for dashboards and BI tools

**Tech Stack:** Python, scikit-learn, Unity Catalog, Databricks SQL, DBR 18.1+

👉 [View detailed documentation](./python_udfs_custom_dependencies/README.md)

---

### 2. [Claude Agent SDK with Databricks](./databricks_claude_agent_sdk_example)

Build autonomous AI agents using Claude models served by Databricks through the Claude Agent SDK. Access Claude AI models (Haiku, Sonnet, Opus) via Databricks Model Serving endpoints with enterprise observability and data governance.

**Highlights:**
- 6 progressive examples from basic to enterprise integration
- MLflow integration for tracking and GenAI evaluation
- MCP integration with Unity Catalog Functions, DBSQL, and Genie
- Complete observability and production-ready patterns

**Tech Stack:** Claude Agent SDK, Databricks Model Serving, MLflow, Model Context Protocol

👉 [View detailed documentation](./databricks_claude_agent_sdk_example/README.md)

---

### 3. [Agent Bricks: Knowledge Assistant](./agent_bricks_ka_example)

Create and manage Databricks Agent Bricks Knowledge Assistants programmatically. Knowledge Assistants enable document-based Q&A using RAG (Retrieval-Augmented Generation) over Unity Catalog Volumes with automatic indexing and citation-backed responses.

**Highlights:**
- Create Knowledge Assistants via REST API and Python SDK
- Index documents from Unity Catalog Volumes (PDF, TXT, MD, DOCX, PPTX)
- Multi-turn conversations with context retention
- Add example questions with guidelines to improve response quality
- Sync and re-index knowledge sources programmatically

**Tech Stack:** Databricks Agent Bricks, REST API, Databricks SDK, Unity Catalog Volumes, Model Serving

👉 [View detailed documentation](./agent_bricks_ka_example/README.md)

---

### 4. [AI Agent Metadata Extractor](./ai_agent_metadata_extract)

Extract and classify metadata about all deployed AI agents, serving endpoints, and Knowledge Assistants in your Databricks workspace. Provides comprehensive visibility into Foundation Models, Agent Bricks, AI Gateway endpoints, UC-registered models, and Knowledge Assistants.

**Highlights:**
- Extract serving endpoints with single API call (fast) or detailed per-endpoint calls
- Classify endpoints by type: FM (PPT/PT), Agent Bricks (KA/MAS/KIE/MS), AI Gateway, Classic ML
- Extract Knowledge Assistants using the dedicated Knowledge Assistants API (`/api/2.1/knowledge-assistants`)
- Enrich Agent Bricks with Tiles API metadata (name, description, instructions)
- Export to JSON and Markdown reports

**Tech Stack:** Databricks SDK, REST API, Python

👉 [View detailed documentation](./ai_agent_metadata_extract/README.md)

---

### 5. [SEC Financial Analyst — Multi-Agent AI System](./agentbricks_oai_sdk_multi_agent_demo)

An end-to-end, production-quality AI agent system that analyzes SEC 10-K filings and financial data. Drop any company's annual reports into a Unity Catalog volume — the pipeline automatically discovers companies using Databricks AI functions, extracts structured financial metrics, loads stock history, and exposes the data to a multi-agent orchestrator that intelligently routes questions across three specialized tools.

**The Use Case:**  
Investment analysts spend hours manually cross-referencing financial filings, market data, and analytical models. This demo automates that workflow with a conversational AI agent:

- *"What are the key risk factors from the 10-K?"* → Knowledge Assistant searches SEC filing PDFs
- *"What is the company's revenue trend over 3 years?"* → Genie Space queries structured financial tables
- *"Should I invest in this company?"* → Supervisor agent orchestrates across all three sources: valuation score (UC function), revenue data (Genie), risk factors (KA)

**Highlights:**

**Agent Bricks — Knowledge Assistant (KA)**
- Programmatically create and manage a KA over SEC PDF filings in Unity Catalog Volumes
- Automatic RAG indexing with citation-backed answers to qualitative questions

**Agent Bricks — Genie Space**
- Natural language SQL interface over gold-layer financial and stock tables
- Agent queries Genie via MCP for quantitative analysis (revenue, EPS, stock performance)

**Supervisor / Multi-Agent Orchestration**
- OpenAI Agents SDK supervisor routes requests to KA, Genie, and UC Functions as tools
- UC Functions handle complex analytical computations (valuation scoring, peer comparison)
- Full MLflow tracing across every agent turn for observability and debugging

**AI-Driven Data Pipeline (Spark Declarative Pipelines)**
- `ai_parse_document` → `ai_classify` → `ai_extract` chain in DLT for fully automatic company discovery from PDFs — no hardcoded tickers or mappings
- `mapInPandas` + `yfinance` in DLT for dynamic stock history loading
- Bronze → Silver → Gold medallion architecture in Unity Catalog

**Deployed as a Databricks App**
- FastAPI server packaged as a Databricks App with OAuth and chat proxy enabled
- `uv run run-sequence --all` drives the complete lifecycle: KA setup → data pipeline → agent deployment

**Tech Stack:** OpenAI Agents SDK, Databricks Agent Bricks (KA + Genie), Spark Declarative Pipelines, AI Functions, Unity Catalog, Databricks Apps, MLflow, FastAPI

👉 [View detailed documentation](./agentbricks_oai_sdk_multi_agent_demo/README.md)

---

## 🚀 Getting Started

Each project is self-contained with its own documentation and dependencies:

1. **Choose a project** from the list above
2. **Navigate to the project folder** and read the README.md for detailed information
3. **Follow the setup instructions** in SETUP.md or README.md
4. **Configure your environment** (`.env` file or config template)
5. **Run the examples**

## 📖 Prerequisites

- **Python 3.9+**
- **Databricks workspace** with appropriate access
- **Git** for cloning this repository

Additional requirements vary by project - see individual project READMEs for details.

## 📁 Repository Structure

```
databricks-examples/
├── README.md                                      # This file
│
├── python_udfs_custom_dependencies/               # ML models as Unity Catalog UDFs
├── databricks_claude_agent_sdk_example/           # Claude Agent SDK progressive examples
├── agent_bricks_ka_example/                       # Agent Bricks Knowledge Assistant
├── ai_agent_metadata_extract/                     # AI endpoint metadata & reporting
└── agentbricks_oai_sdk_multi_agent_demo/          # SEC Financial Analyst Multi-Agent
```

Each project is self-contained with its own `README.md`, `.env.example`/`.env.template`, and dependencies.

## 🎯 Use Cases

### ML & Data Science
- Deploy ML models as UDFs for real-time inference in SQL queries
- Create production-ready data transformations with custom Python libraries

### AI Agents & GenAI
- Build autonomous agents with Claude models and enterprise data access
- Integrate AI agents with Unity Catalog, DBSQL, and Genie for data-driven applications

### Agent Bricks & RAG
- Create Knowledge Assistants for document Q&A with automatic RAG indexing
- Build production-ready document assistants with citation-backed responses

### Multi-Agent Orchestration & Databricks Apps
- Supervisor agent pattern: route user questions across KA, Genie Space, and UC Functions
- Deploy multi-agent systems as full Databricks Apps with OAuth, chat proxy, and MLflow tracing
- End-to-end financial analyst use case: SEC filings + structured data + custom computations

### Governance & Observability
- Extract and classify all AI endpoints across your workspace
- Generate reports on Foundation Models, Agent Bricks, and AI Gateway usage

## 🔐 Security

- **Never commit credentials** to version control
- Use `.env` files or config files stored in your home directory (`~/`)
- Templates in the repo contain only placeholders
- Follow each project's security guidelines

## 🤝 Contributing

This repository contains working examples and demonstrations. Each project is self-contained with its own documentation and dependencies.

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](./LICENSE) file for details.

**You are free to:**
- ✅ Use this code commercially
- ✅ Modify and distribute
- ✅ Use in private projects
- ✅ Use for any purpose

This is open-source software - feel free to reuse, adapt, and build upon these examples!

---

**Repository**: https://github.com/prasadkona/databricks-examples  
**Author**: Prasad Kona  
**Contact**: prasad.kona@gmail.com  
**Last Updated**: March 19, 2026
