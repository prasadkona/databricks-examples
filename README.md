# Databricks Examples

**Author**: Prasad Kona  
**Last Updated**: January 26, 2026

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
├── README.md                                  # This file
│
├── python_udfs_custom_dependencies/           # Python UDFs with custom dependencies
│   ├── README.md                              # Full project documentation
│   ├── requirements.txt                       # Python dependencies
│   └── notebooks/                             # 4 notebooks (training, UDFs, SQL examples)
│
└── databricks_claude_agent_sdk_example/       # Claude Agent SDK demos
    ├── README.md                              # Full project documentation
    ├── SETUP.md                               # Setup instructions
    ├── requirements.txt                       # Python dependencies
    └── notebooks/                             # 6 progressive examples
```

## 🎯 Use Cases

### ML & Data Science
- Deploy ML models as UDFs for real-time inference in SQL queries
- Create production-ready data transformations with custom Python libraries

### AI Agents & GenAI
- Build autonomous agents with Claude models and enterprise data access
- Integrate AI agents with Unity Catalog, DBSQL, and Genie for data-driven applications

## 🔐 Security

- **Never commit credentials** to version control
- Use `.env` files or config files stored in your home directory (`~/`)
- Templates in the repo contain only placeholders
- Follow each project's security guidelines

## 🤝 Contributing

This repository contains working examples and demonstrations. Each project is self-contained with its own documentation and dependencies.

## 📄 License

See individual project folders for license information.

---

**Repository**: https://github.com/prasadkona/databricks-examples  
**Author**: Prasad Kona  
**Contact**: prasad.kona@gmail.com  
**Last Updated**: January 26, 2026
