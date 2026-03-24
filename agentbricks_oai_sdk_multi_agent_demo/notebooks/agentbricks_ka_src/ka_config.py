# Databricks notebook source
# MAGIC %md
# MAGIC # KA Configuration
# MAGIC
# MAGIC KA-specific configuration: name, description, instructions, and example questions.
# MAGIC
# MAGIC Reads `KA_NAME`, `KA_TILE_ID`, `KA_ENDPOINT` from the central config (env vars).
# MAGIC
# MAGIC This module is imported by the step scripts; it is not executed directly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## KA Identity

# COMMAND ----------

import os

KA_NAME = os.getenv("KA_NAME", "SEC_Financial_Analyst_KA")
KA_TILE_ID = os.getenv("KA_TILE_ID", "")
KA_ENDPOINT = os.getenv("KA_ENDPOINT", "")

VOLUME_DATASET_FOLDER = "sec_2024"

# COMMAND ----------

# MAGIC %md
# MAGIC ## KA Description and Instructions
# MAGIC
# MAGIC These are passed to the KA creation API to configure the assistant's behavior.

# COMMAND ----------

KA_DESCRIPTION = (
    "Financial analyst assistant that answers questions about annual reports, "
    "10-K filings, and financial statements for companies like NVIDIA, Apple, "
    "Samsung, and others. Covers revenue, earnings, business segments, risk "
    "factors, and strategic initiatives."
)

KA_INSTRUCTIONS = """\
You are a financial analyst assistant specializing in company annual reports and SEC filings.
You have access to financial filings that may include 10-K reports, annual reports, and financial statements.

Guidelines:
1. **Cite Sources**: Always cite the specific company and document section when providing financial data
2. **Structured Comparisons**: When comparing companies, present data in tables or bullet points
3. **Year-over-Year**: For revenue/earnings questions, include YoY changes when available
4. **Fiscal Year Clarity**: Be aware that different companies have different fiscal year end dates
5. **Be Honest**: If information is not in the documents, clearly state that and list available companies
6. **Financial Terminology**: Use precise terms (revenue, net income, operating income, etc.)
7. **Segment Details**: When discussing business segments, explain what each segment includes
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Questions
# MAGIC
# MAGIC These are added to the KA after creation to seed it with relevant financial queries.
# MAGIC Each example includes a `guideline` that describes the expected answer quality.

# COMMAND ----------

EXAMPLE_QUESTIONS = [
    {
        "question": "What was NVIDIA's total revenue for fiscal year 2024?",
        "guideline": "Should state $60.9 billion, up 126% year-over-year, with Data Center as the primary driver",
    },
    {
        "question": "How does NVIDIA's Data Center revenue compare to Gaming revenue?",
        "guideline": "Should compare Data Center ($47.5B, up 217%) vs Gaming ($10.4B, up 15%)",
    },
    {
        "question": "What are Apple's main product categories?",
        "guideline": "Should list iPhone, Mac, iPad, Wearables/Home/Accessories, and Services",
    },
    {
        "question": "What are the key risk factors mentioned in Apple's 10-K filing?",
        "guideline": "Should reference macroeconomic conditions, supply chain, competition, and regulatory risks",
    },
    {
        "question": "What are Samsung's two main business divisions?",
        "guideline": "Should explain DS (Device Solutions - semiconductors) and DX (Device eXperience - consumer electronics)",
    },
    {
        "question": "Compare the fiscal year 2024 performance of NVIDIA, Apple, and Samsung",
        "guideline": "Should compare revenue figures and highlight key growth drivers for each company",
    },
    {
        "question": "What is NVIDIA's Blackwell platform?",
        "guideline": "Should describe it as NVIDIA's most powerful AI platform for generative AI",
    },
    {
        "question": "What is Apple's Services segment and how did it perform?",
        "guideline": "Should describe Services (App Store, iCloud, Apple Music, etc.) and note growth",
    },
    {
        "question": "What is NVIDIA's net income for FY2024?",
        "guideline": "Should state $33.0 billion, up significantly year-over-year",
    },
    {
        "question": "What accounting standards does Samsung follow?",
        "guideline": "Should mention Korean IFRS (Korean International Financial Reporting Standards)",
    },
]
