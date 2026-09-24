# Test Case Comparison System

## Overview

The Test Case Comparison System is an AI-assisted application that automates the comparison and analysis of test cases across different systems (New Shopping Cart vs Old Booking Engine). It scans and indexes test cases using embeddings, stores them in a vector database (Qdrant), and uses Large Language Models (Gemini) to generate sophisticated comparisons with semantic matching and intelligent analysis.

This repository contains Python modules and helper logic to run the full test case comparison workflow locally.

## Technologies

- **Indexing**: Qdrant (Vector Database for test case embeddings)
- **AI / LLM**: Google Vertex AI integration (gemini-2.5-pro for generation, text-embedding-004 for embeddings)
- **Data Processing**: Pandas, NumPy, openpyxl for XLSX handling
- **Multi-Agent System**: LangGraph for agent orchestration
- **Similarity**: scikit-learn for cosine similarity computation
- **Configuration**: Environment variables via .env

## Prerequisites

- Python 3.8 or higher (3.10+ recommended)
- pip package manager
- Git (for cloning repository)
- Google Cloud Vertex AI access and a service account JSON
- Qdrant Cloud URL and API Key (or local Qdrant instance)

## Usage

### Clone the repository

```bash
git clone <your-repository-url>
cd <your-repo-folder>
```

### Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Install required packages

If this repo includes requirements.txt, install it with:

```bash
pip install -r requirements.txt
```

### Configure credentials and environment

Set required environment variables in a .env file or update your system configuration. Required variables:

- `GOOGLE_APPLICATION_CREDENTIALS` — path to Google service account JSON
- `GCP_PROJECT_ID` — Google Cloud Project ID
- `GCP_LOCATION` — Google Cloud Location (e.g., us-central1)
- `QDRANT_URL` — URL of your Qdrant instance
- `QDRANT_API_KEY` — API Key for Qdrant

### Running the application

**Test Case Indexing:**

```python
from qdrant_indexer import DualTestCaseIndexer

indexer = DualTestCaseIndexer(
    collection_name="test_case_comparisons",
    embedding_model="gemini-embedding-001",
    mode="unified"  # or "separate"
)

# Create collections
indexer.create_collections()

# Index test cases from Excel file
indexer.index_from_excel(
    excel_path="Final_Sheet.xlsx",
    recreate=False
)
```

**Test Case Comparison:**

Use the multi-agent semantic comparison system to automatically match and analyze test cases:

```python
from semantic_comparsion_agent import RetrievalAgent, ComparisonAgent

# Initialize agents with Qdrant and embedding models
retrieval_agent = RetrievalAgent(qdrant_client, embedding_model)
comparison_agent = ComparisonAgent()

# Compare test cases and generate results
results = run_comparison_pipeline(new_test_cases, retrieval_agent, comparison_agent)
```

## Features

- **Unified Indexing**: Index both new and old test cases in a single collection with source tagging or in separate collections
- **Semantic Search**: Find similar test cases using vector embeddings
- **Metadata Extraction**: Automatically extract intent, domain, test type, and complexity from test cases
- **Multi-Agent Analysis**: 
  - Retrieval Agent: Finds top-100 similar candidates
  - Filtering Agent: Narrows down to top-50 using LLM scoring
  - Comparison Agent: Generates detailed semantic matches and analysis
- **Rate Limit Handling**: Built-in exponential backoff for API rate limits
- **XLSX Integration**: Read test cases from and write results to Excel files

## Output and persistence

- **Qdrant Collections**: Persistent vector indices of your test cases for fast retrieval and context-aware comparison
- **Comparison Results**: XLSX files containing detailed match results with similarity scores and AI analysis
- **Processing Logs**: Detailed logs of comparison decisions and reasoning

## Example quick usage

1. Clone repo and create a virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Set `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT_ID`, `QDRANT_URL`, and `QDRANT_API_KEY` in your .env file
4. Create an indexer instance and index your test cases from Excel
5. Use the semantic comparison system to match and analyze test cases
6. Download results in XLSX format

## Configuration

All configurations are externalized to environment variables for security and flexibility. No hardcoded credentials should be present in the source code. Load your configuration from a `.env` file or system environment variables before initializing any components.
