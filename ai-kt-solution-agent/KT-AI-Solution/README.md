# KT Documentation Generator

## Overview
The KT Documentation Generator is an AI-assisted Streamlit application that automates the creation of comprehensive technical documentation for Python, JavaScript, React, and Node.js codebases. It scans and indexes source code using embeddings, stores them in a vector database (Qdrant), and uses Large Language Models (Gemini) to generate system overviews, API references, data models, and deployment guides.
This repository contains the Streamlit app and helper logic to run the full documentation generation workflow locally.

## Technologies

*   **UI / App**: Streamlit (Python)
*   **AI / LLM**: Google Vertex AI integration (`gemini-2.5-pro` for generation, `text-embedding-004` for embeddings)
*   **Persistence**: Qdrant (Vector Database for code embeddings)
*   **Configuration**: Environment variables via `.env`

## Prerequisites

*   Python 3.8 or higher (3.10+ recommended)
*   pip package manager
*   Git (for cloning repository)
*   Google Cloud Vertex AI access and a service account JSON
*   Qdrant Cloud URL and API Key (or local Qdrant instance)

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
Set required environment variables in a `.env` file or update `config.py`. Required variables:

*   `GOOGLE_APPLICATION_CREDENTIALS` — path to Google service account JSON
*   `GCP_PROJECT_ID` — Google Cloud Project ID
*   `GCP_LOCATION` — Google Cloud Location (e.g., `us-central1`)
*   `QDRANT_URL` — URL of your Qdrant instance
*   `QDRANT_API_KEY` — API Key for Qdrant

### Start the app

```bash
streamlit run app.py
```

Open the local Streamlit URL (printed in the terminal). 

1.  **Index Codebase**: Upload a ZIP file or provide a local directory path to your codebase. Click "Start Indexing" to scan and embed the files.
2.  **Generate Documentation**: Switch to the generation tab, select the documentation types you need (System Overview, API Reference, etc.), and click "Generate Documentation".
3.  **Chat with Codebase**: Use the chat interface to ask specific questions about your code.

## Output and persistence

*   **Generated Documentation**: Downloadable ZIP file containing Markdown files for each selected documentation type.
*   **Qdrant Collections**: Persistent vector indices of your codebase for fast retrieval and context-aware generation.

## Example quick usage

1.  Clone repo and create a virtual environment.
2.  Install dependencies.
3.  Set `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT_ID`, and `QDRANT` variables in `.env`.
4.  Run `streamlit run app.py` and point it to a codebase directory.
