# Cypress Code to Business Logic Converter

## Overview

Cypress Code to Business Logic Converter is an AI-powered FastAPI application that automates the translation of Cypress automation test code into clear, business-friendly logic descriptions. It accepts individual Cypress test files (`.js`, `.ts`, `.jsx`, `.tsx`) or ZIP archives, extracts every `it()` test block, sends each block to Azure OpenAI to generate detailed business logic steps and a concise executive summary, and packages the results into formatted Excel workbooks delivered as a single ZIP download — all driven through an intuitive drag-and-drop web interface.

This repository contains the FastAPI backend, an LLM configuration manager supporting multiple providers, and a static HTML/CSS/JavaScript frontend to run the full conversion workflow locally.

## Technologies

- **UI / App:** Vanilla HTML / CSS / JavaScript (static single-page frontend)
- **Backend / API:** FastAPI, Uvicorn
- **AI / LLM:** Azure OpenAI (`openai` Python SDK) — chat completions for code-to-business-logic conversion and summary generation
- **LLM Configuration:** Pydantic-based `ConfigManager` with preset support for Azure OpenAI, Google Vertex AI, OpenAI, and Anthropic Claude
- **Data Processing:** Pandas (DataFrame manipulation), OpenPyXL (Excel generation with formatted column widths)
- **File Handling:** Python `zipfile` / `io` for ZIP extraction and creation, `tempfile` for temporary output storage
- **Utilities:** python-dotenv (environment variable management), python-multipart (file upload handling)

## Prerequisites

- Python 3.10 or higher (3.8+ minimum)
- pip package manager
- Azure OpenAI resource with:
  - API key
  - Endpoint URL (e.g. `https://<resource-name>.openai.azure.com/`)
  - Deployment name (the model deployment you created in Azure)
  - API version (e.g. `2024-02-15-preview`)

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

```bash
pip install -r requriements.txt
```

### Configure Azure OpenAI credentials

You can configure credentials in two ways:

**Option A — Via the web interface (recommended):**
Launch the app (see below), then fill in the Azure OpenAI Configuration form directly in the browser before uploading files.

**Option B — Edit defaults in `main.py`:**
Update the `current_config` dictionary near the top of `main.py` with your Azure OpenAI values:

- `api_key` — Azure OpenAI API key
- `azure_endpoint` — Azure OpenAI endpoint URL
- `deployment_name` — name of your Azure OpenAI model deployment
- `api_version` — Azure OpenAI API version
- `temperature` — LLM sampling temperature (default `0.2`)

### Start the app

```bash
python main.py
```

Open `http://localhost:8001` in your browser. Use the drag-and-drop area to upload Cypress test files (individual `.js`/`.ts`/`.jsx`/`.tsx` files or a `.zip` archive), then click **Convert to Business Logic**. The app will extract every `it()` test block, generate business logic descriptions and summaries via Azure OpenAI, and produce a downloadable ZIP containing one Excel file per source file.

## Output and persistence

- **Excel files** — Each source file produces a `<filename>_output.xlsx` workbook with the following columns: *test_description*, *business_logic_summary*, *business_logic*, *file_path*, *code_line_number*, *test_type*, and *code_snippet*. Column widths are auto-configured for readability.

- **ZIP archive** — All generated Excel files are bundled into a single `business_logic_output_<timestamp>.zip`, preserving the original directory structure when processing ZIP uploads. Files are stored temporarily in the system temp directory and served via the `/api/download/{filename}` endpoint.

## Example quick usage

1. Clone the repo and create a virtual environment.
2. Install dependencies with `pip install -r requriements.txt`.
3. Run `python main.py` to start the server on port 8001.
4. Open `http://localhost:8001` in your browser.
5. Enter your Azure OpenAI API key, endpoint, deployment name, and API version in the configuration form.
6. Upload Cypress test files or a ZIP archive using drag-and-drop.
7. Click **Convert to Business Logic** and wait for processing to complete.
8. Download the generated ZIP containing Excel files with business logic descriptions for every test case.
