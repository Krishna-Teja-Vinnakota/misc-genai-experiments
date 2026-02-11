from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import re
from datetime import datetime
import pandas as pd
from openai import AzureOpenAI
import tempfile
import traceback
import logging
import warnings
import zipfile
import io

# ============================================================================
# CONFIGURE APPLICATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LIFESPAN EVENTS
# ============================================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info("Cypress Code to Business Logic Converter")
    logger.info("=" * 60)
    try:
        initialize_azure_openai()
        logger.info("✓ Application ready")
    except Exception as e:
        logger.warning(f"⚠ Startup initialization deferred: {e}")
    yield
    # Shutdown
    logger.info("Shutting down application...")

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Cypress Code to Business Logic Converter",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
azure_openai_initialized = False
azure_client = None

# Configuration Model
class LLMConfig(BaseModel):
    api_key: str = "YOUR_API_KEY_HERE"
    azure_endpoint: str = "YOUR_AZURE_ENDPOINT_HERE"
    deployment_name: str = "YOUR_DEPLOYMENT_NAME_HERE"
    api_version: str = "YOUR_API_VERSION_HERE"
    temperature: float = 0.2

# Global configuration
current_config = {
    "api_key": "YOUR_API_KEY_HERE",
    "azure_endpoint": "YOUR_AZURE_ENDPOINT_HERE",
    "deployment_name": "YOUR_DEPLOYMENT_NAME_HERE",
    "api_version": "YOUR_API_VERSION_HERE",
    "temperature": 0.2
}

# ============================================================================
# AZURE OPENAI INITIALIZATION
# ============================================================================
def initialize_azure_openai():
    """Initialize Azure OpenAI client"""
    global azure_openai_initialized, azure_client
    
    if azure_openai_initialized and azure_client:
        logger.info("Azure OpenAI already initialized")
        return azure_client
    
    try:
        if not current_config.get("api_key"):
            raise ValueError("Azure OpenAI API key not configured")
        
        if not current_config.get("azure_endpoint"):
            raise ValueError("Azure OpenAI endpoint not configured")
        
        # Initialize Azure OpenAI client
        azure_client = AzureOpenAI(
            api_key=current_config.get("api_key"),
            api_version=current_config.get("api_version", "YOUR_API_VERSION_HERE"),
            azure_endpoint=current_config.get("azure_endpoint")
        )
        
        azure_openai_initialized = True
        logger.info("✓ Azure OpenAI initialized successfully")
        return azure_client
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize Azure OpenAI: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize Azure OpenAI: {str(e)}"
        )

def get_llm():
    """Get Azure OpenAI client"""
    global azure_client
    
    try:
        if not azure_client:
            azure_client = initialize_azure_openai()
        
        logger.info(f"✓ Using Azure OpenAI: {current_config.get('deployment_name')}")
        return azure_client
        
    except Exception as e:
        logger.error(f"✗ Error getting Azure OpenAI client: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get Azure OpenAI client: {str(e)}"
        )

# ============================================================================
# CODE EXTRACTION FUNCTIONS
# ============================================================================
def extract_code_blocks(content: str, file_path: str) -> List[dict]:
    """Extract ONLY 'it' statement code blocks from Cypress test files"""
    blocks = []
    lines = content.split('\n')
    
    # STRICT pattern to identify ONLY 'it' statements with word boundaries
    # This ensures 'it' is a standalone word, not part of another word or inside a string
    it_pattern = r"^\s*it\s*\(\s*['\"](.+?)['\"]"
    
    current_block = []
    current_line_start = 0
    block_description = ""
    indent_level = 0
    brace_count = 0
    in_it_block = False
    
    for i, line in enumerate(lines, 1):
        # Check if line matches 'it' statement pattern at start of line
        match = re.match(it_pattern, line)
        
        if match:
            # Save previous block if exists and has content
            if current_block and len(current_block) > 3:
                blocks.append({
                    'code': '\n'.join(current_block),
                    'line_start': current_line_start,
                    'line_end': i - 1,
                    'description': block_description,
                    'file_path': file_path,
                    'type': 'it'
                })
            
            # Start new block
            current_block = [line]
            current_line_start = i
            block_description = match.group(1)
            indent_level = len(line) - len(line.lstrip())
            brace_count = line.count('{') - line.count('}')
            in_it_block = True
            continue
        
        # Add line to current block if we're inside an 'it' block
        if in_it_block and current_block:
            current_block.append(line)
            
            # Track opening and closing braces
            brace_count += line.count('{') - line.count('}')
            
            # Check if we've closed all braces (end of 'it' block)
            if brace_count == 0 and len(current_block) > 3:
                blocks.append({
                    'code': '\n'.join(current_block),
                    'line_start': current_line_start,
                    'line_end': i,
                    'description': block_description,
                    'file_path': file_path,
                    'type': 'it'
                })
                current_block = []
                block_description = ""
                in_it_block = False
    
    # Add final block if exists
    if current_block and len(current_block) > 3:
        blocks.append({
            'code': '\n'.join(current_block),
            'line_start': current_line_start,
            'line_end': len(lines),
            'description': block_description,
            'file_path': file_path,
            'type': 'it'
        })
    
    logger.info(f"  → Extracted {len(blocks)} 'it' statement blocks")
    return blocks

# ============================================================================
# BUSINESS LOGIC CONVERSION
# ============================================================================
def convert_code_to_business_logic(code_block: str, description: str, client) -> str:
    """Convert code block to business logic using Azure OpenAI"""
    
    system_message = """You are an expert business analyst specialized in converting Cypress automation code 
from The Travel Company into clear, readable business logic descriptions.

Your task is to:
1. Analyze the Cypress test code
2. Extract the business logic and user workflows
3. Convert technical actions into business-friendly language
4. Focus on WHAT the test does from a business perspective, not HOW it does it technically
5. Be concise but comprehensive
6. Structure your response with bullet points for clarity

Format your response as clear business logic without code syntax.
Example format:
• User navigates to the flights search page
• User enters departure city as "New York" and destination as "London"
• User selects 2 adult passengers and 1 child
• User clicks search button
• System displays available flight options
• User verifies that search results are visible and contain at least one flight

Keep descriptions action-oriented and from the user/system perspective."""

    user_message = f"""Test Description: {description}

Convert this Cypress test code into business logic:

{code_block}"""
    
    try:
        response = client.chat.completions.create(
            model=current_config.get("deployment_name", "YOUR_DEPLOYMENT_NAME_HERE"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=current_config.get("temperature", 0.2)
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"  → Converted: {description[:50]}...")
        return result
        
    except Exception as e:
        error_msg = f"Error converting code: {str(e)}"
        logger.error(f"  → {error_msg}")
        return error_msg

def generate_business_logic_summary(business_logic: str, test_description: str, client) -> str:
    """Generate a concise summary of the business logic steps"""
    
    system_message = """You are an expert at creating concise executive summaries of test scenarios.

Your task is to:
1. Create a dense, concise, high-quality summary (1 sentence maximum) and **keep it under 20 words**
2. No need of whole process execution details in summary, just the essence
3. Use language that non-technical stakeholders (product owners, business analysts, managers) can immediately understand


Format: Write a flowing paragraph that summarizes the entire test scenario, not bullet points."""

    user_message = f"""Test: {test_description}

Business Logic Steps:
{business_logic}

Create a concise summary (2-4 sentences) that captures the essence of this test scenario."""
    
    try:
        response = client.chat.completions.create(
            model=current_config.get("deployment_name", "YOUR_DEPLOYMENT_NAME_HERE"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=current_config.get("temperature", 0.2)
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"  → Generated summary for: {test_description[:50]}...")
        return result
        
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        logger.error(f"  → {error_msg}")
        return error_msg

# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================
def extract_js_files_from_zip(zip_file: UploadFile) -> List[tuple]:
    """Extract all .js files from uploaded zip file"""
    js_files = []
    
    try:
        # Read zip file content
        zip_content = zip_file.file.read()
        
        # Open zip file
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
            # Get all file names in the zip
            file_list = zip_ref.namelist()
            
            # Filter for .js, .jsx, .ts, .tsx files
            for file_name in file_list:
                if file_name.endswith(('.js', '.jsx', '.ts', '.tsx')) and not file_name.startswith('__MACOSX'):
                    # Read file content
                    content = zip_ref.read(file_name).decode('utf-8')
                    js_files.append((file_name, content))
                    logger.info(f"  → Extracted: {file_name}")
        
        logger.info(f"✓ Extracted {len(js_files)} JavaScript files from zip")
        return js_files
        
    except Exception as e:
        logger.error(f"✗ Error extracting files from zip: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error extracting files from zip: {str(e)}"
        )

def create_excel_file(results: List[dict], filename: str) -> tuple:
    """Create Excel file with business logic results - returns (filename, filepath)"""
    
    if not results:
        # Create a DataFrame with "No business logic found" message
        df = pd.DataFrame([{
            'test_description': 'N/A',
            'business_logic_summary': 'No business logic found in this code',
            'business_logic': 'No test cases (it blocks) were found in this file',
            'file_path': filename,
            'code_line_number': 'N/A',
            'test_type': 'N/A',
            'code_snippet': 'N/A'
        }])
    else:
        df = pd.DataFrame(results)
    
    # Reorder columns to put summary as 2nd column after test_description
    columns_order = [
        'test_description',
        'business_logic_summary',
        'business_logic',
        'file_path',
        'code_line_number',
        'test_type',
        'code_snippet'
    ]
    df = df[columns_order]
    
    # Generate filename: original_filename_output.xlsx
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Replace any path separators or invalid characters with underscores
    base_name = base_name.replace('/', '_').replace('\\', '_')
    excel_filename = f"{base_name}_output.xlsx"
    excel_path = os.path.join(tempfile.gettempdir(), excel_filename)
    
    # Create Excel with formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Business Logic', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Business Logic']
        column_widths = {
            'test_description': 40,
            'business_logic_summary': 60,
            'business_logic': 80,
            'file_path': 30,
            'code_line_number': 15,
            'test_type': 12,
            'code_snippet': 60
        }
        
        for idx, col in enumerate(df.columns):
            width = column_widths.get(col, 30)
            worksheet.column_dimensions[chr(65 + idx)].width = width
    
    logger.info(f"✓ Excel file created: {excel_filename}")
    return excel_filename, excel_path

def process_single_file(file_name: str, content: str, client) -> List[dict]:
    """Process a single JS file and return business logic results"""
    
    logger.info(f"\nProcessing: {file_name}")
    
    # Extract code blocks
    code_blocks = extract_code_blocks(content, file_name)
    
    if not code_blocks:
        logger.warning(f"  ⚠ No test blocks found in {file_name}")
        return []
    
    results = []
    
    # Convert each block to business logic
    for idx, block in enumerate(code_blocks, 1):
        logger.info(f"  [{idx}/{len(code_blocks)}] Converting: {block['description'][:60]}...")
        
        business_logic = convert_code_to_business_logic(
            block['code'], 
            block['description'],
            client
        )
        
        results.append({
            'test_description': block['description'],
            'business_logic': business_logic,
            'file_path': block['file_path'],
            'code_line_number': f"{block['line_start']}-{block['line_end']}",
            'test_type': block['type'],
            'code_snippet': block['code'][:500] + '...' if len(block['code']) > 500 else block['code']
        })
    
    # Generate summaries for each business logic
    logger.info(f"\nGenerating summaries for {file_name}...")
    for idx, result in enumerate(results, 1):
        logger.info(f"  [{idx}/{len(results)}] Generating summary...")
        
        summary = generate_business_logic_summary(
            result['business_logic'],
            result['test_description'],
            client
        )
        
        result['business_logic_summary'] = summary
    
    logger.info(f"✓ Processed {len(results)} test blocks from {file_name}")
    return results

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/api/configure")
async def configure_llm(config: LLMConfig):
    """Configure the Azure OpenAI settings"""
    global current_config, azure_client, azure_openai_initialized
    
    current_config = config.dict()
    
    # Reset client to force reinitialization with new config
    azure_client = None
    azure_openai_initialized = False
    
    try:
        # Test the configuration
        client = get_llm()
        return {
            "status": "success", 
            "message": "Azure OpenAI configured successfully", 
            "config": {
                "deployment_name": current_config.get("deployment_name"),
                "api_version": current_config.get("api_version"),
                "temperature": current_config.get("temperature"),
                "max_tokens": current_config.get("max_tokens")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current Azure OpenAI configuration (without API key)"""
    return {
        "deployment_name": current_config.get("deployment_name"),
        "api_version": current_config.get("api_version"),
        "temperature": current_config.get("temperature"),
        "max_tokens": current_config.get("max_tokens"),
        "azure_endpoint": current_config.get("azure_endpoint")
    }

@app.post("/api/convert")
async def convert_cypress_code(files: List[UploadFile] = File(...)):
    """Convert uploaded Cypress files (individual or zip) to business logic and generate Excel files in a ZIP"""
    
    try:
        logger.info("=" * 60)
        logger.info(f"Starting conversion for {len(files)} file(s)")
        logger.info("=" * 60)
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Initialize Azure OpenAI
        logger.info("Initializing Azure OpenAI...")
        client = get_llm()
        
        generated_excel_files = []
        all_js_files = []
        
        # Process each uploaded file
        for file in files:
            # Check if it's a zip file
            if file.filename.endswith('.zip'):
                logger.info(f"\n📦 Processing ZIP file: {file.filename}")
                js_files = extract_js_files_from_zip(file)
                all_js_files.extend(js_files)
            elif file.filename.endswith(('.js', '.jsx', '.ts', '.tsx')):
                logger.info(f"\n📄 Processing JS file: {file.filename}")
                content = await file.read()
                content = content.decode('utf-8')
                all_js_files.append((file.filename, content))
            else:
                logger.warning(f"⚠ Skipping unsupported file: {file.filename}")
                continue
        
        if not all_js_files:
            raise HTTPException(
                status_code=400,
                detail="No valid JavaScript files found in uploaded files"
            )
        
        logger.info(f"\n✓ Total JavaScript files to process: {len(all_js_files)}")
        logger.info("=" * 60)
        
        # Process each JS file and create separate Excel files
        for file_num, (file_name, content) in enumerate(all_js_files, 1):
            logger.info(f"\n[{file_num}/{len(all_js_files)}] " + "=" * 50)
            
            try:
                # Process the file
                results = process_single_file(file_name, content, client)
                
                # Create Excel file (even if no results)
                excel_filename, excel_path = create_excel_file(results, file_name)
                generated_excel_files.append({
                    'source_file': file_name,
                    'excel_file': excel_filename,
                    'excel_path': excel_path,
                    'test_blocks_found': len(results)
                })
                
            except Exception as e:
                logger.error(f"✗ Error processing {file_name}: {str(e)}")
                # Create an Excel with error message
                error_results = [{
                    'test_description': 'Error',
                    'business_logic_summary': f'Error processing file: {str(e)}',
                    'business_logic': 'An error occurred while processing this file',
                    'file_path': file_name,
                    'code_line_number': 'N/A',
                    'test_type': 'Error',
                    'code_snippet': 'N/A'
                }]
                excel_filename, excel_path = create_excel_file(error_results, file_name)
                generated_excel_files.append({
                    'source_file': file_name,
                    'excel_file': excel_filename,
                    'excel_path': excel_path,
                    'test_blocks_found': 0,
                    'error': str(e)
                })
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Successfully generated {len(generated_excel_files)} Excel file(s)")
        logger.info("=" * 60)
        
        # Create ZIP file with all Excel files maintaining directory structure
        logger.info("\n📦 Creating output ZIP file...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"business_logic_output_{timestamp}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_info in generated_excel_files:
                source_file = file_info['source_file']
                excel_path = file_info['excel_path']
                
                # Maintain directory structure
                # Get the directory part of the source file
                dir_name = os.path.dirname(source_file)
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                
                # Create the path in the ZIP maintaining directory structure
                if dir_name:
                    zip_excel_path = os.path.join(dir_name, f"{base_name}_output.xlsx")
                else:
                    zip_excel_path = f"{base_name}_output.xlsx"
                
                # Add file to ZIP
                zipf.write(excel_path, zip_excel_path)
                logger.info(f"  → Added: {zip_excel_path}")
        
        logger.info(f"✓ ZIP file created: {zip_filename}")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "message": f"Processed {len(all_js_files)} JavaScript file(s)",
            "generated_files": generated_excel_files,
            "total_excel_files": len(generated_excel_files),
            "zip_download_url": f"/api/download/{zip_filename}",
            "zip_filename": zip_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error processing files: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing files: {str(e)}"
        )

@app.get("/api/download/{filename}")
async def download_excel(filename: str):
    """Download generated Excel or ZIP file"""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(file_path):
        logger.error(f"✗ File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"✓ Downloading: {filename}")
    
    # Determine media type based on file extension
    if filename.endswith('.zip'):
        media_type = "application/zip"
    else:
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "azure_openai_initialized": azure_openai_initialized,
        "model": current_config.get("deployment_name"),
        "api_version": current_config.get("api_version")
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )