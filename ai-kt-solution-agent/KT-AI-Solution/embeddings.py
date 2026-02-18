"""
File scanning, summary generation, and embedding creation
"""
import os
import ast
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from config import *
from prompts import SUMMARY_PROMPT_TEMPLATE

# Initialize Vertex AI once
_vertex_initialized = False

def configure_vertex_ai():
    """Configure Vertex AI with service account credentials"""
    global _vertex_initialized
    
    if _vertex_initialized:
        return
    
    try:
        # Get project ID from environment or credentials file
        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        # Load service account credentials
        credentials_path = GCP_CREDENTIALS_PATH
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
        
        # If project_id not in environment, read from credentials file
        if not project_id:
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
                project_id = creds_data.get('project_id')
        
        if not project_id:
            raise ValueError("project_id not found in environment or credentials file")
        
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize Vertex AI
        vertexai.init(
            project=project_id,
            location=location,  # Use variable instead of hardcoded 'us-central1'
            credentials=credentials
        )
        
        _vertex_initialized = True
        print(f"✅ Vertex AI initialized with project: {project_id}")
        
    except Exception as e:
        print(f"❌ Error configuring Vertex AI: {e}")
        raise


def retry_with_backoff(func, *args, **kwargs):
    """
    Retry function with exponential backoff for rate limiting
    Delays: 25s → 60s → 120s → 240s (4 retries total)
    Handles 429 Resource exhausted errors with progressive wait times
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate" in error_msg or "429" in error_msg or "resource exhausted" in error_msg:
                if attempt < RATE_LIMIT_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"⏳ Rate limit hit. Waiting {delay}s before retry {attempt + 2}/{RATE_LIMIT_RETRIES}...")
                    time.sleep(delay)
                else:
                    print(f"❌ Rate limit exceeded after {RATE_LIMIT_RETRIES} retries")
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise
    return None


def should_exclude(file_path: str) -> bool:
    """Check if file should be excluded based on patterns"""
    path_str = str(file_path)
    path_lower = path_str.lower()
    path_parts = Path(file_path).parts
    path_parts_lower = [part.lower() for part in path_parts]
    
    # Get just the filename
    file_name = Path(file_path).name
    file_name_lower = file_name.lower()
    
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            # File extension pattern (e.g., *.pyc, *.min.js)
            if file_name_lower.endswith(pattern[1:].lower()):
                return True
        else:
            pattern_lower = pattern.lower()
            
            # Check if pattern is an exact filename match
            if pattern_lower == file_name_lower:
                return True
            
            # Check if pattern is a complete directory name in the path
            # This prevents "out" from matching "route" or "layout"
            if pattern_lower in path_parts_lower:
                return True
            
            # For patterns starting with ".", do exact filename match only
            # This handles .env, .git, .gitignore, etc.
            if pattern.startswith('.'):
                # Already checked exact filename match above
                continue
            
            # For patterns with "/" or "\", do substring match (full path patterns)
            if '/' in pattern or '\\' in pattern:
                if pattern_lower in path_lower:
                    return True
    
    # Additional checks for specific file types
    # Exclude test files
    if any(test_pattern in file_name_lower for test_pattern in ['.test.', '.spec.', '_test.', '-test.']):
        return True
    
    # Exclude minified files
    if file_name_lower.endswith('.min.js') or file_name_lower.endswith('.min.css'):
        return True
    
    # Exclude chunk/bundle files
    if '.chunk.' in file_name_lower or '.bundle.' in file_name_lower:
        return True
    
    return False


def scan_codebase(codebase_path: str) -> List[str]:
    """Scan codebase and return list of files to process"""
    files_to_process = []
    codebase_path = Path(codebase_path)
    
    if not codebase_path.exists():
        raise ValueError(f"Codebase path does not exist: {codebase_path}")
    
    # Scan for Python files
    for py_file in codebase_path.rglob("*.py"):
        if not should_exclude(py_file):
            files_to_process.append(str(py_file))
    
    # Scan for JavaScript/TypeScript files
    for ext in JAVASCRIPT_EXTENSIONS:
        for js_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(js_file):
                files_to_process.append(str(js_file))
    
    # Scan for style files
    for ext in STYLE_EXTENSIONS:
        for style_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(style_file):
                files_to_process.append(str(style_file))
    
    # Scan for HTML files
    for ext in HTML_EXTENSIONS:
        for html_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(html_file):
                files_to_process.append(str(html_file))
    
    # Scan for config files
    for ext in CONFIG_EXTENSIONS:
        for config_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(config_file):
                files_to_process.append(str(config_file))
    
    # Scan for infrastructure files
    for infra_file_name in INFRA_FILES:
        for infra_file in codebase_path.rglob(infra_file_name):
            if not should_exclude(infra_file):
                files_to_process.append(str(infra_file))
    
    # Scan for documentation files
    for ext in DOC_EXTENSIONS:
        for doc_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(doc_file):
                files_to_process.append(str(doc_file))
    
    # Scan for SQL files
    for ext in SQL_EXTENSIONS:
        for sql_file in codebase_path.rglob(f"*{ext}"):
            if not should_exclude(sql_file):
                files_to_process.append(str(sql_file))
    
    # Scan for dependency files
    for dep_file_name in DEPENDENCY_FILES:
        for dep_file in codebase_path.rglob(dep_file_name):
            if not should_exclude(dep_file):
                files_to_process.append(str(dep_file))
    
    # Scan for JavaScript config files
    for js_config_name in JS_CONFIG_FILES:
        for js_config_file in codebase_path.rglob(js_config_name):
            if not should_exclude(js_config_file):
                files_to_process.append(str(js_config_file))
    
    # Scan for env example files
    for env_file_name in ENV_FILES:
        for env_file in codebase_path.rglob(env_file_name):
            if not should_exclude(env_file):
                files_to_process.append(str(env_file))
    
    return sorted(list(set(files_to_process)))  # Remove duplicates and sort


def extract_code_snippet(content: str, file_path: str, max_lines: int = 50) -> str:
    """Extract key code snippet from file (first significant class/function)"""
    if file_path.endswith('.py'):
        # Python file
        try:
            tree = ast.parse(content)
            
            # Find first class or significant function
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Get class and first few methods
                    start_line = node.lineno - 1
                    end_line = min(start_line + max_lines, len(content.split('\n')))
                    lines = content.split('\n')[start_line:end_line]
                    return '\n'.join(lines)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Get public function
                    start_line = node.lineno - 1
                    end_line = min(start_line + max_lines, len(content.split('\n')))
                    lines = content.split('\n')[start_line:end_line]
                    return '\n'.join(lines)
            
            # If no class/function found, return first 30 lines
            lines = content.split('\n')[:30]
            return '\n'.join(lines)
        except:
            # If parsing fails, return first 30 lines
            lines = content.split('\n')[:30]
            return '\n'.join(lines)
    
    elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
        # JavaScript/TypeScript file
        try:
            from js_parser import extract_first_significant_code
            return extract_first_significant_code(content, max_lines)
        except:
            # If parsing fails, return first 30 lines
            lines = content.split('\n')[:30]
            return '\n'.join(lines)
    
    else:
        # For other files, return first 30 lines
        lines = content.split('\n')[:30]
        return '\n'.join(lines)


def extract_metadata(file_path: str, content: str) -> Dict:
    """Extract metadata from Python or JavaScript/TypeScript file"""
    
    # Route to appropriate parser based on file type
    if file_path.endswith('.py'):
        return extract_python_metadata(file_path, content)
    elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
        from js_parser import extract_js_metadata
        return extract_js_metadata(file_path, content)
    else:
        # For other files (config, markdown, etc.), return basic metadata
        return {
            "classes": [],
            "functions": [],
            "imports": {"external": [], "internal": []},
            "file_category": "config",
            "lines_of_code": len(content.split('\n')),
            "tags": [],
            "key_concepts": []
        }


def extract_python_metadata(file_path: str, content: str) -> Dict:
    """Extract metadata from Python file"""
    metadata = {
        "classes": [],
        "functions": [],
        "imports": {"external": [], "internal": []},
        "file_category": "unknown",
        "lines_of_code": len(content.split('\n')),
        "tags": [],
        "key_concepts": []
    }
    
    try:
        tree = ast.parse(content)
        
        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                metadata["classes"].append({
                    "name": node.name,
                    "line_start": node.lineno,
                    "methods": methods[:10]  # First 10 methods
                })
        
        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                if not any(node.lineno >= c.get("line_start", 0) for c in metadata["classes"]):
                    # Only top-level functions
                    metadata["functions"].append({
                        "name": node.name,
                        "line_start": node.lineno,
                        "params": [arg.arg for arg in node.args.args]
                    })
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(('src.', 'app.', 'api.')):
                        metadata["imports"]["internal"].append(alias.name)
                    else:
                        metadata["imports"]["external"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith(('src.', 'app.', 'api.')):
                        metadata["imports"]["internal"].append(node.module)
                    else:
                        metadata["imports"]["external"].append(node.module)
        
        # Determine file category
        content_lower = content.lower()
        if any(pattern in content for pattern in ['@app.route', '@router.', 'APIRouter', '@api_view']):
            metadata["file_category"] = "api"
            metadata["tags"].append("api_endpoint")
        elif any(pattern in content for pattern in ['class.*Model', 'Base =', 'db.Column']):
            metadata["file_category"] = "model"
            metadata["tags"].append("database_model")
        elif any(pattern in content_lower for pattern in ['service', 'handler', 'manager']):
            metadata["file_category"] = "service"
            metadata["tags"].append("business_logic")
        elif 'config' in file_path.lower():
            metadata["file_category"] = "config"
            metadata["tags"].append("configuration")
        
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return metadata


def generate_file_summary(file_path: str, content: str, lines_of_code: int) -> str:
    """Generate detailed summary (400-600 words) using Gemini with retry logic"""
    configure_vertex_ai()
    
    # Detect language from file extension
    if file_path.endswith('.py'):
        language = "Python"
    elif file_path.endswith(('.js', '.jsx')):
        language = "JavaScript"
    elif file_path.endswith(('.ts', '.tsx')):
        language = "TypeScript"
    else:
        language = "Configuration/Other"
    
    # Send FULL content to LLM - no truncation
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        file_path=file_path,
        lines_of_code=lines_of_code,
        content=content,  # Full content
        language=language
    )
    
    def _generate():
        model = GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 8000,  # Increased for 400-600 words
                "top_p": SUMMARY_GENERATION_CONFIG["top_p"],
                "top_k": SUMMARY_GENERATION_CONFIG["top_k"]
            }
        )
        return response.text
    
    try:
        summary = retry_with_backoff(_generate)
        return summary
    except Exception as e:
        print(f"❌ Failed to generate summary for {file_path}: {e}")
        return f"This file is located at {file_path}. It contains {len(content.split())} words and {lines_of_code} lines of code."


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using text-embedding-004 with retry logic"""
    configure_vertex_ai()
    
    def _embed():
        # Use Vertex AI text embedding model
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    
    try:
        embedding = retry_with_backoff(_embed)
        return embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * VECTOR_SIZE


def create_embedding_text(file_data: Dict) -> str:
    """Build rich text for embedding from summary, snippet, and metadata"""
    embedding_text = f"""File: {file_data['file_path']}
Category: {file_data['metadata']['file_category']}

Summary:
{file_data['summary']}

Key Components:
Classes: {', '.join([c['name'] for c in file_data['metadata']['classes']])}
Functions: {', '.join([f['name'] for f in file_data['metadata']['functions'][:10]])}

Tags: {', '.join(file_data['metadata']['tags'])}

Dependencies:
External: {', '.join(file_data['metadata']['imports']['external'][:8])}
Internal: {', '.join(file_data['metadata']['imports']['internal'][:8])}

Representative Code:
{file_data['code_snippet'][:500]}
"""
    return embedding_text


def process_file(file_path: str, codebase_root: str, keep_alive_callback=None) -> Optional[Dict]:
    """Process single file: read, summarize, extract metadata, create embedding"""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if len(content.strip()) == 0:
            print(f"⚠️  Skipping empty file: {file_path}")
            return None
        
        # Get relative path
        rel_path = str(Path(file_path).relative_to(codebase_root))
        
        print(f"📄 Processing: {rel_path}")
        
        # Keep connection alive
        if keep_alive_callback:
            keep_alive_callback(rel_path, "reading")
        
        # Extract metadata
        metadata = extract_metadata(file_path, content)
        
        # Extract code snippet
        code_snippet = extract_code_snippet(content, file_path)
        
        # Get lines of code
        lines_of_code = len(content.split('\n'))
        
        # Generate summary (this takes ~30 seconds - keep connection alive)
        print(f"   Generating summary...")
        if keep_alive_callback:
            keep_alive_callback(rel_path, "summarizing")
        summary = generate_file_summary(rel_path, content, lines_of_code)
        
        # Build file data
        file_data = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "relative_path": rel_path,
            "file_size": len(content),
            "lines_of_code": len(content.split('\n')),
            "summary": summary,
            "code_snippet": code_snippet,
            "content": content,
            "metadata": metadata
        }
        
        # Create embedding text
        embedding_text = create_embedding_text(file_data)
        
        # Generate embedding (keep connection alive)
        print(f"   Generating embedding...")
        if keep_alive_callback:
            keep_alive_callback(rel_path, "embedding")
        embedding = generate_embedding(embedding_text)
        
        file_data["embedding"] = embedding
        
        print(f"✅ Completed: {rel_path}\n")
        
        return file_data
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}\n")
        return None


def index_codebase(codebase_path: str, progress_callback=None) -> List[Dict]:
    """Main function to index entire codebase"""
    import sys
    
    print(f"🔍 Scanning codebase: {codebase_path}\n")
    sys.stdout.flush()
    
    # Scan files
    files = scan_codebase(codebase_path)
    print(f"📊 Found {len(files)} files to process\n")
    sys.stdout.flush()
    
    if len(files) == 0:
        raise ValueError("No files found in the codebase")
    
    # Process each file
    processed_files = []
    for i, file_path in enumerate(files, 1):
        # Get relative path for display
        rel_path = str(Path(file_path).relative_to(codebase_path))
        print(f"[{i}/{len(files)}] ", end="")
        sys.stdout.flush()  # Flush after each file to keep connection alive
        
        # Create keep-alive callback to prevent websocket timeout
        def keep_alive(filename, stage):
            if progress_callback:
                # Send intermediate updates to keep connection alive
                progress_callback(i, len(files), f"{filename} ({stage})")
        
        file_data = process_file(file_path, codebase_path, keep_alive)
        
        if file_data:
            processed_files.append(file_data)
        
        # Update progress with filename
        if progress_callback:
            progress_callback(i, len(files), rel_path)
    
    print(f"\n✅ Successfully processed {len(processed_files)} files")
    sys.stdout.flush()
    return processed_files

