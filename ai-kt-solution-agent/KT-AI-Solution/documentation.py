"""
Documentation generation logic
"""
import os
import zipfile
from pathlib import Path
from typing import Dict, List
import vertexai
from vertexai.generative_models import GenerativeModel
from config import *
from prompts import QUERY_CONFIGS, DOC_GENERATION_PROMPT_TEMPLATE
from embeddings import generate_embedding, configure_vertex_ai, retry_with_backoff
from vector_store import get_qdrant_client, search_similar_files


def build_file_context(file_data: Dict, index: int) -> str:
    """Build context string for a single file"""
    metadata = file_data['metadata']
    file_path = file_data['relative_path']
    
    # Detect file type
    is_python = file_path.endswith('.py')
    is_javascript = file_path.endswith(('.js', '.jsx', '.ts', '.tsx'))
    
    # Build key components section based on file type
    if is_python:
        components_section = f"""**Key Components:**
- Classes: {', '.join([c['name'] for c in metadata['classes']]) or 'None'}
- Functions: {', '.join([f['name'] for f in metadata['functions'][:10]]) or 'None'}
- External Dependencies: {', '.join(metadata['imports']['external'][:8]) or 'None'}"""
    
    elif is_javascript:
        # JavaScript/React file
        components_info = []
        if metadata.get('components'):
            components_info.append(f"React Components: {', '.join([c['name'] for c in metadata['components']])}")
        if metadata.get('functions'):
            components_info.append(f"Functions: {', '.join([f['name'] for f in metadata['functions'][:10]])}")
        if metadata.get('classes'):
            components_info.append(f"Classes: {', '.join([c['name'] for c in metadata['classes']])}")
        if metadata.get('hooks_used'):
            components_info.append(f"Hooks: {', '.join(metadata['hooks_used'][:8])}")
        if metadata.get('exports', {}).get('default'):
            components_info.append(f"Default Export: {metadata['exports']['default']}")
        
        components_section = f"""**Key Components:**
- {chr(10).join(components_info) if components_info else 'None'}
- External Dependencies: {', '.join(metadata['imports']['external'][:8]) or 'None'}"""
    
    else:
        # Other file types
        components_section = f"""**Key Components:**
- External Dependencies: {', '.join(metadata['imports']['external'][:8]) or 'None'}"""
    
    context = f"""
## File {index}: {file_data['relative_path']}

**Summary:**
{file_data['summary']}

**File Info:**
- Size: {file_data['file_size']} bytes
- Lines: {file_data['lines_of_code']}
- Category: {metadata['file_category']}

{components_section}

**Complete Source Code:**
```
{file_data['content']}
```

{'='*80}
"""
    return context


def generate_documentation(doc_type: str, collection_name: str) -> str:
    """
    Generate documentation for a specific type
    
    Steps:
    1. Get query configuration for doc type
    2. Generate query embedding
    3. Search for similar files (top 20-30)
    4. Build context with full file content
    5. Send to Gemini for documentation generation
    6. Return markdown documentation
    """
    configure_vertex_ai()
    
    print(f"\n{'='*80}")
    print(f"📄 Generating {DOC_TYPE_LABELS[doc_type]} Documentation")
    print(f"{'='*80}\n")
    
    # Get query configuration
    query_config = QUERY_CONFIGS[doc_type]
    query_text = query_config["query"]
    min_score = query_config.get("min_score", 0.60)
    max_results = query_config.get("max_results", 25)
    
    # Generate query embedding
    print(f"🔍 Creating query embedding...")
    query_embedding = generate_embedding(query_text)
    
    # Search for similar files with smart selection
    print(f"🔎 Searching for relevant files (min_score={min_score})...")
    client = get_qdrant_client()
    similar_files = search_similar_files(
        client=client,
        collection_name=collection_name,
        query_embedding=query_embedding,
        min_score=min_score,
        max_results=max_results
    )
    
    if not similar_files:
        return f"# {DOC_TYPE_LABELS[doc_type]}\n\nNo relevant files found for this documentation type."
    
    print(f"✅ Found {len(similar_files)} relevant files (scores: {similar_files[0]['score']:.3f} to {similar_files[-1]['score']:.3f})")
    
    # Limit to top 20 files to avoid token limits
    selected_files = similar_files[:20]
    print(f"📚 Selecting top {len(selected_files)} files for detailed analysis")
    
    # Build context with full file content
    import sys
    print(f"📝 Building context...")
    sys.stdout.flush()
    
    file_contexts = ""
    for i, file_data in enumerate(selected_files, 1):
        print(f"   Adding file {i}/{len(selected_files)}: {file_data['relative_path']}")
        sys.stdout.flush()  # Force flush for each file
        file_contexts += build_file_context(file_data, i)
    
    # Build prompt
    prompt = DOC_GENERATION_PROMPT_TEMPLATE.format(
        doc_type_label=DOC_TYPE_LABELS[doc_type],
        file_count=len(selected_files),
        file_contexts=file_contexts
    )
    
    # Calculate approximate token count
    approx_tokens = len(prompt.split()) * 1.3
    print(f"📊 Context size: ~{int(approx_tokens):,} tokens")
    
    if approx_tokens > 100000:
        print(f"⚠️  Warning: Context is large, may hit token limits")
    
    # Generate documentation with retry logic
    print(f"🤖 Generating documentation with Gemini...")
    
    def _generate_documentation():
        model = GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": DOC_GENERATION_CONFIG["temperature"],
                "max_output_tokens": DOC_GENERATION_CONFIG["max_output_tokens"],
                "top_p": DOC_GENERATION_CONFIG["top_p"],
                "top_k": DOC_GENERATION_CONFIG["top_k"]
            }
        )
        return response.text
    
    try:
        documentation = retry_with_backoff(_generate_documentation)
        print(f"✅ Documentation generated successfully ({len(documentation.split())} words)\n")
        return documentation
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error generating documentation: {error_msg}")
        return f"# {DOC_TYPE_LABELS[doc_type]}\n\nError generating documentation: {error_msg}"


def generate_all_documentation(selected_doc_types: List[str] = None, collection_name: str = None, progress_callback=None) -> Dict[str, str]:
    """Generate all selected documentation types"""
    import sys
    
    if selected_doc_types is None:
        selected_doc_types = DOC_TYPES
    
    print(f"\n{'='*80}")
    print(f"📚 Generating Documentation")
    print(f"{'='*80}")
    print(f"Selected types: {', '.join([DOC_TYPE_LABELS[dt] for dt in selected_doc_types])}\n")
    sys.stdout.flush()  # Force flush to keep connection alive
    
    docs = {}
    total_types = len(selected_doc_types)
    
    for i, doc_type in enumerate(selected_doc_types, 1):
        print(f"\n[{i}/{total_types}] Generating {DOC_TYPE_LABELS[doc_type]}...")
        sys.stdout.flush()  # Force flush
        
        # Update progress at the start of each doc type
        if progress_callback:
            progress_callback(i, total_types, doc_type)
        
        documentation = generate_documentation(doc_type, collection_name)
        docs[doc_type] = documentation
        
        # Force flush after each document
        sys.stdout.flush()
    
    print(f"\n{'='*80}")
    print(f"✅ All documentation generated successfully")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    return docs


def create_markdown_zip(docs: Dict[str, str], output_path: str = None) -> str:
    """Create ZIP file with markdown documentation"""
    print(f"\n📦 Creating documentation ZIP file...")
    
    # Generate unique filename with timestamp if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"kt_documentation_{timestamp}.zip"
    
    # Create temporary directory structure
    temp_dir = Path("temp_docs")
    temp_dir.mkdir(exist_ok=True)
    
    # Create index.md
    index_content = "# Documentation\n\n"
    index_content += "## Table of Contents\n\n"
    for doc_type, _ in docs.items():
        index_content += f"- [{DOC_TYPE_LABELS[doc_type]}]({doc_type}.md)\n"
    
    with open(temp_dir / "index.md", "w", encoding="utf-8") as f:
        f.write(index_content)
    
    # Create individual documentation files
    for doc_type, content in docs.items():
        filename = f"{doc_type}.md"
        filepath = temp_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"   ✅ Created {filename}")
    
    # Create ZIP file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for md_file in temp_dir.glob("*.md"):
            zipf.write(md_file, md_file.name)
    
    # Cleanup temp directory
    for file in temp_dir.glob("*.md"):
        file.unlink()
    temp_dir.rmdir()
    
    print(f"✅ Documentation ZIP created: {output_path}")
    return output_path


def create_index_page() -> str:
    """Create a simple index/overview page"""
    index_content = """# KT Documentation

This documentation was automatically generated using AI to help with knowledge transfer.

## Documentation Sections

### 1. System Overview & Architecture
High-level architecture and system structure

### 2. API Reference
All API endpoints and their specifications

### 3. Data Models & Database Schema
Database models and data structures

### 4. Business Logic & Workflows
Core business processes and workflows

### 5. External Integrations
Third-party services and integrations

### 6. Deployment & Operations
Deployment setup and infrastructure

---

*Generated with KT Documentation Generator*
"""
    return index_content

