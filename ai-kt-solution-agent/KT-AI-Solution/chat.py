"""
Chat with Codebase - Q&A Interface
"""
import vertexai
from vertexai.generative_models import GenerativeModel
from typing import List, Dict
from embeddings import generate_embedding, configure_vertex_ai
from vector_store import get_qdrant_client, search_similar_files
from config import *
from prompts import CHAT_QUERY_CONFIG


def chat_with_codebase(user_question: str, collection_name: str) -> Dict[str, any]:
    """
    Answer user questions about the codebase using RAG
    
    Args:
        user_question: User's question
        collection_name: Collection to search in
        
    Returns:
        Dictionary with answer and relevant files
    """
    configure_vertex_ai()
    
    try:
        import sys
        print(f"\n💬 Processing question: {user_question[:100]}...")
        sys.stdout.flush()
        
        # Generate query embedding
        print(f"🔍 Creating query embedding...")
        sys.stdout.flush()
        query_embedding = generate_embedding(user_question)
        
        # Search for relevant files
        print(f"🔎 Searching for relevant files...")
        sys.stdout.flush()
        client = get_qdrant_client()
        relevant_files = search_similar_files(
            client=client,
            collection_name=collection_name,
            query_embedding=query_embedding,
            min_score=CHAT_QUERY_CONFIG["min_score"],
            max_results=CHAT_QUERY_CONFIG["max_results"]
        )
        
        if not relevant_files:
            return {
                "answer": "I couldn't find relevant files to answer your question. Try:\n\n- Being more specific (mention class/file names if you know them)\n- Asking about specific components like VectorSearch, FileTracker, IncrementalIndexer\n- Rephrasing your question with different keywords\n\nExample: Instead of 'What external services?', try 'What is Qdrant used for?'",
                "relevant_files": [],
                "file_count": 0
            }
        
        print(f"✅ Found {len(relevant_files)} relevant files")
        
        # Build context for chat
        context = build_chat_context(user_question, relevant_files)
        
        # Generate answer
        print(f"🤖 Generating answer...")
        model = GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            contents=context,
            generation_config={
                "temperature": 0.2,  # Slightly higher for conversational tone
                "max_output_tokens": 4000,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        answer = response.text
        
        # Extract file references for UI
        file_refs = []
        for file_data in relevant_files[:5]:  # Top 5 files
            file_refs.append({
                "file_path": file_data["relative_path"],
                "score": file_data["score"],
                "summary": file_data["summary"][:200] + "..."
            })
        
        print(f"✅ Answer generated ({len(answer.split())} words)")
        
        return {
            "answer": answer,
            "relevant_files": file_refs,
            "file_count": len(relevant_files)
        }
        
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "relevant_files": [],
            "file_count": 0
        }


def build_chat_context(user_question: str, relevant_files: List[Dict]) -> str:
    """Build context for chat response"""
    
    context = f"""You are a helpful AI assistant that answers questions about a codebase.

User Question: {user_question}

I will provide you with relevant files from the codebase. Analyze them and provide a clear, detailed answer to the user's question.

Guidelines:
1. Answer the question directly and clearly
2. Reference specific files, classes, and functions when relevant
3. Include code snippets ONLY when they help explain your answer
4. Use markdown formatting with code blocks (```language```)
5. Be conversational but technical
6. If the question can't be fully answered from the provided files, say so

---

# Relevant Files from Codebase:

"""
    
    for i, file_data in enumerate(relevant_files, 1):
        context += f"""
## File {i}: {file_data['relative_path']}
**Similarity Score:** {file_data['score']:.3f}

**Summary:**
{file_data['summary']}

**Code:**
```python
{file_data['content'][:3000]}  # First 3000 chars
```

---

"""
    
    context += """
Now, answer the user's question based on the above codebase files. Be specific and helpful.
"""
    
    return context


def get_file_explanation(file_path: str, collection_name: str) -> Dict[str, any]:
    """Get detailed explanation of a specific file"""
    configure_vertex_ai()
    
    try:
        # Search for the specific file
        client = get_qdrant_client()
        
        # Use file path as query
        query_embedding = generate_embedding(f"explain the file {file_path}")
        
        results = search_similar_files(
            client=client,
            collection_name=collection_name,
            query_embedding=query_embedding,
            min_score=0.3,  # Lower threshold for exact file match
            max_results=5
        )
        
        # Find exact match
        target_file = None
        for result in results:
            if file_path in result['file_path']:
                target_file = result
                break
        
        if not target_file:
            return {
                "explanation": f"File '{file_path}' not found in the collection.",
                "file_found": False
            }
        
        # Generate explanation
        prompt = f"""Provide a detailed explanation of this file:

File: {target_file['relative_path']}

Summary: {target_file['summary']}

Code:
```python
{target_file['content']}
```

Explain:
1. What this file does
2. Key classes and functions
3. How it fits in the system
4. Important dependencies
"""
        
        model = GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4000
            }
        )
        
        return {
            "explanation": response.text,
            "file_found": True,
            "file_data": target_file
        }
        
    except Exception as e:
        return {
            "explanation": f"Error getting file explanation: {str(e)}",
            "file_found": False
        }

