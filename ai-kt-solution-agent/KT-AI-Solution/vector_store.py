"""
Qdrant vector database operations
"""
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import *
from pathlib import Path
import hashlib
import re


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return client


def get_collection_name_from_path(codebase_path: str) -> str:
    """Generate collection name from repository path"""
    # Get last directory name (repo name)
    repo_name = Path(codebase_path).name
    
    # Clean the name: lowercase, replace spaces/special chars with underscores
    clean_name = re.sub(r'[^a-z0-9_]', '_', repo_name.lower())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    
    # Add prefix
    collection_name = f"{COLLECTION_NAME_PREFIX}{clean_name}"
    
    return collection_name


def list_all_collections(client: QdrantClient) -> List[Dict]:
    """List all collections in Qdrant"""
    try:
        collections = client.get_collections().collections
        
        result = []
        for collection in collections:
            # Get collection info
            try:
                info = client.get_collection(collection_name=collection.name)
                result.append({
                    "name": collection.name,
                    "vectors_count": info.points_count,
                    "status": info.status.name if hasattr(info.status, 'name') else str(info.status)
                })
            except:
                result.append({
                    "name": collection.name,
                    "vectors_count": 0,
                    "status": "unknown"
                })
        
        return result
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []


def create_collection(client: QdrantClient, collection_name: str):
    """Create collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if collection_exists:
            print(f"✅ Collection already exists: {collection_name}")
            return True
        
        # Create collection
        print(f"🆕 Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        raise


def delete_collection(client: QdrantClient, collection_name: str) -> bool:
    """Delete a collection"""
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"🗑️  Successfully deleted collection: {collection_name}")
        return True
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        return False


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if collection exists"""
    try:
        collections = client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return False


def generate_point_id(file_path: str) -> int:
    """Generate unique ID for file using hash converted to integer"""
    # Generate hash and convert to integer for Qdrant compatibility
    hash_hex = hashlib.sha256(file_path.encode()).hexdigest()
    # Take first 16 hex chars and convert to int (64-bit integer)
    return int(hash_hex[:16], 16)


def store_embeddings(client: QdrantClient, collection_name: str, file_data_list: List[Dict]):
    """Store file embeddings in Qdrant"""
    if not file_data_list:
        print("⚠️  No files to store")
        return
    
    print(f"\n💾 Storing {len(file_data_list)} files in collection '{collection_name}'...")
    
    points = []
    for file_data in file_data_list:
        # Create point ID from file path
        point_id = generate_point_id(file_data["file_path"])
        
        # Prepare payload (exclude embedding from payload)
        payload = {
            "file_path": file_data["file_path"],
            "file_name": file_data["file_name"],
            "relative_path": file_data["relative_path"],
            "file_size": file_data["file_size"],
            "lines_of_code": file_data["lines_of_code"],
            "summary": file_data["summary"],
            "code_snippet": file_data["code_snippet"],
            "content": file_data["content"],
            "metadata": file_data["metadata"]
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=file_data["embedding"],
            payload=payload
        )
        points.append(point)
    
    # Batch upsert
    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        print(f"✅ Successfully stored {len(points)} files in collection '{collection_name}'")
        
        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"📊 Total points in collection: {collection_info.points_count}")
        
    except Exception as e:
        print(f"❌ Error storing embeddings: {e}")
        raise


def search_similar_files(
    client: QdrantClient,
    collection_name: str,
    query_embedding: List[float],
    min_score: float = 0.60,
    max_results: int = 25
) -> List[Dict]:
    """
    Smart semantic search with score threshold
    
    Logic:
    1. Search with high score threshold (0.60)
    2. If many files match (>max_results), return top max_results
    3. If few files match (<max_results), return all matches
    """
    try:
        # First search with generous limit to see all potential matches
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=100,  # Get more results to apply logic
            score_threshold=min_score,  # Only files above threshold
            with_payload=True
        )
        
        print(f"🔍 Found {len(results)} files with similarity score > {min_score}")
        
        # Apply smart selection logic
        if len(results) > max_results:
            # Many files match - take top max_results
            selected_results = results[:max_results]
            print(f"📊 Selected top {max_results} files (many matches)")
        else:
            # Few files match - take all
            selected_results = results
            print(f"📊 Selected all {len(results)} files (few matches)")
        
        # Convert results to list of dicts
        similar_files = []
        for result in selected_results:
            file_data = {
                "score": result.score,
                **result.payload
            }
            similar_files.append(file_data)
        
        if similar_files:
            print(f"   Score range: {similar_files[0]['score']:.3f} to {similar_files[-1]['score']:.3f}")
        
        return similar_files
        
    except Exception as e:
        print(f"❌ Error searching similar files: {e}")
        raise


def get_collection_stats(client: QdrantClient, collection_name: str) -> Dict:
    """Get statistics about a specific collection"""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        
        stats = {
            "collection_name": collection_name,
            "total_files": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance.name
        }
        
        return stats
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")
        return {}


def test_connection() -> bool:
    """Test Qdrant connection"""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        print(f"✅ Successfully connected to Qdrant")
        print(f"📊 Found {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False

