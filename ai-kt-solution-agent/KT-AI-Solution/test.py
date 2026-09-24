#!/usr/bin/env python3
"""
Qdrant Collection Downloader
============================
Simple script to list and download entire Qdrant collections as JSON files.

Requirements:
    pip install qdrant-client python-dotenv

Setup:
    Create .env file with:
    QDRANT_URL=https://your-qdrant-instance.com
    QDRANT_API_KEY=your-api-key-here

Usage:
    python qdrant_downloader.py
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def main():
    print("🚀 Qdrant Collection Downloader")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        print("\nCreate .env file with:")
        print("QDRANT_URL=https://your-qdrant-instance.com")
        print("QDRANT_API_KEY=your-api-key-here")
        return
    
    try:
        # Connect to Qdrant
        print(f"🔌 Connecting to {qdrant_url}...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Get all collections
        print("📋 Fetching collections...")
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if not collection_names:
            print("❌ No collections found!")
            return
        
        # Display collections
        print(f"\n📚 Found {len(collection_names)} collections:")
        print("-" * 30)
        for i, name in enumerate(collection_names, 1):
            # Get collection info
            info = client.get_collection(name)
            print(f"{i:2d}. {name:<20} ({info.points_count:,} points)")
        print("-" * 30)
        
        # Get user choice
        while True:
            try:
                choice = input(f"\n🎯 Enter collection number (1-{len(collection_names)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    return
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(collection_names):
                    selected_collection = collection_names[choice_num - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(collection_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get collection details
        collection_info = client.get_collection(selected_collection)
        total_points = collection_info.points_count
        
        print(f"\n📊 Collection: {selected_collection}")
        print(f"📈 Points: {total_points:,}")
        print(f"🎯 Vector Size: {collection_info.config.params.vectors.size}")
        print(f"📏 Distance: {collection_info.config.params.vectors.distance.value}")
        
        # Confirm download
        if total_points > 1000:
            confirm = input(f"\n⚠️  Large collection ({total_points:,} points). Continue? (y/n): ")
            if confirm.lower() not in ['y', 'yes']:
                print("❌ Download cancelled")
                return
        
        # Download collection
        print(f"\n⬇️  Downloading {selected_collection}...")
        all_points = []
        offset = None
        batch_size = 100
        
        while True:
            # Get batch of points
            points, next_offset = client.scroll(
                collection_name=selected_collection,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                break
            
            # Convert points to JSON format
            for point in points:
                all_points.append({
                    'id': str(point.id),
                    'vector': point.vector,
                    'payload': point.payload
                })
            
            # Show progress
            print(f"📦 Downloaded: {len(all_points):,}/{total_points:,} points", end='\r')
            
            offset = next_offset
            if not next_offset:
                break
        
        print(f"\n✅ Download complete: {len(all_points):,} points")
        
        # Create output data
        output_data = {
            'collection_name': selected_collection,
            'download_timestamp': datetime.now().isoformat(),
            'total_points': len(all_points),
            'collection_info': {
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'points_count': total_points
            },
            'points': all_points
        }
        
        # Save to file
        filename = f"{selected_collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print(f"💾 Saving to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Show file info
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"✅ Successfully saved to: {filename}")
        print(f"📁 File size: {file_size:.2f} MB")
        print(f"📊 Total points: {len(all_points):,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")