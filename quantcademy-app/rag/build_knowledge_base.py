"""
Script to fetch content from links.md and build the knowledge base.
Run this to populate the knowledge base with content from the Foundation module links.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.content_fetcher import fetch_all_content
from rag.knowledge_base_v2 import add_fetched_content_chunks, get_knowledge_base_stats

def main():
    """Main function to fetch content and build knowledge base."""
    print("=" * 60)
    print("QuantCademy Knowledge Base Builder")
    print("=" * 60)
    
    # Path to links.md
    links_file = Path(__file__).parent / "links.md"
    
    if not links_file.exists():
        print(f"Error: {links_file} not found!")
        return
    
    print(f"\nReading links from: {links_file}")
    print("This will fetch content from all URLs in links.md...")
    print("(This may take several minutes)\n")
    
    # Fetch all content
    fetched_content = fetch_all_content(str(links_file), delay=1.0)
    
    if not fetched_content:
        print("\nNo content was fetched. Check your internet connection and URLs.")
        return
    
    print(f"\n✓ Fetched {len(fetched_content)} chunks from {len(set(c['url'] for c in fetched_content))} URLs")
    
    # Add to knowledge base
    print("\nAdding chunks to knowledge base...")
    add_fetched_content_chunks(fetched_content)
    
    # Get stats
    stats = get_knowledge_base_stats()
    
    print("\n" + "=" * 60)
    print("Knowledge Base Statistics:")
    print("=" * 60)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"\nBy tier:")
    for tier, count in stats['by_tier'].items():
        print(f"  {tier}: {count}")
    print(f"\nBy category:")
    for category, count in stats['by_category'].items():
        print(f"  {category}: {count}")
    print(f"\nBy type:")
    for chunk_type, count in stats['by_type'].items():
        print(f"  {chunk_type}: {count}")
    
    # Save fetched content to JSON for persistence (optional)
    cache_file = Path(__file__).parent / "fetched_content_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(fetched_content, f, indent=2)
    print(f"\n✓ Saved fetched content cache to: {cache_file}")
    print("\nNote: The knowledge base is now updated in memory.")
    print("Restart the app to use the new content in the RAG system.")


if __name__ == "__main__":
    main()
