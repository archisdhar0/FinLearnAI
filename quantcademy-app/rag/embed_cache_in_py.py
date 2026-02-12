"""
Script to convert fetched_content_cache.json into Python code
that can be embedded directly in knowledge_base_v2.py
"""

import json
from pathlib import Path

def generate_python_chunks(cache_file: str = "rag/fetched_content_cache.json"):
    """Generate Python code from cache JSON."""
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        print(f"Cache file not found: {cache_file}")
        return None
    
    with open(cache_path, 'r') as f:
        fetched_content = json.load(f)
    
    if not fetched_content:
        print("Cache file is empty")
        return None
    
    # Generate Python code
    python_code = []
    python_code.append("# =========================================================\n")
    python_code.append("# FETCHED CONTENT FROM LINKS.MD\n")
    python_code.append("# Auto-generated from fetched_content_cache.json\n")
    python_code.append("# Run: python -m rag.embed_cache_in_py to regenerate\n")
    python_code.append("# =========================================================\n\n")
    
    for item in fetched_content:
        lesson_id = item.get('lesson_id', '')
        module_id = item.get('module_id', 'foundations')
        url = item.get('url', '')
        source = item.get('source', 'Unknown')
        title = item.get('title', '')
        content = item.get('content', '')
        chunk_idx = item.get('chunk_index', 0)
        total_chunks = item.get('total_chunks', 1)
        
        if not content:
            continue
        
        # Escape content for Python string
        content_escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        
        # Generate chunk code
        doc_id = url.replace("https://", "").replace("http://", "").replace("/", "_")[:50]
        section = f"{title} - Part {chunk_idx + 1}" if total_chunks > 1 else title
        
        python_code.append(f'    CHUNKED_KNOWLEDGE_BASE.append(Chunk(\n')
        python_code.append(f'        id=generate_chunk_id("{doc_id}", "{section}", {chunk_idx}),\n')
        python_code.append(f'        document_id="{doc_id}",\n')
        python_code.append(f'        content="""{content_escaped}""",\n')
        python_code.append(f'        chunk_type="concept",\n')
        python_code.append(f'        section="{section}",\n')
        python_code.append(f'        source="{source}",\n')
        python_code.append(f'        source_tier=get_source_tier("{source}"),\n')
        python_code.append(f'        url="{url}",\n')
        python_code.append(f'        category="basics",\n')
        python_code.append(f'        difficulty="beginner",\n')
        python_code.append(f'        key_terms=[],\n')
        python_code.append(f'        lesson_id="{lesson_id}",\n')
        python_code.append(f'        module_id="{module_id}"\n')
        python_code.append(f'    ))\n\n')
    
    return ''.join(python_code)


def embed_in_knowledge_base():
    """Generate Python code and show where to insert it."""
    python_code = generate_python_chunks()
    
    if not python_code:
        return
    
    # Show the code
    print("=" * 60)
    print("Generated Python code for knowledge_base_v2.py")
    print("=" * 60)
    print("\nAdd this code to build_chunked_knowledge_base() function")
    print("right before the closing of the function (before the final comment).\n")
    print(python_code)
    
    # Also save to a file for easy copy-paste
    output_file = Path("rag/fetched_content_embedded.py")
    with open(output_file, 'w') as f:
        f.write(python_code)
    
    print(f"\n✓ Also saved to: {output_file}")
    print("\nTo embed:")
    print("1. Open knowledge_base_v2.py")
    print("2. Find build_chunked_knowledge_base() function")
    print("3. Add the code from fetched_content_embedded.py before the function ends")
    print("4. Remove or comment out: load_fetched_content_from_cache()")


if __name__ == "__main__":
    embed_in_knowledge_base()
