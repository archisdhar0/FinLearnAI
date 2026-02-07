"""
QuantCademy Vector Store
Semantic search using sentence-transformers and ChromaDB.
Provides much better retrieval than keyword matching.
"""

import os
import hashlib
from typing import List, Dict, Tuple, Optional

# Flag to track if dependencies are available
VECTOR_STORE_AVAILABLE = False
IMPORT_ERROR = None

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)

from .knowledge_base import KNOWLEDGE_BASE, get_all_documents


# Configuration
COLLECTION_NAME = "quantcademy_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")


class VectorStore:
    """
    Vector store for semantic search over the knowledge base.
    Uses ChromaDB for storage and sentence-transformers for embeddings.
    """
    
    def __init__(self, persist: bool = True):
        """
        Initialize the vector store.
        
        Args:
            persist: Whether to persist the database to disk
        """
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError(
                f"Vector store dependencies not available: {IMPORT_ERROR}\n"
                "Install with: pip install chromadb sentence-transformers"
            )
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        if persist:
            self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "QuantCademy financial education knowledge base"}
        )
        
        print(f"Vector store initialized. Collection has {self.collection.count()} documents.")
    
    def _compute_content_hash(self) -> str:
        """Compute hash of knowledge base content for change detection."""
        content = str(sorted(KNOWLEDGE_BASE.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def index_knowledge_base(self, force: bool = False) -> int:
        """
        Index all documents from the knowledge base.
        
        Args:
            force: Force re-indexing even if content hasn't changed
            
        Returns:
            Number of documents indexed
        """
        # Check if already indexed
        current_hash = self._compute_content_hash()
        existing_count = self.collection.count()
        
        # Get existing metadata to check hash
        if existing_count > 0 and not force:
            try:
                existing = self.collection.get(limit=1, include=["metadatas"])
                if existing["metadatas"] and existing["metadatas"][0].get("kb_hash") == current_hash:
                    print(f"Knowledge base already indexed ({existing_count} documents). Skipping.")
                    return existing_count
            except Exception:
                pass
        
        print(f"Indexing knowledge base ({len(KNOWLEDGE_BASE)} documents)...")
        
        # Clear existing if re-indexing
        if existing_count > 0:
            # Delete all existing documents
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        
        # Prepare documents for indexing
        documents = []
        metadatas = []
        ids = []
        
        for doc_id, doc in KNOWLEDGE_BASE.items():
            # Create searchable text combining title, content, and key terms
            searchable_text = f"""
            {doc['title']}
            
            {doc['content']}
            
            Key topics: {', '.join(doc.get('key_terms', []))}
            """
            
            documents.append(searchable_text.strip())
            metadatas.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "source": doc["source"],
                "category": doc.get("category", "general"),
                "difficulty": doc.get("difficulty", "beginner"),
                "kb_hash": current_hash  # For change detection
            })
            ids.append(doc_id)
        
        # Generate embeddings and add to collection
        print("Generating embeddings...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True).tolist()
        
        print("Adding to vector store...")
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(documents)} documents.")
        return len(documents)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category: str = None,
        difficulty: str = None
    ) -> List[Dict]:
        """
        Semantic search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            category: Filter by category
            difficulty: Filter by difficulty
            
        Returns:
            List of matching documents with scores
        """
        # Build where clause for filtering
        where = None
        where_clauses = []
        
        if category:
            where_clauses.append({"category": category})
        if difficulty:
            where_clauses.append({"difficulty": difficulty})
        
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Get full document from knowledge base
                full_doc = KNOWLEDGE_BASE.get(doc_id, {})
                
                formatted_results.append({
                    "id": doc_id,
                    "title": results["metadatas"][0][i].get("title", ""),
                    "source": results["metadatas"][0][i].get("source", ""),
                    "category": results["metadatas"][0][i].get("category", ""),
                    "difficulty": results["metadatas"][0][i].get("difficulty", ""),
                    "content": full_doc.get("content", ""),
                    "key_terms": full_doc.get("key_terms", []),
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "relevance_score": 1 - (results["distances"][0][i] if results["distances"] else 0)
                })
        
        return formatted_results
    
    def search_with_context(
        self,
        query: str,
        n_results: int = 3,
        max_context_length: int = 4000
    ) -> Tuple[List[Dict], str]:
        """
        Search and return formatted context for LLM.
        
        Args:
            query: Search query
            n_results: Number of results
            max_context_length: Maximum context length in characters
            
        Returns:
            Tuple of (results list, formatted context string)
        """
        results = self.search(query, n_results=n_results)
        
        # Format context for LLM
        context_parts = []
        total_length = 0
        
        for result in results:
            context = f"""
---
SOURCE: {result['title']} ({result['source']})
RELEVANCE: {result['relevance_score']:.2f}

{result['content'].strip()}
---
"""
            if total_length + len(context) > max_context_length:
                break
            
            context_parts.append(context)
            total_length += len(context)
        
        return results, "\n".join(context_parts)


# Singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(initialize: bool = True) -> Optional[VectorStore]:
    """
    Get the singleton vector store instance.
    
    Args:
        initialize: Whether to initialize if not already done
        
    Returns:
        VectorStore instance or None if not available
    """
    global _vector_store_instance
    
    if not VECTOR_STORE_AVAILABLE:
        return None
    
    if _vector_store_instance is None and initialize:
        try:
            _vector_store_instance = VectorStore()
            # Auto-index on first use
            _vector_store_instance.index_knowledge_base()
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None
    
    return _vector_store_instance


def semantic_search(
    query: str,
    n_results: int = 5,
    category: str = None,
    difficulty: str = None
) -> List[Dict]:
    """
    Convenience function for semantic search.
    Falls back to keyword search if vector store unavailable.
    """
    store = get_vector_store()
    
    if store:
        return store.search(query, n_results, category, difficulty)
    else:
        # Fallback to keyword search
        from .knowledge_base import search_knowledge_base
        results = search_knowledge_base(query, category, difficulty)
        return [
            {
                "id": doc_id,
                "title": doc["title"],
                "source": doc["source"],
                "category": doc.get("category", ""),
                "difficulty": doc.get("difficulty", ""),
                "content": doc["content"],
                "key_terms": doc.get("key_terms", []),
                "relevance_score": score / 50  # Normalize keyword score
            }
            for doc_id, score, doc in results
        ]


def get_context_for_query(query: str, n_results: int = 3) -> str:
    """
    Get formatted context for LLM from a query.
    """
    store = get_vector_store()
    
    if store:
        _, context = store.search_with_context(query, n_results)
        return context
    else:
        # Fallback
        from .knowledge_base import search_knowledge_base, format_context_for_llm
        results = search_knowledge_base(query)
        return format_context_for_llm(results)


# CLI for indexing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage QuantCademy vector store")
    parser.add_argument("--reindex", action="store_true", help="Force re-index the knowledge base")
    parser.add_argument("--search", type=str, help="Test search query")
    args = parser.parse_args()
    
    if not VECTOR_STORE_AVAILABLE:
        print(f"Error: {IMPORT_ERROR}")
        print("Install dependencies: pip install chromadb sentence-transformers")
        exit(1)
    
    store = VectorStore()
    
    if args.reindex:
        store.index_knowledge_base(force=True)
    else:
        store.index_knowledge_base()
    
    if args.search:
        print(f"\nSearching for: '{args.search}'")
        results = store.search(args.search)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} (score: {result['relevance_score']:.3f})")
            print(f"   Source: {result['source']}")
            print(f"   Category: {result['category']}")
