"""
QuantCademy RAG Module
Retrieval-Augmented Generation for financial education.

Components:
- knowledge_base: Curated financial education content
- vector_store: Semantic search using embeddings
- ollama_agent: LLM integration with RAG
"""

from .knowledge_base import (
    KNOWLEDGE_BASE,
    search_knowledge_base,
    get_document_by_id,
    get_documents_by_category,
    get_documents_for_module,
    format_context_for_llm,
    get_all_documents
)

from .ollama_agent import (
    check_ollama_status,
    chat_with_ollama,
    get_quick_response,
    get_rag_context,
    QUICK_RESPONSES
)

# Try to import vector store (optional, requires extra dependencies)
try:
    from .vector_store import (
        VectorStore,
        semantic_search,
        get_context_for_query,
        get_vector_store,
        VECTOR_STORE_AVAILABLE
    )
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    VectorStore = None
    semantic_search = None
    get_context_for_query = None
    get_vector_store = None

__all__ = [
    # Knowledge Base
    'KNOWLEDGE_BASE',
    'search_knowledge_base',
    'get_document_by_id',
    'get_documents_by_category',
    'get_documents_for_module',
    'format_context_for_llm',
    'get_all_documents',
    
    # Ollama Agent
    'check_ollama_status',
    'chat_with_ollama',
    'get_quick_response',
    'get_rag_context',
    'QUICK_RESPONSES',
    
    # Vector Store (optional)
    'VECTOR_STORE_AVAILABLE',
    'VectorStore',
    'semantic_search',
    'get_context_for_query',
    'get_vector_store',
]
