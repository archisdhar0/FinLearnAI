"""
QuantCademy RAG Module
Retrieval-Augmented Generation for financial education.

Components:
- knowledge_base: Curated financial education content
- vector_store: Semantic search using embeddings
- llm_provider: Multi-provider LLM support (Gemini, Ollama)
- ollama_agent: Main RAG integration
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

# Try to import LLM provider
try:
    from .llm_provider import (
        check_llm_status,
        chat_with_llm,
        LLM_PROVIDER,
        GEMINI_AVAILABLE
    )
except ImportError:
    check_llm_status = None
    chat_with_llm = None
    LLM_PROVIDER = "none"
    GEMINI_AVAILABLE = False

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
    
    # LLM Agent
    'check_ollama_status',
    'chat_with_ollama',
    'get_quick_response',
    'get_rag_context',
    'QUICK_RESPONSES',
    
    # LLM Provider
    'check_llm_status',
    'chat_with_llm',
    'LLM_PROVIDER',
    'GEMINI_AVAILABLE',
    
    # Vector Store (optional)
    'VECTOR_STORE_AVAILABLE',
    'VectorStore',
    'semantic_search',
    'get_context_for_query',
    'get_vector_store',
]
