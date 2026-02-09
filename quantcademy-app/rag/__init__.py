"""
QuantCademy RAG Module (Capstone-Grade)
Retrieval-Augmented Generation for financial education.

Components:
- knowledge_base: Legacy knowledge base (keyword search)
- knowledge_base_v2: Capstone-grade chunked knowledge with source tiering
- retrieval: Advanced retrieval (hybrid BM25 + semantic, reranking, confidence)
- llm_provider: Multi-provider LLM support (Gemini, Ollama)
- ollama_agent: Main RAG integration
"""

# Legacy knowledge base (for backwards compatibility)
from .knowledge_base import (
    KNOWLEDGE_BASE,
    search_knowledge_base,
    get_document_by_id,
    get_documents_by_category,
    get_documents_for_module,
    format_context_for_llm,
    get_all_documents
)

# Main agent interface
from .ollama_agent import (
    check_ollama_status,
    chat_with_ollama,
    get_quick_response,
    get_rag_context,
    QUICK_RESPONSES,
    ADVANCED_RAG_AVAILABLE
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

# Try to import advanced retrieval system (capstone-grade)
ADVANCED_RETRIEVAL_AVAILABLE = False
try:
    from .retrieval import (
        retrieve_with_citations,
        format_context_with_citations,
        RetrievalResponse,
        RetrievalResult,
        get_retriever,
        ADVANCED_RETRIEVAL_AVAILABLE
    )
except ImportError as e:
    print(f"Advanced retrieval not available: {e}")
    retrieve_with_citations = None
    format_context_with_citations = None
    RetrievalResponse = None
    RetrievalResult = None
    get_retriever = None

# Try to import chunked knowledge base v2
try:
    from .knowledge_base_v2 import (
        Chunk,
        SourceTier,
        get_all_chunks,
        get_chunks_by_tier,
        get_chunk_by_id,
        get_source_tier,
        get_tier_label,
        get_knowledge_base_stats
    )
    KNOWLEDGE_BASE_V2_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_V2_AVAILABLE = False
    Chunk = None
    SourceTier = None
    get_all_chunks = None
    get_chunks_by_tier = None
    get_chunk_by_id = None
    get_source_tier = None
    get_tier_label = None
    get_knowledge_base_stats = None

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
    # Legacy Knowledge Base
    'KNOWLEDGE_BASE',
    'search_knowledge_base',
    'get_document_by_id',
    'get_documents_by_category',
    'get_documents_for_module',
    'format_context_for_llm',
    'get_all_documents',
    
    # Knowledge Base v2 (Capstone)
    'KNOWLEDGE_BASE_V2_AVAILABLE',
    'Chunk',
    'SourceTier',
    'get_all_chunks',
    'get_chunks_by_tier',
    'get_chunk_by_id',
    'get_source_tier',
    'get_tier_label',
    'get_knowledge_base_stats',
    
    # LLM Agent
    'check_ollama_status',
    'chat_with_ollama',
    'get_quick_response',
    'get_rag_context',
    'QUICK_RESPONSES',
    'ADVANCED_RAG_AVAILABLE',
    
    # Advanced Retrieval (Capstone)
    'ADVANCED_RETRIEVAL_AVAILABLE',
    'retrieve_with_citations',
    'format_context_with_citations',
    'RetrievalResponse',
    'RetrievalResult',
    'get_retriever',
    
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
