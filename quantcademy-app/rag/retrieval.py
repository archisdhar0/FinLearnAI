"""
QuantCademy Advanced Retrieval System
Capstone-grade RAG with:
- Hybrid retrieval (BM25 + semantic embeddings)
- Reranking (top 20 → top 5)
- Confidence gating + safe refusal
- Multi-query decomposition
- Citation-required answers
"""

import os
import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Import chunked knowledge base
from .knowledge_base_v2 import (
    Chunk, 
    get_all_chunks, 
    SourceTier, 
    get_tier_label,
    get_chunk_by_id
)

# Try to import ML dependencies
ADVANCED_RETRIEVAL_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    ADVANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    pass


# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Faster reranker model - BAAI's BGE reranker is optimized and faster than ms-marco
RERANK_MODEL = "BAAI/bge-reranker-base"
BM25_K1 = 1.5
BM25_B = 0.75
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for a valid answer
TOP_K_INITIAL = 10  # Reduced from 20 for faster reranking
TOP_K_FINAL = 5     # After reranking


@dataclass
class RetrievalResult:
    """A single retrieval result with all metadata."""
    chunk: Chunk
    bm25_score: float
    semantic_score: float
    combined_score: float
    rerank_score: float
    final_score: float
    
    def get_citation(self) -> str:
        """Get formatted citation."""
        return self.chunk.get_citation()
    
    def to_dict(self) -> Dict:
        return {
            **self.chunk.to_dict(),
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "combined_score": self.combined_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "citation": self.get_citation()
        }


@dataclass 
class RetrievalResponse:
    """Complete retrieval response with confidence and citations."""
    results: List[RetrievalResult]
    confidence: float
    is_confident: bool
    citations: List[str]
    refusal_reason: Optional[str] = None
    subqueries: Optional[List[str]] = None
    
    def get_context(self, max_chunks: int = 5) -> str:
        """Format results as context for LLM."""
        if not self.is_confident:
            return ""
        
        context_parts = []
        for i, result in enumerate(self.results[:max_chunks]):
            chunk = result.chunk
            context_parts.append(f"""
[SOURCE {i+1}: {chunk.source} - {get_tier_label(chunk.source_tier)}]
Section: {chunk.section}
Type: {chunk.chunk_type}

{chunk.content}

---""")
        
        return "\n".join(context_parts)
    
    def get_citation_string(self) -> str:
        """Get formatted citations for response."""
        if not self.citations:
            return ""
        
        unique_citations = list(dict.fromkeys(self.citations))  # Preserve order, remove dupes
        return "Sources: " + "; ".join(unique_citations[:5])


class BM25:
    """BM25 implementation for keyword-based retrieval."""
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_lengths = [len(self._tokenize(doc)) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.doc_freqs = self._compute_doc_freqs()
        self.idf = self._compute_idf()
        self.N = len(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _compute_doc_freqs(self) -> Dict[str, int]:
        """Compute document frequencies for each term."""
        doc_freqs = Counter()
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            doc_freqs.update(tokens)
        return doc_freqs
    
    def _compute_idf(self) -> Dict[str, float]:
        """Compute IDF for each term."""
        idf = {}
        N = len(self.documents)
        for term, df in self.doc_freqs.items():
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query-document pair."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(self.documents[doc_idx])
        doc_length = self.doc_lengths[doc_idx]
        
        term_freqs = Counter(doc_tokens)
        score = 0.0
        
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            tf = term_freqs.get(term, 0)
            idf = self.idf[term]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search and return top-k document indices with scores."""
        scores = [(i, self.score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class AdvancedRetriever:
    """
    Advanced retrieval system with hybrid search, reranking, and confidence gating.
    """
    
    def __init__(self):
        self.chunks = get_all_chunks()
        self.chunk_texts = [c.content for c in self.chunks]
        
        # Initialize BM25
        self.bm25 = BM25(self.chunk_texts)
        
        # Initialize semantic search if available
        self.encoder = None
        self.reranker = None
        self.embeddings = None
        
        if ADVANCED_RETRIEVAL_AVAILABLE:
            print("Loading embedding model...")
            self.encoder = SentenceTransformer(EMBEDDING_MODEL)
            print("Loading reranker model...")
            self.reranker = CrossEncoder(RERANK_MODEL)
            print("Computing embeddings...")
            self.embeddings = self.encoder.encode(self.chunk_texts, show_progress_bar=True)
            print(f"Advanced retriever ready with {len(self.chunks)} chunks.")
        else:
            print("Advanced retrieval not available. Using BM25 only.")
    
    def _semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Semantic search using embeddings."""
        if self.encoder is None or self.embeddings is None:
            return []
        
        query_embedding = self.encoder.encode([query])[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Rerank candidates using cross-encoder."""
        if self.reranker is None or not candidates:
            return candidates
        
        # Prepare pairs for reranking
        pairs = [(query, self.chunk_texts[idx]) for idx, _ in candidates]
        
        # Get rerank scores
        scores = self.reranker.predict(pairs)
        
        # Combine with indices
        reranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _hybrid_search(self, query: str, top_k: int = TOP_K_INITIAL, current_lesson_id: Optional[str] = None) -> List[RetrievalResult]:
        """
        Hybrid search combining BM25 and semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            current_lesson_id: Optional lesson ID to prioritize (e.g., "what_is_investing")
        """
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k)
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Semantic search
        if ADVANCED_RETRIEVAL_AVAILABLE:
            semantic_results = self._semantic_search(query, top_k=top_k)
            semantic_scores = {idx: score for idx, score in semantic_results}
        else:
            semantic_scores = {}
        
        # Get all candidate indices
        all_indices = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Normalize scores
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1
        semantic_max = max(semantic_scores.values()) if semantic_scores else 1
        
        # Combine scores (weighted average)
        results = []
        for idx in all_indices:
            bm25_norm = bm25_scores.get(idx, 0) / bm25_max if bm25_max > 0 else 0
            semantic_norm = semantic_scores.get(idx, 0) / semantic_max if semantic_max > 0 else 0
            
            # Hybrid score: 40% BM25, 60% semantic
            combined = 0.4 * bm25_norm + 0.6 * semantic_norm if semantic_scores else bm25_norm
            
            # Boost by source tier
            tier_boost = int(self.chunks[idx].source_tier) / 5.0  # 0.2 to 1.0
            combined = combined * (0.8 + 0.2 * tier_boost)
            
            # PRIORITIZE: Boost chunks from current lesson by 50%
            # This ensures content relevant to the current lesson is prioritized
            if current_lesson_id and self.chunks[idx].lesson_id == current_lesson_id:
                combined = combined * 1.5
            
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                bm25_score=bm25_scores.get(idx, 0),
                semantic_score=semantic_scores.get(idx, 0),
                combined_score=combined,
                rerank_score=0,
                final_score=combined
            ))
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
    
    def _compute_confidence(self, results: List[RetrievalResult]) -> float:
        """
        Compute confidence score based on retrieval quality.
        More strict - returns low confidence for weak matches.
        """
        if not results:
            return 0.0
        
        # Get top result score
        top_score = results[0].final_score if results else 0
        
        # STRICT: If top score is very low, immediately return low confidence
        # This catches queries with no relevant content
        if top_score < 0.25:
            return 0.1
        
        # Check if BM25 score is near zero (no keyword match at all)
        top_bm25 = results[0].bm25_score if results else 0
        if top_bm25 < 0.5:  # Very weak keyword match
            return min(top_score * 0.5, 0.3)  # Cap at 30%
        
        # Score gap between top results (higher gap = more confident)
        if len(results) >= 2:
            score_gap = results[0].final_score - results[1].final_score
        else:
            score_gap = 0
        
        # Average tier of top results (higher tier = more confident)
        avg_tier = sum(int(r.chunk.source_tier) for r in results[:3]) / min(3, len(results))
        tier_factor = avg_tier / 5.0
        
        # Number of results above threshold (more strict: 0.4 instead of 0.3)
        good_results = sum(1 for r in results if r.final_score > 0.4)
        coverage_factor = min(good_results / 3.0, 1.0)
        
        # Weighted combination
        confidence = (
            0.4 * top_score +
            0.2 * min(score_gap * 2, 1.0) +
            0.2 * tier_factor +
            0.2 * coverage_factor
        )
        
        return min(confidence, 1.0)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K_FINAL,
        min_confidence: float = CONFIDENCE_THRESHOLD,
        current_lesson_id: Optional[str] = None
    ) -> RetrievalResponse:
        """
        Main retrieval function with hybrid search, reranking, and confidence gating.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            current_lesson_id: Optional lesson ID to prioritize (e.g., "what_is_investing")
        """
        # Step 1: Hybrid search (BM25 + semantic)
        candidates = self._hybrid_search(query, top_k=TOP_K_INITIAL, current_lesson_id=current_lesson_id)
        
        if not candidates:
            return RetrievalResponse(
                results=[],
                confidence=0.0,
                is_confident=False,
                citations=[],
                refusal_reason="No relevant information found in the knowledge base."
            )
        
        # Step 2: Rerank top candidates
        if ADVANCED_RETRIEVAL_AVAILABLE and self.reranker:
            candidate_tuples = [(i, r.combined_score) for i, r in enumerate(candidates)]
            pairs = [(query, candidates[i].chunk.content) for i, _ in candidate_tuples[:TOP_K_INITIAL]]
            
            if pairs:
                rerank_scores = self.reranker.predict(pairs)
                
                for i, score in enumerate(rerank_scores):
                    candidates[i].rerank_score = float(score)
                    # Final score: combine hybrid and rerank
                    candidates[i].final_score = 0.3 * candidates[i].combined_score + 0.7 * (float(score) + 10) / 20
                
                candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Step 3: Take top K
        top_results = candidates[:top_k]
        
        # Step 4: Compute confidence
        confidence = self._compute_confidence(top_results)
        is_confident = confidence >= min_confidence
        
        # Step 5: Generate citations
        citations = [r.get_citation() for r in top_results if r.final_score > 0.2]
        
        # Step 6: Check for refusal
        refusal_reason = None
        if not is_confident:
            refusal_reason = (
                f"I don't have enough reliable information in my sources to answer this question confidently. "
                f"(Confidence: {confidence:.0%}, threshold: {min_confidence:.0%}). "
                f"Please try rephrasing or ask about a different topic."
            )
        
        return RetrievalResponse(
            results=top_results,
            confidence=confidence,
            is_confident=is_confident,
            citations=citations,
            refusal_reason=refusal_reason
        )


class MultiQueryRetriever:
    """
    Handles complex questions by decomposing into subqueries.
    """
    
    def __init__(self, base_retriever: AdvancedRetriever):
        self.retriever = base_retriever
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query needs decomposition."""
        # Indicators of complexity
        complex_patterns = [
            r'\b(and|also|as well as)\b.*\?',  # Multiple questions
            r'\b(compare|difference|vs|versus)\b',  # Comparison
            r'\b(how|why).*\b(and|then)\b',  # Multi-step
            r'\b(what|which).*\b(best|better)\b.*\b(for|if)\b',  # Conditional
            r'\?.*\?',  # Multiple question marks
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, query.lower()):
                return True
        
        # Length-based heuristic
        if len(query.split()) > 20:
            return True
        
        return False
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into 2-4 subqueries.
        This is a rule-based approach. For production, use an LLM.
        """
        subqueries = []
        query_lower = query.lower()
        
        # Pattern 1: "What is X and how does Y work?"
        if ' and ' in query_lower and '?' in query:
            parts = re.split(r'\band\b', query, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip().rstrip('?').strip()
                if len(part) > 10:
                    if not part.endswith('?'):
                        part += '?'
                    subqueries.append(part)
        
        # Pattern 2: "Compare X and Y" or "X vs Y"
        compare_match = re.search(r'(compare|difference|vs|versus)\s+(.+?)\s+(and|vs|versus|with)\s+(.+?)[\?]?$', query_lower)
        if compare_match:
            term1 = compare_match.group(2).strip()
            term2 = compare_match.group(4).strip().rstrip('?')
            subqueries.append(f"What is {term1}?")
            subqueries.append(f"What is {term2}?")
            subqueries.append(f"How does {term1} compare to {term2}?")
        
        # Pattern 3: "How do I X if Y?"
        conditional_match = re.search(r'(how|what|should).*\b(if|when|while)\b', query_lower)
        if conditional_match and not subqueries:
            subqueries.append(query)  # Keep original
            # Extract the condition
            condition_match = re.search(r'\b(if|when|while)\s+(.+?)[\?]?$', query_lower)
            if condition_match:
                condition = condition_match.group(2).strip().rstrip('?')
                subqueries.append(f"What about {condition}?")
        
        # If no patterns matched or too few subqueries, use original
        if len(subqueries) < 2:
            subqueries = [query]
        
        # Limit to 4 subqueries
        return subqueries[:4]
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K_FINAL,
        min_confidence: float = CONFIDENCE_THRESHOLD,
        current_lesson_id: Optional[str] = None
    ) -> RetrievalResponse:
        """
        Retrieve with optional query decomposition for complex questions.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            current_lesson_id: Optional lesson ID to prioritize (e.g., "what_is_investing")
        """
        subqueries = None
        
        # Check if we need to decompose
        if self._is_complex_query(query):
            subqueries = self._decompose_query(query)
        else:
            subqueries = [query]
        
        # Retrieve for each subquery
        all_results = []
        all_citations = []
        
        for subquery in subqueries:
            response = self.retriever.retrieve(subquery, top_k=top_k, current_lesson_id=current_lesson_id)
            all_results.extend(response.results)
            all_citations.extend(response.citations)
        
        # Deduplicate by chunk ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.chunk.id not in seen_ids:
                seen_ids.add(result.chunk.id)
                unique_results.append(result)
        
        # Re-sort by score
        unique_results.sort(key=lambda x: x.final_score, reverse=True)
        top_results = unique_results[:top_k]
        
        # Compute overall confidence
        confidence = self.retriever._compute_confidence(top_results)
        is_confident = confidence >= min_confidence
        
        # Deduplicate citations
        unique_citations = list(dict.fromkeys(all_citations))
        
        refusal_reason = None
        if not is_confident:
            refusal_reason = (
                f"I don't have enough reliable information to answer this question confidently. "
                f"(Confidence: {confidence:.0%}). Please try a more specific question."
            )
        
        return RetrievalResponse(
            results=top_results,
            confidence=confidence,
            is_confident=is_confident,
            citations=unique_citations,
            refusal_reason=refusal_reason,
            subqueries=subqueries if len(subqueries) > 1 else None
        )


# Singleton instances
_advanced_retriever: Optional[AdvancedRetriever] = None
_multi_query_retriever: Optional[MultiQueryRetriever] = None


def get_retriever() -> MultiQueryRetriever:
    """Get the singleton retriever instance."""
    global _advanced_retriever, _multi_query_retriever
    
    if _multi_query_retriever is None:
        _advanced_retriever = AdvancedRetriever()
        _multi_query_retriever = MultiQueryRetriever(_advanced_retriever)
    
    return _multi_query_retriever


def retrieve_with_citations(
    query: str,
    top_k: int = TOP_K_FINAL,
    min_confidence: float = CONFIDENCE_THRESHOLD,
    current_lesson_id: Optional[str] = None
) -> RetrievalResponse:
    """
    Main retrieval function with all advanced features.
    
    Args:
        query: Search query
        top_k: Number of results to return
        min_confidence: Minimum confidence threshold
        current_lesson_id: Optional lesson ID to prioritize (e.g., "what_is_investing")
    
    Returns a RetrievalResponse with:
    - results: List of chunks with scores
    - confidence: Overall confidence score
    - is_confident: Whether to answer or refuse
    - citations: List of source citations
    - refusal_reason: Why we can't answer (if not confident)
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k, min_confidence, current_lesson_id)


def format_context_with_citations(response: RetrievalResponse) -> Tuple[str, str]:
    """
    Format retrieval response into context and citation strings.
    
    Returns:
        Tuple of (context_for_llm, citation_string)
    """
    if not response.is_confident:
        return "", response.refusal_reason or ""
    
    context = response.get_context()
    citations = response.get_citation_string()
    
    return context, citations


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test advanced retrieval")
    parser.add_argument("query", nargs="?", default="What is an ETF and how does it compare to a mutual fund?")
    args = parser.parse_args()
    
    print(f"\nQuery: {args.query}\n")
    
    response = retrieve_with_citations(args.query)
    
    print(f"Confidence: {response.confidence:.1%}")
    print(f"Is Confident: {response.is_confident}")
    print(f"Subqueries: {response.subqueries}")
    print(f"\nCitations: {response.get_citation_string()}")
    
    if response.refusal_reason:
        print(f"\nRefusal: {response.refusal_reason}")
    else:
        print(f"\nTop {len(response.results)} Results:")
        for i, result in enumerate(response.results, 1):
            print(f"\n{i}. [{result.chunk.source}] {result.chunk.section}")
            print(f"   Type: {result.chunk.chunk_type}, Tier: {get_tier_label(result.chunk.source_tier)}")
            print(f"   Scores: BM25={result.bm25_score:.3f}, Semantic={result.semantic_score:.3f}, Final={result.final_score:.3f}")
