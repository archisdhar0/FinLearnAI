"""
RAG validation script.

Run from repo root (or anywhere) with the app as sibling folder:
  python rag_validation/validate_rag.py

Expects layout:
  FinLearnAI/
    quantcademy-app/   (app with rag/, .env)
    rag_validation/
      validate_rag.py
      test_set.json
      validation_results.json (written)

Reads test_set.json from this folder, calls the app's chat_with_ollama(stream=False),
and validates with semantic similarity (and optional refusal + retrieval checks).
"""

import json
import os
import sys
from pathlib import Path

# This folder (rag_validation) and the app root (quantcademy-app)
_THIS_DIR = Path(__file__).resolve().parent
_APP_ROOT = _THIS_DIR.parent / "quantcademy-app"
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# Load app .env when script is run from outside quantcademy-app
if _APP_ROOT.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_APP_ROOT / ".env")
    except ImportError:
        pass

# Run with app as cwd so RAG/knowledge base paths and .env resolve correctly
if _APP_ROOT.exists():
    os.chdir(_APP_ROOT.resolve())

from rag.ollama_agent import chat_with_ollama, get_rag_context

# Optional: retrieval-level validation
try:
    from rag.retrieval import retrieve_with_citations
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False


def load_test_set(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_model_answer(question: str) -> str:
    response = chat_with_ollama(question, user_profile=None, stream=False)
    if hasattr(response, "__iter__") and not isinstance(response, str):
        return "".join(response)
    return response or ""


_SIM_MODEL = None


def _get_sim_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SIM_MODEL


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between embeddings. Returns -1.0 on error."""
    import numpy as np
    try:
        model = _get_sim_model()
        emb_a = model.encode([text_a])
        emb_b = model.encode([text_b])
        a, b = emb_a[0], emb_b[0]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception:
        return -1.0


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return (
        "don't have enough" in t
        or "can't recommend" in t
        or "not enough reliable" in t
        or "index fund" in t
        or "i don't have enough" in t
        or "cannot recommend" in t
    )


def key_fact_coverage(golden: str, answer: str) -> float:
    """
    Fraction of golden sentences (or phrases) that appear or are paraphrased in answer.
    Simple version: sentence overlap. Returns 0.0–1.0.
    """
    if not golden or not answer:
        return 0.0
    golden_lower = golden.lower()
    answer_lower = answer.lower()
    # Split into sentences (rough)
    sentences = [s.strip() for s in golden.replace(".\n", ". ").split(". ") if s.strip()]
    if not sentences:
        return 1.0 if answer_lower.strip() else 0.0
    found = 0
    for sent in sentences:
        if len(sent) < 10:
            continue
        # Check if any 5-word window from sentence appears in answer (paraphrase-friendly)
        words = sent.split()
        for i in range(max(1, len(words) - 4)):
            phrase = " ".join(words[i : i + 5])
            if phrase in answer_lower:
                found += 1
                break
    return found / len(sentences) if sentences else 0.0


def get_retrieval_sources(question: str) -> list:
    """Return list of citation/source strings for the question. Empty if retrieval unavailable."""
    if not RETRIEVAL_AVAILABLE:
        return []
    try:
        response = retrieve_with_citations(question)
        return list(response.citations) if response.citations else []
    except Exception:
        return []


def main():
    test_path = _THIS_DIR / "test_set.json"
    if not test_path.exists():
        print(f"Test set not found: {test_path}")
        sys.exit(1)

    tests = load_test_set(str(test_path))
    results = []
    sim_model = None

    for item in tests:
        q = item.get("question", "")
        # Support both "golden" and "answer" as the reference answer
        golden = item.get("golden") or item.get("answer") or ""
        refusal_expected = item.get("refusal_expected", False)
        expected_sources = item.get("expected_sources", [])
        id_ = item.get("id", len(results) + 1)

        answer = get_model_answer(q)
        sim = -1.0
        coverage = 0.0
        retrieval_ok = True

        if refusal_expected:
            passed = is_refusal(answer)
        else:
            sim = semantic_similarity(answer, golden)
            coverage = key_fact_coverage(golden, answer)
            # Pass if similarity >= 0.7 or key-fact coverage is high
            passed = sim >= 0.7 or coverage >= 0.5

        if expected_sources and RETRIEVAL_AVAILABLE:
            sources = get_retrieval_sources(q)
            citation_str = " ".join(sources).lower()
            retrieval_ok = any(exp.lower() in citation_str for exp in expected_sources)

        results.append({
            "id": id_,
            "question": q,
            "golden": golden,
            "answer": answer,
            "similarity": round(sim, 4) if sim >= 0 else None,
            "key_fact_coverage": round(coverage, 4),
            "refusal_expected": refusal_expected,
            "retrieval_ok": retrieval_ok if expected_sources else None,
            "passed": passed and (retrieval_ok if expected_sources else True),
        })

    # Print summary
    passed_count = sum(1 for r in results if r["passed"])
    sim_scores = [r["similarity"] for r in results if r.get("similarity") is not None]
    n_sim = len(sim_scores)
    if n_sim:
        mean_sim = sum(sim_scores) / n_sim
        min_sim = min(sim_scores)
        max_sim = max(sim_scores)
        print(f"\nRAG validation: {passed_count}/{len(results)} passed")
        print(f"Semantic similarity (cosine, {n_sim} Q&A items): mean={mean_sim:.3f}  min={min_sim:.3f}  max={max_sim:.3f}\n")
    else:
        print(f"\nRAG validation: {passed_count}/{len(results)} passed\n")
    for r in results:
        sim_str = f" sim={r['similarity']}" if r.get("similarity") is not None else ""
        ref_str = " [refusal]" if r.get("refusal_expected") else ""
        ret_str = " retrieval_ok" if r.get("retrieval_ok") is True else (" retrieval_fail" if r.get("retrieval_ok") is False else "")
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['id']} {status}{sim_str}{ref_str}{ret_str}  {r['question'][:60]}...")

    # Write full results
    out_path = _THIS_DIR / "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to: {out_path}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
