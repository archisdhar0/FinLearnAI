# Content Fetcher & Knowledge Base Builder

This system fetches content from URLs in `links.md` and stores it in the knowledge base for the RAG tutor.

## How It Works

1. **Content Fetcher** (`content_fetcher.py`): Fetches web content from URLs in `links.md`
2. **Knowledge Base** (`knowledge_base_v2.py`): Stores chunks with lesson/module metadata
3. **Retrieval System** (`retrieval.py`): Prioritizes chunks based on current lesson

## Setup

1. Install dependencies:
```bash
pip install beautifulsoup4
```

2. Run the content fetcher:
```bash
cd quantcademy-app
python -m rag.build_knowledge_base
```

This will:
- Read URLs from `rag/links.md`
- Fetch content from each URL
- Chunk the content intelligently
- Add chunks to the knowledge base with proper metadata
- Save a cache file for faster reloads

## Lesson Prioritization

When a user is viewing a specific lesson (e.g., "What is Investing?"), the RAG system will prioritize content from that lesson by boosting matching chunks by 50%.

To use this in your code:

```python
from rag.retrieval import retrieve_with_citations

# When user is on "What is Investing?" lesson
response = retrieve_with_citations(
    query="What is the difference between saving and investing?",
    current_lesson_id="what_is_investing"  # This boosts relevant chunks
)
```

## Lesson ID Mapping

The Foundation module lessons map as follows:
- "What is investing?" → `what_is_investing`
- "What you're actually buying" → `what_youre_actually_buying`
- "How Markets Function" → `how_markets_function`
- "Time and Compounding" → `time_compounding`
- "The Basics of Risk" → `basics_of_risk`
- "Accounts and Setup" → `accounts_setup`
- "First Time Investor Mindset" → `first_time_mindset`

## Caching

Fetched content is cached in `rag/fetched_content_cache.json`. To reload from cache on app startup, uncomment this line in `knowledge_base_v2.py`:

```python
load_fetched_content_from_cache()
```

## Notes

- The fetcher respects rate limits (1 second delay between requests)
- Content is automatically chunked into ~2000 character pieces with overlap
- Source tiers are automatically assigned based on domain
- Chunks include metadata: lesson_id, module_id, source, category, difficulty, key_terms
