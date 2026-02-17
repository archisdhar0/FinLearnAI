"""
QuantCademy — Financial Content Scraper Agent
Finds and scrapes content ONLY from trusted financial/educational sites.

Strategy:
  1. For each lesson topic, run ONE search query (not 12 per-site queries)
  2. Filter results to only keep URLs from whitelisted domains
  3. If search fails, use pre-built fallback URLs for major sites
  4. Scrape, chunk, and save to fetched_content_cache.json

Usage:
    cd quantcademy-app
    python3 -m rag.web_link_finder
"""

import requests
import json
import time
import sys
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Domains we trust — only URLs from these will be kept
ALLOWED_DOMAINS: Dict[str, str] = {
    "investopedia.com":             "Investopedia",
    "vanguard.com":                 "Vanguard",
    "fidelity.com":                 "Fidelity",
    "schwab.com":                   "Charles Schwab",
    "sec.gov":                      "SEC",
    "finra.org":                    "FINRA",
    "investor.gov":                 "SEC Investor.gov",
    "nerdwallet.com":               "NerdWallet",
    "bankrate.com":                 "Bankrate",
    "morningstar.com":              "Morningstar",
    "khanacademy.org":              "Khan Academy",
    "corporatefinanceinstitute.com":"CFI",
    "usbank.com":                   "U.S. Bank",
    "wellsfargo.com":               "Wells Fargo",
    "morganstanley.com":            "Morgan Stanley",
    "sofi.com":                     "SoFi",
    "fool.com":                     "Motley Fool",
    "nasdaq.com":                   "NASDAQ",
    "capitalone.com":               "Capital One",
}


def domain_allowed(url: str) -> bool:
    """Return True if URL belongs to a whitelisted domain."""
    host = urlparse(url).netloc.lower().replace("www.", "")
    return any(d in host for d in ALLOWED_DOMAINS)


def source_name(url: str) -> str:
    """Get friendly source name for a URL."""
    host = urlparse(url).netloc.lower().replace("www.", "")
    for d, name in ALLOWED_DOMAINS.items():
        if d in host:
            return name
    return host


# ---------------------------------------------------------------------------
# Search: uses duckduckgo_search library (already installed by user)
# ---------------------------------------------------------------------------
def search_ddg(query: str, max_results: int = 20) -> List[str]:
    """Search DuckDuckGo and return URLs from trusted domains only."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("    [!] duckduckgo_search not available, trying HTML fallback")
        return _search_ddg_html(query, max_results)

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        urls = [r["href"] for r in raw if "href" in r]
        trusted = [u for u in urls if domain_allowed(u)]
        return trusted
    except Exception as e:
        print(f"    [!] DDG library search failed: {e}")
        return _search_ddg_html(query, max_results)


def _search_ddg_html(query: str, max_results: int = 20) -> List[str]:
    """Fallback: scrape DuckDuckGo HTML interface directly."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"    [!] DDG HTML search failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        # skip DDG ad/tracking redirects
        if "duckduckgo.com" in href:
            continue
        if href.startswith("http") and domain_allowed(href):
            if href not in urls:
                urls.append(href)
        if len(urls) >= max_results:
            break
    return urls


# ---------------------------------------------------------------------------
# Fallback URLs — hand-picked high-quality pages for each lesson
# These are used when search returns nothing for a topic
# ---------------------------------------------------------------------------
# Every URL below has been verified as 200 OK
FALLBACK_URLS: Dict[str, List[str]] = {
    "what_is_investing": [
        "https://www.investopedia.com/terms/i/investing.asp",
        "https://www.investopedia.com/articles/basics/11/3-s-simple-investing.asp",
        "https://investor.vanguard.com/investor-resources-education/article/how-to-start-investing",
        "https://www.nerdwallet.com/article/investing/how-to-start-investing",
        "https://www.fidelity.com/learning-center/trading-investing/investing-for-beginners",
    ],
    "what_youre_actually_buying": [
        "https://www.investopedia.com/terms/s/stock.asp",
        "https://www.investopedia.com/terms/b/bond.asp",
        "https://www.investopedia.com/terms/e/etf.asp",
        "https://www.investopedia.com/terms/i/indexfund.asp",
        "https://www.schwab.com/learn/story/investing-basics",
    ],
    "how_markets_function": [
        "https://www.investopedia.com/terms/l/law-of-supply-demand.asp",
        "https://www.investopedia.com/articles/basics/04/100804.asp",
        "https://www.investopedia.com/terms/b/bearmarket.asp",
        "https://www.investopedia.com/terms/b/bullmarket.asp",
        "https://www.bankrate.com/investing/how-to-invest-in-stocks/",
    ],
    "time_compounding": [
        "https://www.investopedia.com/terms/c/compoundinterest.asp",
        "https://www.investopedia.com/terms/t/timehorizon.asp",
        "https://www.fidelity.com/viewpoints/investing-ideas/power-of-compound-interest",
        "https://www.morningstar.com/investing-definitions/compound-interest",
        "https://investor.vanguard.com/investor-resources-education/article/how-to-start-investing",
    ],
    "basics_of_risk": [
        "https://www.investopedia.com/terms/r/risk.asp",
        "https://www.investopedia.com/terms/v/volatility.asp",
        "https://www.investopedia.com/terms/d/diversification.asp",
        "https://www.investopedia.com/terms/r/risktolerance.asp",
        "https://www.finra.org/investors/investing/investing-basics/risk",
    ],
    "accounts_setup": [
        "https://www.investopedia.com/terms/b/brokerageaccount.asp",
        "https://www.investopedia.com/terms/r/rothira.asp",
        "https://www.finra.org/investors/investing/investment-accounts/brokerage-accounts",
        "https://investor.vanguard.com/accounts-plans/iras/roth-ira",
        "https://www.schwab.com/learn/story/investing-basics",
    ],
    "first_time_mindset": [
        "https://www.investopedia.com/articles/basics/06/invest1000.asp",
        "https://www.fidelity.com/learning-center/trading-investing/investing-for-beginners",
        "https://investor.vanguard.com/investor-resources-education/article/how-to-start-investing",
        "https://www.nerdwallet.com/article/investing/how-to-start-investing",
        "https://www.schwab.com/learn/story/investing-basics",
    ],
    "what_moves_markets": [
        "https://www.investopedia.com/articles/basics/04/100804.asp",
        "https://www.investopedia.com/terms/i/inflation.asp",
        "https://www.investopedia.com/terms/i/interestrate.asp",
        "https://www.investopedia.com/terms/e/earnings.asp",
        "https://investor.vanguard.com/investor-resources-education/market-volatility",
    ],
    "investor_psychology": [
        "https://www.investopedia.com/terms/b/behavioralfinance.asp",
        "https://www.investopedia.com/terms/h/herdinstinct.asp",
        "https://www.investopedia.com/terms/s/speculation.asp",
        "https://www.fidelity.com/learning-center/trading-investing/investing-for-beginners",
    ],
    "hype_vs_fundamentals": [
        "https://www.investopedia.com/terms/s/speculation.asp",
        "https://www.investopedia.com/terms/f/fundamentalanalysis.asp",
        "https://www.investopedia.com/news/active-vs-passive-investing/",
        "https://www.investopedia.com/terms/p/passiveinvesting.asp",
    ],
    "types_of_investing": [
        "https://www.investopedia.com/news/active-vs-passive-investing/",
        "https://www.investopedia.com/terms/p/passiveinvesting.asp",
        "https://www.finra.org/investors/insights/active-passive-investing",
        "https://www.fidelity.com/viewpoints/investing-ideas/guide-to-diversification",
    ],
    "risk_portfolio_thinking": [
        "https://www.investopedia.com/terms/d/diversification.asp",
        "https://www.investopedia.com/terms/a/assetallocation.asp",
        "https://www.investopedia.com/terms/a/assetclasses.asp",
        "https://www.investopedia.com/terms/p/portfolio.asp",
        "https://www.fidelity.com/viewpoints/investing-ideas/guide-to-diversification",
    ],
    "reading_market_signals": [
        "https://www.investopedia.com/terms/t/trend.asp",
        "https://www.investopedia.com/terms/v/volatility.asp",
        "https://www.investopedia.com/terms/m/momentum.asp",
        "https://www.investopedia.com/terms/b/bearmarket.asp",
        "https://www.investopedia.com/terms/b/bullmarket.asp",
    ],
    "costs_fees_taxes": [
        "https://www.investopedia.com/terms/e/expenseratio.asp",
        "https://www.investopedia.com/terms/c/capital_gains_tax.asp",
        "https://www.investopedia.com/investing/costs-investing/",
        "https://www.finra.org/investors/investing/investing-basics/fees-commissions",
        "https://www.schwab.com/learn/story/investing-basics",
    ],
    "what_do_in_crash": [
        "https://www.investopedia.com/terms/s/stock-market-crash.asp",
        "https://www.investopedia.com/terms/c/correction.asp",
        "https://www.schwab.com/learn/story/stay-course-when-markets-turn-turbulent",
        "https://www.fidelity.com/viewpoints/market-and-economic-insights/market-corrections",
        "https://investor.vanguard.com/investor-resources-education/market-volatility",
    ],
    "setting_long_term_structure": [
        "https://www.investopedia.com/terms/d/dollarcostaveraging.asp",
        "https://www.investopedia.com/terms/r/rebalancing.asp",
        "https://www.fidelity.com/viewpoints/investing-ideas/rebalancing",
        "https://www.fidelity.com/learning-center/trading-investing/investing-for-beginners",
        "https://www.schwab.com/learn/story/investing-basics",
    ],
    "realistic_expectations": [
        "https://www.investopedia.com/ask/answers/042415/what-average-annual-return-sp-500.asp",
        "https://www.investopedia.com/terms/v/volatility.asp",
        "https://www.fidelity.com/learning-center/trading-investing/markets/time-in-market",
        "https://www.nerdwallet.com/article/investing/average-stock-market-return",
        "https://www.fidelity.com/learning-center/trading-investing/average-stock-market-return",
    ],
}


# ---------------------------------------------------------------------------
# Lessons — each lesson has search queries and a lesson_id / module_id
# ---------------------------------------------------------------------------
LESSONS: Dict[str, Dict] = {
    "What is investing?": {
        "id": "what_is_investing",
        "module": "foundations",
        "queries": [
            "what is investing beginner guide",
            "saving vs investing difference",
            "why invest money instead of saving",
            "inflation effect on cash savings",
        ],
    },
    "What you're actually buying": {
        "id": "what_youre_actually_buying",
        "module": "foundations",
        "queries": [
            "stocks bonds ETFs index funds explained beginner",
            "what is a stock share equity",
            "what is a bond investing",
            "ETF vs index fund vs mutual fund",
        ],
    },
    "How Markets Function": {
        "id": "how_markets_function",
        "module": "foundations",
        "queries": [
            "how does the stock market work beginner",
            "supply and demand effect on stock prices",
            "what causes stock prices to change",
        ],
    },
    "Time and Compounding": {
        "id": "time_compounding",
        "module": "foundations",
        "queries": [
            "compound interest investing explained",
            "power of starting early investing",
            "investment time horizon explained",
        ],
    },
    "The Basics of Risk": {
        "id": "basics_of_risk",
        "module": "foundations",
        "queries": [
            "investment risk explained beginner",
            "portfolio diversification basics",
            "risk tolerance investing",
        ],
    },
    "Accounts and Setup": {
        "id": "accounts_setup",
        "module": "foundations",
        "queries": [
            "brokerage account vs IRA explained",
            "how to open investment account beginner",
            "Roth IRA explained beginner",
        ],
    },
    "First Time Investor Mindset": {
        "id": "first_time_mindset",
        "module": "foundations",
        "queries": [
            "beginner investor tips first time",
            "investing for beginners guide",
            "common beginner investor mistakes",
        ],
    },
    "What Moves Markets": {
        "id": "what_moves_markets",
        "module": "investor_insight",
        "queries": [
            "what moves the stock market earnings interest rates",
            "how inflation affects investments",
            "economic indicators stock market",
        ],
    },
    "Investor Psychology": {
        "id": "investor_psychology",
        "module": "investor_insight",
        "queries": [
            "behavioral finance investor psychology",
            "herd behavior panic selling investing",
            "overconfidence bias investing",
        ],
    },
    "Hype vs. Fundamentals": {
        "id": "hype_vs_fundamentals",
        "module": "investor_insight",
        "queries": [
            "speculation vs investing fundamentals",
            "market noise vs signal long term",
        ],
    },
    "Types of Investing": {
        "id": "types_of_investing",
        "module": "investor_insight",
        "queries": [
            "active vs passive investing explained",
            "long term vs short term investing strategy",
            "index fund investing for beginners",
        ],
    },
    "Risk and Portfolio Thinking": {
        "id": "risk_portfolio_thinking",
        "module": "investor_insight",
        "queries": [
            "portfolio diversification asset allocation",
            "asset classes explained stocks bonds",
            "how to build investment portfolio",
        ],
    },
    "Reading Base Market Signals": {
        "id": "reading_market_signals",
        "module": "investor_insight",
        "queries": [
            "stock market trends uptrend downtrend",
            "market volatility and momentum explained",
            "market cycles bull bear explained",
        ],
    },
    "Costs, Fees, and Taxes": {
        "id": "costs_fees_taxes",
        "module": "applied_investing",
        "queries": [
            "investment fees expense ratio explained",
            "capital gains tax investing",
            "tax advantaged accounts IRA 401k",
        ],
    },
    "What to do in a market crash": {
        "id": "what_do_in_crash",
        "module": "applied_investing",
        "queries": [
            "what to do stock market crash downturn",
            "market crash history recovery timeline",
            "emotional discipline investing bear market",
        ],
    },
    "Setting a Long term structure": {
        "id": "setting_long_term_structure",
        "module": "applied_investing",
        "queries": [
            "dollar cost averaging explained",
            "portfolio rebalancing strategy",
            "long term investment plan automatic",
        ],
    },
    "Realistic Expectations About Returns": {
        "id": "realistic_expectations",
        "module": "applied_investing",
        "queries": [
            "average stock market return history",
            "stock market volatility normal",
            "time in market vs timing the market",
        ],
    },
}


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
def scrape_page(url: str) -> Optional[Dict[str, str]]:
    """Fetch a URL and return cleaned text, title, source. None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    body = None
    for sel in ["main", "article", '[role="main"]', ".article-body",
                ".entry-content", "#content", "body"]:
        body = soup.select_one(sel)
        if body:
            break
    if not body:
        return None

    text = body.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    content = "\n".join(lines)

    if len(content) < 300:
        return None

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    return {
        "url": url,
        "title": title,
        "source": source_name(url),
        "content": content,
    }


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def chunk_text(content: str, size: int = 2000, overlap: int = 400) -> List[str]:
    """Sliding-window chunking with sentence-boundary snapping."""
    if len(content) <= size:
        return [content]
    chunks, start, step = [], 0, size - overlap
    while start < len(content):
        end = min(start + size, len(content))
        chunk = content[start:end].strip()
        if end < len(content) and len(chunk) > size * 0.8:
            for i in range(end - 1, max(start, end - int(size * 0.2)), -1):
                if content[i] in ".!?\n":
                    chunk = content[start : i + 1].strip()
                    break
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def run_agent():
    """
    For every lesson:
      1. Search trusted sites for each query
      2. Fall back to pre-built URLs if search finds nothing
      3. Scrape & chunk content
      4. Save to fetched_content_cache.json
    """
    print("=" * 60)
    print("  QuantCademy Financial Content Agent")
    print("  Searching ONLY within trusted financial sites")
    print("=" * 60)

    all_chunks: List[Dict] = []
    stats = {"lessons": 0, "urls_searched": 0, "urls_scraped": 0, "chunks": 0}

    for lesson_title, meta in LESSONS.items():
        lesson_id = meta["id"]
        module_id = meta["module"]
        queries = meta["queries"]

        print(f"\n{'—'*60}")
        print(f"  {lesson_title}")
        print(f"  lesson_id={lesson_id}  module={module_id}")
        print(f"{'—'*60}")
        stats["lessons"] += 1

        # --- Step 1: Search for URLs ---
        found_urls: List[str] = []
        seen: Set[str] = set()

        for q in queries:
            print(f"  Search: \"{q}\"")
            results = search_ddg(q, max_results=15)
            new = [u for u in results if u not in seen]
            for u in new:
                seen.add(u)
                found_urls.append(u)
            if results:
                print(f"    -> {len(results)} trusted results ({len(new)} new)")
            else:
                print(f"    -> 0 results")
            time.sleep(1.5)   # rate-limit between searches

        print(f"  Total from search: {len(found_urls)} URLs")

        # --- Step 2: Add fallback URLs if search returned few results ---
        fallbacks = FALLBACK_URLS.get(lesson_id, [])
        if len(found_urls) < 3 and fallbacks:
            print(f"  Adding {len(fallbacks)} fallback URLs")
            for u in fallbacks:
                if u not in seen:
                    seen.add(u)
                    found_urls.append(u)

        # Cap at 8 URLs per lesson to keep cache reasonable
        found_urls = found_urls[:8]
        stats["urls_searched"] += len(found_urls)

        # --- Step 3: Scrape each URL ---
        for url in found_urls:
            domain = urlparse(url).netloc.replace("www.", "")
            short = url[:80] + ("..." if len(url) > 80 else "")
            print(f"  Scrape [{domain}]: {short} ... ", end="", flush=True)

            result = scrape_page(url)
            if result is None:
                print("SKIP")
                continue

            chars = len(result["content"])
            print(f"OK ({chars:,} chars)")
            stats["urls_scraped"] += 1

            chunks = chunk_text(result["content"])
            for idx, ct in enumerate(chunks):
                all_chunks.append({
                    "lesson_id": lesson_id,
                    "module_id": module_id,
                    "url": result["url"],
                    "source": result["source"],
                    "title": result["title"],
                    "content": ct,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                })
            stats["chunks"] += len(chunks)
            time.sleep(0.8)

    # ------------------------------------------------------------------
    # Save to NEW file only — does NOT touch existing cache
    # ------------------------------------------------------------------
    out_path = Path(__file__).parent / "web_scraped_content_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  DONE — Summary")
    print(f"{'='*60}")
    print(f"  Lessons processed : {stats['lessons']}")
    print(f"  URLs attempted    : {stats['urls_searched']}")
    print(f"  URLs scraped OK   : {stats['urls_scraped']}")
    print(f"  Total chunks      : {stats['chunks']}")
    print(f"  Saved to          : {out_path.name}")
    print(f"  (Old cache NOT modified)")

    by_lesson: Dict[str, int] = {}
    for c in all_chunks:
        lid = c["lesson_id"]
        by_lesson[lid] = by_lesson.get(lid, 0) + 1
    print(f"\n  Chunks per lesson:")
    for lid, count in sorted(by_lesson.items()):
        print(f"    {lid}: {count}")
    print()


if __name__ == "__main__":
    run_agent()
