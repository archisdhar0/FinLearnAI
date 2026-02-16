"""
Content Fetcher for QuantCademy Knowledge Base
Fetches content from URLs in links.md and stores it in the knowledge base.
"""

import re
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time

# Mapping of lesson titles to lesson IDs from learning_modules.py
LESSON_MAPPING = {
    # Foundation module
    "What is investing?": "what_is_investing",
    "What you're actually buying": "what_youre_actually_buying",
    "How Markets Function": "how_markets_function",
    "Time and Compounding": "time_compounding",
    "The Basics of Risk": "basics_of_risk",
    "Accounts and Setup": "accounts_setup",
    "First Time Investor Mindset": "first_time_mindset",
    # Investor Insight module
    "What Moves Markets": "what_moves_markets",
    "Investor Psychology": "investor_psychology",
    "Hype vs. Fundamentals": "hype_vs_fundamentals",
    "Types of Investing": "types_of_investing",
    "Risk and Portfolio Thinking": "risk_portfolio_thinking",
    "Reading Base Market Signals": "reading_market_signals",
    # Applied Investing module
    "Costs, Fees, and Taxes": "costs_fees_taxes",
    "What to do in a market crash": "what_do_in_crash",
    "Setting a Long term structure": "setting_long_term_structure",
    "Realistic Expectations About Returns": "realistic_expectations",
}

# Mapping of lesson IDs to module IDs
LESSON_TO_MODULE = {
    # Foundation module
    "what_is_investing": "foundations",
    "what_youre_actually_buying": "foundations",
    "how_markets_function": "foundations",
    "time_compounding": "foundations",
    "basics_of_risk": "foundations",
    "accounts_setup": "foundations",
    "first_time_mindset": "foundations",
    # Investor Insight module
    "what_moves_markets": "investor_insight",
    "investor_psychology": "investor_insight",
    "hype_vs_fundamentals": "investor_insight",
    "types_of_investing": "investor_insight",
    "risk_portfolio_thinking": "investor_insight",
    "reading_market_signals": "investor_insight",
    # Applied Investing module
    "costs_fees_taxes": "applied_investing",
    "what_do_in_crash": "applied_investing",
    "setting_long_term_structure": "applied_investing",
    "realistic_expectations": "applied_investing",
}


def extract_domain_name(url: str) -> str:
    """Extract a clean domain name from URL for source attribution."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    
    # Map common domains to source names
    domain_map = {
        "wellsfargo.com": "Wells Fargo",
        "morganstanley.com": "Morgan Stanley",
        "usbank.com": "U.S. Bank",
        "investopedia.com": "Investopedia",
        "andrewtemte.com": "Andrew Temte",
        "usalearning.gov": "FinRed",
        "westernsouthern.com": "Western Southern",
        "theweek.com": "The Week",
        "rasmussen.edu": "Rasmussen University",
        "stackexchange.com": "Stack Exchange",
        "vanguard.com": "Vanguard",
        "dfi.wa.gov": "DFI Washington",
        "nasdaq.com": "NASDAQ",
        "betterexplained.com": "Better Explained",
        "econlib.org": "EconLib",
        "disnat.com": "Disnat",
        "stlouisfed.org": "St. Louis Fed",
        "wealthify.com": "Wealthify",
        "investor.gov": "SEC Investor.gov",
        "visualcapitalist.com": "Visual Capitalist",
        "burrowscap.com": "Burrows Capital",
        "fmtrust.bank": "FM Trust Bank",
        "bluemountaininvest.com": "Blue Mountain",
        "sbisecurities.in": "SBI Securities",
        "finra.org": "FINRA",
        "fidelity.com": "Fidelity",
        "intuit.com": "TurboTax",
        "sec.gov": "SEC",
        "khanacademy.org": "Khan Academy",
        "riverbridge.com": "Riverbridge",
        "corporatefinanceinstitute.com": "CFI",
        "arqwealth.com": "ARQ Wealth",
        "ls.berkeley.edu": "UC Berkeley",
        "thedecisionlab.com": "The Decision Lab",
        "sofi.com": "SoFi",
        "beutelgoodman.com": "Beutel Goodman",
        "financialresearch.gov": "OFR",
        "leelynsmith.com": "Lee Lyn Smith",
        "kayne.com": "Kayne Anderson",
        "whittiertrust.com": "Whittier Trust",
        "cfcapllc.com": "CF Capital",
        "ishares.com": "iShares",
        "schwab.com": "Charles Schwab",
        "nl.vanguard": "Vanguard",
        "burlingbank.com": "Burling Bank",
        "bankrate.com": "Bankrate",
        "experian.com": "Experian",
        "smartasset.com": "SmartAsset",
        "cambridgeassociates.com": "Cambridge Associates",
        "capitalgroup.com": "Capital Group",
        "fnbo.com": "First National Bank",
        "morningstar.com": "Morningstar",
        "mfs.com": "MFS",
        "heygotrade.com": "HeyGoTrade",
        "researchaffiliates.com": "Research Affiliates",
        "nerdwallet.com": "NerdWallet",
        "ajbell.co.uk": "AJ Bell",
        "financialsuccess.fsu.edu": "FSU Financial Success",
    }
    
    for key, value in domain_map.items():
        if key in domain:
            return value
    
    # Fallback: capitalize domain
    return domain.split(".")[0].replace("-", " ").title()


def fetch_url_content(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Fetch content from a URL and extract main text.
    Returns dict with 'content', 'title', and 'source' or None if failed.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', 'body']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return None
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        # Get title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else extract_domain_name(url)
        
        # No need to truncate - sliding window chunking will handle long content
        
        return {
            'content': content,
            'title': title_text,
            'source': extract_domain_name(url),
            'url': url
        }
    
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def parse_links_file(file_path: str) -> Dict[str, List[str]]:
    """
    Parse links.md file and return dict mapping lesson titles to URLs.
    Handles both main lessons (ending with colon or matching known titles) and subsections.
    Subsections are grouped under their parent lesson.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    lessons = {}
    current_lesson = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a URL
        if line.startswith('http'):
            if current_lesson and current_lesson in LESSON_MAPPING:
                lessons[current_lesson].append(line)
            elif current_lesson:
                # This is a URL under a subsection - find the parent lesson
                # Check if current_lesson is a subsection of a known lesson
                parent_lesson = None
                for known_lesson in LESSON_MAPPING.keys():
                    # Simple heuristic: if subsection text appears in known lesson or vice versa
                    if known_lesson.lower() in current_lesson.lower() or current_lesson.lower() in known_lesson.lower():
                        parent_lesson = known_lesson
                        break
                
                if parent_lesson:
                    if parent_lesson not in lessons:
                        lessons[parent_lesson] = []
                    lessons[parent_lesson].append(line)
            continue
        
        # Check if it's a lesson title (ends with colon)
        if line.endswith(':'):
            potential_lesson = line.rstrip(':')
            if potential_lesson in LESSON_MAPPING:
                current_lesson = potential_lesson
                if current_lesson not in lessons:
                    lessons[current_lesson] = []
            else:
                # Unknown section with colon - might be a subsection, keep current_lesson
                pass
        # Check if it matches a known lesson title (without colon)
        elif line in LESSON_MAPPING:
            current_lesson = line
            if current_lesson not in lessons:
                lessons[current_lesson] = []
        # Otherwise, it might be a subsection header - keep current_lesson if it exists
        # (URLs under subsections will be handled above)
    
    return lessons


def chunk_content(content: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
    """
    Split content into chunks using a sliding window approach.
    
    Uses a true sliding window that moves across the text with overlap,
    ensuring all content is covered and context is preserved across boundaries.
    
    Args:
        content: The text content to chunk
        chunk_size: Target size for each chunk (characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    step_size = chunk_size - overlap  # How much to move the window forward
    
    while start < len(content):
        # Calculate end position
        end = min(start + chunk_size, len(content))
        
        # Extract chunk
        chunk = content[start:end].strip()
        
        # If this is not the last chunk, try to break at a sentence boundary
        if end < len(content) and len(chunk) > chunk_size * 0.8:  # Only if chunk is substantial
            # Look backwards from end for sentence boundary (within last 20% of chunk)
            search_start = max(start, end - int(chunk_size * 0.2))
            for i in range(end - 1, search_start, -1):
                if content[i] in '.!?\n':
                    # Found sentence boundary - adjust end
                    end = i + 1
                    chunk = content[start:end].strip()
                    break
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Move window forward by step_size (sliding window)
        start += step_size
        
        # If we're at the end but haven't covered everything, ensure we get the last bit
        if start >= len(content) and end < len(content):
            # Get remaining content
            remaining = content[end:].strip()
            if remaining:
                chunks.append(remaining)
            break
    
    return chunks


def fetch_all_content(links_file: str = "rag/links.md", delay: float = 1.0) -> List[Dict]:
    """
    Fetch all content from links.md and return list of content dicts.
    Each dict has: lesson_id, module_id, url, source, title, content, chunks
    """
    lessons = parse_links_file(links_file)
    all_content = []
    
    for lesson_title, urls in lessons.items():
        lesson_id = LESSON_MAPPING.get(lesson_title)
        if not lesson_id:
            print(f"Warning: No lesson_id mapping for '{lesson_title}' - skipping")
            continue
        
        # Get module_id from lesson_id
        module_id = LESSON_TO_MODULE.get(lesson_id, "foundations")
        
        print(f"\nFetching content for: {lesson_title} ({lesson_id}, module: {module_id})")
        
        for url in urls:
            print(f"  Fetching: {url}")
            result = fetch_url_content(url)
            
            if result:
                # Split into chunks
                chunks = chunk_content(result['content'])
                
                for idx, chunk_text in enumerate(chunks):
                    all_content.append({
                        'lesson_id': lesson_id,
                        'module_id': module_id,
                        'url': result['url'],
                        'source': result['source'],
                        'title': result['title'],
                        'content': chunk_text,
                        'chunk_index': idx,
                        'total_chunks': len(chunks)
                    })
                
                print(f"    ✓ Fetched {len(chunks)} chunk(s)")
            else:
                print(f"    ✗ Failed to fetch")
            
            # Be polite - delay between requests
            time.sleep(delay)
    
    return all_content


if __name__ == "__main__":
    # Test the fetcher
    print("Testing content fetcher...")
    content_list = fetch_all_content()
    print(f"\nTotal chunks fetched: {len(content_list)}")
    
    # Print summary
    by_lesson = {}
    for item in content_list:
        lesson = item['lesson_id']
        by_lesson[lesson] = by_lesson.get(lesson, 0) + 1
    
    print("\nChunks by lesson:")
    for lesson, count in by_lesson.items():
        print(f"  {lesson}: {count} chunks")
