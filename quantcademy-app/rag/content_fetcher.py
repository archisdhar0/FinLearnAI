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
    "What is investing?": "what_is_investing",
    "What you're actually buying": "what_youre_actually_buying",
    "How Markets Function": "how_markets_function",
    "Time and Compounding": "time_compounding",
    "The Basics of Risk": "basics_of_risk",
    "Accounts and Setup": "accounts_setup",
    "First Time Investor Mindset": "first_time_mindset",
}

MODULE_ID = "foundations"


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
        
        # Limit content length (keep first 10000 chars to avoid huge chunks)
        if len(content) > 10000:
            content = content[:10000] + "... [Content truncated]"
        
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
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    lessons = {}
    current_lesson = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a lesson title (ends with colon)
        if line.endswith(':'):
            current_lesson = line.rstrip(':')
            lessons[current_lesson] = []
        # Check if it's a URL
        elif line.startswith('http'):
            if current_lesson:
                lessons[current_lesson].append(line)
    
    return lessons


def chunk_content(content: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split content into chunks with overlap for better context preservation.
    """
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(content):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start, end - 200), -1):
                if content[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward with overlap
        start = end - overlap
        if start >= len(content):
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
            print(f"Warning: No lesson_id mapping for '{lesson_title}'")
            continue
        
        print(f"\nFetching content for: {lesson_title} ({lesson_id})")
        
        for url in urls:
            print(f"  Fetching: {url}")
            result = fetch_url_content(url)
            
            if result:
                # Split into chunks
                chunks = chunk_content(result['content'])
                
                for idx, chunk_text in enumerate(chunks):
                    all_content.append({
                        'lesson_id': lesson_id,
                        'module_id': MODULE_ID,
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
