"""
News Fetcher - Polygon.io News API wrapper for stock news retrieval.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NewsArticle:
    """Represents a single news article."""
    title: str
    description: str
    published: datetime
    source: str
    url: str
    ticker: str
    keywords: List[str]


class NewsFetcher:
    """
    Fetches stock news from Polygon.io API.
    
    Usage:
        fetcher = NewsFetcher(api_key="your_key")
        articles = fetcher.get_news("AAPL", limit=10)
    """
    
    BASE_URL = "https://api.polygon.io/v2/reference/news"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the news fetcher.
        
        Args:
            api_key: Polygon.io API key. If not provided, reads from POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var or pass api_key.")
    
    def get_news(
        self, 
        ticker: str, 
        limit: int = 10,
        days_back: int = 7,
        order: str = "desc"
    ) -> List[NewsArticle]:
        """
        Fetch news articles for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            limit: Maximum number of articles to return (max 1000)
            days_back: How many days back to search
            order: Sort order - "desc" (newest first) or "asc"
            
        Returns:
            List of NewsArticle objects
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            "ticker": ticker.upper(),
            "limit": min(limit, 1000),
            "order": order,
            "published_utc.gte": start_date.strftime("%Y-%m-%d"),
            "published_utc.lte": end_date.strftime("%Y-%m-%d"),
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("results", []):
                try:
                    # Parse published date
                    pub_str = item.get("published_utc", "")
                    if pub_str:
                        # Handle different date formats
                        try:
                            published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        except:
                            published = datetime.now()
                    else:
                        published = datetime.now()
                    
                    article = NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        published=published,
                        source=item.get("publisher", {}).get("name", "Unknown"),
                        url=item.get("article_url", ""),
                        ticker=ticker.upper(),
                        keywords=item.get("keywords", [])
                    )
                    articles.append(article)
                except Exception as e:
                    print(f"[News] Error parsing article: {e}")
                    continue
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"[News] API request failed for {ticker}: {e}")
            return []
    
    def get_news_batch(
        self, 
        tickers: List[str], 
        limit_per_ticker: int = 5,
        days_back: int = 7
    ) -> Dict[str, List[NewsArticle]]:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            limit_per_ticker: Max articles per ticker
            days_back: How many days back to search
            
        Returns:
            Dict mapping ticker -> list of articles
        """
        results = {}
        for ticker in tickers:
            articles = self.get_news(ticker, limit=limit_per_ticker, days_back=days_back)
            results[ticker] = articles
        return results
    
    def get_market_news(self, limit: int = 20, days_back: int = 3) -> List[NewsArticle]:
        """
        Fetch general market news (not ticker-specific).
        
        Args:
            limit: Maximum number of articles
            days_back: How many days back to search
            
        Returns:
            List of NewsArticle objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            "limit": min(limit, 100),
            "order": "desc",
            "published_utc.gte": start_date.strftime("%Y-%m-%d"),
            "published_utc.lte": end_date.strftime("%Y-%m-%d"),
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("results", []):
                try:
                    pub_str = item.get("published_utc", "")
                    try:
                        published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    except:
                        published = datetime.now()
                    
                    # Get first ticker if available
                    tickers = item.get("tickers", [])
                    ticker = tickers[0] if tickers else "MARKET"
                    
                    article = NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        published=published,
                        source=item.get("publisher", {}).get("name", "Unknown"),
                        url=item.get("article_url", ""),
                        ticker=ticker,
                        keywords=item.get("keywords", [])
                    )
                    articles.append(article)
                except Exception as e:
                    continue
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"[News] Market news request failed: {e}")
            return []


# Test function
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load from parent's .env
    load_dotenv("../.env")
    load_dotenv("../../quantcademy-app/.env")
    
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY to test")
        exit(1)
    
    fetcher = NewsFetcher(api_key)
    
    print("\n=== AAPL News ===")
    articles = fetcher.get_news("AAPL", limit=5)
    for article in articles:
        print(f"\n📰 {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Date: {article.published.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\nTotal articles: {len(articles)}")
