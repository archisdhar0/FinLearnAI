"""
Sentiment Analyzer - FinBERT-based sentiment analysis for financial news.

FinBERT is a pre-trained NLP model specifically designed for financial text.
It classifies text as: positive, negative, or neutral.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0-1
    scores: Dict[str, float]  # All class probabilities


@dataclass  
class StockSentiment:
    """Aggregated sentiment for a stock based on multiple articles."""
    ticker: str
    overall_sentiment: str
    overall_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float
    num_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    articles: List[SentimentResult]


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    
    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Apple reports record quarterly earnings")
        print(result.sentiment, result.confidence)
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detects if not specified.
        """
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the best available device."""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def load_model(self):
        """Load FinBERT model and tokenizer."""
        if self._loaded:
            return
            
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            print(f"[Sentiment] Loading FinBERT on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print("[Sentiment] FinBERT loaded successfully")
            
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )
        except Exception as e:
            print(f"[Sentiment] Failed to load model: {e}")
            raise
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze (headline, description, etc.)
            
        Returns:
            SentimentResult with sentiment label and confidence
        """
        if not self._loaded:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # FinBERT labels: ['positive', 'negative', 'neutral']
        labels = ['positive', 'negative', 'neutral']
        scores = {label: float(prob) for label, prob in zip(labels, probs)}
        
        # Get predicted class
        pred_idx = np.argmax(probs)
        sentiment = labels[pred_idx]
        confidence = float(probs[pred_idx])
        
        return SentimentResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            sentiment=sentiment,
            confidence=confidence,
            scores=scores
        )
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of SentimentResult objects
        """
        if not self._loaded:
            self.load_model()
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
            labels = ['positive', 'negative', 'neutral']
            
            for j, text in enumerate(batch):
                scores = {label: float(prob) for label, prob in zip(labels, probs[j])}
                pred_idx = np.argmax(probs[j])
                
                results.append(SentimentResult(
                    text=text[:200] + "..." if len(text) > 200 else text,
                    sentiment=labels[pred_idx],
                    confidence=float(probs[j][pred_idx]),
                    scores=scores
                ))
        
        return results
    
    def analyze_stock(
        self, 
        ticker: str, 
        articles: List[Dict],
        use_title_only: bool = False
    ) -> StockSentiment:
        """
        Analyze sentiment for a stock based on its news articles.
        
        Args:
            ticker: Stock ticker symbol
            articles: List of article dicts with 'title' and optionally 'description'
            use_title_only: If True, only analyze titles (faster, less accurate)
            
        Returns:
            StockSentiment with aggregated sentiment
        """
        if not articles:
            return StockSentiment(
                ticker=ticker,
                overall_sentiment='neutral',
                overall_score=0.0,
                confidence=0.0,
                num_articles=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                articles=[]
            )
        
        # Prepare texts for analysis
        texts = []
        for article in articles:
            if use_title_only:
                text = article.get('title', '')
            else:
                title = article.get('title', '')
                desc = article.get('description', '')
                text = f"{title}. {desc}" if desc else title
            if text:
                texts.append(text)
        
        if not texts:
            return StockSentiment(
                ticker=ticker,
                overall_sentiment='neutral',
                overall_score=0.0,
                confidence=0.0,
                num_articles=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                articles=[]
            )
        
        # Analyze all texts
        results = self.analyze_batch(texts)
        
        # Count sentiments
        positive_count = sum(1 for r in results if r.sentiment == 'positive')
        negative_count = sum(1 for r in results if r.sentiment == 'negative')
        neutral_count = sum(1 for r in results if r.sentiment == 'neutral')
        
        # Calculate overall score (-1 to +1)
        # Weight by confidence
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = result.confidence
            if result.sentiment == 'positive':
                total_score += weight
            elif result.sentiment == 'negative':
                total_score -= weight
            # neutral contributes 0
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall sentiment
        if overall_score > 0.15:
            overall_sentiment = 'positive'
        elif overall_score < -0.15:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in results])
        
        return StockSentiment(
            ticker=ticker,
            overall_sentiment=overall_sentiment,
            overall_score=round(overall_score, 3),
            confidence=round(avg_confidence, 3),
            num_articles=len(results),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            articles=results
        )
    
    def get_sentiment_signal(self, stock_sentiment: StockSentiment) -> Tuple[str, int]:
        """
        Convert sentiment to a trading signal.
        
        Args:
            stock_sentiment: StockSentiment result
            
        Returns:
            Tuple of (signal: 'BULLISH'/'BEARISH'/'NEUTRAL', strength: 0-100)
        """
        score = stock_sentiment.overall_score
        confidence = stock_sentiment.confidence
        
        # Calculate strength (0-100)
        strength = int(abs(score) * confidence * 100)
        strength = min(strength, 100)
        
        if score > 0.2 and confidence > 0.5:
            return ('BULLISH', strength)
        elif score < -0.2 and confidence > 0.5:
            return ('BEARISH', strength)
        else:
            return ('NEUTRAL', strength)


# Test function
if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    
    analyzer = SentimentAnalyzer()
    
    # Test single analysis
    test_texts = [
        "Apple reports record quarterly earnings, stock surges 5%",
        "Company faces massive lawsuit, investors flee",
        "Market remains stable amid economic uncertainty",
        "Tesla announces major partnership with leading automaker",
        "Bank of America cuts workforce by 10,000 employees"
    ]
    
    print("\n=== Single Text Analysis ===")
    for text in test_texts:
        result = analyzer.analyze(text)
        emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}[result.sentiment]
        print(f"\n{emoji} {result.sentiment.upper()} ({result.confidence:.1%})")
        print(f"   \"{text}\"")
        print(f"   Scores: +{result.scores['positive']:.2f} / -{result.scores['negative']:.2f} / ={result.scores['neutral']:.2f}")
    
    # Test stock analysis
    print("\n\n=== Stock Sentiment Analysis ===")
    fake_articles = [
        {"title": "Apple beats earnings expectations", "description": "Revenue up 15% year over year"},
        {"title": "iPhone sales disappoint in China", "description": "Competition from local brands intensifies"},
        {"title": "Apple announces new AI features", "description": "Siri gets major upgrade with GPT integration"},
    ]
    
    stock_result = analyzer.analyze_stock("AAPL", fake_articles)
    signal, strength = analyzer.get_sentiment_signal(stock_result)
    
    print(f"\nTicker: {stock_result.ticker}")
    print(f"Overall: {stock_result.overall_sentiment} (score: {stock_result.overall_score:+.2f})")
    print(f"Confidence: {stock_result.confidence:.1%}")
    print(f"Articles: {stock_result.num_articles} (+{stock_result.positive_count}/-{stock_result.negative_count}/={stock_result.neutral_count})")
    print(f"Signal: {signal} (strength: {strength}%)")
