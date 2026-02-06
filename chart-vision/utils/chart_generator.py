"""
Chart Generator - Creates labeled chart images from stock data
Uses Polygon.io API for data (requires API key)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from PIL import Image
import time


class PolygonDataFetcher:
    """Fetch stock data from Polygon.io API."""
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Polygon.io API key. If not provided, uses POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Either:\n"
                "  1. Pass api_key parameter\n"
                "  2. Set POLYGON_API_KEY environment variable\n"
                "  3. Create .env file with POLYGON_API_KEY=your_key"
            )
        
        try:
            from polygon import RESTClient
            self.client = RESTClient(self.api_key)
        except ImportError:
            raise ImportError("Install polygon-api-client: pip install polygon-api-client")
    
    def get_daily_bars(
        self, 
        ticker: str, 
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Polygon.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        try:
            bars = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000
            )
            
            if not bars:
                return None
            
            data = []
            for bar in bars:
                data.append({
                    'Date': datetime.fromtimestamp(bar.timestamp / 1000),
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None


def draw_candlestick_chart(
    df: pd.DataFrame,
    save_path: str,
    figsize: tuple = (8, 6),
    dpi: int = 100,
    show_volume: bool = False
) -> bool:
    """
    Draw a candlestick chart using matplotlib (no mplfinance dependency).
    
    Args:
        df: DataFrame with OHLCV data
        save_path: Path to save the image
        figsize: Figure size
        dpi: Resolution
        show_volume: Whether to show volume bars
    
    Returns:
        True if successful
    """
    try:
        # Prepare data
        df = df.reset_index()
        dates = range(len(df))
        
        # Create figure
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Set dark background
        fig.patch.set_facecolor('#1a1a2e')
        ax1.set_facecolor('#1a1a2e')
        
        # Draw candlesticks
        width = 0.6
        width2 = 0.1
        
        up = df[df['Close'] >= df['Open']]
        down = df[df['Close'] < df['Open']]
        
        # Up candles (green)
        up_color = '#22c55e'
        for idx in up.index:
            i = dates[idx]
            o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
            
            # Body
            ax1.add_patch(Rectangle((i - width/2, o), width, c - o, 
                                    facecolor=up_color, edgecolor=up_color))
            # Wick
            ax1.plot([i, i], [l, o], color=up_color, linewidth=1)
            ax1.plot([i, i], [c, h], color=up_color, linewidth=1)
        
        # Down candles (red)
        down_color = '#ef4444'
        for idx in down.index:
            i = dates[idx]
            o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
            
            # Body
            ax1.add_patch(Rectangle((i - width/2, c), width, o - c,
                                    facecolor=down_color, edgecolor=down_color))
            # Wick
            ax1.plot([i, i], [l, c], color=down_color, linewidth=1)
            ax1.plot([i, i], [o, h], color=down_color, linewidth=1)
        
        # Clean up axes (remove labels for training images)
        ax1.set_xlim(-1, len(df))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        
        # Volume bars
        if show_volume:
            ax2.set_facecolor('#1a1a2e')
            colors = [up_color if df.loc[i, 'Close'] >= df.loc[i, 'Open'] else down_color 
                     for i in df.index]
            ax2.bar(dates, df['Volume'], color=colors, width=width)
            ax2.set_xticks([])
            ax2.set_yticks([])
            for spine in ax2.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, facecolor='#1a1a2e', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Error drawing chart: {e}")
        return False


class ChartGenerator:
    """Generate stock chart images with automatic trend labels."""
    
    # Popular tickers for diverse training data
    TICKERS = [
        # Large caps
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'NFLX', 'ADBE', 'CRM',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'GLD',
        # Mid caps
        'SQ', 'ROKU', 'SNAP', 'UBER', 'COIN',
    ]
    
    def __init__(self, output_dir: str = "data/raw", api_key: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_fetcher = PolygonDataFetcher(api_key=api_key)
    
    def download_data(self, ticker: str, years: int = 3) -> pd.DataFrame:
        """Download historical data for a ticker."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        return self.data_fetcher.get_daily_bars(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    def classify_trend(self, prices: np.ndarray, window: int = 20) -> str:
        """
        Classify the trend of a price series.
        
        Returns: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(prices) < window:
            return 'sideways'
        
        # Calculate linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize slope by price level
        norm_slope = slope / np.mean(prices) * 100
        
        # Calculate volatility for sideways detection
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Thresholds
        if norm_slope > 0.15 and volatility < 0.04:
            return 'uptrend'
        elif norm_slope < -0.15 and volatility < 0.04:
            return 'downtrend'
        else:
            return 'sideways'
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 10) -> dict:
        """
        Find support and resistance levels in the data.
        Returns coordinates for labeling.
        """
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local maxima (resistance) and minima (support)
        try:
            from scipy.signal import argrelextrema
            resistance_idx = argrelextrema(highs, np.greater, order=window)[0]
            support_idx = argrelextrema(lows, np.less, order=window)[0]
        except ImportError:
            # Fallback without scipy
            resistance_idx = []
            support_idx = []
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    resistance_idx.append(i)
                if lows[i] == min(lows[i-window:i+window+1]):
                    support_idx.append(i)
            resistance_idx = np.array(resistance_idx)
            support_idx = np.array(support_idx)
        
        # Cluster nearby levels
        def cluster_levels(indices, prices, threshold=0.02):
            if len(indices) == 0:
                return []
            
            levels = prices[indices]
            clusters = []
            used = set()
            
            for i, level in enumerate(levels):
                if i in used:
                    continue
                    
                cluster = [level]
                for j, other in enumerate(levels):
                    if j != i and j not in used:
                        if abs(other - level) / level < threshold:
                            cluster.append(other)
                            used.add(j)
                
                clusters.append(float(np.mean(cluster)))
                used.add(i)
            
            return sorted(clusters)
        
        return {
            'support': cluster_levels(support_idx, lows),
            'resistance': cluster_levels(resistance_idx, highs),
            'support_idx': support_idx.tolist() if hasattr(support_idx, 'tolist') else list(support_idx),
            'resistance_idx': resistance_idx.tolist() if hasattr(resistance_idx, 'tolist') else list(resistance_idx)
        }
    
    def generate_chart_image(
        self, 
        df: pd.DataFrame, 
        save_path: str,
        show_volume: bool = False,
        figsize: tuple = (8, 6),
        dpi: int = 100
    ) -> bool:
        """Generate a candlestick chart image."""
        return draw_candlestick_chart(
            df, save_path, figsize=figsize, dpi=dpi, show_volume=show_volume
        )
    
    def generate_dataset(
        self,
        num_samples_per_class: int = 500,
        window_size: int = 60,  # Days per chart
        tickers: list = None
    ) -> dict:
        """
        Generate a balanced dataset of chart images.
        
        Args:
            num_samples_per_class: Target samples for each trend class
            window_size: Number of trading days per chart
            tickers: List of tickers to use (default: self.TICKERS)
        
        Returns:
            Dictionary with dataset statistics
        """
        tickers = tickers or self.TICKERS
        
        labels = {'uptrend': [], 'downtrend': [], 'sideways': []}
        metadata = []
        
        print(f"Generating dataset with {num_samples_per_class} samples per class...")
        print(f"Using {len(tickers)} tickers, window size: {window_size} days")
        print(f"Output directory: {self.output_dir}")
        
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            # Rate limiting for Polygon free tier
            time.sleep(0.25)
            
            df = self.download_data(ticker, years=3)
            
            if df is None or len(df) < window_size + 10:
                print(f"  Skipping {ticker} - insufficient data")
                continue
            
            # Slide through the data
            for start_idx in range(0, len(df) - window_size, window_size // 2):
                # Check if we have enough samples
                min_samples = min(len(v) for v in labels.values())
                if min_samples >= num_samples_per_class:
                    print("\nReached target samples for all classes!")
                    break
                
                end_idx = start_idx + window_size
                window_df = df.iloc[start_idx:end_idx].copy()
                
                # Classify the trend
                trend = self.classify_trend(window_df['Close'].values)
                
                # Skip if we have enough of this class
                if len(labels[trend]) >= num_samples_per_class:
                    continue
                
                # Generate unique filename
                timestamp = window_df.index[0].strftime('%Y%m%d')
                filename = f"{ticker}_{timestamp}_{trend}_{len(labels[trend])}.png"
                filepath = self.output_dir / filename
                
                # Generate the chart image
                if self.generate_chart_image(window_df, str(filepath)):
                    labels[trend].append(str(filepath))
                    
                    # Find S/R levels for this window
                    sr_levels = self.find_support_resistance(window_df)
                    
                    metadata.append({
                        'filename': filename,
                        'ticker': ticker,
                        'start_date': str(window_df.index[0].date()),
                        'end_date': str(window_df.index[-1].date()),
                        'trend': trend,
                        'support_levels': sr_levels['support'],
                        'resistance_levels': sr_levels['resistance'],
                        'price_range': [float(window_df['Low'].min()), float(window_df['High'].max())]
                    })
                    
                    print(f"  ✓ Generated {filename} ({trend})")
            
            # Early exit if done
            if all(len(v) >= num_samples_per_class for v in labels.values()):
                break
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save labels
        labels_path = self.output_dir / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        stats = {
            'total_samples': sum(len(v) for v in labels.values()),
            'uptrend': len(labels['uptrend']),
            'downtrend': len(labels['downtrend']),
            'sideways': len(labels['sideways']),
            'metadata_path': str(metadata_path),
            'labels_path': str(labels_path)
        }
        
        print(f"\n{'='*50}")
        print(f"✅ Dataset generated!")
        print(f"   Total: {stats['total_samples']} images")
        print(f"   Uptrend: {stats['uptrend']}")
        print(f"   Downtrend: {stats['downtrend']}")
        print(f"   Sideways: {stats['sideways']}")
        print(f"{'='*50}")
        
        return stats


def main():
    """Generate the training dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate chart training data')
    parser.add_argument('--api-key', type=str, help='Polygon.io API key')
    parser.add_argument('--samples', type=int, default=100, help='Samples per class')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        print("❌ Polygon API key required!")
        print("\nOptions:")
        print("  1. Pass --api-key YOUR_KEY")
        print("  2. Set environment variable: export POLYGON_API_KEY=your_key")
        print("\nGet a free API key at: https://polygon.io/")
        return
    
    generator = ChartGenerator(output_dir=args.output, api_key=api_key)
    
    # Generate dataset
    stats = generator.generate_dataset(
        num_samples_per_class=args.samples,
        window_size=60,
    )
    
    print(f"\nDataset saved to: {args.output}/")


if __name__ == "__main__":
    main()
