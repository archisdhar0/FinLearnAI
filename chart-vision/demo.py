"""
Chart Vision Demo - Compare Price-Based vs Image-Based S/R Detection
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='S/R Detection Demo')
    parser.add_argument('--ticker', '-t', type=str, default='SPY', 
                        help='Stock ticker to analyze (default: SPY)')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 CHART VISION - S/R DETECTION COMPARISON")
    print("   Price-Based vs Computer Vision (Image-Based)")
    print("=" * 60)
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        print("\n⚠️  No Polygon API key - using simulated data")
        demo_comparison_simulated()
    else:
        demo_comparison_real(api_key, ticker=args.ticker)


def demo_comparison_simulated():
    """Compare both methods using simulated data."""
    from models.sr_detector import SupportResistanceDetector
    from utils.chart_generator import draw_candlestick_chart
    import pandas as pd
    
    print("\n1️⃣ Generating simulated price data with clear S/R levels...")
    
    # Generate price data with obvious S/R
    np.random.seed(42)
    n = 60
    
    # Create price movement with S/R behavior
    prices = [150.0]
    for i in range(1, n):
        change = np.random.randn() * 1.5
        
        # Resistance at ~170 (selling pressure)
        if prices[-1] > 168:
            change -= 2.0
        # Support at ~145 (buying pressure)
        if prices[-1] < 147:
            change += 2.0
            
        prices.append(max(140, min(175, prices[-1] + change)))
    
    closes = np.array(prices)
    highs = closes + np.abs(np.random.randn(n)) * 1.5
    lows = closes - np.abs(np.random.randn(n)) * 1.5
    opens = closes + np.random.randn(n) * 0.5
    
    # Create DataFrame for chart generation
    dates = pd.date_range('2024-01-01', periods=n)
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    run_comparison(df, 'SIMULATED')


def demo_comparison_real(api_key: str, ticker: str = 'SPY'):
    """Compare both methods using real stock data."""
    from utils.chart_generator import PolygonDataFetcher
    from datetime import datetime, timedelta
    
    print(f"\n1️⃣ Fetching real {ticker} data from Polygon...")
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    df = fetcher.get_daily_bars(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if df is None or len(df) < 30:
        print(f"   ⚠️ Could not fetch enough {ticker} data, using simulated")
        demo_comparison_simulated()
        return
    
    print(f"   ✅ Got {len(df)} days of {ticker} data")
    
    run_comparison(df, ticker)


def run_comparison(df, ticker: str):
    """Run both detection methods and compare."""
    from models.sr_detector import SupportResistanceDetector
    from utils.chart_generator import draw_candlestick_chart
    import cv2
    
    output_dir = Path('data/demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== METHOD 1: PRICE-BASED ====================
    print("\n" + "=" * 60)
    print("2️⃣ METHOD 1: PRICE-BASED S/R Detection")
    print("   (Using actual OHLC numbers)")
    print("=" * 60)
    
    detector = SupportResistanceDetector()
    
    price_results = detector.detect_from_prices(
        highs=df['High'].values,
        lows=df['Low'].values,
        closes=df['Close'].values,
        window=5,
        num_levels=4
    )
    
    print("\n   🟢 SUPPORT (Price-Based):")
    for level in price_results['support']:
        print(f"      ${level['price']:.2f} (touched {level['touches']}x)")
    
    print("\n   🔴 RESISTANCE (Price-Based):")
    for level in price_results['resistance']:
        print(f"      ${level['price']:.2f} (touched {level['touches']}x)")
    
    # ==================== GENERATE CHART IMAGE ====================
    print("\n" + "=" * 60)
    print("3️⃣ Generating candlestick chart image...")
    print("=" * 60)
    
    chart_path = output_dir / f'{ticker}_candlestick.png'
    draw_candlestick_chart(df, str(chart_path), figsize=(10, 6), dpi=150)
    print(f"   ✅ Saved: {chart_path}")
    
    # ==================== METHOD 2: IMAGE-BASED (CV) ====================
    print("\n" + "=" * 60)
    print("4️⃣ METHOD 2: IMAGE-BASED S/R Detection (Computer Vision)")
    print("   (Analyzing chart pixels)")
    print("=" * 60)
    
    image_results = detector.detect_from_image(str(chart_path))
    
    if 'error' in image_results:
        print(f"   ⚠️ Error: {image_results['error']}")
    else:
        print("\n   🟢 SUPPORT (Image-Based):")
        for level in image_results.get('support', []):
            y = level['y'] if isinstance(level, dict) else level
            strength = level.get('strength', 0) if isinstance(level, dict) else 0
            print(f"      y={y}px (strength: {strength:.0f})")
        
        print("\n   🔴 RESISTANCE (Image-Based):")
        for level in image_results.get('resistance', []):
            y = level['y'] if isinstance(level, dict) else level
            strength = level.get('strength', 0) if isinstance(level, dict) else 0
            print(f"      y={y}px (strength: {strength:.0f})")
    
    # ==================== SIDE-BY-SIDE VISUALIZATION ====================
    print("\n" + "=" * 60)
    print("5️⃣ Creating side-by-side comparison visualization...")
    print("=" * 60)
    
    create_comparison_visualization(
        df, chart_path, price_results, image_results, ticker, output_dir
    )


def create_comparison_visualization(df, chart_path, price_results, image_results, ticker, output_dir):
    """Create a side-by-side comparison of both methods."""
    import cv2
    
    # Load the chart image
    chart_img = cv2.imread(str(chart_path))
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_BGR2RGB)
    img_height, img_width = chart_img.shape[:2]
    
    # Get price range for coordinate conversion
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_range = price_max - price_min
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # ==================== LEFT: Price-Based ====================
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    
    # Plot price
    ax1.fill_between(range(len(df)), df['Low'].values, df['High'].values, 
                     alpha=0.3, color='#6366f1')
    ax1.plot(df['Close'].values, color='#6366f1', linewidth=2)
    
    # Draw S/R from price-based method
    for level in price_results['support'][:3]:
        ax1.axhline(y=level['price'], color='#22c55e', linestyle='--', 
                    linewidth=2, alpha=0.8)
        ax1.text(len(df)+1, level['price'], f"${level['price']:.0f}", 
                color='#22c55e', fontsize=10, va='center', fontweight='bold')
    
    for level in price_results['resistance'][:3]:
        ax1.axhline(y=level['price'], color='#ef4444', linestyle='--',
                    linewidth=2, alpha=0.8)
        ax1.text(len(df)+1, level['price'], f"${level['price']:.0f}",
                color='#ef4444', fontsize=10, va='center', fontweight='bold')
    
    ax1.set_title('METHOD 1: Price-Based\n(Analyzes OHLC numbers)', 
                  fontsize=12, fontweight='bold', color='#22c55e')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # ==================== CENTER: Chart Image with CV Detection ====================
    ax2 = axes[1]
    ax2.imshow(chart_img)
    
    # Draw S/R from image-based method
    for level in image_results.get('support', [])[:3]:
        y = level['y'] if isinstance(level, dict) else level
        ax2.axhline(y=y, color='#22c55e', linestyle='--', linewidth=3, alpha=0.8)
    
    for level in image_results.get('resistance', [])[:3]:
        y = level['y'] if isinstance(level, dict) else level
        ax2.axhline(y=y, color='#ef4444', linestyle='--', linewidth=3, alpha=0.8)
    
    ax2.set_title('METHOD 2: Computer Vision\n(Analyzes chart pixels)', 
                  fontsize=12, fontweight='bold', color='#f59e0b')
    ax2.axis('off')
    
    # ==================== RIGHT: Combined Overlay ====================
    ax3 = axes[2]
    ax3.imshow(chart_img)
    
    # Convert price-based levels to image y-coordinates
    # Note: In images, y=0 is TOP, so we need to invert
    margin_top = img_height * 0.1  # Approximate chart margins
    margin_bottom = img_height * 0.1
    chart_height = img_height - margin_top - margin_bottom
    
    def price_to_y(price):
        """Convert price to image y-coordinate."""
        normalized = (price - price_min) / price_range
        return margin_top + chart_height * (1 - normalized)  # Inverted
    
    # Draw price-based (solid lines)
    for level in price_results['support'][:3]:
        y = price_to_y(level['price'])
        ax3.axhline(y=y, color='#22c55e', linestyle='-', linewidth=3, alpha=0.9)
        ax3.text(10, y-10, f"Price: ${level['price']:.0f}", color='#22c55e', 
                fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    for level in price_results['resistance'][:3]:
        y = price_to_y(level['price'])
        ax3.axhline(y=y, color='#ef4444', linestyle='-', linewidth=3, alpha=0.9)
        ax3.text(10, y-10, f"Price: ${level['price']:.0f}", color='#ef4444',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Draw image-based (dashed lines)
    for level in image_results.get('support', [])[:3]:
        y = level['y'] if isinstance(level, dict) else level
        ax3.axhline(y=y, color='#86efac', linestyle=':', linewidth=2, alpha=0.8)
        ax3.text(img_width-100, y-10, f"CV: y={y}", color='#86efac', fontsize=8)
    
    for level in image_results.get('resistance', [])[:3]:
        y = level['y'] if isinstance(level, dict) else level
        ax3.axhline(y=y, color='#fca5a5', linestyle=':', linewidth=2, alpha=0.8)
        ax3.text(img_width-100, y-10, f"CV: y={y}", color='#fca5a5', fontsize=8)
    
    ax3.set_title('COMBINED: Both Methods\n(Solid=Price, Dotted=CV)', 
                  fontsize=12, fontweight='bold', color='#8b5cf6')
    ax3.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#22c55e', linestyle='-', linewidth=2, label='Price-Based Support'),
        Line2D([0], [0], color='#ef4444', linestyle='-', linewidth=2, label='Price-Based Resistance'),
        Line2D([0], [0], color='#86efac', linestyle=':', linewidth=2, label='CV Support'),
        Line2D([0], [0], color='#fca5a5', linestyle=':', linewidth=2, label='CV Resistance'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    
    plt.suptitle(f'{ticker} - S/R Detection: Price-Based vs Computer Vision',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    comparison_path = output_dir / f'{ticker}_sr_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n   ✅ Comparison saved: {comparison_path}")
    
    # ==================== PRINT COMPARISON ====================
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  METHOD           │  WHAT IT USES      │  BEST FOR          │
    ├─────────────────────────────────────────────────────────────┤
    │  Price-Based      │  OHLC numbers      │  ✅ Most accurate   │
    │                   │  from API          │  Exact $ values    │
    ├─────────────────────────────────────────────────────────────┤
    │  Computer Vision  │  Chart image       │  When you only     │
    │  (Image-Based)    │  pixels (RGB/HSV)  │  have a screenshot │
    └─────────────────────────────────────────────────────────────┘
    
    💡 The Price-Based method is more accurate because:
       - Works with exact dollar values
       - No image compression/noise issues
       - Can calculate precise entry/exit targets
    
    💡 Computer Vision is useful when:
       - You only have a chart screenshot
       - Analyzing charts from Twitter/Reddit
       - Building a "chart scanner" tool
    """)


if __name__ == "__main__":
    main()
