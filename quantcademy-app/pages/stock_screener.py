"""
Stock Screener Dashboard
Real-time stock analysis with AI-powered signals (S/R + Trend + Sentiment placeholder)
Uses Polygon API for market data and trained CV models for chart analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Add chart-vision to path
CHART_VISION_PATH = Path(__file__).parent.parent.parent / "chart-vision"
sys.path.insert(0, str(CHART_VISION_PATH))

# Try to import required modules
POLYGON_AVAILABLE = False
MODELS_AVAILABLE = False

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torchvision.transforms as transforms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    from train_sr_model_v2 import SRZoneModel
    from train_trend_model_v2 import TrendModelV2, CLASSES as TREND_CLASSES
    
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_IMPORT_ERROR = str(e)

# ============================================================================
# Configuration
# ============================================================================

# Popular stocks for screener
WATCHLIST_STOCKS = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Finance": ["JPM", "BAC", "GS", "V", "MA", "AXP"],
    "Consumer": ["WMT", "HD", "NKE", "SBUX", "MCD", "COST"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE"],
    "Growth": ["TSLA", "NFLX", "CRM", "ADBE", "SQ", "SHOP"],
}

ALL_STOCKS = []
for stocks in WATCHLIST_STOCKS.values():
    ALL_STOCKS.extend(stocks)

NUM_ZONES = 10  # Must match training

# ============================================================================
# Data Fetching
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker: str, api_key: str, days: int = 30) -> pd.DataFrame:
    """Fetch stock data from Polygon API."""
    try:
        client = RESTClient(api_key)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra days for weekends
        
        bars = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
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
        
        # Get last N days
        return df.tail(days)
        
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_stock_quote(ticker: str, api_key: str) -> dict:
    """Fetch latest quote for a stock."""
    try:
        client = RESTClient(api_key)
        
        # Get previous day's close
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        bars = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            limit=5
        )
        
        if not bars or len(bars) < 2:
            return None
        
        latest = bars[-1]
        prev = bars[-2]
        
        change = latest.close - prev.close
        change_pct = (change / prev.close) * 100
        
        return {
            'price': latest.close,
            'change': change,
            'change_pct': change_pct,
            'high': latest.high,
            'low': latest.low,
            'volume': latest.volume,
            'open': latest.open,
            'prev_close': prev.close
        }
        
    except Exception as e:
        return None


# ============================================================================
# Chart Generation
# ============================================================================

def generate_chart_image(df: pd.DataFrame, figsize=(6, 4), dpi=100) -> Image.Image:
    """Generate a candlestick chart image from DataFrame."""
    df = df.reset_index()
    dates = range(len(df))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dark background
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Draw candlesticks
    width = 0.6
    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]
    
    up_color = '#22c55e'
    down_color = '#ef4444'
    
    # Up candles
    for idx in up.index:
        i = dates[idx]
        o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
        ax.add_patch(Rectangle((i - width/2, o), width, c - o, 
                               facecolor=up_color, edgecolor=up_color))
        ax.plot([i, i], [l, o], color=up_color, linewidth=1)
        ax.plot([i, i], [c, h], color=up_color, linewidth=1)
    
    # Down candles
    for idx in down.index:
        i = dates[idx]
        o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
        ax.add_patch(Rectangle((i - width/2, c), width, o - c,
                               facecolor=down_color, edgecolor=down_color))
        ax.plot([i, i], [l, c], color=down_color, linewidth=1)
        ax.plot([i, i], [o, h], color=down_color, linewidth=1)
    
    ax.set_xlim(-1, len(df))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor='#1a1a2e', 
                bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)


# ============================================================================
# AI Model Predictions
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained AI models."""
    models = {}
    
    # S/R Model
    sr_path = CHART_VISION_PATH / "checkpoints" / "sr_zone_model_best.pt"
    if sr_path.exists():
        try:
            model = SRZoneModel(num_zones=NUM_ZONES)
            checkpoint = torch.load(sr_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models['sr'] = model
        except Exception as e:
            st.warning(f"Could not load S/R model: {e}")
    
    # Trend Model
    trend_path = CHART_VISION_PATH / "checkpoints" / "trend_model_v2_best.pt"
    if trend_path.exists():
        try:
            model = TrendModelV2()
            checkpoint = torch.load(trend_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models['trend'] = model
        except Exception as e:
            st.warning(f"Could not load Trend model: {e}")
    
    return models


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)


def analyze_chart(image: Image.Image, models: dict, price_range: tuple) -> dict:
    """Run AI analysis on a chart image."""
    results = {
        'trend': None,
        'trend_confidence': 0,
        'support_zones': [],
        'resistance_zones': [],
        'signal': 'NEUTRAL',
        'signal_strength': 0
    }
    
    if not models:
        return results
    
    img_tensor = preprocess_image(image)
    
    # Trend prediction
    if 'trend' in models:
        with torch.no_grad():
            class_logits, slope_pred = models['trend'](img_tensor)
            probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            
            results['trend'] = TREND_CLASSES[pred_idx]
            results['trend_confidence'] = float(probs[pred_idx])
            results['trend_probs'] = {cls: float(probs[i]) for i, cls in enumerate(TREND_CLASSES)}
    
    # S/R prediction
    if 'sr' in models:
        with torch.no_grad():
            logits = models['sr'](img_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            num_zones = len(probs) // 2
            support_probs = probs[:num_zones]
            resistance_probs = probs[num_zones:]
            
            # Convert zones to price levels
            price_min, price_max = price_range
            price_step = (price_max - price_min) / num_zones
            
            for i, prob in enumerate(support_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    results['support_zones'].append({
                        'price': round(price, 2),
                        'confidence': float(prob)
                    })
            
            for i, prob in enumerate(resistance_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    results['resistance_zones'].append({
                        'price': round(price, 2),
                        'confidence': float(prob)
                    })
    
    # Calculate combined signal
    signal_score = 0
    
    if results['trend'] == 'uptrend':
        signal_score += results['trend_confidence'] * 40
    elif results['trend'] == 'downtrend':
        signal_score -= results['trend_confidence'] * 40
    
    # Near support = bullish, near resistance = bearish
    if results['support_zones']:
        signal_score += 15
    if results['resistance_zones']:
        signal_score -= 15
    
    # Placeholder for sentiment (to be added later)
    # signal_score += sentiment_score * 30
    
    if signal_score > 20:
        results['signal'] = 'BUY'
        results['signal_strength'] = min(signal_score, 100)
    elif signal_score < -20:
        results['signal'] = 'SELL'
        results['signal_strength'] = min(abs(signal_score), 100)
    else:
        results['signal'] = 'HOLD'
        results['signal_strength'] = 50
    
    return results


# ============================================================================
# UI Components
# ============================================================================

def render_stock_card(ticker: str, quote: dict, analysis: dict, chart_image: Image.Image):
    """Render a stock card with quote, chart, and AI analysis."""
    
    # Determine colors
    if quote and quote['change'] >= 0:
        price_color = "#22c55e"
        arrow = "↗"
    else:
        price_color = "#ef4444"
        arrow = "↘"
    
    signal = analysis.get('signal', 'NEUTRAL')
    signal_colors = {
        'BUY': '#22c55e',
        'SELL': '#ef4444',
        'HOLD': '#f59e0b',
        'NEUTRAL': '#6b7280'
    }
    signal_color = signal_colors.get(signal, '#6b7280')
    
    # Card container
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid #3d3d5c;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <h2 style="margin: 0; color: white; font-size: 1.5rem;">{ticker}</h2>
                <p style="margin: 0; color: #9ca3af; font-size: 0.9rem;">
                    {quote.get('volume', 0):,.0f} vol
                </p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; color: {price_color}; font-size: 1.8rem; font-weight: bold;">
                    ${quote.get('price', 0):.2f}
                </p>
                <p style="margin: 0; color: {price_color}; font-size: 1rem;">
                    {arrow} {quote.get('change', 0):+.2f} ({quote.get('change_pct', 0):+.1f}%)
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(chart_image, use_container_width=True)
    
    with col2:
        # AI Signal
        st.markdown(f"""
        <div style="background: {signal_color}22; border: 2px solid {signal_color};
                    border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 1rem;">
            <p style="margin: 0; color: {signal_color}; font-size: 1.5rem; font-weight: bold;">
                {signal}
            </p>
            <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">
                AI Signal ({analysis.get('signal_strength', 0):.0f}% confidence)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trend
        trend = analysis.get('trend', 'unknown')
        trend_conf = analysis.get('trend_confidence', 0) * 100
        trend_icons = {'uptrend': '📈', 'downtrend': '📉', 'sideways': '➡️'}
        trend_colors = {'uptrend': '#22c55e', 'downtrend': '#ef4444', 'sideways': '#f59e0b'}
        
        st.markdown(f"""
        <div style="background: #2d2d44; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
            <p style="margin: 0; color: #9ca3af; font-size: 0.75rem;">TREND</p>
            <p style="margin: 0; color: {trend_colors.get(trend, '#6b7280')}; font-size: 1rem; font-weight: bold;">
                {trend_icons.get(trend, '❓')} {trend.upper()} ({trend_conf:.0f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # S/R Levels
        support = analysis.get('support_zones', [])
        resistance = analysis.get('resistance_zones', [])
        
        if support:
            support_str = ", ".join([f"${s['price']:.2f}" for s in support[:2]])
            st.markdown(f"""
            <div style="background: #22c55e22; border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; color: #22c55e; font-size: 0.85rem;">
                    🟢 Support: {support_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if resistance:
            resist_str = ", ".join([f"${r['price']:.2f}" for r in resistance[:2]])
            st.markdown(f"""
            <div style="background: #ef444422; border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; color: #ef4444; font-size: 0.85rem;">
                    🔴 Resistance: {resist_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment placeholder
        st.markdown("""
        <div style="background: #3d3d5c; border-radius: 8px; padding: 0.5rem; opacity: 0.6;">
            <p style="margin: 0; color: #9ca3af; font-size: 0.75rem;">
                📰 Sentiment: Coming Soon
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_market_overview(api_key: str):
    """Render market overview with major indices."""
    indices = ["SPY", "QQQ", "DIA", "IWM"]
    
    cols = st.columns(len(indices))
    
    for i, ticker in enumerate(indices):
        quote = fetch_stock_quote(ticker, api_key)
        
        if quote:
            color = "#22c55e" if quote['change'] >= 0 else "#ef4444"
            arrow = "↗" if quote['change'] >= 0 else "↘"
            
            with cols[i]:
                st.markdown(f"""
                <div style="background: #2d2d44; border-radius: 12px; padding: 1rem; text-align: center;">
                    <p style="margin: 0; color: #9ca3af; font-size: 0.9rem;">{ticker}</p>
                    <p style="margin: 0; color: white; font-size: 1.3rem; font-weight: bold;">
                        ${quote['price']:.2f}
                    </p>
                    <p style="margin: 0; color: {color}; font-size: 0.9rem;">
                        {arrow} {quote['change_pct']:+.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)


def render_screener_table(stocks: list, api_key: str, models: dict):
    """Render a screener table with all stocks."""
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(stocks):
        status.text(f"Analyzing {ticker}...")
        progress.progress((i + 1) / len(stocks))
        
        # Fetch data
        quote = fetch_stock_quote(ticker, api_key)
        df = fetch_stock_data(ticker, api_key, days=30)
        
        if quote and df is not None and len(df) > 10:
            # Generate chart and analyze
            chart_img = generate_chart_image(df)
            price_range = (df['Low'].min(), df['High'].max())
            analysis = analyze_chart(chart_img, models, price_range)
            
            results.append({
                'Ticker': ticker,
                'Price': f"${quote['price']:.2f}",
                'Change': f"{quote['change_pct']:+.2f}%",
                'Trend': analysis.get('trend', 'N/A').upper(),
                'Trend Conf': f"{analysis.get('trend_confidence', 0)*100:.0f}%",
                'Signal': analysis.get('signal', 'N/A'),
                'Signal Str': f"{analysis.get('signal_strength', 0):.0f}%",
                '_change_val': quote['change_pct'],
                '_signal': analysis.get('signal', 'NEUTRAL'),
                '_chart': chart_img,
                '_analysis': analysis,
                '_quote': quote
            })
        
        time.sleep(0.1)  # Rate limiting
    
    progress.empty()
    status.empty()
    
    return results


# ============================================================================
# Main Page
# ============================================================================

def main():
    st.set_page_config(
        page_title="Stock Screener | QuantCademy",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f1a;
    }
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 AI Stock Screener</h1>
        <p>Real-time analysis with Computer Vision + NLP (coming soon)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check requirements
    if not POLYGON_AVAILABLE:
        st.error("""
        **Polygon API client not installed.**
        
        ```bash
        pip install polygon-api-client
        ```
        """)
        return
    
    if not MODELS_AVAILABLE:
        st.warning(f"""
        **AI models not available.** Chart analysis will be limited.
        
        Train models first:
        ```bash
        cd chart-vision
        python train_sr_model_v2.py --mode both --samples 800
        python train_trend_model_v2.py --mode both --samples 400
        ```
        """)
    
    # API Key from environment (hidden from users)
    api_key = os.environ.get('POLYGON_API_KEY', '')
    
    with st.sidebar:
        st.markdown("### Settings")
        st.markdown("---")
        
        # Watchlist selection
        st.markdown("### Watchlist")
        selected_category = st.selectbox(
            "Category",
            options=list(WATCHLIST_STOCKS.keys()),
            index=0
        )
        
        selected_stocks = st.multiselect(
            "Stocks",
            options=WATCHLIST_STOCKS[selected_category],
            default=WATCHLIST_STOCKS[selected_category][:4]
        )
        
        # Custom ticker
        st.markdown("---")
        custom_ticker = st.text_input("Add Custom Ticker", placeholder="e.g., PLTR")
        if custom_ticker and custom_ticker.upper() not in selected_stocks:
            selected_stocks.append(custom_ticker.upper())
        
        st.markdown("---")
        
        # View mode
        view_mode = st.radio(
            "View Mode",
            options=["Cards", "Table"],
            index=0
        )
    
    if not api_key:
        st.error("Stock Screener is currently unavailable. Please try again later.")
        return
    
    # Load models
    models = {}
    if MODELS_AVAILABLE:
        with st.spinner("Loading AI models..."):
            models = load_models()
    
    # Market Overview
    st.markdown("### Market Overview")
    render_market_overview(api_key)
    
    st.markdown("---")
    
    # Main content
    if not selected_stocks:
        st.info("Select stocks from the sidebar to analyze.")
        return
    
    st.markdown(f"### Analyzing {len(selected_stocks)} Stocks")
    
    if view_mode == "Cards":
        # Card view - detailed analysis
        for ticker in selected_stocks:
            with st.spinner(f"Analyzing {ticker}..."):
                quote = fetch_stock_quote(ticker, api_key)
                df = fetch_stock_data(ticker, api_key, days=30)
                
                if quote and df is not None and len(df) > 10:
                    chart_img = generate_chart_image(df, figsize=(8, 5))
                    price_range = (df['Low'].min(), df['High'].max())
                    analysis = analyze_chart(chart_img, models, price_range)
                    
                    render_stock_card(ticker, quote, analysis, chart_img)
                else:
                    st.warning(f"Could not fetch data for {ticker}")
            
            time.sleep(0.2)  # Rate limiting
    
    else:
        # Table view - quick overview
        results = render_screener_table(selected_stocks, api_key, models)
        
        if results:
            # Create display dataframe
            display_df = pd.DataFrame([{
                'Ticker': r['Ticker'],
                'Price': r['Price'],
                'Change': r['Change'],
                'Trend': r['Trend'],
                'Confidence': r['Trend Conf'],
                'Signal': r['Signal'],
                'Strength': r['Signal Str']
            } for r in results])
            
            # Style the dataframe
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #22c55e33; color: #22c55e'
                elif val == 'SELL':
                    return 'background-color: #ef444433; color: #ef4444'
                else:
                    return 'background-color: #f59e0b33; color: #f59e0b'
            
            def color_change(val):
                if val.startswith('+'):
                    return 'color: #22c55e'
                elif val.startswith('-'):
                    return 'color: #ef4444'
                return ''
            
            styled_df = display_df.style.applymap(
                color_signal, subset=['Signal']
            ).applymap(
                color_change, subset=['Change']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Show detailed view for selected stock
            st.markdown("---")
            st.markdown("### Detailed View")
            
            selected_ticker = st.selectbox(
                "Select stock for details",
                options=[r['Ticker'] for r in results]
            )
            
            selected_result = next((r for r in results if r['Ticker'] == selected_ticker), None)
            
            if selected_result:
                render_stock_card(
                    selected_result['Ticker'],
                    selected_result['_quote'],
                    selected_result['_analysis'],
                    selected_result['_chart']
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
        <p>AI signals are for educational purposes only. Not financial advice.</p>
        <p>Data provided by Polygon.io | Models trained on historical chart patterns</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
