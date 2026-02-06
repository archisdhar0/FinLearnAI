"""
QuantCademy - Full Learning Modules
Based on Figma design: Market Mechanics → Advanced Options
Each module and lesson is clickable with proper navigation.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Learning Modules | QuantCademy",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .module-btn {
        background: #16213e;
        border: none;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
        cursor: pointer;
        border-left: 4px solid #4ade80;
    }
    .module-btn:hover {
        background: #1e3a5f;
    }
    .lesson-btn {
        background: #1a1a2e;
        border-left: 2px solid #4b5563;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0 0.25rem 1rem;
    }
    .level-beginner { border-left-color: #4ade80 !important; }
    .level-intermediate { border-left-color: #fbbf24 !important; }
    .level-advanced { border-left-color: #f97316 !important; }
    .level-expert { border-left-color: #ef4444 !important; }
    .badge-beginner { color: #4ade80; }
    .badge-intermediate { color: #fbbf24; }
    .badge-advanced { color: #f97316; }
    .badge-expert { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state.current_module = None
if 'current_lesson' not in st.session_state:
    st.session_state.current_lesson = None
if 'completed_lessons' not in st.session_state:
    st.session_state.completed_lessons = set()

# Module definitions
MODULES = {
    "market_mechanics": {
        "number": "1",
        "title": "Market Mechanics",
        "level": "beginner",
        "icon": "🏛️",
        "description": "Foundational knowledge of market structure, participants, and order execution.",
        "lessons": [
            {"id": "stock_equity", "title": "What is a Stock & Equity?"},
            {"id": "order_book", "title": "The Order Book (Bid/Ask/Spread)"},
            {"id": "order_types", "title": "Market vs Limit vs Stop Orders"},
            {"id": "liquidity", "title": "Liquidity Providers vs Takers"},
            {"id": "exchanges", "title": "Exchanges vs Dark Pools"}
        ]
    },
    "macro_economics": {
        "number": "2",
        "title": "Macro Economics",
        "level": "beginner",
        "icon": "🌍",
        "description": "Understanding the global economic forces that drive asset prices.",
        "lessons": [
            {"id": "interest_rates", "title": "Interest Rates & The Fed"},
            {"id": "inflation", "title": "Inflation & Purchasing Power"},
            {"id": "gdp", "title": "GDP & Economic Cycles"},
            {"id": "currency", "title": "Currency & Exchange Rates"},
            {"id": "geopolitics", "title": "Geopolitical Risk Factors"}
        ]
    },
    "technical_analysis": {
        "number": "3",
        "title": "Technical Analysis",
        "level": "intermediate",
        "icon": "📊",
        "description": "Mastering chart reading, indicators, and price action patterns.",
        "lessons": [
            {"id": "candlesticks", "title": "Candlestick Patterns"},
            {"id": "support_resistance", "title": "Support & Resistance"},
            {"id": "moving_averages", "title": "Moving Averages (SMA, EMA)"},
            {"id": "indicators", "title": "RSI, MACD & Momentum"},
            {"id": "volume", "title": "Volume Analysis"},
            {"id": "chart_patterns", "title": "Chart Patterns"}
        ]
    },
    "fundamental_analysis": {
        "number": "3.5",
        "title": "Fundamental Analysis",
        "level": "intermediate",
        "icon": "📈",
        "description": "Evaluating the intrinsic value of a company using financial statements and ratios.",
        "lessons": [
            {"id": "financial_statements", "title": "Reading Financial Statements"},
            {"id": "pe_ratio", "title": "P/E Ratio & Valuation Metrics"},
            {"id": "growth", "title": "Revenue & Earnings Growth"},
            {"id": "balance_sheet", "title": "Balance Sheet Analysis"},
            {"id": "cash_flow", "title": "Cash Flow Analysis"},
            {"id": "moats", "title": "Competitive Moats"}
        ]
    },
    "quant_strategies": {
        "number": "4",
        "title": "Quant Strategies",
        "level": "advanced",
        "icon": "🤖",
        "description": "Systematic trading, algorithm design, and statistical edges.",
        "lessons": [
            {"id": "factor_investing", "title": "Factor Investing"},
            {"id": "mean_reversion", "title": "Mean Reversion Strategies"},
            {"id": "momentum", "title": "Momentum Strategies"},
            {"id": "stat_arb", "title": "Statistical Arbitrage"},
            {"id": "backtesting", "title": "Backtesting & Validation"},
            {"id": "risk_systems", "title": "Risk Management Systems"}
        ]
    },
    "advanced_options": {
        "number": "5",
        "title": "Advanced Options",
        "level": "expert",
        "icon": "⚡",
        "description": "Complex derivative strategies for income generation and portfolio protection.",
        "lessons": [
            {"id": "greeks", "title": "Options Greeks Deep Dive"},
            {"id": "vertical_spreads", "title": "Vertical Spreads"},
            {"id": "iron_condor", "title": "Iron Condors & Butterflies"},
            {"id": "calendar_spreads", "title": "Calendar & Diagonal Spreads"},
            {"id": "vol_trading", "title": "Volatility Trading"},
            {"id": "hedging", "title": "Portfolio Hedging"}
        ]
    }
}


def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📚 Learning Modules</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Master investing from the fundamentals to advanced strategies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check what view to show
    if st.session_state.current_lesson:
        render_lesson_view()
    elif st.session_state.current_module:
        render_module_view()
    else:
        render_modules_list()


def render_modules_list():
    """Show all modules as clickable cards."""
    st.markdown("### Select a Module to Begin")
    
    # Progress
    total_lessons = sum(len(m['lessons']) for m in MODULES.values())
    completed = len(st.session_state.completed_lessons)
    st.progress(completed / total_lessons if total_lessons > 0 else 0)
    st.caption(f"Progress: {completed}/{total_lessons} lessons completed")
    
    st.markdown("---")
    
    # Module grid
    cols = st.columns(2)
    
    for i, (mod_id, mod) in enumerate(MODULES.items()):
        with cols[i % 2]:
            level_color = {
                'beginner': '#4ade80',
                'intermediate': '#fbbf24', 
                'advanced': '#f97316',
                'expert': '#ef4444'
            }.get(mod['level'], '#6366f1')
            
            # Count completed lessons in this module
            mod_completed = sum(1 for lesson in mod['lessons'] 
                              if f"{mod_id}_{lesson['id']}" in st.session_state.completed_lessons)
            
            st.markdown(f"""
            <div style="background: #16213e; border-radius: 12px; padding: 1.5rem; 
                        margin-bottom: 1rem; border-left: 4px solid {level_color};">
                <span style="color: {level_color}; font-size: 0.75rem; font-weight: 600; 
                             text-transform: uppercase;">{mod['level']}</span>
                <h3 style="color: white; margin: 0.5rem 0;">{mod['icon']} {mod['number']}. {mod['title']}</h3>
                <p style="color: #9ca3af; font-size: 0.9rem; margin: 0.5rem 0;">{mod['description']}</p>
                <p style="color: #6b7280; font-size: 0.8rem; margin: 0;">
                    📝 {len(mod['lessons'])} lessons | ✅ {mod_completed} completed
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"▶️ Start {mod['title']}", key=f"start_{mod_id}", use_container_width=True):
                st.session_state.current_module = mod_id
                st.rerun()


def render_module_view():
    """Show lessons within a module."""
    mod_id = st.session_state.current_module
    mod = MODULES[mod_id]
    
    level_color = {
        'beginner': '#4ade80',
        'intermediate': '#fbbf24',
        'advanced': '#f97316',
        'expert': '#ef4444'
    }.get(mod['level'], '#6366f1')
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_module = None
            st.rerun()
    
    # Module header
    st.markdown(f"""
    <div style="background: #16213e; border-radius: 12px; padding: 2rem; margin: 1rem 0;
                border-left: 4px solid {level_color};">
        <span style="color: {level_color}; font-size: 0.85rem; font-weight: 600; 
                     text-transform: uppercase;">{mod['level']}</span>
        <h2 style="color: white; margin: 0.5rem 0;">{mod['icon']} {mod['number']}. {mod['title']}</h2>
        <p style="color: #9ca3af;">{mod['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Lessons")
    st.markdown("Click on a lesson to start learning:")
    
    for i, lesson in enumerate(mod['lessons']):
        lesson_key = f"{mod_id}_{lesson['id']}"
        is_completed = lesson_key in st.session_state.completed_lessons
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            status = "✅" if is_completed else f"{i+1}."
            if st.button(
                f"{status} {lesson['title']}", 
                key=f"lesson_{lesson_key}",
                use_container_width=True
            ):
                st.session_state.current_lesson = lesson['id']
                st.rerun()
        
        with col2:
            if is_completed:
                st.markdown("✅")


def render_lesson_view():
    """Show individual lesson content."""
    mod_id = st.session_state.current_module
    lesson_id = st.session_state.current_lesson
    mod = MODULES[mod_id]
    
    # Find current lesson
    lesson = next((l for l in mod['lessons'] if l['id'] == lesson_id), None)
    if not lesson:
        st.error("Lesson not found")
        return
    
    lesson_idx = next(i for i, l in enumerate(mod['lessons']) if l['id'] == lesson_id)
    lesson_key = f"{mod_id}_{lesson_id}"
    
    # Navigation header
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("← Back to Module", use_container_width=True):
            st.session_state.current_lesson = None
            st.rerun()
    
    with col3:
        if lesson_idx < len(mod['lessons']) - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.current_lesson = mod['lessons'][lesson_idx + 1]['id']
                st.rerun()
    
    # Lesson header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
                border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <p style="color: #6366f1; margin: 0;">{mod['icon']} {mod['title']} / Lesson {lesson_idx + 1}</p>
        <h2 style="color: white; margin: 0.5rem 0;">{lesson['title']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Render lesson content
    render_lesson_content(mod_id, lesson_id)
    
    # Completion button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if lesson_key in st.session_state.completed_lessons:
            st.success("✅ Lesson completed!")
            if st.button("📖 Review Again", use_container_width=True):
                pass  # Already showing content
        else:
            if st.button("✅ Mark as Complete", type="primary", use_container_width=True):
                st.session_state.completed_lessons.add(lesson_key)
                st.balloons()
                st.success("Lesson completed! 🎉")
                
                # Auto-advance to next lesson
                if lesson_idx < len(mod['lessons']) - 1:
                    st.session_state.current_lesson = mod['lessons'][lesson_idx + 1]['id']
                    st.rerun()


def render_lesson_content(mod_id, lesson_id):
    """Render the actual lesson content."""
    
    # ==================== MODULE 1: MARKET MECHANICS ====================
    if mod_id == "market_mechanics":
        
        if lesson_id == "stock_equity":
            st.markdown("""
            ## What is a Stock?
            
            A **stock** (also called equity or share) represents **ownership** in a company.
            
            When you buy a stock, you become a **shareholder** — you literally own a tiny piece of that company.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Key Terms
                
                | Term | Definition |
                |------|------------|
                | **Share** | A single unit of ownership |
                | **Shareholder** | Someone who owns shares |
                | **Market Cap** | Total value (Price × Shares) |
                | **Float** | Shares available to trade |
                """)
            
            with col2:
                fig = go.Figure(go.Pie(
                    values=[30, 25, 20, 15, 10],
                    labels=['Institutions', 'Mutual Funds', 'Retail', 'Insiders', 'ETFs'],
                    hole=0.4
                ))
                fig.update_layout(title="Typical Stock Ownership", height=300, margin=dict(t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### How Shareholders Make Money
            
            1. **💰 Capital Appreciation** — The stock price goes up, you sell for profit
            2. **💵 Dividends** — Company pays you a share of profits (usually quarterly)
            
            ### Why Companies Issue Stock
            
            - Raise money without debt
            - Reward employees with equity
            - Use stock for acquisitions
            
            > 💡 **Key Insight**: When you buy an index fund like VTI, you're buying tiny pieces of ~4,000 companies at once!
            """)
            
        elif lesson_id == "order_book":
            st.markdown("""
            ## The Order Book
            
            The **order book** is where all buy and sell orders are collected. It shows supply and demand in real-time.
            
            ### Key Terms
            
            - **Bid**: Highest price buyers will pay
            - **Ask**: Lowest price sellers will accept  
            - **Spread**: Gap between bid and ask
            - **Depth**: Number of shares at each price level
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Bids (Buyers)")
                bids = pd.DataFrame({
                    'Price': ['$99.95', '$99.90', '$99.85', '$99.80'],
                    'Size': [500, 1200, 800, 2000],
                })
                st.dataframe(bids, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("### 🔴 Asks (Sellers)")
                asks = pd.DataFrame({
                    'Price': ['$100.00', '$100.05', '$100.10', '$100.15'],
                    'Size': [300, 900, 1500, 600],
                })
                st.dataframe(asks, hide_index=True, use_container_width=True)
            
            st.info("**Spread** = $100.00 - $99.95 = $0.05 (0.05%)")
            
            st.markdown("""
            ### Why Spread Matters
            
            - **Tight spread** (penny stocks: bad, Apple: good) = High liquidity, easy to trade
            - **Wide spread** = Low liquidity, harder to get fair price
            - You "pay" the spread every time you trade
            """)
            
        elif lesson_id == "order_types":
            st.markdown("""
            ## Order Types Explained
            
            Different order types give you different levels of control over your trades.
            """)
            
            st.markdown("""
            | Order Type | How It Works | Best For |
            |------------|--------------|----------|
            | **Market** | Buy/sell immediately at best available price | Speed, guaranteed fill |
            | **Limit** | Buy/sell only at your specified price or better | Price control |
            | **Stop** | Becomes market order when price hits trigger | Limiting losses |
            | **Stop-Limit** | Becomes limit order when price hits trigger | Precise exit |
            """)
            
            st.markdown("### Example: Stock at $100")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Market Buy**: "Buy now!"
                - Fills at ~$100.05 (the ask)
                - ✅ Guaranteed to fill
                - ⚠️ Might pay more in volatile markets
                """)
                
                st.markdown("""
                **Limit Buy at $98**: "Only buy at $98 or less"
                - Waits for price to drop
                - ✅ Price control
                - ⚠️ Might never fill
                """)
            
            with col2:
                st.markdown("""
                **Stop Sell at $95**: "If price drops to $95, sell!"
                - Protects against big losses
                - ✅ Automatic protection
                - ⚠️ Can trigger on temporary dips
                """)
                
                st.markdown("""
                **Stop-Limit Sell at $95/$94**: 
                - Triggers at $95, sells only at $94+
                - ✅ More control
                - ⚠️ Might not fill if price gaps down
                """)
            
            st.warning("⚠️ **Pro Tip**: For long-term investing, market orders are usually fine. Limit orders matter more for active trading.")
            
        elif lesson_id == "liquidity":
            st.markdown("""
            ## Liquidity: Makers vs Takers
            
            **Liquidity** = How easily you can buy/sell without moving the price.
            
            ### Two Roles in Every Trade
            
            | Role | What They Do | Order Type | Gets |
            |------|--------------|------------|------|
            | **Maker** | Adds liquidity (places limit orders) | Limit | Often rebates |
            | **Taker** | Removes liquidity (hits existing orders) | Market | Pays fees |
            
            ### Why Liquidity Matters
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                #### High Liquidity (Good)
                - Apple, S&P 500 ETFs
                - Tight spreads ($0.01)
                - Easy to trade large amounts
                - Price is "efficient"
                """)
            
            with col2:
                st.markdown("""
                #### Low Liquidity (Risky)
                - Penny stocks, small-caps
                - Wide spreads (5-10%+)
                - Your order moves the price
                - Harder to exit positions
                """)
            
            st.success("💡 **For beginners**: Stick to liquid investments like VTI, VOO, SPY. You'll get better prices.")
            
        elif lesson_id == "exchanges":
            st.markdown("""
            ## Exchanges vs Dark Pools
            
            Not all trades happen on the NYSE or NASDAQ!
            
            ### Public Exchanges
            - **NYSE, NASDAQ** - Where most retail orders go
            - Transparent order books
            - Regulated, fair access
            
            ### Dark Pools (~40% of volume!)
            - Private trading venues
            - Orders are **hidden**
            - Used by institutions for large orders
            
            ### Why Dark Pools Exist
            
            Imagine you're a mutual fund buying 1 million shares:
            
            1. **On public exchange**: Everyone sees huge buyer → Price jumps → You pay more
            2. **In dark pool**: Order is hidden → Price doesn't move → Better execution
            
            ### Should You Care?
            
            **As a retail investor: No.** Your orders are small enough that regular exchanges work fine.
            
            Dark pools are for institutions moving millions of dollars.
            """)
    
    # ==================== MODULE 2: MACRO ECONOMICS ====================
    elif mod_id == "macro_economics":
        
        if lesson_id == "interest_rates":
            st.markdown("""
            ## Interest Rates & The Federal Reserve
            
            The **Fed** (Federal Reserve) controls interest rates, which affect everything in the economy.
            
            ### The Fed Funds Rate
            - Rate banks charge each other overnight
            - Currently: ~5.25-5.50%
            - The Fed raises/lowers this to control inflation and growth
            """)
            
            # Rate history chart
            years = list(range(2015, 2025))
            rates = [0.25, 0.5, 1.0, 1.75, 2.5, 0.25, 0.25, 0.25, 4.5, 5.5]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=rates, mode='lines+markers',
                                     line=dict(color='#6366f1', width=3)))
            fig.update_layout(title="Fed Funds Rate History", height=300,
                             xaxis_title="Year", yaxis_title="Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### How Rates Affect Investments
            
            | Asset | Rates UP | Rates DOWN |
            |-------|----------|------------|
            | **Stocks** | Often fall (growth costs more) | Often rise |
            | **Bonds** | Prices fall, yields rise | Prices rise |
            | **Real Estate** | Mortgages expensive | More affordable |
            | **Cash/Savings** | Earns more | Earns less |
            
            > 💡 "Don't fight the Fed" - When rates are dropping, it's usually good for stocks.
            """)
            
        elif lesson_id == "inflation":
            st.markdown("""
            ## Inflation & Purchasing Power
            
            **Inflation** = Rising prices = Your money buys less over time.
            
            ### The Fed's Target: 2%
            
            Why not 0%? A little inflation:
            - Encourages spending (prices will rise later)
            - Makes debt easier to pay off
            - Deflation is actually worse (Japan's "lost decades")
            
            ### The Impact on Your Money
            """)
            
            # Inflation impact chart
            years = np.arange(0, 31)
            value_2pct = 100 * (1 - 0.02) ** years
            value_3pct = 100 * (1 - 0.03) ** years
            value_5pct = 100 * (1 - 0.05) ** years
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=value_2pct, name='2% inflation', line=dict(color='#4ade80')))
            fig.add_trace(go.Scatter(x=years, y=value_3pct, name='3% inflation', line=dict(color='#fbbf24')))
            fig.add_trace(go.Scatter(x=years, y=value_5pct, name='5% inflation', line=dict(color='#ef4444')))
            fig.update_layout(title="$100 Purchasing Power Over Time", height=300,
                             xaxis_title="Years", yaxis_title="Real Value ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.error("⚠️ At 3% inflation, $100 today = $74 in 10 years. **This is why you MUST invest!**")
            
        else:
            st.info(f"📝 Full content for **{lesson_id}** coming soon!")
            st.markdown("Check back later or ask the AI Tutor about this topic!")
    
    # ==================== MODULE 3: TECHNICAL ANALYSIS ====================
    elif mod_id == "technical_analysis":
        
        if lesson_id == "candlesticks":
            st.markdown("""
            ## Candlestick Patterns
            
            Candlesticks show price action over a time period (day, hour, minute).
            
            ### Anatomy of a Candlestick
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ```
                     │ ← High (wick)
                     │
                ┌────┴────┐
                │  GREEN  │ ← Body
                │  (UP)   │   Open to Close
                └────┬────┘
                     │
                     │ ← Low (wick)
                ```
                **Green/White** = Close > Open (price went UP)
                """)
            
            with col2:
                st.markdown("""
                ```
                     │ ← High (wick)
                     │
                ┌────┴────┐
                │   RED   │ ← Body
                │ (DOWN)  │   Open to Close
                └────┬────┘
                     │
                     │ ← Low (wick)
                ```
                **Red/Black** = Close < Open (price went DOWN)
                """)
            
            # Sample candlestick chart
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=30)
            opens = 100 + np.cumsum(np.random.randn(30) * 1.5)
            closes = opens + np.random.randn(30) * 2
            highs = np.maximum(opens, closes) + np.abs(np.random.randn(30)) * 0.5
            lows = np.minimum(opens, closes) - np.abs(np.random.randn(30)) * 0.5
            
            fig = go.Figure(data=[go.Candlestick(x=dates, open=opens, high=highs, low=lows, close=closes)])
            fig.update_layout(title="Sample Candlestick Chart", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Common Patterns
            
            | Pattern | Shape | Meaning |
            |---------|-------|---------|
            | **Doji** | + shape, tiny body | Indecision |
            | **Hammer** | Small body, long lower wick | Potential reversal UP |
            | **Shooting Star** | Small body, long upper wick | Potential reversal DOWN |
            | **Engulfing** | Large candle swallows previous | Strong reversal signal |
            """)
            
        elif lesson_id == "moving_averages":
            st.markdown("""
            ## Moving Averages
            
            Moving averages smooth price data to show the trend direction.
            
            ### Types
            
            - **SMA (Simple)**: Average of last N prices (equal weight)
            - **EMA (Exponential)**: More weight to recent prices (faster reaction)
            
            ### Common Moving Averages
            
            - **50-day MA**: Medium-term trend
            - **200-day MA**: Long-term trend
            - **Price above 200 MA** = Bullish, below = Bearish
            """)
            
            # MA chart
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
            ma_50 = pd.Series(prices).rolling(50).mean()
            ma_200 = pd.Series(prices).rolling(200).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=prices, name='Price', line=dict(color='#94a3b8', width=1)))
            fig.add_trace(go.Scatter(y=ma_50, name='50 MA', line=dict(color='#22c55e', width=2)))
            fig.add_trace(go.Scatter(y=ma_200, name='200 MA', line=dict(color='#ef4444', width=2)))
            fig.update_layout(title="Price with Moving Averages", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Trading Signals
            
            - **Golden Cross**: 50 MA crosses ABOVE 200 MA → Bullish
            - **Death Cross**: 50 MA crosses BELOW 200 MA → Bearish
            
            > ⚠️ **Warning**: These signals often lag. By the time you see a Golden Cross, much of the move has happened.
            """)
            
        else:
            st.info(f"📝 Full content for this lesson coming soon!")
            st.markdown("Ask the AI Tutor about this topic for immediate help!")
    
    # ==================== DEFAULT FOR OTHER MODULES ====================
    else:
        st.info(f"📝 Content for this lesson is being developed.")
        st.markdown("""
        In the meantime, you can:
        - Ask the **AI Tutor** about this topic
        - Explore other completed lessons
        - Check back soon for updates!
        """)
        
        if st.button("🤖 Ask AI Tutor", use_container_width=True):
            st.session_state.current_lesson = None
            st.session_state.current_module = None
            st.switch_page("pages/ai_tutor.py")


if __name__ == "__main__":
    main()
