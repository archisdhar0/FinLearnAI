"""
QuantCademy - Complete Learning Modules
Comprehensive financial education from beginner to expert.
Content sourced from: Investopedia, Vanguard, Fidelity, SEC, FINRA, Bogleheads
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
    .source-tag {
        background: #1e3a5f;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .key-concept {
        background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #6366f1;
        margin: 1rem 0;
    }
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
        "source": "SEC, Investopedia, FINRA",
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
        "source": "Federal Reserve, Investopedia, Fidelity",
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
        "source": "Investopedia, TradingView, CMT Association",
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
        "source": "Fidelity, Morningstar, SEC EDGAR",
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
        "source": "AQR Capital, Two Sigma Research, Academic Papers",
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
        "source": "CBOE, Options Industry Council, Tastytrade",
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
                <p style="color: #4b5563; font-size: 0.7rem; margin-top: 0.5rem;">
                    📖 Sources: {mod.get('source', 'Various')}
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
        <p style="color: #4b5563; font-size: 0.8rem;">📖 Sources: {mod.get('source', 'Various')}</p>
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
        render_market_mechanics_content(lesson_id)
    
    # ==================== MODULE 2: MACRO ECONOMICS ====================
    elif mod_id == "macro_economics":
        render_macro_economics_content(lesson_id)
    
    # ==================== MODULE 3: TECHNICAL ANALYSIS ====================
    elif mod_id == "technical_analysis":
        render_technical_analysis_content(lesson_id)
    
    # ==================== MODULE 3.5: FUNDAMENTAL ANALYSIS ====================
    elif mod_id == "fundamental_analysis":
        render_fundamental_analysis_content(lesson_id)
    
    # ==================== MODULE 4: QUANT STRATEGIES ====================
    elif mod_id == "quant_strategies":
        render_quant_strategies_content(lesson_id)
    
    # ==================== MODULE 5: ADVANCED OPTIONS ====================
    elif mod_id == "advanced_options":
        render_advanced_options_content(lesson_id)


# ==================== MODULE 1: MARKET MECHANICS ====================
def render_market_mechanics_content(lesson_id):
    """Render Market Mechanics module content."""
    
    if lesson_id == "stock_equity":
        st.markdown("*📖 Source: SEC Investor.gov, Investopedia*")
        
        st.markdown("""
        ## What is a Stock?
        
        A **stock** (also called equity or share) represents **ownership** in a company.
        When you buy a stock, you become a **shareholder** — you literally own a tiny piece of that company.
        
        Think of it this way: If a company is a pizza, each stock is a slice. When you buy shares,
        you're buying slices of that pizza.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Key Terms
            
            | Term | Definition |
            |------|------------|
            | **Share** | A single unit of ownership |
            | **Shareholder** | Someone who owns shares |
            | **Market Cap** | Total value (Price × Shares Outstanding) |
            | **Float** | Shares available to trade publicly |
            | **Outstanding Shares** | Total shares that exist |
            """)
        
        with col2:
            fig = go.Figure(go.Pie(
                values=[30, 25, 20, 15, 10],
                labels=['Institutions', 'Mutual Funds', 'Retail', 'Insiders', 'ETFs'],
                hole=0.4,
                marker_colors=['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe']
            ))
            fig.update_layout(
                title="Typical Stock Ownership Breakdown",
                height=300,
                margin=dict(t=50, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 💰 How Shareholders Make Money
        
        **1. Capital Appreciation**
        - The stock price goes up over time
        - You sell for more than you paid
        - Example: Buy at $100, sell at $150 = $50 profit per share
        
        **2. Dividends**
        - Company pays you a share of its profits
        - Usually paid quarterly (4x per year)
        - Not all companies pay dividends (growth companies often don't)
        - Example: $1.00 dividend × 100 shares = $100 per quarter
        """)
        
        st.info("""
        💡 **Key Insight from Vanguard**: When you buy a total stock market index fund like VTI,
        you're buying tiny pieces of ~4,000 companies at once! This provides instant diversification.
        """)
        
        st.markdown("""
        ### 🏢 Why Companies Issue Stock
        
        1. **Raise capital without debt** - No interest payments required
        2. **Reward employees** - Stock options attract top talent
        3. **Make acquisitions** - Use stock as currency to buy other companies
        4. **Provide liquidity for founders** - Early investors can sell shares
        
        ### ⚠️ Shareholder Rights
        
        As a shareholder, you typically have:
        - **Voting rights** on major company decisions
        - **Right to dividends** if the company pays them
        - **Claim on assets** if the company is liquidated (after creditors)
        - **Right to information** via annual reports and SEC filings
        """)
        
    elif lesson_id == "order_book":
        st.markdown("*📖 Source: SEC, FINRA, Investopedia*")
        
        st.markdown("""
        ## The Order Book
        
        The **order book** is where all buy and sell orders are collected. It shows
        supply and demand in real-time and is the mechanism that determines stock prices.
        
        ### 🔑 Key Terms
        
        - **Bid**: The highest price buyers are willing to pay
        - **Ask** (or Offer): The lowest price sellers will accept  
        - **Spread**: The gap between bid and ask
        - **Depth**: Number of shares at each price level
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Bids (Buyers)")
            bids = pd.DataFrame({
                'Price': ['$99.95', '$99.90', '$99.85', '$99.80', '$99.75'],
                'Size': [500, 1200, 800, 2000, 1500],
                'Orders': [3, 8, 5, 12, 7]
            })
            st.dataframe(bids, hide_index=True, use_container_width=True)
            st.caption("Buyers waiting to purchase at these prices")
        
        with col2:
            st.markdown("### 🔴 Asks (Sellers)")
            asks = pd.DataFrame({
                'Price': ['$100.00', '$100.05', '$100.10', '$100.15', '$100.20'],
                'Size': [300, 900, 1500, 600, 2200],
                'Orders': [2, 6, 9, 4, 11]
            })
            st.dataframe(asks, hide_index=True, use_container_width=True)
            st.caption("Sellers waiting to sell at these prices")
        
        st.info("**Spread** = $100.00 - $99.95 = **$0.05** (0.05%)")
        
        st.markdown("""
        ### 📊 Why the Spread Matters
        
        The spread is effectively a **transaction cost**. Every time you trade, you "pay" the spread:
        
        | Spread Type | Example | What It Means |
        |-------------|---------|---------------|
        | **Tight** (good) | $0.01 on Apple | High liquidity, easy to trade |
        | **Wide** (costly) | $0.50 on penny stock | Low liquidity, harder to get fair price |
        
        ### 🎯 How Prices Move
        
        1. **More buyers than sellers** → Buyers bid higher to get filled → Price rises
        2. **More sellers than buyers** → Sellers ask lower to get filled → Price falls
        3. **Big news** → Orders flood one side → Rapid price movement
        
        > 💡 **Pro Tip from Fidelity**: For most long-term investors, the spread is negligible.
        > Focus on transaction costs when trading illiquid securities or in large sizes.
        """)
        
    elif lesson_id == "order_types":
        st.markdown("*📖 Source: Fidelity, Charles Schwab, Investopedia*")
        
        st.markdown("""
        ## Order Types Explained
        
        Different order types give you different levels of control over price and execution speed.
        Understanding when to use each is crucial for effective trading.
        """)
        
        st.markdown("""
        ### 📋 Order Type Comparison
        
        | Order Type | How It Works | Guaranteed Fill? | Best For |
        |------------|--------------|------------------|----------|
        | **Market** | Buy/sell immediately at best available price | ✅ Yes | Speed, small orders |
        | **Limit** | Buy/sell only at your specified price or better | ❌ No | Price control |
        | **Stop** | Becomes market order when price hits trigger | ✅ Once triggered | Limiting losses |
        | **Stop-Limit** | Becomes limit order when price hits trigger | ❌ No | Precise exit points |
        | **Trailing Stop** | Stop price trails the market price | ✅ Once triggered | Protecting gains |
        """)
        
        st.markdown("### 📈 Example: Stock Trading at $100")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Market Buy
            *"Buy now at whatever price!"*
            
            - Fills immediately at ~$100.05 (the ask)
            - ✅ Guaranteed to fill
            - ⚠️ Might pay more in volatile markets
            - 👍 Best for: Urgent trades, liquid stocks
            
            ---
            
            #### Limit Buy at $98
            *"Only buy if price drops to $98 or less"*
            
            - Order waits until price reaches $98
            - ✅ You control your entry price
            - ⚠️ Might never fill if price doesn't drop
            - 👍 Best for: Patient buyers, volatile markets
            """)
        
        with col2:
            st.markdown("""
            #### Stop Sell at $95
            *"If price drops to $95, sell immediately!"*
            
            - Triggers when price hits $95
            - Then executes as market order
            - ✅ Automatic loss protection
            - ⚠️ Can trigger on temporary dips
            - 👍 Best for: Downside protection
            
            ---
            
            #### Stop-Limit Sell at $95/$94
            *"If price drops to $95, sell but only at $94 or better"*
            
            - Triggers at $95, becomes limit at $94
            - ✅ More price control
            - ⚠️ Might not fill if price gaps down
            - 👍 Best for: Avoiding forced sales at bad prices
            """)
        
        st.warning("""
        ⚠️ **Important Note from Fidelity**: During volatile markets or gaps (overnight moves), 
        stop orders can execute at prices significantly different from your stop price. 
        This is called "slippage."
        """)
        
        st.markdown("""
        ### 🎯 Which Order Should You Use?
        
        | Scenario | Recommended Order |
        |----------|-------------------|
        | Long-term investing, buying ETFs | **Market** (simple, fast) |
        | Want to buy on a dip | **Limit** (set your price) |
        | Already own stock, want protection | **Stop** or **Trailing Stop** |
        | Very volatile stock | **Limit** (avoid slippage) |
        | Exiting a position precisely | **Stop-Limit** |
        """)
        
    elif lesson_id == "liquidity":
        st.markdown("*📖 Source: SEC, FINRA, Investopedia*")
        
        st.markdown("""
        ## Liquidity: Makers vs Takers
        
        **Liquidity** = How easily you can buy or sell an asset without significantly moving its price.
        
        High liquidity means you can trade large amounts quickly at fair prices.
        Low liquidity means your trades may move the market or you may struggle to exit positions.
        """)
        
        st.markdown("""
        ### 🎭 Two Roles in Every Trade
        
        | Role | What They Do | Order Type | Exchange Treatment |
        |------|--------------|------------|-------------------|
        | **Market Maker** | Adds orders to the book | Limit orders | Often receives rebates |
        | **Market Taker** | Removes orders from book | Market orders | Pays fees |
        
        **Makers provide liquidity** - They put orders in the book for others to trade against.
        
        **Takers consume liquidity** - They fill existing orders immediately.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ High Liquidity (Ideal)
            
            **Examples**: Apple, Microsoft, S&P 500 ETFs (SPY, VOO)
            
            - Tight spreads ($0.01)
            - Deep order books (millions of shares)
            - Can trade large amounts easily
            - Price accurately reflects value
            - Low transaction costs
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Low Liquidity (Risky)
            
            **Examples**: Penny stocks, small-cap stocks, exotic ETFs
            
            - Wide spreads (5-10%+)
            - Thin order books
            - Your order moves the price
            - Difficult to exit positions
            - High hidden costs
            """)
        
        st.markdown("""
        ### 📊 Measuring Liquidity
        
        | Metric | What It Shows | Good Sign |
        |--------|---------------|-----------|
        | **Average Volume** | Shares traded daily | Higher = better |
        | **Bid-Ask Spread** | Gap between buy/sell | Smaller = better |
        | **Market Depth** | Orders at each price | Deeper = better |
        | **Price Impact** | How much your order moves price | Lower = better |
        """)
        
        # Liquidity comparison chart
        fig = go.Figure()
        categories = ['Apple (AAPL)', 'Mid-Cap Stock', 'Small Cap', 'Penny Stock']
        spreads = [0.01, 0.05, 0.25, 2.00]
        colors = ['#4ade80', '#fbbf24', '#f97316', '#ef4444']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=spreads,
            marker_color=colors,
            text=[f'{s:.2f}%' for s in spreads],
            textposition='outside'
        ))
        fig.update_layout(
            title="Typical Bid-Ask Spreads by Stock Type",
            yaxis_title="Spread (%)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        💡 **Beginner Advice from Vanguard**: Stick to liquid investments like VTI, VOO, or SPY. 
        You'll always get fair prices and can buy/sell easily. Avoid penny stocks and 
        thinly-traded securities until you understand liquidity risk.
        """)
        
    elif lesson_id == "exchanges":
        st.markdown("*📖 Source: SEC, NYSE, NASDAQ, Investopedia*")
        
        st.markdown("""
        ## Exchanges vs Dark Pools
        
        Not all trades happen on the public exchanges you see on TV!
        Understanding market structure helps you know where your orders go.
        
        ### 🏛️ Public Exchanges
        
        These are the traditional, regulated marketplaces:
        
        | Exchange | Description | Notable Listings |
        |----------|-------------|------------------|
        | **NYSE** | New York Stock Exchange, largest by market cap | Berkshire, JPMorgan, Walmart |
        | **NASDAQ** | Electronic exchange, tech-heavy | Apple, Microsoft, Google, Amazon |
        | **CBOE** | Chicago Board Options Exchange | Options, VIX |
        | **IEX** | "Investors Exchange" - designed for fairness | Various |
        
        **Key Features:**
        - Transparent order books
        - Regulated by SEC
        - Fair access rules
        - Public price discovery
        """)
        
        st.markdown("""
        ### 🌑 Dark Pools (~40% of US Trading Volume!)
        
        Dark pools are **private trading venues** where orders are hidden until executed.
        
        **Why They Exist:**
        
        Imagine you're a pension fund buying 1 million shares of Apple:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### On Public Exchange ❌
            1. You start buying
            2. Everyone sees huge buyer
            3. Traders front-run you
            4. Price jumps before you finish
            5. You pay more than necessary
            """)
        
        with col2:
            st.markdown("""
            #### In Dark Pool ✅
            1. Your order is hidden
            2. No one knows you're buying
            3. You match with hidden sellers
            4. Price doesn't move
            5. Better execution price
            """)
        
        st.markdown("""
        ### 🔄 Payment for Order Flow (PFOF)
        
        When you place an order at Robinhood, Schwab, or TD Ameritrade:
        
        1. Your broker **sells your order** to a market maker (Citadel, Virtu)
        2. The market maker fills your order (often at slight improvement)
        3. Market maker profits from the spread
        4. You get "free" trading
        
        **Is this bad?** It's debated:
        - ✅ You often get price improvement over the public quote
        - ⚠️ Critics argue you might get even better prices elsewhere
        - 📊 SEC requires brokers to disclose execution quality
        """)
        
        st.info("""
        💡 **For Retail Investors**: The venue your order goes to rarely matters for small trades.
        Focus on keeping costs low and investing consistently. Market structure is mainly 
        important for institutional traders moving large amounts.
        """)


# ==================== MODULE 2: MACRO ECONOMICS ====================
def render_macro_economics_content(lesson_id):
    """Render Macro Economics module content."""
    
    if lesson_id == "interest_rates":
        st.markdown("*📖 Source: Federal Reserve, Investopedia, Fidelity*")
        
        st.markdown("""
        ## Interest Rates & The Federal Reserve
        
        The **Federal Reserve** (the Fed) is the central bank of the United States.
        Its decisions on interest rates affect everything from your savings account to the stock market.
        
        ### 🏛️ What Does the Fed Do?
        
        The Fed has a **dual mandate**:
        1. **Maximum employment** - Keep unemployment low
        2. **Price stability** - Keep inflation around 2%
        
        Their main tool? **The Federal Funds Rate** - the rate banks charge each other for overnight loans.
        """)
        
        # Rate history chart
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        rates = [0.25, 0.50, 1.00, 1.75, 2.50, 0.25, 0.25, 4.50, 5.25, 5.50]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=rates,
            mode='lines+markers',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title="Federal Funds Rate History (2015-2024)",
            height=350,
            xaxis_title="Year",
            yaxis_title="Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 How Rate Changes Affect Markets
        
        | Asset Class | Rates **UP** ⬆️ | Rates **DOWN** ⬇️ |
        |-------------|-----------------|-------------------|
        | **Stocks** | Often fall (borrowing costs more) | Often rise (cheaper to grow) |
        | **Bonds** | Prices fall, yields rise | Prices rise, yields fall |
        | **Real Estate** | Mortgages expensive → fewer buyers | More affordable → demand up |
        | **Savings** | Higher interest on deposits | Lower interest earned |
        | **Dollar** | Strengthens vs other currencies | Weakens |
        
        ### 🔄 The Rate Cycle
        
        1. **Economy overheating** → Fed raises rates → Slows borrowing → Cools economy
        2. **Economy weak** → Fed cuts rates → Encourages borrowing → Stimulates growth
        3. **Crisis** → Fed cuts to near zero → Emergency stimulus
        """)
        
        st.warning("""
        ⚠️ **"Don't Fight the Fed"** - This classic Wall Street saying means: 
        When the Fed is cutting rates, it's usually good for stocks. 
        When they're raising aggressively, be cautious.
        """)
        
        st.markdown("""
        ### 🎯 Key Takeaways for Investors
        
        - **Higher rates** = Growth stocks (tech) often struggle more than value stocks
        - **Lower rates** = Bonds are less attractive, stocks benefit
        - **Rate expectations** matter as much as actual changes
        - Watch Fed meeting dates and Jerome Powell's speeches
        """)
        
    elif lesson_id == "inflation":
        st.markdown("*📖 Source: Bureau of Labor Statistics, Federal Reserve, Investopedia*")
        
        st.markdown("""
        ## Inflation & Purchasing Power
        
        **Inflation** = The rate at which prices rise over time, reducing your money's purchasing power.
        
        If you keep $100 in cash and inflation is 3%, after one year that $100 only buys 
        what $97 could buy before. Over time, this compounds dramatically.
        
        ### 📊 The Fed's Target: 2% Inflation
        
        Why not 0%? A little inflation is actually healthy:
        - Encourages spending (prices will rise later)
        - Makes debt easier to pay off over time
        - **Deflation** (falling prices) is actually worse - see Japan's "lost decades"
        """)
        
        # Inflation impact visualization
        years = np.arange(0, 31)
        value_2pct = 100 * (0.98) ** years
        value_3pct = 100 * (0.97) ** years
        value_5pct = 100 * (0.95) ** years
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=value_2pct, name='2% inflation', 
                                 line=dict(color='#4ade80', width=2)))
        fig.add_trace(go.Scatter(x=years, y=value_3pct, name='3% inflation', 
                                 line=dict(color='#fbbf24', width=2)))
        fig.add_trace(go.Scatter(x=years, y=value_5pct, name='5% inflation', 
                                 line=dict(color='#ef4444', width=2)))
        fig.update_layout(
            title="$100 Purchasing Power Over Time",
            height=350,
            xaxis_title="Years",
            yaxis_title="Real Value ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("""
        ⚠️ **The Silent Wealth Destroyer**: At 3% inflation, your cash loses HALF its purchasing 
        power in about 24 years. This is why you MUST invest - cash savings alone won't preserve wealth.
        """)
        
        st.markdown("""
        ### 📈 Measuring Inflation
        
        | Index | What It Measures | Used By |
        |-------|------------------|---------|
        | **CPI** (Consumer Price Index) | Basket of consumer goods | Most common measure |
        | **Core CPI** | CPI excluding food & energy | Fed's preferred view |
        | **PCE** (Personal Consumption Expenditures) | Broader measure | Fed's official target |
        | **PPI** (Producer Price Index) | Wholesale prices | Leading indicator |
        
        ### 🛡️ Inflation-Hedging Investments
        
        | Investment | How It Helps |
        |------------|--------------|
        | **Stocks** | Companies can raise prices → earnings grow |
        | **TIPS** | Treasury bonds indexed to CPI |
        | **Real Estate** | Property values and rents rise with inflation |
        | **Commodities** | Raw materials rise with inflation |
        | **I-Bonds** | Government savings bonds indexed to CPI |
        """)
        
        st.success("""
        💡 **Key Insight**: Over the long term, stocks have been the best inflation hedge, 
        returning ~10% annually vs ~3% inflation. A diversified portfolio protects your 
        purchasing power far better than cash.
        """)
        
    elif lesson_id == "gdp":
        st.markdown("*📖 Source: Bureau of Economic Analysis, Investopedia, Federal Reserve*")
        
        st.markdown("""
        ## GDP & Economic Cycles
        
        **GDP (Gross Domestic Product)** = The total value of all goods and services produced 
        in a country over a specific period. It's the primary measure of economic health.
        
        ### 📊 GDP Components
        
        GDP = **C + I + G + (X - M)**
        
        | Component | Description | % of US GDP |
        |-----------|-------------|-------------|
        | **C** - Consumer Spending | Households buying stuff | ~68% |
        | **I** - Business Investment | Companies investing | ~18% |
        | **G** - Government Spending | Federal + state + local | ~17% |
        | **X-M** - Net Exports | Exports minus imports | ~-3% (deficit) |
        """)
        
        # GDP growth chart
        years = list(range(2015, 2025))
        gdp_growth = [2.9, 1.6, 2.4, 2.9, 2.3, -2.8, 5.9, 2.1, 2.5, 2.8]
        colors = ['#4ade80' if g > 0 else '#ef4444' for g in gdp_growth]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=gdp_growth, marker_color=colors))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="US Real GDP Growth Rate (%)",
            height=300,
            xaxis_title="Year",
            yaxis_title="Growth Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔄 The Business Cycle
        
        Economies move through predictable phases:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📈 Expansion
            - GDP growing
            - Unemployment falling
            - Consumer confidence high
            - Stocks typically rise
            
            #### 🔝 Peak
            - Economy at maximum output
            - Inflation may be rising
            - Fed may raise rates
            - Stocks may be overvalued
            """)
        
        with col2:
            st.markdown("""
            #### 📉 Contraction (Recession)
            - GDP declining for 2+ quarters
            - Unemployment rising
            - Businesses cutting costs
            - Stocks typically fall
            
            #### 🔻 Trough
            - Economy at lowest point
            - Unemployment peaks
            - Fed cuts rates
            - Stocks often turn around here
            """)
        
        st.info("""
        💡 **Investment Tip from Fidelity**: Different sectors perform better at different 
        cycle stages. Cyclical stocks (consumer discretionary, industrials) outperform 
        in early expansion. Defensive stocks (utilities, healthcare) outperform in recession.
        """)
        
    elif lesson_id == "currency":
        st.markdown("*📖 Source: Federal Reserve, IMF, Investopedia*")
        
        st.markdown("""
        ## Currency & Exchange Rates
        
        Exchange rates affect international investments, imports/exports, and the global economy.
        Understanding currency dynamics is essential for diversified portfolios.
        
        ### 💱 What Determines Exchange Rates?
        
        | Factor | Effect on Dollar |
        |--------|------------------|
        | **Higher US interest rates** | Dollar strengthens (attracts capital) |
        | **Strong US economy** | Dollar strengthens (investment flows in) |
        | **Higher US inflation** | Dollar weakens (purchasing power falls) |
        | **Trade deficit** | Dollar weakens (more dollars going abroad) |
        | **Safe haven demand** | Dollar strengthens (crisis = buy USD) |
        """)
        
        st.markdown("""
        ### 🌍 Impact on Your Investments
        
        **International Stocks (VXUS, VEU, etc.)**
        
        When you own international stocks, you're exposed to **currency risk**:
        
        | Scenario | Impact on Your International Holdings |
        |----------|---------------------------------------|
        | Dollar strengthens | Foreign stocks worth less in USD (hurts returns) |
        | Dollar weakens | Foreign stocks worth more in USD (boosts returns) |
        
        **Example**: If European stocks rise 10% in euros, but the euro falls 5% vs dollar,
        your return is only about 5% in USD terms.
        """)
        
        st.markdown("""
        ### 🏭 Currency & Corporate Earnings
        
        US multinational companies are also affected:
        
        - **Strong dollar** = Foreign revenue worth less when converted → Hurts earnings
        - **Weak dollar** = Foreign revenue worth more → Boosts earnings
        
        Companies like Apple, Coca-Cola, and McDonald's earn 50%+ of revenue abroad!
        
        ### 🛡️ Should You Hedge Currency?
        
        | Approach | Pros | Cons |
        |----------|------|------|
        | **Unhedged** (most common) | Simpler, natural diversification | Currency volatility |
        | **Hedged** | Removes currency risk | Costs money, removes potential gains |
        
        **Vanguard's view**: For long-term investors, currency movements tend to even out. 
        Hedging adds costs and complexity that usually aren't worth it for most investors.
        """)
        
    elif lesson_id == "geopolitics":
        st.markdown("*📖 Source: Council on Foreign Relations, IMF, Investopedia*")
        
        st.markdown("""
        ## Geopolitical Risk Factors
        
        Geopolitical events can cause sudden market volatility. Understanding these risks 
        helps you stay calm during turbulent times.
        
        ### 🌍 Types of Geopolitical Risk
        
        | Risk Type | Examples | Market Impact |
        |-----------|----------|---------------|
        | **War/Conflict** | Russia-Ukraine, Middle East | Energy spikes, flight to safety |
        | **Trade Wars** | US-China tariffs | Supply chain disruption |
        | **Sanctions** | Russia sanctions | Sector-specific impacts |
        | **Elections** | Major policy shifts | Uncertainty, volatility |
        | **Regulatory** | Tech regulation, antitrust | Industry-specific |
        | **Pandemic** | COVID-19 | Global economic shock |
        """)
        
        st.markdown("""
        ### 📊 Historical Market Reactions
        
        | Event | Initial Drop | Recovery Time |
        |-------|--------------|---------------|
        | 9/11 Attacks (2001) | -12% | 1 month |
        | Iraq War Start (2003) | -5% | 1 month |
        | COVID Crash (2020) | -34% | 5 months |
        | Russia-Ukraine (2022) | -13% | 3 months |
        
        **Key Pattern**: Markets initially panic, then recover as uncertainty resolves.
        """)
        
        st.success("""
        💡 **Historical Perspective from Vanguard**: Since 1945, there have been dozens of 
        major geopolitical crises. In almost every case, investors who stayed invested 
        were rewarded. Panic selling during crises is usually the wrong move.
        """)
        
        st.markdown("""
        ### 🛡️ How to Handle Geopolitical Risk
        
        1. **Stay diversified** - Don't concentrate in regions/sectors exposed to specific risks
        2. **Maintain perspective** - Markets have recovered from every crisis in history
        3. **Avoid panic selling** - Selling into a crisis locks in losses
        4. **Consider rebalancing** - Crises create opportunities to buy quality assets cheap
        5. **Keep cash reserves** - Emergency fund prevents forced selling
        
        ### ⚠️ What NOT to Do
        
        - Don't try to predict geopolitical events (experts can't either)
        - Don't make portfolio changes based on news headlines
        - Don't assume "this time is different" - it rarely is
        """)


# ==================== MODULE 3: TECHNICAL ANALYSIS ====================
def render_technical_analysis_content(lesson_id):
    """Render Technical Analysis module content."""
    
    if lesson_id == "candlesticks":
        st.markdown("*📖 Source: Investopedia, TradingView, CMT Association*")
        
        st.markdown("""
        ## Candlestick Patterns
        
        Candlesticks originated in 18th century Japan for rice trading. They show price action 
        over a time period (day, hour, minute) in a visually intuitive way.
        
        ### 🕯️ Anatomy of a Candlestick
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ```
                 │ ← High (upper wick/shadow)
                 │
            ┌────┴────┐
            │  GREEN  │ ← Body (Open to Close)
            │  (UP)   │   Close is ABOVE Open
            └────┬────┘
                 │
                 │ ← Low (lower wick/shadow)
            ```
            **Green/White** = Bullish (price went UP)
            """)
        
        with col2:
            st.markdown("""
            ```
                 │ ← High (upper wick/shadow)
                 │
            ┌────┴────┐
            │   RED   │ ← Body (Open to Close)
            │ (DOWN)  │   Close is BELOW Open
            └────┬────┘
                 │
                 │ ← Low (lower wick/shadow)
            ```
            **Red/Black** = Bearish (price went DOWN)
            """)
        
        # Generate sample candlestick data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30)
        opens = 100 + np.cumsum(np.random.randn(30) * 1.5)
        closes = opens + np.random.randn(30) * 2
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(30)) * 0.5
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(30)) * 0.5
        
        fig = go.Figure(data=[go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            increasing_line_color='#4ade80',
            decreasing_line_color='#ef4444'
        )])
        fig.update_layout(
            title="Sample Candlestick Chart",
            height=400,
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📊 Key Single-Candle Patterns
        
        | Pattern | Shape | Signal | Reliability |
        |---------|-------|--------|-------------|
        | **Doji** | + shape, tiny body | Indecision, potential reversal | Medium |
        | **Hammer** | Small body, long lower wick | Bullish reversal | High |
        | **Shooting Star** | Small body, long upper wick | Bearish reversal | High |
        | **Marubozu** | Full body, no wicks | Strong trend continuation | High |
        | **Spinning Top** | Small body, equal wicks | Indecision | Low |
        
        ### 📈 Key Multi-Candle Patterns
        
        | Pattern | Description | Signal |
        |---------|-------------|--------|
        | **Bullish Engulfing** | Green candle fully engulfs prior red | Strong bullish reversal |
        | **Bearish Engulfing** | Red candle fully engulfs prior green | Strong bearish reversal |
        | **Morning Star** | Down, small, up (3 candles) | Bullish reversal |
        | **Evening Star** | Up, small, down (3 candles) | Bearish reversal |
        | **Three White Soldiers** | Three consecutive green candles | Strong bullish |
        | **Three Black Crows** | Three consecutive red candles | Strong bearish |
        """)
        
        st.warning("""
        ⚠️ **Important**: Candlestick patterns work best when:
        1. They occur at key support/resistance levels
        2. They're confirmed by volume
        3. They're used with other indicators
        
        No pattern works 100% of the time!
        """)
        
    elif lesson_id == "support_resistance":
        st.markdown("*📖 Source: Investopedia, TradingView, Technical Analysis of Stock Trends*")
        
        st.markdown("""
        ## Support & Resistance
        
        **Support** and **Resistance** are price levels where buying or selling pressure 
        tends to concentrate, causing price to pause or reverse.
        
        ### 📉 Support
        - A price level where **buying interest** is strong enough to overcome selling pressure
        - Price tends to "bounce" off support
        - Think of it as a "floor" under the price
        - If broken, often becomes new resistance
        
        ### 📈 Resistance
        - A price level where **selling interest** is strong enough to overcome buying pressure
        - Price tends to "reject" at resistance
        - Think of it as a "ceiling" above the price
        - If broken, often becomes new support
        """)
        
        # Create support/resistance visualization
        np.random.seed(123)
        x = np.arange(100)
        y = 100 + np.cumsum(np.random.randn(100) * 0.5)
        # Add artificial S/R bounces
        y = np.clip(y, 97, 105)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price', line=dict(color='#94a3b8')))
        fig.add_hline(y=105, line_dash="dash", line_color="#ef4444", 
                      annotation_text="Resistance $105")
        fig.add_hline(y=97, line_dash="dash", line_color="#4ade80", 
                      annotation_text="Support $97")
        fig.update_layout(
            title="Support and Resistance Example",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 How to Identify S/R Levels
        
        | Method | Description |
        |--------|-------------|
        | **Historical highs/lows** | Previous peaks and troughs |
        | **Round numbers** | $100, $50, etc. (psychological) |
        | **Moving averages** | 50-day, 200-day MA act as dynamic S/R |
        | **Fibonacci levels** | 38.2%, 50%, 61.8% retracements |
        | **Volume profile** | High-volume price levels |
        | **Trendlines** | Diagonal S/R connecting highs or lows |
        
        ### 📊 Trading Support & Resistance
        
        | Scenario | Strategy |
        |----------|----------|
        | Price approaches support | Consider buying (with stop below support) |
        | Price approaches resistance | Consider selling/taking profits |
        | Support breaks | May signal more downside (stop loss triggered) |
        | Resistance breaks | May signal breakout (potential entry) |
        """)
        
        st.info("""
        💡 **Pro Tip**: The more times a level is tested, the more significant it becomes.
        However, each test also weakens it slightly. A level tested 4-5 times may eventually break.
        """)
        
    elif lesson_id == "moving_averages":
        st.markdown("*📖 Source: Investopedia, TradingView, Fidelity*")
        
        st.markdown("""
        ## Moving Averages (SMA & EMA)
        
        Moving averages smooth price data to reveal the underlying trend direction.
        They're among the most widely used technical indicators.
        
        ### 📊 Types of Moving Averages
        
        | Type | Calculation | Characteristics |
        |------|-------------|-----------------|
        | **SMA** (Simple) | Average of last N prices | Equal weight to all prices |
        | **EMA** (Exponential) | Weighted toward recent prices | Faster reaction to changes |
        | **WMA** (Weighted) | Linear weighting | Between SMA and EMA |
        """)
        
        # Generate MA chart
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
        ma_20 = pd.Series(prices).rolling(20).mean()
        ma_50 = pd.Series(prices).rolling(50).mean()
        ma_200 = pd.Series(prices).rolling(200).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, name='Price', line=dict(color='#94a3b8', width=1)))
        fig.add_trace(go.Scatter(y=ma_20, name='20 MA', line=dict(color='#a78bfa', width=2)))
        fig.add_trace(go.Scatter(y=ma_50, name='50 MA', line=dict(color='#22c55e', width=2)))
        fig.add_trace(go.Scatter(y=ma_200, name='200 MA', line=dict(color='#ef4444', width=2)))
        fig.update_layout(
            title="Price with Multiple Moving Averages",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Common Moving Averages
        
        | MA Period | Use Case |
        |-----------|----------|
        | **10-20 day** | Short-term trend, active trading |
        | **50 day** | Medium-term trend, swing trading |
        | **200 day** | Long-term trend, investing |
        
        ### ⚔️ Classic Trading Signals
        
        | Signal | What It Means | Reliability |
        |--------|---------------|-------------|
        | **Golden Cross** | 50 MA crosses ABOVE 200 MA | Bullish (moderate) |
        | **Death Cross** | 50 MA crosses BELOW 200 MA | Bearish (moderate) |
        | **Price above 200 MA** | Long-term uptrend | Bullish context |
        | **Price below 200 MA** | Long-term downtrend | Bearish context |
        """)
        
        st.warning("""
        ⚠️ **Limitations of Moving Averages**:
        - They **lag** - by the time you see a signal, much of the move has happened
        - They generate **false signals** in choppy, sideways markets
        - They work best in **trending markets**
        
        Use them as part of a broader analysis, not as standalone signals.
        """)
        
    elif lesson_id == "indicators":
        st.markdown("*📖 Source: Investopedia, TradingView, CMT Association*")
        
        st.markdown("""
        ## RSI, MACD & Momentum Indicators
        
        Momentum indicators measure the speed and strength of price movements.
        They help identify overbought/oversold conditions and trend strength.
        
        ### 📊 RSI (Relative Strength Index)
        
        RSI measures the speed and magnitude of recent price changes to evaluate 
        overbought or oversold conditions. It oscillates between 0 and 100.
        """)
        
        # Generate RSI example
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 1)
        
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rsi, name='RSI', line=dict(color='#6366f1', width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="RSI (14-period)",
            height=250,
            yaxis=dict(range=[0, 100]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        | RSI Level | Interpretation | Potential Action |
        |-----------|----------------|------------------|
        | **Above 70** | Overbought | Watch for reversal, don't buy |
        | **Below 30** | Oversold | Watch for bounce, potential buy |
        | **50** | Neutral | Trend direction unclear |
        
        ### 📈 MACD (Moving Average Convergence Divergence)
        
        MACD shows the relationship between two moving averages and helps identify 
        trend changes and momentum.
        
        **Components:**
        - **MACD Line** = 12-period EMA - 26-period EMA
        - **Signal Line** = 9-period EMA of MACD Line
        - **Histogram** = MACD Line - Signal Line
        
        **Signals:**
        - MACD crosses ABOVE signal line = **Bullish**
        - MACD crosses BELOW signal line = **Bearish**
        - Histogram growing = Momentum increasing
        - Histogram shrinking = Momentum decreasing
        """)
        
        st.markdown("""
        ### 🔄 Divergence
        
        **Divergence** occurs when price and indicator move in opposite directions - 
        a powerful warning signal.
        
        | Type | What Happens | Signal |
        |------|--------------|--------|
        | **Bullish Divergence** | Price makes lower low, RSI makes higher low | Potential bottom |
        | **Bearish Divergence** | Price makes higher high, RSI makes lower high | Potential top |
        """)
        
        st.info("""
        💡 **Pro Tip**: Don't rely on any single indicator. The best traders use multiple 
        indicators together with price action and volume for confirmation.
        """)
        
    elif lesson_id == "volume":
        st.markdown("*📖 Source: Investopedia, CMT Association, Technical Analysis of Stock Trends*")
        
        st.markdown("""
        ## Volume Analysis
        
        **Volume** = The number of shares traded in a given period. It shows the 
        strength of conviction behind price moves.
        
        ### 📊 Volume Principles
        
        | Scenario | Interpretation |
        |----------|----------------|
        | Price UP + Volume UP | Strong bullish move (confirmed) |
        | Price UP + Volume DOWN | Weak rally (potential reversal) |
        | Price DOWN + Volume UP | Strong selling pressure (confirmed) |
        | Price DOWN + Volume DOWN | Weak decline (potential bounce) |
        
        ### 🔑 Key Insight
        
        **"Volume precedes price"** - Often, volume will spike before a major price move.
        Watch for unusual volume as a warning sign of impending movement.
        """)
        
        # Generate volume chart
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=60)
        prices = 100 + np.cumsum(np.random.randn(60) * 1)
        volume = np.random.randint(1000000, 5000000, 60)
        volume[40:45] = volume[40:45] * 3  # Volume spike
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=volume, name='Volume', marker_color='#6366f1', opacity=0.7))
        fig.update_layout(
            title="Volume with Spike Example",
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Volume Indicators
        
        | Indicator | What It Shows |
        |-----------|---------------|
        | **OBV** (On-Balance Volume) | Cumulative volume flow |
        | **Volume MA** | Average volume to spot unusual activity |
        | **VWAP** | Volume-weighted average price (institutional benchmark) |
        | **A/D Line** | Accumulation/Distribution |
        
        ### ⚠️ Volume Red Flags
        
        - **Breakout on low volume** = Likely false breakout
        - **New highs on declining volume** = Weakening trend
        - **Huge volume spike** = Potential climax (reversal coming)
        """)
        
    elif lesson_id == "chart_patterns":
        st.markdown("*📖 Source: Edwards & Magee, Investopedia, CMT Association*")
        
        st.markdown("""
        ## Chart Patterns
        
        Chart patterns are formations that appear on price charts and suggest future 
        price movements. They're based on the idea that market psychology creates 
        recurring patterns.
        
        ### 📊 Reversal Patterns
        
        These patterns signal a potential trend change:
        
        | Pattern | Signal | Characteristics |
        |---------|--------|-----------------|
        | **Head & Shoulders** | Bearish reversal | Three peaks, middle highest |
        | **Inverse H&S** | Bullish reversal | Three troughs, middle lowest |
        | **Double Top** | Bearish reversal | Two peaks at similar level |
        | **Double Bottom** | Bullish reversal | Two troughs at similar level |
        | **Triple Top/Bottom** | Strong reversal | Three tests of level |
        
        ### 📈 Continuation Patterns
        
        These patterns suggest the trend will continue:
        
        | Pattern | Signal | Characteristics |
        |---------|--------|-----------------|
        | **Flag** | Continuation | Small rectangle against trend |
        | **Pennant** | Continuation | Small triangle after sharp move |
        | **Wedge** | Continuation/Reversal | Converging trendlines |
        | **Triangle** | Breakout | Symmetrical, ascending, or descending |
        | **Cup & Handle** | Bullish continuation | U-shape with small pullback |
        
        ### 🎯 Trading Patterns
        
        1. **Identify the pattern** - Wait for it to fully form
        2. **Wait for confirmation** - Breakout with volume
        3. **Set entry** - After breakout confirmation
        4. **Set stop loss** - Below pattern support/above resistance
        5. **Set target** - Often equals the height of the pattern
        """)
        
        st.warning("""
        ⚠️ **Pattern Trading Challenges**:
        - Patterns are **subjective** - different traders see different things
        - Many patterns **fail** - not all breakouts follow through
        - **Hindsight bias** - patterns are easier to see after the fact
        
        Always use stops and manage risk when trading patterns!
        """)


# ==================== MODULE 3.5: FUNDAMENTAL ANALYSIS ====================
def render_fundamental_analysis_content(lesson_id):
    """Render Fundamental Analysis module content."""
    
    if lesson_id == "financial_statements":
        st.markdown("*📖 Source: SEC EDGAR, Fidelity, Morningstar*")
        
        st.markdown("""
        ## Reading Financial Statements
        
        Fundamental analysis starts with understanding a company's financial statements.
        Public companies file these with the SEC and they're available for free.
        
        ### 📋 The Three Core Statements
        
        | Statement | What It Shows | Time Period |
        |-----------|---------------|-------------|
        | **Income Statement** | Revenue, expenses, profit | Quarter or year |
        | **Balance Sheet** | Assets, liabilities, equity | Point in time |
        | **Cash Flow Statement** | Where cash came from/went | Quarter or year |
        
        ### 📊 Income Statement (P&L)
        
        Shows profitability over a period:
        
        ```
        Revenue (Sales)                    $100,000,000
        - Cost of Goods Sold (COGS)        - $40,000,000
        ─────────────────────────────────────────────────
        = Gross Profit                      $60,000,000
        - Operating Expenses               - $30,000,000
        ─────────────────────────────────────────────────
        = Operating Income (EBIT)           $30,000,000
        - Interest Expense                  - $5,000,000
        ─────────────────────────────────────────────────
        = Pre-Tax Income                    $25,000,000
        - Taxes                             - $5,000,000
        ─────────────────────────────────────────────────
        = Net Income                        $20,000,000
        ```
        
        ### 🏦 Balance Sheet
        
        Shows financial position at a moment:
        
        **Assets = Liabilities + Shareholders' Equity**
        
        | Assets (What company owns) | Liabilities (What company owes) |
        |----------------------------|--------------------------------|
        | Cash & equivalents | Accounts payable |
        | Accounts receivable | Short-term debt |
        | Inventory | Long-term debt |
        | Property & equipment | Other liabilities |
        | Intangible assets | **Shareholders' Equity** |
        
        ### 💵 Cash Flow Statement
        
        Shows actual cash movements:
        
        | Section | What It Includes |
        |---------|------------------|
        | **Operating** | Cash from business operations |
        | **Investing** | Capital expenditures, acquisitions |
        | **Financing** | Debt, dividends, stock buybacks |
        """)
        
        st.success("""
        💡 **Key Insight**: Net income can be manipulated with accounting tricks, 
        but cash flow is harder to fake. Always compare net income to operating cash flow.
        If cash flow is consistently lower than net income, investigate why.
        """)
        
    elif lesson_id == "pe_ratio":
        st.markdown("*📖 Source: Morningstar, Fidelity, Investopedia*")
        
        st.markdown("""
        ## P/E Ratio & Valuation Metrics
        
        Valuation metrics help you determine if a stock is cheap, fairly valued, or expensive
        relative to its earnings, assets, or growth.
        
        ### 📊 Price-to-Earnings (P/E) Ratio
        
        **P/E = Stock Price / Earnings Per Share (EPS)**
        
        | P/E Range | Interpretation |
        |-----------|----------------|
        | **Below 10** | Possibly undervalued or troubled |
        | **10-20** | Reasonable for mature companies |
        | **20-30** | Premium valuation, growth expected |
        | **Above 30** | High growth expectations or overvalued |
        """)
        
        # P/E comparison chart
        companies = ['Value Stock', 'S&P 500 Avg', 'Growth Stock', 'High-Growth Tech']
        pe_ratios = [12, 22, 35, 80]
        colors = ['#4ade80', '#fbbf24', '#f97316', '#ef4444']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=companies, y=pe_ratios, marker_color=colors))
        fig.update_layout(
            title="P/E Ratio Comparison",
            height=300,
            yaxis_title="P/E Ratio",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Other Key Valuation Metrics
        
        | Metric | Formula | Best For |
        |--------|---------|----------|
        | **P/E** | Price / EPS | Most companies |
        | **Forward P/E** | Price / Next Year's EPS | Growth companies |
        | **PEG** | P/E / Growth Rate | Comparing growth stocks |
        | **P/S** | Price / Sales | Unprofitable companies |
        | **P/B** | Price / Book Value | Banks, asset-heavy companies |
        | **EV/EBITDA** | Enterprise Value / EBITDA | Comparing across capital structures |
        
        ### 🎯 PEG Ratio (P/E to Growth)
        
        **PEG = P/E / Annual EPS Growth Rate**
        
        | PEG | Interpretation |
        |-----|----------------|
        | **Below 1** | Potentially undervalued |
        | **Around 1** | Fairly valued |
        | **Above 2** | Potentially overvalued |
        
        Example: Company with P/E of 30 and 30% growth rate → PEG = 1.0 (fair)
        """)
        
        st.warning("""
        ⚠️ **Valuation Caveats**:
        - Compare within industries (tech vs utilities have different norms)
        - Consider growth rates (high P/E may be justified)
        - Look at trends over time, not just current values
        - Cheap stocks can stay cheap (value traps)
        """)
        
    elif lesson_id == "growth":
        st.markdown("*📖 Source: Morningstar, Fidelity, Company Filings*")
        
        st.markdown("""
        ## Revenue & Earnings Growth
        
        Growth is the engine of stock returns. Understanding how to analyze 
        growth helps you identify winning companies.
        
        ### 📈 Revenue Growth
        
        Revenue (sales) is the "top line" - all growth starts here.
        
        | Growth Rate | Assessment |
        |-------------|------------|
        | **< 5%** | Slow growth, mature company |
        | **5-15%** | Moderate growth |
        | **15-25%** | Strong growth |
        | **> 25%** | High growth (often tech, disruptors) |
        
        ### 💰 Earnings Growth
        
        Earnings (profits) are the "bottom line" - what shareholders actually own.
        
        **Quality of earnings matters:**
        - Is growth from revenue increase or cost cutting?
        - Is it sustainable or one-time?
        - Does cash flow support reported earnings?
        """)
        
        # Growth comparison
        years = ['2020', '2021', '2022', '2023', '2024']
        revenue = [100, 115, 135, 160, 190]
        earnings = [10, 12, 15, 20, 25]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=revenue, name='Revenue', 
                                 line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=years, y=earnings, name='Earnings', 
                                 line=dict(color='#4ade80', width=3)))
        fig.update_layout(
            title="Revenue vs Earnings Growth Example",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 Key Growth Metrics
        
        | Metric | What to Look For |
        |--------|------------------|
        | **Revenue CAGR** | Compound annual growth rate (3-5 year) |
        | **EPS Growth** | Earnings per share trend |
        | **Margin Expansion** | Growing profits faster than revenue |
        | **TAM Growth** | Is the total addressable market growing? |
        | **Market Share** | Is the company gaining vs competitors? |
        
        ### ⚠️ Growth Red Flags
        
        - Slowing growth rates quarter over quarter
        - Revenue growth without profit growth
        - Growth only from acquisitions
        - Unsustainable customer acquisition costs
        - Growth dependent on one product/customer
        """)
        
    elif lesson_id == "balance_sheet":
        st.markdown("*📖 Source: Fidelity, Morningstar, Warren Buffett's Letters*")
        
        st.markdown("""
        ## Balance Sheet Analysis
        
        The balance sheet shows financial health at a point in time.
        Strong balance sheets protect companies during downturns.
        
        ### 🏦 Key Balance Sheet Metrics
        
        | Metric | Formula | What It Shows |
        |--------|---------|---------------|
        | **Current Ratio** | Current Assets / Current Liabilities | Short-term liquidity |
        | **Quick Ratio** | (Current Assets - Inventory) / Current Liabilities | Immediate liquidity |
        | **Debt-to-Equity** | Total Debt / Shareholders' Equity | Leverage level |
        | **Book Value** | Assets - Liabilities | Net worth |
        | **Working Capital** | Current Assets - Current Liabilities | Operating cushion |
        
        ### 📊 Healthy Ranges
        
        | Metric | Healthy Range | Warning Sign |
        |--------|---------------|--------------|
        | **Current Ratio** | > 1.5 | < 1.0 |
        | **Quick Ratio** | > 1.0 | < 0.5 |
        | **Debt/Equity** | < 1.0 | > 2.0 |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Signs of Strength
            - Growing cash position
            - Decreasing debt levels
            - Increasing book value
            - No goodwill impairments
            - Consistent inventory turnover
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Warning Signs
            - Declining cash, rising debt
            - Large goodwill/intangibles
            - Rising accounts receivable
            - Inventory buildup
            - Off-balance sheet liabilities
            """)
        
        st.info("""
        💡 **Buffett's Rule**: "When a management team with a reputation for brilliance 
        tackles a business with a reputation for bad economics, it is the reputation 
        of the business that remains intact." Look for strong balance sheets first.
        """)
        
    elif lesson_id == "cash_flow":
        st.markdown("*📖 Source: Fidelity, Morningstar, SEC*")
        
        st.markdown("""
        ## Cash Flow Analysis
        
        **"Cash is king."** The cash flow statement shows actual cash movements, 
        which is harder to manipulate than earnings.
        
        ### 💵 Three Types of Cash Flow
        
        | Type | What It Includes | What to Look For |
        |------|------------------|------------------|
        | **Operating (CFO)** | Cash from business | Positive, growing |
        | **Investing (CFI)** | CapEx, acquisitions | Usually negative (investing in growth) |
        | **Financing (CFF)** | Debt, dividends, buybacks | Varies by strategy |
        
        ### 📊 Free Cash Flow (FCF)
        
        **FCF = Operating Cash Flow - Capital Expenditures**
        
        This is the cash available to:
        - Pay dividends
        - Buy back stock
        - Pay down debt
        - Make acquisitions
        - Invest in growth
        
        **Free Cash Flow Yield = FCF / Market Cap**
        - Above 5% = potentially undervalued
        - Below 2% = expensive or reinvesting heavily
        """)
        
        # Cash flow visualization
        categories = ['Net Income', 'Operating CF', 'Free Cash Flow']
        values = [100, 120, 80]
        colors = ['#6366f1', '#4ade80', '#fbbf24']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors))
        fig.update_layout(
            title="Cash Flow Comparison (Healthy Company)",
            height=300,
            yaxis_title="$ Millions",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 Cash Flow Quality Checks
        
        | Check | Healthy Sign | Warning Sign |
        |-------|--------------|--------------|
        | **CFO vs Net Income** | CFO > Net Income | CFO << Net Income |
        | **FCF Trend** | Growing over time | Declining or negative |
        | **CapEx/Depreciation** | Roughly equal | CapEx much higher (playing catch-up) |
        | **Working Capital** | Stable or improving | Deteriorating rapidly |
        """)
        
        st.warning("""
        ⚠️ **Red Flag**: If a company reports profits but burns cash, investigate. 
        Common causes: aggressive revenue recognition, inventory issues, or 
        unsustainable working capital practices.
        """)
        
    elif lesson_id == "moats":
        st.markdown("*📖 Source: Morningstar, Warren Buffett's Letters, Pat Dorsey*")
        
        st.markdown("""
        ## Competitive Moats
        
        A **moat** is a sustainable competitive advantage that protects a company's 
        profits from competitors - like the moat around a castle.
        
        ### 🏰 Types of Economic Moats
        
        | Moat Type | Description | Examples |
        |-----------|-------------|----------|
        | **Network Effects** | Product gets better with more users | Visa, Facebook, eBay |
        | **Switching Costs** | Painful for customers to switch | Microsoft, Adobe, Salesforce |
        | **Intangible Assets** | Brands, patents, licenses | Coca-Cola, Disney, Pfizer |
        | **Cost Advantage** | Produce cheaper than competitors | Walmart, Amazon, Costco |
        | **Efficient Scale** | Market only supports few players | Railroads, utilities |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Signs of a Moat
            
            - High and stable profit margins
            - Returns on capital above 15%
            - Pricing power (can raise prices)
            - Market share stability or gains
            - Customer retention above 90%
            - High barriers to entry
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Moat Erosion Signs
            
            - Declining margins over time
            - Losing market share
            - Needing to compete on price
            - Technology disruption
            - Regulatory changes
            - New well-funded competitors
            """)
        
        st.markdown("""
        ### 📊 Moat Metrics
        
        | Metric | Moat Indication | No Moat |
        |--------|-----------------|---------|
        | **ROIC** | > 15% sustained | < 10% or declining |
        | **Gross Margin** | > 40% stable | < 20% or falling |
        | **Customer Retention** | > 90% | < 80% |
        | **Market Share** | Growing or stable | Declining |
        """)
        
        st.success("""
        💡 **Buffett's Advice**: "The key to investing is not assessing how much an industry 
        is going to affect society, but rather determining the competitive advantage of 
        any given company and, above all, the durability of that advantage."
        """)


# ==================== MODULE 4: QUANT STRATEGIES ====================
def render_quant_strategies_content(lesson_id):
    """Render Quant Strategies module content."""
    
    if lesson_id == "factor_investing":
        st.markdown("*📖 Source: AQR Capital, Fama-French Research, MSCI*")
        
        st.markdown("""
        ## Factor Investing
        
        **Factor investing** identifies characteristics (factors) that explain 
        differences in returns across assets. It's the foundation of modern 
        quantitative investing.
        
        ### 📊 The Classic Factors
        
        | Factor | Description | Academic Support |
        |--------|-------------|------------------|
        | **Market (Beta)** | Exposure to overall market | CAPM (1960s) |
        | **Size** | Small caps outperform large caps | Fama-French (1992) |
        | **Value** | Cheap stocks beat expensive | Fama-French (1992) |
        | **Momentum** | Recent winners keep winning | Jegadeesh-Titman (1993) |
        | **Quality** | Profitable companies outperform | Novy-Marx (2013) |
        | **Low Volatility** | Less risky stocks outperform | Black (1972) |
        """)
        
        # Factor returns visualization
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low Vol']
        premiums = [7.0, 2.0, 3.5, 6.0, 3.0, 2.5]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=factors, y=premiums, marker_color='#6366f1'))
        fig.update_layout(
            title="Historical Factor Premiums (Annual %)",
            height=300,
            yaxis_title="Premium (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Factor ETFs for Individual Investors
        
        | Factor | ETF Examples | Expense Ratio |
        |--------|--------------|---------------|
        | **Value** | VTV, IWD, VLUE | 0.04-0.15% |
        | **Size (Small)** | VB, IJR, IWM | 0.05-0.20% |
        | **Momentum** | MTUM, QMOM | 0.15-0.25% |
        | **Quality** | QUAL, SPHQ | 0.15-0.30% |
        | **Low Vol** | USMV, SPLV | 0.15-0.25% |
        | **Multi-Factor** | LRGF, VFMF | 0.10-0.20% |
        
        ### ⚠️ Factor Investing Caveats
        
        1. **Factors can underperform for years** - Value was "dead" 2010-2020
        2. **Crowding risk** - Popular factors may be arbitraged away
        3. **Implementation costs** - Transaction costs erode premiums
        4. **Factor timing is hard** - Don't try to time factor rotations
        """)
        
        st.info("""
        💡 **Practical Advice**: For most investors, a simple value tilt (like VTV) or 
        multi-factor fund provides factor exposure without complexity. Don't over-engineer it.
        """)
        
    elif lesson_id == "mean_reversion":
        st.markdown("*📖 Source: Academic Research, AQR Capital, Two Sigma*")
        
        st.markdown("""
        ## Mean Reversion Strategies
        
        **Mean reversion** is the theory that prices tend to return to their average 
        over time. When something moves too far from normal, it tends to snap back.
        
        ### 📊 The Concept
        
        If a stock is:
        - **Far below average** → May be oversold → Potential buy
        - **Far above average** → May be overbought → Potential sell
        
        ### 🎯 Mean Reversion Indicators
        
        | Indicator | Mean Reversion Signal |
        |-----------|----------------------|
        | **RSI** | Below 30 (oversold) or above 70 (overbought) |
        | **Bollinger Bands** | Price at lower/upper band |
        | **Z-Score** | Standard deviations from mean |
        | **Pairs Spread** | Spread between correlated assets widens |
        """)
        
        # Mean reversion example
        np.random.seed(42)
        x = np.arange(200)
        mean = 100
        prices = mean + np.cumsum(np.random.randn(200)) * 0.5
        # Add mean-reverting component
        prices = prices - 0.1 * (prices - mean) + np.random.randn(200) * 0.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, name='Price', line=dict(color='#94a3b8')))
        fig.add_hline(y=mean, line_dash="dash", line_color="#6366f1", annotation_text="Mean")
        fig.add_hline(y=mean+5, line_dash="dot", line_color="#ef4444", annotation_text="+2σ")
        fig.add_hline(y=mean-5, line_dash="dot", line_color="#4ade80", annotation_text="-2σ")
        fig.update_layout(
            title="Mean Reversion Example",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Example Strategy: RSI Mean Reversion
        
        1. **Buy signal**: RSI drops below 30 (oversold)
        2. **Sell signal**: RSI rises above 70 (overbought)
        3. **Risk management**: Stop loss if price continues against you
        
        ### ⚠️ The Catch: Trending Markets
        
        Mean reversion **fails** when markets trend:
        - In 2008, "cheap" stocks kept getting cheaper
        - Catching a falling knife = buying into a crash
        
        **"The market can stay irrational longer than you can stay solvent."** - Keynes
        """)
        
        st.warning("""
        ⚠️ **Key Risk**: Mean reversion assumes prices will return to normal. 
        But sometimes the "mean" itself has shifted. A stock down 50% might 
        deserve to be down 50% due to fundamental changes.
        """)
        
    elif lesson_id == "momentum":
        st.markdown("*📖 Source: AQR Capital, Cliff Asness Research, Academic Papers*")
        
        st.markdown("""
        ## Momentum Strategies
        
        **Momentum** is the tendency for recent winners to keep winning and recent 
        losers to keep losing. It's one of the most robust anomalies in finance.
        
        ### 📊 The Evidence
        
        - Documented across stocks, bonds, currencies, and commodities
        - Works in markets globally
        - Persisted for 200+ years of data
        - ~6% annual premium historically
        
        ### 🎯 Types of Momentum
        
        | Type | Lookback Period | Holding Period |
        |------|-----------------|----------------|
        | **Short-term** | 1 week - 1 month | Days to weeks |
        | **Intermediate** | 3-6 months | 1-3 months |
        | **Long-term** | 6-12 months | 3-6 months |
        
        **The classic strategy**: Buy stocks with best 12-month returns (excluding last month),
        hold for 3-6 months.
        """)
        
        st.markdown("""
        ### 📈 Simple Momentum Rules
        
        1. **Trend Following**: Buy when price > 200-day moving average
        2. **Relative Momentum**: Buy top 10% performers, sell bottom 10%
        3. **Dual Momentum**: Combine absolute and relative momentum
        
        ### ⚠️ Momentum Risks
        
        | Risk | Description |
        |------|-------------|
        | **Momentum Crashes** | Sharp reversals (2009, 2020) |
        | **High Turnover** | Frequent trading = high costs |
        | **Crowding** | Too many momentum traders |
        | **Crash Risk** | Momentum does worst when market recovers |
        
        ### 💡 Practical Implementation
        
        For individual investors, consider:
        - **MTUM** - iShares Momentum ETF
        - **QMOM** - Alpha Architect Momentum
        - **Simple rule**: Only hold stocks trading above their 200-day MA
        """)
        
    elif lesson_id == "stat_arb":
        st.markdown("*📖 Source: Quantitative Trading, Two Sigma, Renaissance Technologies*")
        
        st.markdown("""
        ## Statistical Arbitrage
        
        **Statistical arbitrage** (stat arb) exploits pricing inefficiencies between 
        related securities using mathematical models. It's a mainstay of hedge fund strategies.
        
        ### 📊 Core Concepts
        
        **Pairs Trading**: The classic stat arb strategy
        
        1. Find two stocks that move together (high correlation)
        2. When their relationship diverges (spread widens)
        3. Short the outperformer, long the underperformer
        4. Wait for relationship to normalize
        5. Close both positions for profit
        """)
        
        # Pairs trading visualization
        np.random.seed(42)
        x = np.arange(100)
        common_factor = np.cumsum(np.random.randn(100)) * 0.5
        stock_a = 100 + common_factor + np.cumsum(np.random.randn(100)) * 0.2
        stock_b = 100 + common_factor + np.cumsum(np.random.randn(100)) * 0.2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=stock_a, name='Stock A', line=dict(color='#6366f1')))
        fig.add_trace(go.Scatter(y=stock_b, name='Stock B', line=dict(color='#4ade80')))
        fig.update_layout(
            title="Pairs Trading: Two Correlated Stocks",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Common Stat Arb Approaches
        
        | Strategy | Description | Risk Level |
        |----------|-------------|------------|
        | **Pairs Trading** | Two correlated stocks | Moderate |
        | **Index Arbitrage** | ETF vs underlying stocks | Low |
        | **Sector Neutral** | Long/short within sectors | Moderate |
        | **Market Neutral** | Zero beta overall | Low market risk |
        
        ### ⚠️ Why It's Hard
        
        1. **Requires sophisticated tools** - Math, programming, data
        2. **Alpha decay** - Strategies stop working as they get crowded
        3. **Model risk** - Relationships can permanently break
        4. **Execution matters** - Need low-cost, fast trading
        5. **Competition** - You're competing with PhDs and supercomputers
        """)
        
        st.info("""
        💡 **Reality Check**: Successful stat arb requires institutional-grade infrastructure, 
        data, and execution. For individual investors, factor-based ETFs capture similar 
        concepts with much less complexity.
        """)
        
    elif lesson_id == "backtesting":
        st.markdown("*📖 Source: Quantitative Trading Research, Systematic Trading*")
        
        st.markdown("""
        ## Backtesting & Validation
        
        **Backtesting** tests a trading strategy on historical data to see how it 
        would have performed. It's essential but full of traps.
        
        ### 📊 The Backtesting Process
        
        1. **Define strategy rules** precisely
        2. **Gather clean historical data**
        3. **Run simulation** on past data
        4. **Analyze results** (returns, drawdowns, Sharpe)
        5. **Validate** with out-of-sample testing
        6. **Paper trade** before real money
        
        ### ⚠️ Backtesting Pitfalls
        
        | Pitfall | Description | Solution |
        |---------|-------------|----------|
        | **Overfitting** | Curve-fitting to historical noise | Keep rules simple |
        | **Survivorship Bias** | Only testing stocks that survived | Use delisting data |
        | **Look-Ahead Bias** | Using future data in decisions | Careful data handling |
        | **Transaction Costs** | Ignoring fees, slippage | Realistic cost modeling |
        | **Data Snooping** | Testing many strategies until one works | Out-of-sample testing |
        """)
        
        st.markdown("""
        ### 📈 Validation Framework
        
        ```
        Total Data
        ├── In-Sample (60%) - Develop strategy
        ├── Validation (20%) - Tune parameters  
        └── Out-of-Sample (20%) - Final test (ONLY ONCE!)
        ```
        
        ### 🎯 Key Metrics to Evaluate
        
        | Metric | Good Value | What It Measures |
        |--------|------------|------------------|
        | **Sharpe Ratio** | > 1.0 | Risk-adjusted return |
        | **Max Drawdown** | < 20% | Worst peak-to-trough |
        | **Win Rate** | > 50% | Percentage of winning trades |
        | **Profit Factor** | > 1.5 | Gross profit / gross loss |
        | **Calmar Ratio** | > 1.0 | Return / max drawdown |
        """)
        
        st.error("""
        ⚠️ **Golden Rule**: A backtest that looks too good is probably wrong. 
        The more impressive the results, the more skeptical you should be. 
        Real-world performance is almost always worse than backtests.
        """)
        
    elif lesson_id == "risk_systems":
        st.markdown("*📖 Source: Risk Management, AQR Capital, Professional Trading*")
        
        st.markdown("""
        ## Risk Management Systems
        
        **"Risk management is more important than return management."**
        
        Professional traders spend more time on risk than on finding trades.
        
        ### 📊 Position Sizing
        
        Never risk too much on any single trade:
        
        | Method | Rule | Example |
        |--------|------|---------|
        | **Fixed Percentage** | Risk 1-2% per trade | $100K account → max $2K risk |
        | **Kelly Criterion** | Optimal f = (p*b - q) / b | Math-based sizing |
        | **Volatility-Based** | Size inversely to volatility | Smaller positions in volatile markets |
        
        ### 🛡️ Risk Limits
        
        | Limit Type | Description |
        |------------|-------------|
        | **Position Limit** | Max % in any single stock |
        | **Sector Limit** | Max % in any sector |
        | **Drawdown Limit** | Reduce risk after losses |
        | **Correlation Limit** | Diversify across uncorrelated bets |
        | **Leverage Limit** | Max borrowed amount |
        """)
        
        st.markdown("""
        ### 📈 Stop Loss Strategies
        
        | Type | How It Works |
        |------|--------------|
        | **Fixed Stop** | Sell if down X% |
        | **ATR Stop** | Stop based on volatility (2x ATR) |
        | **Trailing Stop** | Stop moves up with price |
        | **Time Stop** | Exit if no movement in X days |
        
        ### 🎯 The 2% Rule
        
        Never risk more than 2% of your account on any single trade.
        
        Example:
        - Account: $50,000
        - Max risk per trade: $1,000 (2%)
        - If stop loss is 10% below entry → Position size: $10,000
        """)
        
        st.success("""
        💡 **Professional Wisdom**: "Amateurs focus on returns. Professionals focus on risk."
        The best traders are those who survive long enough for their edge to play out.
        Preservation of capital always comes first.
        """)


# ==================== MODULE 5: ADVANCED OPTIONS ====================
def render_advanced_options_content(lesson_id):
    """Render Advanced Options module content."""
    
    if lesson_id == "greeks":
        st.markdown("*📖 Source: CBOE, Options Industry Council, Natenberg*")
        
        st.markdown("""
        ## Options Greeks Deep Dive
        
        The **Greeks** measure how an option's price changes in response to various 
        factors. Understanding Greeks is essential for options trading.
        
        ### 📊 The Five Greeks
        
        | Greek | Measures | Range | Call | Put |
        |-------|----------|-------|------|-----|
        | **Delta (Δ)** | Price sensitivity to stock move | 0 to 1 / -1 to 0 | Positive | Negative |
        | **Gamma (Γ)** | Rate of change of delta | Always positive | Same | Same |
        | **Theta (Θ)** | Time decay per day | Usually negative | Negative | Negative |
        | **Vega (ν)** | Sensitivity to volatility | Always positive | Same | Same |
        | **Rho (ρ)** | Sensitivity to interest rates | Varies | Positive | Negative |
        """)
        
        st.markdown("""
        ### 🎯 Delta Explained
        
        Delta tells you how much the option price moves for a $1 stock move.
        
        | Delta | Interpretation |
        |-------|----------------|
        | **0.50** | At-the-money (50% chance of expiring ITM) |
        | **0.80** | Deep in-the-money |
        | **0.20** | Out-of-the-money |
        | **1.00** | So deep ITM it acts like stock |
        
        **Delta as hedge ratio**: To delta-hedge 1 call with 0.50 delta, short 50 shares.
        
        ### ⏰ Theta Explained
        
        Theta is the daily cost of holding an option. It accelerates near expiration.
        
        - **Long options**: Theta works against you
        - **Short options**: Theta works for you
        - **At expiration**: All time value = 0
        """)
        
        # Theta decay visualization
        days = np.arange(60, 0, -1)
        theta_decay = np.sqrt(days) / np.sqrt(60) * 5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=60-days, y=theta_decay, fill='tozeroy',
                                 line=dict(color='#ef4444'),
                                 fillcolor='rgba(239, 68, 68, 0.3)'))
        fig.update_layout(
            title="Option Time Value Decay",
            height=300,
            xaxis_title="Days Until Expiration",
            yaxis_title="Time Value ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Vega Explained
        
        Vega measures sensitivity to implied volatility (IV).
        
        - High IV = Options are expensive
        - Low IV = Options are cheap
        - Vega is highest for ATM options with longer expirations
        
        **Trading Tip**: Buy options when IV is low, sell when IV is high.
        """)
        
    elif lesson_id == "vertical_spreads":
        st.markdown("*📖 Source: CBOE, Options Playbook, Tastytrade*")
        
        st.markdown("""
        ## Vertical Spreads
        
        A **vertical spread** involves buying and selling options of the same type 
        (calls or puts) with the same expiration but different strikes.
        
        ### 📊 The Four Vertical Spreads
        
        | Spread | Construction | Outlook | Max Profit | Max Loss |
        |--------|--------------|---------|------------|----------|
        | **Bull Call** | Buy lower call, sell higher call | Bullish | Strike diff - premium | Premium paid |
        | **Bear Put** | Buy higher put, sell lower put | Bearish | Strike diff - premium | Premium paid |
        | **Bull Put** | Sell higher put, buy lower put | Bullish | Premium received | Strike diff - premium |
        | **Bear Call** | Sell lower call, buy higher call | Bearish | Premium received | Strike diff - premium |
        
        ### 🎯 Bull Call Spread Example
        
        Stock at $100, moderately bullish:
        
        - Buy 100 call @ $3.00
        - Sell 105 call @ $1.50
        - Net cost: $1.50
        - Max profit: $5.00 - $1.50 = **$3.50** (if stock > $105)
        - Max loss: **$1.50** (if stock < $100)
        - Breakeven: $101.50
        """)
        
        # P/L diagram
        stock_prices = np.arange(95, 115, 0.5)
        bull_call_pl = np.minimum(np.maximum(stock_prices - 100, 0), 5) - 1.5 - \
                       np.minimum(np.maximum(stock_prices - 105, 0), 0)
        bull_call_pl = np.minimum(np.maximum(stock_prices - 100, 0) - 
                                  np.maximum(stock_prices - 105, 0), 5) - 1.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=bull_call_pl * 100,
                                 fill='tozeroy', line=dict(color='#4ade80'),
                                 fillcolor='rgba(74, 222, 128, 0.3)'))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Bull Call Spread P/L Diagram",
            height=300,
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### ✅ Advantages of Spreads
        
        - **Defined risk** - Know max loss upfront
        - **Lower cost** - Cheaper than single options
        - **Theta benefit** - Short leg offsets some decay
        - **Lower breakeven** - Than buying calls outright
        
        ### ⚠️ Disadvantages
        
        - **Capped profit** - Can't make unlimited gains
        - **Complexity** - Two legs to manage
        - **Assignment risk** - Short leg can be assigned
        """)
        
    elif lesson_id == "iron_condor":
        st.markdown("*📖 Source: CBOE, Tastytrade, Options Alpha*")
        
        st.markdown("""
        ## Iron Condors & Butterflies
        
        These are **neutral strategies** that profit when the stock stays in a range.
        They're popular for income generation.
        
        ### 🦅 Iron Condor
        
        Combines a bull put spread and bear call spread:
        
        ```
        Sell put  @ 95  │
        Buy put   @ 90  │ Bull Put Spread
        ─────────────────
        Sell call @ 105 │
        Buy call  @ 110 │ Bear Call Spread
        ```
        
        **Profit zone**: Stock stays between 95 and 105
        **Max profit**: Premium collected
        **Max loss**: Width of spread - premium
        """)
        
        # Iron condor P/L
        stock_prices = np.arange(85, 120, 0.5)
        ic_pl = np.zeros_like(stock_prices)
        premium = 2.0
        
        for i, price in enumerate(stock_prices):
            put_spread = max(95 - price, 0) - max(90 - price, 0)
            call_spread = max(price - 105, 0) - max(price - 110, 0)
            ic_pl[i] = premium - put_spread - call_spread
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=ic_pl * 100,
                                 fill='tozeroy', line=dict(color='#6366f1'),
                                 fillcolor='rgba(99, 102, 241, 0.3)'))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Iron Condor P/L Diagram",
            height=300,
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🦋 Butterfly Spread
        
        Similar to iron condor but with overlapping strikes:
        
        ```
        Buy 1  put/call @ 95
        Sell 2 put/call @ 100
        Buy 1  put/call @ 105
        ```
        
        **Max profit**: At exactly $100 (middle strike)
        **Max loss**: Premium paid
        
        ### 📊 When to Use Each
        
        | Strategy | Best When | Win Rate | Profit Potential |
        |----------|-----------|----------|------------------|
        | **Iron Condor** | Low volatility expected | Higher | Lower per trade |
        | **Butterfly** | Price pin expected | Lower | Higher per trade |
        """)
        
    elif lesson_id == "calendar_spreads":
        st.markdown("*📖 Source: CBOE, Options Playbook, Professional Trading*")
        
        st.markdown("""
        ## Calendar & Diagonal Spreads
        
        These spreads use different expiration dates to profit from time decay 
        and volatility differences.
        
        ### 📅 Calendar Spread (Time Spread)
        
        Same strike, different expirations:
        
        - **Sell** near-term option (faster theta decay)
        - **Buy** longer-term option (slower theta decay)
        
        **Profit from**: Time decay differential
        **Best when**: Stock stays near strike, IV increases
        
        ### 📐 Diagonal Spread
        
        Different strikes AND different expirations:
        
        - **Sell** near-term, out-of-money option
        - **Buy** longer-term, in-the-money option
        
        **Combines**: Directional bias + time decay
        """)
        
        st.markdown("""
        ### 📊 Calendar Spread Example
        
        Stock at $100:
        - Sell 30-day $100 call @ $2.00
        - Buy 60-day $100 call @ $3.50
        - Net cost: $1.50
        
        **Scenarios:**
        
        | Stock Price | 30 Days Later | Result |
        |-------------|---------------|--------|
        | $100 | Near-term expires worthless | Keep $2, long call has value |
        | $95 | Both decline | Small loss |
        | $105 | Near-term ITM | May need to roll |
        
        ### ⚠️ Key Risks
        
        1. **Movement away from strike** - Both calendars and diagonals lose if stock moves too far
        2. **IV collapse** - Hurts long option more than short
        3. **Early assignment** - Short option can be assigned
        4. **Management required** - Not set-and-forget
        """)
        
    elif lesson_id == "vol_trading":
        st.markdown("*📖 Source: Volatility Trading by Sinclair, CBOE VIX, Tastytrade*")
        
        st.markdown("""
        ## Volatility Trading
        
        Instead of betting on direction, you can bet on **volatility** itself.
        
        ### 📊 Implied vs Realized Volatility
        
        | Type | Definition | Trading Implication |
        |------|------------|---------------------|
        | **Implied (IV)** | Market's expected future volatility | What you're paying for |
        | **Realized (HV)** | Actual historical volatility | What actually happens |
        
        **The edge**: IV is often higher than realized volatility 
        (volatility risk premium). Sellers have a long-term advantage.
        """)
        
        # VIX chart example
        np.random.seed(42)
        vix = 20 + np.cumsum(np.random.randn(100) * 0.5)
        vix = np.clip(vix, 12, 40)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vix, name='VIX', line=dict(color='#f97316', width=2)))
        fig.add_hline(y=20, line_dash="dash", line_color="#4ade80", annotation_text="Historical Average ~20")
        fig.update_layout(
            title="VIX (Volatility Index) Example",
            height=300,
            yaxis_title="VIX Level",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Volatility Strategies
        
        | Strategy | Position | Profits When |
        |----------|----------|--------------|
        | **Long Straddle** | Buy call + put same strike | Big move either direction |
        | **Short Straddle** | Sell call + put same strike | Stock doesn't move |
        | **Long Strangle** | Buy OTM call + put | Big move, cheaper than straddle |
        | **Short Strangle** | Sell OTM call + put | Stock stays in range |
        
        ### 📈 VIX as Fear Gauge
        
        | VIX Level | Market Condition |
        |-----------|------------------|
        | **Below 15** | Complacent, low fear |
        | **15-20** | Normal |
        | **20-30** | Elevated concern |
        | **Above 30** | Fear/panic |
        | **Above 40** | Crisis levels |
        """)
        
        st.warning("""
        ⚠️ **Risk Warning**: Selling volatility (short straddles/strangles) has 
        unlimited risk. A single market crash can wipe out years of gains. 
        Always use defined-risk structures or strict position sizing.
        """)
        
    elif lesson_id == "hedging":
        st.markdown("*📖 Source: Options as a Strategic Investment, CBOE, Professional Risk Management*")
        
        st.markdown("""
        ## Portfolio Hedging
        
        Options can protect your portfolio from downside risk while maintaining 
        upside potential. Here are the key hedging strategies.
        
        ### 🛡️ Protective Put (Portfolio Insurance)
        
        Buy puts against your stock holdings:
        
        - **Own 100 shares** of SPY at $450
        - **Buy 1 put** at $430 strike for $5
        - **Max loss**: $450 - $430 + $5 = $25 (5.5%)
        - **Upside**: Unlimited (minus put cost)
        
        **Cost**: 1-3% of portfolio value per year
        
        ### 📊 Collar Strategy
        
        Protective put + covered call to reduce cost:
        
        ```
        Own stock      @ $100
        Buy put        @ $95  ($2)
        Sell call      @ $105 ($2)
        Net cost: $0 (zero-cost collar)
        ```
        
        **Tradeoff**: Give up upside above $105 to pay for downside protection below $95.
        """)
        
        # Collar P/L
        stock_prices = np.arange(85, 120, 0.5)
        stock_pl = stock_prices - 100
        collar_pl = np.minimum(np.maximum(stock_prices, 95), 105) - 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=stock_pl, name='Stock Only',
                                 line=dict(color='#94a3b8', dash='dash')))
        fig.add_trace(go.Scatter(x=stock_prices, y=collar_pl, name='Collar',
                                 line=dict(color='#4ade80', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Stock vs Collar P/L",
            height=300,
            xaxis_title="Stock Price",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Hedging Strategies Compared
        
        | Strategy | Cost | Protection | Upside |
        |----------|------|------------|--------|
        | **Protective Put** | High | Full below strike | Unlimited |
        | **Collar** | Low/Zero | Full below put strike | Capped at call strike |
        | **Put Spread** | Medium | Partial | Unlimited |
        | **VIX Calls** | Variable | Portfolio-wide | N/A (hedge only) |
        
        ### 📅 When to Hedge
        
        | Scenario | Hedge? |
        |----------|--------|
        | Long time horizon | Usually no - ride out volatility |
        | Near retirement | Consider collars |
        | Concentrated position | Yes - reduce single-stock risk |
        | Expecting volatility | Evaluate cost/benefit |
        | Market at all-time highs | Hedging is cheap when IV is low |
        """)
        
        st.info("""
        💡 **Professional Perspective**: Most long-term investors don't need to hedge.
        Time and diversification provide natural protection. Hedging makes sense for 
        concentrated positions, short time horizons, or when you can't afford a drawdown.
        """)


if __name__ == "__main__":
    main()
