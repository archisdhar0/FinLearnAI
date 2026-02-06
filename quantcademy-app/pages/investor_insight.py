"""
QuantCademy - Investor Insight Track
Deeper understanding of markets and investment selection
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag.knowledge_base import get_documents_by_category, KNOWLEDGE_BASE
except ImportError:
    KNOWLEDGE_BASE = {}
    def get_documents_by_category(cat): return []

st.set_page_config(
    page_title="Investor Insight | QuantCademy",
    page_icon="💡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .module-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    .concept-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
    }
    .comparison-table {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Navigation
    tab1, tab2, tab3 = st.tabs([
        "📈 Understanding Markets",
        "🎯 Smart ETF Selection", 
        "⚖️ Passive vs Active"
    ])
    
    with tab1:
        render_market_understanding()
    
    with tab2:
        render_etf_selection()
    
    with tab3:
        render_passive_active()


def render_market_understanding():
    """Module: Understanding the Market."""
    st.markdown("""
    <div class="module-header">
        <h1>📈 Understanding the Market</h1>
        <p>Learn how markets work, what drives prices, and why you don't need to predict them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key concept 1: Price Discovery
    st.markdown("### 💡 How Prices Are Set")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>Supply & Demand</h4>
            <p>Stock prices are determined by buyers and sellers:</p>
            <ul>
                <li><strong>More buyers than sellers</strong> → Price rises</li>
                <li><strong>More sellers than buyers</strong> → Price falls</li>
                <li>This happens in <strong>milliseconds</strong></li>
            </ul>
            <p><em>You don't need to predict this - own the whole market via index funds!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Simple supply/demand visualization
        prices = np.linspace(50, 150, 100)
        demand = 100 - 0.5 * prices + 50
        supply = 0.5 * prices
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=demand, name='Demand', line=dict(color='#ef4444')))
        fig.add_trace(go.Scatter(x=prices, y=supply, name='Supply', line=dict(color='#10b981')))
        fig.add_vline(x=100, line_dash="dash", annotation_text="Equilibrium Price")
        fig.update_layout(
            title="Price Discovery",
            xaxis_title="Price ($)",
            yaxis_title="Quantity",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Key concept 2: Market Indices
    st.markdown("### 📊 What Are Market Indices?")
    
    indices_data = {
        "Index": ["S&P 500", "Total Stock Market", "Dow Jones", "NASDAQ-100", "Russell 2000"],
        "What It Tracks": [
            "500 largest US companies",
            "~4,000 US companies (all sizes)",
            "30 large industrial companies",
            "100 largest non-financial NASDAQ stocks",
            "2,000 small US companies"
        ],
        "% of US Market": ["~80%", "~100%", "~25%", "~40%", "~10%"],
        "Best For": [
            "Core US exposure",
            "Complete US market",
            "Outdated, not recommended",
            "Tech-heavy exposure",
            "Small company exposure"
        ],
        "Example ETF": ["VOO, SPY", "VTI, ITOT", "DIA", "QQQ", "IWM"]
    }
    
    st.dataframe(pd.DataFrame(indices_data), use_container_width=True, hide_index=True)
    
    st.info("💡 **Pro tip:** Total Stock Market (VTI) or S&P 500 (VOO) are the best choices for most investors.")
    
    st.markdown("---")
    
    # Key concept 3: Market Efficiency
    st.markdown("### 🎲 The Efficient Market Hypothesis")
    
    st.markdown("""
    <div class="concept-card">
        <h4>What It Means For You</h4>
        <p>The <strong>Efficient Market Hypothesis (EMH)</strong> suggests that stock prices 
        already reflect all available information. This has important implications:</p>
        <ul>
            <li>📰 <strong>News is priced in instantly</strong> - By the time you hear about something, the price has already moved</li>
            <li>🔮 <strong>Predictions don't work</strong> - Nobody can consistently predict which stocks will outperform</li>
            <li>💰 <strong>Past performance ≠ future results</strong> - Last year's winners often become next year's losers</li>
            <li>📈 <strong>Index funds win</strong> - If you can't beat the market, just own it!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Active manager performance chart
    st.markdown("#### Evidence: Active Managers vs Index")
    
    years = [1, 3, 5, 10, 15, 20]
    underperform_pct = [64, 71, 82, 85, 90, 94]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{y} Year{'s' if y > 1 else ''}" for y in years],
        y=underperform_pct,
        marker_color=['#f59e0b' if p < 80 else '#ef4444' for p in underperform_pct],
        text=[f"{p}%" for p in underperform_pct],
        textposition='outside'
    ))
    fig.update_layout(
        title="% of Active Funds That Underperform Their Index (S&P 500)",
        yaxis_title="Percentage Underperforming",
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("**Takeaway:** Over 90% of professional fund managers fail to beat a simple index fund over 15+ years. Don't try to outsmart the market - just own it!")
    
    # Quiz
    st.markdown("---")
    st.markdown("### 📝 Check Your Understanding")
    
    q1 = st.radio(
        "If a company announces great earnings, you should:",
        [
            "Buy immediately before the price goes up",
            "Recognize that the price has likely already adjusted",
            "Wait a few days for the news to 'settle'",
            "Sell your other stocks to buy more"
        ]
    )
    
    if st.button("Check Answer", key="market_quiz"):
        if q1 == "Recognize that the price has likely already adjusted":
            st.success("✅ Correct! In efficient markets, prices adjust almost instantly to new information.")
        else:
            st.error("❌ Not quite. By the time you hear news, the market has already reacted. That's why index investing beats stock picking.")


def render_etf_selection():
    """Module: Smart ETF Selection."""
    st.markdown("""
    <div class="module-header" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
        <h1>🎯 Smart ETF Selection</h1>
        <p>Not all ETFs are created equal. Learn how to pick the right ones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 What to Look For in an ETF")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>✅ Good Signs</h4>
            <ul>
                <li><strong>Low expense ratio</strong> (< 0.20%)</li>
                <li><strong>High trading volume</strong> (liquidity)</li>
                <li><strong>Large assets under management</strong> (> $1B)</li>
                <li><strong>Tracks a broad index</strong></li>
                <li><strong>From reputable provider</strong> (Vanguard, Fidelity, Schwab, iShares)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="concept-card" style="border-left-color: #ef4444;">
            <h4>🚩 Red Flags</h4>
            <ul>
                <li><strong>High expense ratio</strong> (> 0.50%)</li>
                <li><strong>Low trading volume</strong> (wide bid-ask spread)</li>
                <li><strong>Narrow/niche focus</strong> (single country, sector)</li>
                <li><strong>Leveraged or inverse</strong> (2x, 3x, -1x)</li>
                <li><strong>Complex strategies</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Recommended Core ETFs")
    
    etf_data = {
        "ETF": ["VTI", "VXUS", "BND", "VOO", "VT"],
        "Name": [
            "Vanguard Total Stock Market",
            "Vanguard Total International",
            "Vanguard Total Bond Market",
            "Vanguard S&P 500",
            "Vanguard Total World Stock"
        ],
        "Expense Ratio": ["0.03%", "0.07%", "0.03%", "0.03%", "0.07%"],
        "Holdings": ["4,000+", "8,000+", "10,000+", "500", "9,000+"],
        "Best For": [
            "Core US stocks",
            "International diversification",
            "Bond allocation",
            "Large-cap US focus",
            "One-fund simplicity"
        ],
        "Alternatives": [
            "ITOT, SWTSX",
            "IXUS, SWISX",
            "AGG, SCHZ",
            "SPY, IVV",
            "SPGM"
        ]
    }
    
    st.dataframe(pd.DataFrame(etf_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 💰 The Cost of Fees")
    
    st.markdown("Expense ratios seem small, but they compound over time:")
    
    # Fee impact calculator
    col1, col2 = st.columns(2)
    
    with col1:
        initial = st.number_input("Initial Investment ($)", value=10000, step=1000)
        years = st.slider("Years", 5, 40, 30)
        return_rate = st.slider("Expected Return (%)", 4, 12, 7)
    
    with col2:
        low_fee = 0.03
        high_fee = 1.00
        
        low_final = initial * ((1 + return_rate/100 - low_fee/100) ** years)
        high_final = initial * ((1 + return_rate/100 - high_fee/100) ** years)
        difference = low_final - high_final
        
        st.markdown(f"""
        <div class="concept-card">
            <h4>Fee Impact Over {years} Years</h4>
            <p><strong>Low-cost fund (0.03%):</strong> ${low_final:,.0f}</p>
            <p><strong>High-cost fund (1.00%):</strong> ${high_final:,.0f}</p>
            <p style="color: #ef4444; font-size: 1.5rem;"><strong>Cost of fees: ${difference:,.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    years_range = np.arange(0, years + 1)
    low_growth = initial * ((1 + return_rate/100 - low_fee/100) ** years_range)
    high_growth = initial * ((1 + return_rate/100 - high_fee/100) ** years_range)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_range, y=low_growth, name='Low-cost (0.03%)', line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=years_range, y=high_growth, name='High-cost (1.00%)', line=dict(color='#ef4444', width=3)))
    fig.add_annotation(
        x=years, y=(low_growth[-1] + high_growth[-1])/2,
        text=f"${difference:,.0f} difference!",
        showarrow=True, arrowhead=2
    )
    fig.update_layout(
        title="Growth Comparison: Low vs High Fees",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template='plotly_white',
        height=400,
        yaxis=dict(tickformat='$,.0f')
    )
    st.plotly_chart(fig, use_container_width=True)


def render_passive_active():
    """Module: Passive vs Active Investing."""
    st.markdown("""
    <div class="module-header" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
        <h1>⚖️ Passive vs Active Investing</h1>
        <p>The evidence is clear - but let's understand why.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>📈 Passive Investing</h4>
            <p><strong>Strategy:</strong> Buy and hold index funds that track the entire market.</p>
            <p><strong>Philosophy:</strong> "If you can't beat 'em, join 'em."</p>
            <ul>
                <li>✅ Very low fees (0.03-0.10%)</li>
                <li>✅ Automatic diversification</li>
                <li>✅ No research required</li>
                <li>✅ Tax efficient</li>
                <li>✅ Beats 90% of professionals</li>
            </ul>
            <p><strong>Example:</strong> Buy VTI and hold forever.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="concept-card" style="border-left-color: #f59e0b;">
            <h4>🎯 Active Investing</h4>
            <p><strong>Strategy:</strong> Pick stocks or funds that will beat the market.</p>
            <p><strong>Philosophy:</strong> "I can find the winners."</p>
            <ul>
                <li>❌ Higher fees (0.5-2.0%)</li>
                <li>❌ Requires research and time</li>
                <li>❌ Higher taxes from trading</li>
                <li>❌ 90% underperform long-term</li>
                <li>⚠️ Can be fun with "play money"</li>
            </ul>
            <p><strong>Example:</strong> Pick individual stocks or active mutual funds.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 The Evidence")
    
    # Create comparison data
    categories = ['US Large Cap', 'US Mid Cap', 'US Small Cap', 'International', 'Bonds']
    underperform_15yr = [92, 95, 97, 89, 91]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=underperform_15yr,
        orientation='h',
        marker_color=['#ef4444' if p > 90 else '#f59e0b' for p in underperform_15yr],
        text=[f"{p}%" for p in underperform_15yr],
        textposition='outside'
    ))
    fig.update_layout(
        title="% of Active Funds Underperforming Index (15-Year Period)",
        xaxis_title="Percentage Underperforming",
        template='plotly_white',
        height=350,
        xaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="concept-card" style="border-left-color: #6366f1;">
        <h4>🏆 Why Warren Buffett Recommends Index Funds</h4>
        <blockquote>
            "A low-cost index fund is the most sensible equity investment for the great majority of investors."
        </blockquote>
        <p>In his 2013 letter, Buffett revealed that his will instructs 90% of his wife's inheritance 
        be invested in an S&P 500 index fund. In 2017, he won a $1 million bet that an index fund 
        would beat hedge funds over 10 years.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 The Verdict")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #d1fae5; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #059669;">✅ For 95% of People</h3>
            <p><strong>Go Passive</strong></p>
            <p>Index funds + buy and hold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #d97706;">⚠️ If You Enjoy It</h3>
            <p><strong>Core + Satellite</strong></p>
            <p>90% index, 10% "play money"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #fee2e2; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #dc2626;">❌ Avoid</h3>
            <p><strong>100% Active</strong></p>
            <p>Day trading, stock picking</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.success("**Your action:** Open a brokerage account, buy VTI (or a target-date fund), set up automatic investments, and ignore the market noise.")


if __name__ == "__main__":
    main()
