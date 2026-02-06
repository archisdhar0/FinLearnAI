"""
QuantCademy - AI-Powered Investing Education Platform
MVP with personalized, simulation-backed learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('.')

from data.curriculum import CURRICULUM, QUIZ_QUESTIONS, PERSONALIZATION
from simulations.portfolio_sim import (
    monte_carlo_simulation,
    calculate_portfolio_stats,
    probability_of_loss_by_horizon,
    what_if_stop_contributing,
    inflation_adjusted_comparison,
    historical_drawdown_examples,
    ASSET_PARAMS
)

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="QuantCademy | Learn Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e1b4b;
        --light: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Module cards */
    .module-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .module-card:hover {
        transform: translateY(-2px);
    }
    
    .module-card.completed {
        border-left-color: #10b981;
    }
    
    .module-card.locked {
        opacity: 0.6;
        border-left-color: #9ca3af;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e2e8f0;
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.5s ease;
    }
    
    /* Quiz styling */
    .quiz-option {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quiz-option:hover {
        border-color: #6366f1;
        background: #f8fafc;
    }
    
    .quiz-option.correct {
        border-color: #10b981;
        background: #ecfdf5;
    }
    
    .quiz-option.incorrect {
        border-color: #ef4444;
        background: #fef2f2;
    }
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .insight-box.danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: #ef4444;
    }
    
    .insight-box.success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': None,
        'horizon_years': None,
        'monthly_contribution': None,
        'initial_investment': None,
        'emergency_fund': None,
        'risk_tolerance': None,
        'loss_sensitivity': None,
        'completed_modules': [],
        'quiz_scores': {},
        'portfolio_weights': None
    }

if 'current_module' not in st.session_state:
    st.session_state.current_module = None

if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_personalization_key():
    """Get personalization settings based on user profile."""
    profile = st.session_state.user_profile
    
    # Determine horizon category
    if profile['horizon_years']:
        if profile['horizon_years'] < 5:
            horizon_key = 'short_horizon'
        elif profile['horizon_years'] < 15:
            horizon_key = 'medium_horizon'
        else:
            horizon_key = 'long_horizon'
    else:
        horizon_key = 'medium_horizon'
    
    # Determine loss sensitivity
    if profile['loss_sensitivity']:
        if profile['loss_sensitivity'] > 7:
            sensitivity_key = 'high_loss_sensitivity'
        else:
            sensitivity_key = 'low_loss_sensitivity'
    else:
        sensitivity_key = 'low_loss_sensitivity'
    
    return horizon_key, sensitivity_key


def render_progress():
    """Render progress bar in sidebar."""
    completed = len(st.session_state.user_profile['completed_modules'])
    total = sum(len(section['modules']) for section in CURRICULUM.values())
    progress = completed / total if total > 0 else 0
    
    st.sidebar.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: white; font-size: 0.9rem;">Progress</span>
            <span style="color: white; font-size: 0.9rem;">{completed}/{total} modules</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress*100}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_outcome_chart(sim_results, years):
    """Create beautiful outcome band chart."""
    months = np.arange(0, years * 12 + 1)
    years_labels = months / 12
    
    fig = go.Figure()
    
    # Add bands (worst to best)
    fig.add_trace(go.Scatter(
        x=years_labels, y=sim_results['percentiles']['p95'],
        fill=None, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years_labels, y=sim_results['percentiles']['p5'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(99, 102, 241, 0.1)',
        name='90% of outcomes'
    ))
    
    fig.add_trace(go.Scatter(
        x=years_labels, y=sim_results['percentiles']['p75'],
        fill=None, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years_labels, y=sim_results['percentiles']['p25'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(99, 102, 241, 0.2)',
        name='50% of outcomes'
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=years_labels, y=sim_results['percentiles']['p50'],
        mode='lines', line=dict(color='#6366f1', width=3),
        name='Median outcome'
    ))
    
    # Total contributed line
    contributed = np.linspace(
        sim_results['total_contributed'] / (years * 12 + 1),
        sim_results['total_contributed'],
        years * 12 + 1
    )
    initial = st.session_state.user_profile.get('initial_investment', 1000)
    monthly = st.session_state.user_profile.get('monthly_contribution', 500)
    contributed = initial + monthly * months
    
    fig.add_trace(go.Scatter(
        x=years_labels, y=contributed,
        mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
        name='Total contributed'
    ))
    
    fig.update_layout(
        title=dict(text='Your Portfolio: Range of Possible Outcomes', font=dict(size=20)),
        xaxis_title='Years',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">📈 QuantCademy</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Learn Investing Your Way</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show progress if onboarding complete
    if st.session_state.onboarding_complete:
        render_progress()
    
    # Navigation
    st.sidebar.markdown("---")
    
    nav_options = ["🏠 Dashboard"]
    if st.session_state.onboarding_complete:
        nav_options.extend([
            "🎯 Module 1: Goals & Timeline",
            "📊 Module 2: Risk Explained",
            "🏗️ Module 3: Build Portfolio",
            "🔮 Module 4: Outcome Simulator"
        ])
    
    page = st.sidebar.radio("Navigate", nav_options, label_visibility="collapsed")
    
    # User profile summary
    if st.session_state.onboarding_complete:
        profile = st.session_state.user_profile
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Your Profile</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.25rem 0;">
                🎯 Horizon: {profile.get('horizon_years', '?')} years
            </p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.25rem 0;">
                💰 Monthly: ${profile.get('monthly_contribution', 0):,.0f}
            </p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.25rem 0;">
                📊 Risk: {profile.get('risk_tolerance', '?')}/10
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages
    if page == "🏠 Dashboard":
        render_dashboard()
    elif "Module 1" in page:
        render_module_1()
    elif "Module 2" in page:
        render_module_2()
    elif "Module 3" in page:
        render_module_3()
    elif "Module 4" in page:
        render_module_4()


# ============================================================
# DASHBOARD
# ============================================================
def render_dashboard():
    """Main dashboard / onboarding."""
    
    if not st.session_state.onboarding_complete:
        # Onboarding flow
        st.markdown("""
        <div class="main-header">
            <h1>Welcome to QuantCademy 📈</h1>
            <p>Your personalized journey to confident investing starts here.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Let's personalize your learning path")
        st.markdown("Answer a few questions so we can tailor the experience to **your** situation.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("What should we call you?", placeholder="Your name")
            
            horizon = st.slider(
                "How many years until you need this money?",
                min_value=1, max_value=40, value=15,
                help="This is your investment time horizon"
            )
            
            initial = st.number_input(
                "How much can you invest to start? ($)",
                min_value=0, max_value=1000000, value=1000, step=100
            )
        
        with col2:
            monthly = st.number_input(
                "How much can you invest monthly? ($)",
                min_value=0, max_value=50000, value=500, step=50
            )
            
            emergency = st.selectbox(
                "Do you have 3-6 months of expenses saved (emergency fund)?",
                ["Yes, fully funded", "Partially (1-3 months)", "No, not yet"]
            )
            
            risk = st.slider(
                "How comfortable are you with seeing your balance drop temporarily?",
                min_value=1, max_value=10, value=5,
                help="1 = Very uncomfortable, 10 = Very comfortable"
            )
        
        st.markdown("---")
        
        loss_q = st.radio(
            "If your portfolio dropped 20% in value, what would you do?",
            [
                "Sell immediately to prevent further losses",
                "Feel anxious but hold",
                "Stay calm - this is normal",
                "Get excited and invest more at lower prices"
            ]
        )
        
        loss_map = {
            "Sell immediately to prevent further losses": 9,
            "Feel anxious but hold": 7,
            "Stay calm - this is normal": 4,
            "Get excited and invest more at lower prices": 2
        }
        
        if st.button("🚀 Start My Learning Journey", type="primary", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name or "Investor",
                'horizon_years': horizon,
                'monthly_contribution': monthly,
                'initial_investment': initial,
                'emergency_fund': emergency,
                'risk_tolerance': risk,
                'loss_sensitivity': loss_map.get(loss_q, 5)
            })
            st.session_state.onboarding_complete = True
            st.rerun()
    
    else:
        # Main dashboard
        profile = st.session_state.user_profile
        
        st.markdown(f"""
        <div class="main-header">
            <h1>Welcome back, {profile['name']}! 👋</h1>
            <p>Continue your investing journey where you left off.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{profile['horizon_years']}</div>
                <div class="stat-label">Year Horizon</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">${profile['monthly_contribution']:,.0f}</div>
                <div class="stat-label">Monthly Investment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            completed = len(profile['completed_modules'])
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{completed}/4</div>
                <div class="stat-label">Modules Complete</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calculate projected value
            if profile.get('portfolio_weights'):
                sim = monte_carlo_simulation(
                    profile['initial_investment'],
                    profile['monthly_contribution'],
                    profile['portfolio_weights'],
                    profile['horizon_years']
                )
                projected = sim['final_median']
            else:
                projected = profile['initial_investment'] + profile['monthly_contribution'] * profile['horizon_years'] * 12
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">${projected:,.0f}</div>
                <div class="stat-label">Projected Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📚 Your Learning Path")
        
        # Module cards
        modules = CURRICULUM['foundations']['modules']
        
        for i, module in enumerate(modules):
            is_completed = module['id'] in profile['completed_modules']
            is_locked = i > 0 and modules[i-1]['id'] not in profile['completed_modules']
            
            status_class = "completed" if is_completed else ("locked" if is_locked else "")
            status_badge = "✅ Complete" if is_completed else ("🔒 Locked" if is_locked else "▶️ Start")
            
            st.markdown(f"""
            <div class="module-card {status_class}">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <span style="font-size: 2rem;">{module['icon']}</span>
                        <h3 style="margin: 0.5rem 0 0.25rem 0;">{module['title']}</h3>
                        <p style="color: #6366f1; font-size: 0.9rem; margin: 0;">{module['subtitle']}</p>
                        <p style="color: #64748b; margin: 0.5rem 0;">{module['description']}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {'#10b981' if is_completed else '#6366f1'}; color: white; 
                               padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                            {status_badge}
                        </span>
                        <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem;">⏱️ {module['duration']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MODULE 1: GOALS & TIMELINE
# ============================================================
def render_module_1():
    """Module 1: Your Goal + Timeline."""
    profile = st.session_state.user_profile
    horizon_key, sensitivity_key = get_personalization_key()
    personalization = PERSONALIZATION[horizon_key]
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Module 1: Your Goals & Timeline</h1>
        <p>Understanding WHY you're investing shapes HOW you should invest.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personalized message
    st.markdown(f"""
    <div class="insight-box success">
        <strong>Personalized for you:</strong> {personalization['key_message']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Why Does Time Horizon Matter?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Short Horizon (< 5 years)**
        - Less time to recover from drops
        - Prioritize capital preservation
        - Lower stock allocation recommended
        
        **Long Horizon (15+ years)**
        - Time to ride out volatility
        - Can afford more risk for more return
        - Stocks historically always recovered
        """)
    
    with col2:
        # Probability of loss chart
        weights_conservative = {"us_stocks": 40, "intl_stocks": 10, "bonds": 40, "cash": 10}
        weights_aggressive = {"us_stocks": 70, "intl_stocks": 15, "bonds": 10, "cash": 5}
        
        prob_results = probability_of_loss_by_horizon(weights_aggressive)
        
        horizons = list(prob_results.keys())
        prob_loss = [prob_results[h]['prob_loss'] for h in horizons]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{h} years" for h in horizons],
            y=prob_loss,
            marker_color=['#ef4444' if p > 20 else '#f59e0b' if p > 10 else '#10b981' for p in prob_loss],
            text=[f"{p:.0f}%" for p in prob_loss],
            textposition='outside'
        ))
        fig.update_layout(
            title="Probability of Loss by Time Horizon (80/20 Stock/Bond)",
            yaxis_title="Probability of Being Down (%)",
            template='plotly_white',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 💰 Your Money Buckets")
    
    st.markdown("""
    Before investing, let's organize your money into buckets:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2rem;">🛡️</div>
            <h4>Emergency Fund</h4>
            <p style="font-size: 0.9rem; color: #64748b;">3-6 months expenses<br/>Keep in savings account</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2rem;">🎯</div>
            <h4>Near-Term Goals</h4>
            <p style="font-size: 0.9rem; color: #64748b;">< 5 years away<br/>Conservative investments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2rem;">📈</div>
            <h4>Long-Term Growth</h4>
            <p style="font-size: 0.9rem; color: #64748b;">5+ years away<br/>Can take more risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Emergency fund warning
    if profile['emergency_fund'] != "Yes, fully funded":
        st.markdown("""
        <div class="insight-box danger">
            <strong>⚠️ Important:</strong> You indicated your emergency fund isn't fully funded. 
            Consider building this FIRST before investing aggressively. Without it, you might be 
            forced to sell investments at the worst time.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quiz
    st.markdown("### 📝 Quick Check")
    
    if 'goal_timeline' in QUIZ_QUESTIONS:
        q = QUIZ_QUESTIONS['goal_timeline'][0]
        answer = st.radio(q['question'], q['options'], key='m1_quiz')
        
        if st.button("Check Answer", key='m1_check'):
            if q['options'].index(answer) == q['correct']:
                st.success("✅ Correct! " + q['options'][q['correct']])
            else:
                wrong_idx = q['options'].index(answer)
                st.error("❌ " + q['misconception_if_wrong'].get(wrong_idx, "Not quite right."))
    
    st.markdown("---")
    
    if st.button("✅ Complete Module 1", type="primary", use_container_width=True):
        if 'goal_timeline' not in st.session_state.user_profile['completed_modules']:
            st.session_state.user_profile['completed_modules'].append('goal_timeline')
        st.success("Module 1 completed! Moving to Risk Explained...")
        st.rerun()


# ============================================================
# MODULE 2: RISK EXPLAINED
# ============================================================
def render_module_2():
    """Module 2: Risk Explained With Your Numbers."""
    profile = st.session_state.user_profile
    horizon_key, sensitivity_key = get_personalization_key()
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Module 2: Risk, Explained</h1>
        <p>Risk isn't one thing. Let's explore what it actually means for YOUR money.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personalization based on loss sensitivity
    if profile['loss_sensitivity'] > 7:
        st.markdown("""
        <div class="insight-box">
            <strong>Personalized for you:</strong> Since you're more sensitive to losses, we'll focus on 
            worst-case scenarios and drawdowns. Understanding these will help you stay invested when it matters.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📉 What is a Drawdown?")
    
    st.markdown("""
    A **drawdown** is the decline from a portfolio's peak to its lowest point before recovering.
    
    It's not the same as volatility - drawdowns show you the *actual pain* you might experience.
    """)
    
    # Historical drawdowns
    st.markdown("### 📜 Real Historical Drawdowns")
    
    drawdowns = historical_drawdown_examples()
    
    cols = st.columns(2)
    for i, dd in enumerate(drawdowns):
        with cols[i % 2]:
            color = "#ef4444" if dd['drawdown'] < -40 else "#f59e0b" if dd['drawdown'] < -30 else "#10b981"
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
                        border-left: 4px solid {color};">
                <h4 style="margin: 0;">{dd['event']}</h4>
                <p style="font-size: 1.5rem; color: {color}; margin: 0.5rem 0;">{dd['drawdown']}%</p>
                <p style="font-size: 0.85rem; color: #64748b;">Recovery: {dd['recovery_months']} months</p>
                <p style="font-size: 0.9rem; margin: 0;"><em>{dd['lesson']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎚️ See Risk With YOUR Numbers")
    
    st.markdown("Adjust the stock allocation to see how it affects your potential outcomes:")
    
    stock_pct = st.slider("Stock Allocation (%)", 0, 100, 60)
    bond_pct = 100 - stock_pct
    
    weights = {
        "us_stocks": stock_pct * 0.7,
        "intl_stocks": stock_pct * 0.3,
        "bonds": bond_pct * 0.9,
        "cash": bond_pct * 0.1
    }
    
    stats = calculate_portfolio_stats(weights)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Annual Return", f"{stats['expected_return']*100:.1f}%")
    with col2:
        st.metric("Annual Volatility", f"{stats['volatility']*100:.1f}%")
    with col3:
        worst_year = -stats['volatility'] * 2  # Rough estimate
        st.metric("Worst Year (estimate)", f"{worst_year*100:.0f}%")
    
    # Simulation with current settings
    sim = monte_carlo_simulation(
        profile['initial_investment'],
        profile['monthly_contribution'],
        weights,
        profile['horizon_years'],
        n_simulations=500
    )
    
    # Show outcome bands
    fig = create_outcome_chart(sim, profile['horizon_years'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">${sim['final_median']:,.0f}</div>
            <div class="stat-label">Median Outcome</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #f59e0b;">${sim['final_p10']:,.0f}</div>
            <div class="stat-label">Worst 10% Outcome</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #6366f1;">${sim['final_p90']:,.0f}</div>
            <div class="stat-label">Best 10% Outcome</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        prob_color = "#10b981" if sim['prob_loss'] < 10 else "#f59e0b" if sim['prob_loss'] < 25 else "#ef4444"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: {prob_color};">{sim['prob_loss']:.0f}%</div>
            <div class="stat-label">Chance of Loss</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Save risk profile
    risk_profile = "Conservative" if stock_pct < 40 else "Moderate" if stock_pct < 70 else "Aggressive"
    
    st.markdown(f"### Your Risk Profile: **{risk_profile}** ({stock_pct}% stocks / {bond_pct}% bonds)")
    
    if st.button("✅ Complete Module 2", type="primary", use_container_width=True):
        if 'risk_explained' not in st.session_state.user_profile['completed_modules']:
            st.session_state.user_profile['completed_modules'].append('risk_explained')
        st.session_state.user_profile['portfolio_weights'] = weights
        st.success("Module 2 completed! Your risk profile has been saved.")
        st.rerun()


# ============================================================
# MODULE 3: BUILD PORTFOLIO
# ============================================================
def render_module_3():
    """Module 3: Build Your First Portfolio."""
    profile = st.session_state.user_profile
    
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Module 3: Build Your First Portfolio</h1>
        <p>A simple, evidence-based approach that beats most professionals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 The 3-Fund Portfolio Strategy")
    
    st.markdown("""
    You don't need to pick stocks. You don't need 20 different funds. 
    
    A simple **3-fund portfolio** gives you:
    - ✅ Diversification across thousands of companies
    - ✅ Exposure to US and international markets
    - ✅ Bond allocation for stability
    - ✅ Ultra-low fees (0.03-0.07%)
    """)
    
    # Recommended allocation based on profile
    horizon_key, _ = get_personalization_key()
    recommended = PERSONALIZATION[horizon_key]['recommended_allocation']
    
    st.markdown(f"""
    <div class="insight-box success">
        <strong>Recommended for your {profile['horizon_years']}-year horizon:</strong><br/>
        {recommended['stocks']}% Stocks / {recommended['bonds']}% Bonds / {recommended['cash']}% Cash
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Customize Your Allocation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        us_stocks = st.slider("US Stocks (VTI)", 0, 100, int(recommended['stocks'] * 0.7))
        intl_stocks = st.slider("International Stocks (VXUS)", 0, 100 - us_stocks, int(recommended['stocks'] * 0.3))
        bonds = st.slider("US Bonds (BND)", 0, 100 - us_stocks - intl_stocks, recommended['bonds'])
        cash = 100 - us_stocks - intl_stocks - bonds
        
        st.markdown(f"**Cash/Money Market:** {cash}%")
    
    with col2:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['US Stocks (VTI)', 'Intl Stocks (VXUS)', 'Bonds (BND)', 'Cash'],
            values=[us_stocks, intl_stocks, bonds, cash],
            hole=0.4,
            marker_colors=['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
        )])
        fig.update_layout(
            title="Your Portfolio Allocation",
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    weights = {
        "us_stocks": us_stocks,
        "intl_stocks": intl_stocks,
        "bonds": bonds,
        "cash": cash
    }
    
    # Portfolio stats
    stats = calculate_portfolio_stats(weights)
    
    st.markdown("### 📈 Your Portfolio Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Return", f"{stats['expected_return']*100:.1f}%/year")
    with col2:
        st.metric("Volatility", f"{stats['volatility']*100:.1f}%/year")
    with col3:
        st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    with col4:
        total_stocks = us_stocks + intl_stocks
        st.metric("Risk Level", "Conservative" if total_stocks < 40 else "Moderate" if total_stocks < 70 else "Aggressive")
    
    st.markdown("---")
    st.markdown("### 🏦 Where to Buy These ETFs")
    
    st.markdown("""
    | ETF | Name | Expense Ratio | Where to Buy |
    |-----|------|---------------|--------------|
    | **VTI** | Vanguard Total Stock Market | 0.03% | Vanguard, Fidelity, Schwab |
    | **VXUS** | Vanguard Total International | 0.07% | Vanguard, Fidelity, Schwab |
    | **BND** | Vanguard Total Bond Market | 0.03% | Vanguard, Fidelity, Schwab |
    
    **Alternative equivalents:**
    - Fidelity: FSKAX, FTIHX, FXNAX (even lower fees!)
    - Schwab: SWTSX, SWISX, SCHZ
    """)
    
    st.markdown("---")
    
    # Monthly investment breakdown
    monthly = profile['monthly_contribution']
    st.markdown(f"### 💰 Your ${monthly:,.0f}/month Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.5rem; color: #4CAF50;">${monthly * us_stocks / 100:,.0f}</div>
            <div class="stat-label">VTI (US Stocks)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.5rem; color: #2196F3;">${monthly * intl_stocks / 100:,.0f}</div>
            <div class="stat-label">VXUS (Intl)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.5rem; color: #FF9800;">${monthly * bonds / 100:,.0f}</div>
            <div class="stat-label">BND (Bonds)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.5rem; color: #9E9E9E;">${monthly * cash / 100:,.0f}</div>
            <div class="stat-label">Cash Reserve</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("✅ Save Portfolio & Complete Module 3", type="primary", use_container_width=True):
        if 'first_portfolio' not in st.session_state.user_profile['completed_modules']:
            st.session_state.user_profile['completed_modules'].append('first_portfolio')
        st.session_state.user_profile['portfolio_weights'] = weights
        st.success("Portfolio saved! Let's see what could happen...")
        st.rerun()


# ============================================================
# MODULE 4: OUTCOME SIMULATOR
# ============================================================
def render_module_4():
    """Module 4: What Could Happen? (Outcome Simulator)."""
    profile = st.session_state.user_profile
    weights = profile.get('portfolio_weights', {"us_stocks": 50, "intl_stocks": 10, "bonds": 30, "cash": 10})
    
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Module 4: What Could Happen?</h1>
        <p>See the range of possible outcomes for YOUR portfolio and decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run simulation
    sim = monte_carlo_simulation(
        profile['initial_investment'],
        profile['monthly_contribution'],
        weights,
        profile['horizon_years'],
        n_simulations=1000
    )
    
    # Main outcome chart
    fig = create_outcome_chart(sim, profile['horizon_years'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gain = (sim['final_median'] / sim['total_contributed'] - 1) * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">${sim['final_median']:,.0f}</div>
            <div class="stat-label">Median Outcome (+{gain:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">${sim['total_contributed']:,.0f}</div>
            <div class="stat-label">Total Contributed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        prob_color = "#10b981" if sim['prob_loss'] < 10 else "#f59e0b" if sim['prob_loss'] < 25 else "#ef4444"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: {prob_color};">{sim['prob_loss']:.0f}%</div>
            <div class="stat-label">Chance of Loss</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #6366f1;">{sim['prob_double']:.0f}%</div>
            <div class="stat-label">Chance to Double</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ What If I Stop Contributing?")
    
    stop_month = st.slider(
        "Stop contributing after how many months?",
        min_value=1,
        max_value=profile['horizon_years'] * 12,
        value=12
    )
    
    comparison = what_if_stop_contributing(
        profile['initial_investment'],
        profile['monthly_contribution'],
        weights,
        profile['horizon_years'],
        stop_month
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #10b981;">
            <h4>Continue Contributing</h4>
            <div class="stat-value" style="color: #10b981;">${comparison['continue_median']:,.0f}</div>
            <div class="stat-label">Contributed: ${comparison['continue_contributed']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #ef4444;">
            <h4>Stop After {stop_month} Months</h4>
            <div class="stat-value" style="color: #ef4444;">${comparison['stopped_median']:,.0f}</div>
            <div class="stat-label">Contributed: ${comparison['stopped_contributed']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-box danger">
        <strong>Cost of Stopping:</strong> You'd have <strong>${comparison['difference']:,.0f} less</strong> 
        ({comparison['difference_pct']:.0f}% less) by stopping contributions after {stop_month} months.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Probability of Loss Over Time")
    
    prob_results = probability_of_loss_by_horizon(weights, [1, 3, 5, 10, 15, 20])
    
    # Create chart
    horizons = list(prob_results.keys())
    prob_loss = [prob_results[h]['prob_loss'] for h in horizons]
    expected_gain = [prob_results[h]['expected_gain'] for h in horizons]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=[f"{h}yr" for h in horizons],
            y=prob_loss,
            name="Prob. of Loss",
            marker_color=['#ef4444' if p > 20 else '#f59e0b' if p > 10 else '#10b981' for p in prob_loss]
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=[f"{h}yr" for h in horizons],
            y=expected_gain,
            name="Expected Gain",
            line=dict(color='#6366f1', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Why Time in Market Matters",
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_yaxes(title_text="Probability of Loss (%)", secondary_y=False)
    fig.update_yaxes(title_text="Expected Gain (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box success">
        <strong>Key Insight:</strong> The longer you stay invested, the lower your probability of loss 
        AND the higher your expected gains. This is why we say "time IN the market beats timing the market."
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("✅ Complete All Foundation Modules!", type="primary", use_container_width=True):
        if 'what_could_happen' not in st.session_state.user_profile['completed_modules']:
            st.session_state.user_profile['completed_modules'].append('what_could_happen')
        st.balloons()
        st.success("🎉 Congratulations! You've completed the Foundations track!")
        st.rerun()


# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()
