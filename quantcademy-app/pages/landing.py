"""
Landing Page - FinLearn AI
Beautiful home page with branding, features, and authentication
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth import signup, login, get_user, save_progress, load_progress


def render_landing_page():
    """Render the landing/home page."""
    
    # Hero Section
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .landing-hero {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .landing-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .brand-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .hero-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .hero-subtext {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    .cta-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .cta-primary {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        transition: transform 0.2s, box-shadow 0.2s;
        display: inline-block;
    }
    
    .cta-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.4);
    }
    
    .cta-secondary {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        border: 1px solid rgba(255,255,255,0.2);
        transition: background 0.2s;
        display: inline-block;
    }
    
    .cta-secondary:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Features Section */
    .features-section {
        margin-bottom: 3rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #3d3d5c;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Stats Section */
    .stats-section {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 3rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        text-align: center;
    }
    
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .stat-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    
    /* Testimonials */
    .testimonial-card {
        background: #1e1e2e;
        border-radius: 16px;
        padding: 2rem;
        border-left: 4px solid #6366f1;
    }
    
    .testimonial-text {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
        font-size: 1.1rem;
        font-style: italic;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .testimonial-author {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Footer */
    .landing-footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    
    <div class="landing-hero">
        <div class="hero-content">
            <div class="brand-logo">FinLearn AI</div>
            <p class="hero-tagline">Break Into Investing</p>
            <p class="hero-subtext">
                Master the markets with AI-powered education. Learn investing fundamentals, 
                analyze charts with computer vision, and build confidence with personalized simulations.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="features-section">
        <h2 class="section-title">Why FinLearn AI?</h2>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI-Powered Tutor</div>
                <div class="feature-desc">
                    Ask any investing question and get accurate, sourced answers from our RAG-based AI tutor trained on trusted financial content.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Chart Vision</div>
                <div class="feature-desc">
                    Upload any stock chart and our CNN models detect support/resistance levels and trend direction automatically.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Stock Screener</div>
                <div class="feature-desc">
                    Real-time analysis of stocks with AI-generated BUY/HOLD/SELL signals combining multiple indicators.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Personalized Learning</div>
                <div class="feature-desc">
                    Tailored curriculum based on your goals, risk tolerance, and time horizon. Learn at your own pace.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔮</div>
                <div class="feature-title">Monte Carlo Simulator</div>
                <div class="feature-desc">
                    See thousands of possible outcomes for your portfolio. Understand risk before you invest.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Explainable AI</div>
                <div class="feature-desc">
                    Understand why our models make predictions with Grad-CAM visualizations. No black boxes here.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-section">
        <div class="stats-grid">
            <div>
                <div class="stat-number">3</div>
                <div class="stat-label">AI Models</div>
            </div>
            <div>
                <div class="stat-number">80%+</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
            <div>
                <div class="stat-number">50+</div>
                <div class="stat-label">Trusted Sources</div>
            </div>
            <div>
                <div class="stat-number">∞</div>
                <div class="stat-label">Learning Potential</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Testimonial
    st.markdown("""
    <div class="testimonial-card">
        <p class="testimonial-text">
            "I went from knowing nothing about investing to confidently building my first portfolio. 
            The AI tutor answered all my 'dumb' questions without judgment, and the chart analyzer 
            helped me understand what I was looking at."
        </p>
        <p class="testimonial-author">— First-time Investor</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="landing-footer">
        <p>FinLearn AI is for educational purposes only. Not financial advice.</p>
        <p>© 2024 FinLearn AI. Built with ❤️ for aspiring investors.</p>
    </div>
    """, unsafe_allow_html=True)


def render_auth_form():
    """Render login/signup forms."""
    
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 20px;
        border: 1px solid #3d3d5c;
    }
    
    .auth-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .auth-divider {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
    }
    
    .auth-divider::before, .auth-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #3d3d5c;
    }
    
    .auth-divider span {
        padding: 0 1rem;
        color: #64748b;
        font-size: 0.85rem;
    }
    
    .social-btn {
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #3d3d5c;
        background: transparent;
        color: #e2e8f0;
        font-size: 1rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        transition: background 0.2s;
    }
    
    .social-btn:hover {
        background: rgba(255,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Auth tabs
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])
    
    with tab1:
        st.markdown('<p class="auth-title">Welcome Back</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            col1, col2 = st.columns(2)
            with col1:
                remember = st.checkbox("Remember me")
            with col2:
                st.markdown("<p style='text-align: right; color: #6366f1; font-size: 0.9rem;'>Forgot password?</p>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            
            if submitted:
                if email and password:
                    result = login(email, password)
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result['user']['id']
                        st.session_state.user_email = result['user']['email']
                        st.session_state.user_name = result['user']['name']
                        
                        # Load saved progress
                        progress = load_progress(result['user']['id'])
                        if progress:
                            for key, value in progress.items():
                                st.session_state[key] = value
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result.get('error', 'Login failed'))
                else:
                    st.error("Please enter email and password")
        
        st.markdown("""
        <div class="auth-divider"><span>or continue with</span></div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔵 Google", use_container_width=True):
                st.info("Google OAuth coming soon")
        with col2:
            if st.button("⚫ GitHub", use_container_width=True):
                st.info("GitHub OAuth coming soon")
    
    with tab2:
        st.markdown('<p class="auth-title">Create Account</p>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="signup_password")
            confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
            
            agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            
            if submitted:
                if not all([name, email, password, confirm]):
                    st.error("Please fill in all fields")
                elif password != confirm:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not agree:
                    st.error("Please agree to the terms")
                else:
                    result = signup(email, password, name)
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result['user']['id']
                        st.session_state.user_email = result['user']['email']
                        st.session_state.user_name = result['user']['name']
                        st.success("Account created! Welcome to FinLearn AI!")
                        st.rerun()
                    else:
                        st.error(result.get('error', 'Signup failed'))


def main():
    """Main landing page."""
    st.set_page_config(
        page_title="FinLearn AI | Break Into Investing",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide sidebar on landing page
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #0f0f1a; }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if authenticated
    if st.session_state.get('authenticated', False):
        st.switch_page("app.py")
        return
    
    # Render landing page
    render_landing_page()
    
    st.markdown("---")
    
    # Auth section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_auth_form()


if __name__ == "__main__":
    main()
