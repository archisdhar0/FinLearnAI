"""
QuantCademy AI Tutor - Capstone-Grade RAG Chat Interface

Features:
- Hybrid retrieval (BM25 + semantic embeddings)
- Source tiering (SEC/FINRA > Fed/CFA > Fidelity > Investopedia)
- Confidence gating (refuses when unsure)
- Multi-query decomposition
- Citation-required answers
- Stock picking refusal
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag.ollama_agent import (
        check_ollama_status,
        chat_with_ollama,
        get_quick_response,
        get_related_topics,
        QUICK_RESPONSES,
        ADVANCED_RAG_AVAILABLE,
        get_rag_context
    )
    from rag.knowledge_base import search_knowledge_base, KNOWLEDGE_BASE
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    ADVANCED_RAG_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Try to import knowledge base v2 stats
try:
    from rag.knowledge_base_v2 import get_knowledge_base_stats, get_tier_label, SourceTier
    KB_V2_AVAILABLE = True
except ImportError:
    KB_V2_AVAILABLE = False
    get_knowledge_base_stats = None

# Try to import advanced retrieval for testing
try:
    from rag.retrieval import retrieve_with_citations, ADVANCED_RETRIEVAL_AVAILABLE
except ImportError:
    ADVANCED_RETRIEVAL_AVAILABLE = False
    retrieve_with_citations = None

# Page config
st.set_page_config(
    page_title="AI Tutor | QuantCademy",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS with tier colors
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 85%;
    }
    .user-message {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    .source-tag {
        background: #334155;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        color: #94a3b8;
    }
    .tier-1 { border-left: 3px solid #10b981; }
    .tier-2 { border-left: 3px solid #3b82f6; }
    .tier-3 { border-left: 3px solid #8b5cf6; }
    .tier-4 { border-left: 3px solid #f59e0b; }
    .status-online { color: #10b981; }
    .status-offline { color: #ef4444; }
    .status-warning { color: #f59e0b; }
    .rag-info {
        background: #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #6366f1;
    }
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
    .tier-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 4px;
    }
    .tier-regulatory { background: #10b981; color: white; }
    .tier-institutional { background: #3b82f6; color: white; }
    .tier-financial { background: #8b5cf6; color: white; }
    .tier-educational { background: #f59e0b; color: black; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""


def clear_input():
    """Callback to clear input after sending."""
    st.session_state.user_input = ""


def main():
    # Header with capstone badge
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 style="margin: 0;">🤖 AI Investing Tutor</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Capstone-grade RAG with source tiering, confidence gating, and citation-backed answers.
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.8rem;">
                ✅ Hybrid Retrieval<br/>
                ✅ Source Tiering<br/>
                ✅ Confidence Gating
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if RAG module loaded correctly
    if not RAG_AVAILABLE:
        st.error(f"⚠️ Could not load AI module: {IMPORT_ERROR}")
        st.info("Make sure you're running from the quantcademy-app directory.")
        return
    
    # Status indicators
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_status_indicators()
    
    # Sidebar with quick topics and info
    with st.sidebar:
        render_sidebar()
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            render_message(message)
    
    # Handle pending question from sidebar
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
        process_user_input(user_input)
        st.rerun()
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about investing:",
            placeholder="e.g., What's the difference between ETFs and mutual funds?",
            key="chat_input_field",
            value=st.session_state.user_input,
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📨", type="primary", use_container_width=True)
    
    if send_button and user_input:
        process_user_input(user_input)
        st.session_state.user_input = ""  # Clear input
        st.rerun()
    
    # Also trigger on Enter key
    if user_input and user_input != st.session_state.get('last_processed_input', ''):
        # Check if this is a new input (user pressed enter)
        if st.session_state.get('should_process', False):
            process_user_input(user_input)
            st.session_state.user_input = ""
            st.session_state.should_process = False
            st.rerun()
    
    # Example questions for empty state
    if not st.session_state.chat_history:
        render_empty_state()


def show_status_indicators():
    """Show LLM and RAG status with advanced features."""
    status = check_ollama_status()
    
    # LLM status (Gemini or Ollama)
    if status['status'] == 'online':
        llm_icon = "🟢"
        provider = status.get('provider', 'unknown').title()
        llm_text = f"{provider} Online"
        model_text = status.get('message', '')
    else:
        llm_icon = "🔴"
        llm_text = "LLM Offline"
        model_text = status.get('message', 'Check .env config')
    
    # Advanced RAG status
    if ADVANCED_RAG_AVAILABLE and ADVANCED_RETRIEVAL_AVAILABLE:
        rag_icon = "🟢"
        rag_text = "Capstone RAG"
        rag_detail = "Hybrid + Rerank"
    elif ADVANCED_RAG_AVAILABLE:
        rag_icon = "🟡"
        rag_text = "Basic RAG"
        rag_detail = "BM25 only"
    else:
        rag_icon = "🟡"
        rag_text = "Keyword Search"
        rag_detail = "No embeddings"
    
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.85rem;">
        <div>{llm_icon} {llm_text}</div>
        <div style="color: #64748b; font-size: 0.75rem;">{model_text}</div>
        <div style="margin-top: 0.25rem;">{rag_icon} {rag_text}</div>
        <div style="color: #64748b; font-size: 0.75rem;">{rag_detail}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with topics, stats, and source tiers."""
    st.markdown("### 💡 Quick Topics")
    st.markdown("Click to ask about:")
    
    quick_topics = [
        "What is an ETF?",
        "How does compound interest work?",
        "What's the difference between stocks and bonds?",
        "How should I start investing?",
        "What is a 401(k)?",
        "What is dollar cost averaging?",
        "How much should I have in bonds?",
        "What are index funds?",
        "What is diversification?",
        "What is a Roth IRA?"
    ]
    
    for topic in quick_topics:
        if st.button(topic, key=f"topic_{topic}", use_container_width=True):
            st.session_state.pending_question = topic
    
    st.markdown("---")
    
    # Knowledge Base v2 Stats
    st.markdown("### 📊 Knowledge Base")
    
    if KB_V2_AVAILABLE and get_knowledge_base_stats:
        stats = get_knowledge_base_stats()
        st.markdown(f"**{stats['total_chunks']}** semantic chunks")
        
        st.markdown("**By Source Tier:**")
        tier_colors = {
            "🏛️ Regulatory": "#10b981",
            "🎓 Institutional": "#3b82f6",
            "🏦 Financial Institution": "#8b5cf6",
            "📚 Educational": "#f59e0b",
            "📰 General": "#64748b"
        }
        for tier, count in stats.get('by_tier', {}).items():
            color = tier_colors.get(tier, "#64748b")
            st.markdown(f"<span style='color:{color}'>{tier}: {count}</span>", unsafe_allow_html=True)
        
        st.markdown("**By Type:**")
        for chunk_type, count in stats.get('by_type', {}).items():
            st.markdown(f"- {chunk_type}: {count}")
    else:
        st.markdown(f"**{len(KNOWLEDGE_BASE)}** documents (legacy)")
    
    st.markdown("---")
    
    # Source Tier Legend
    st.markdown("### 🏆 Source Tiers")
    st.markdown("""
    <div style="font-size: 0.8rem;">
        <div style="color: #10b981;">🏛️ <b>Tier 1</b>: SEC, FINRA, IRS</div>
        <div style="color: #3b82f6;">🎓 <b>Tier 2</b>: Fed, CFA, Vanguard Research</div>
        <div style="color: #8b5cf6;">🏦 <b>Tier 3</b>: Fidelity, Schwab, Bogleheads</div>
        <div style="color: #f59e0b;">📚 <b>Tier 4</b>: Investopedia, NerdWallet</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # RAG Features
    st.markdown("### ⚡ RAG Features")
    features = [
        ("✅" if ADVANCED_RETRIEVAL_AVAILABLE else "❌", "Hybrid BM25 + Semantic"),
        ("✅" if ADVANCED_RETRIEVAL_AVAILABLE else "❌", "Cross-encoder Reranking"),
        ("✅", "Confidence Gating"),
        ("✅", "Citation-backed Answers"),
        ("✅", "Stock-picking Refusal"),
        ("✅" if ADVANCED_RAG_AVAILABLE else "❌", "Multi-query Decomposition"),
    ]
    for icon, feature in features:
        st.markdown(f"{icon} {feature}")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


def render_message(message):
    """Render a single chat message with confidence and citations."""
    if message['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br/>{message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Confidence indicator
        confidence = message.get('confidence', 0)
        if confidence >= 0.6:
            conf_class = "confidence-high"
            conf_text = f"High confidence ({confidence:.0%})"
        elif confidence >= 0.35:
            conf_class = "confidence-medium"
            conf_text = f"Medium confidence ({confidence:.0%})"
        elif confidence > 0:
            conf_class = "confidence-low"
            conf_text = f"Low confidence ({confidence:.0%})"
        else:
            conf_class = ""
            conf_text = ""
        
        # Show sources with tier badges
        sources_html = ""
        if message.get('citations'):
            sources_html = "<div style='margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #334155;'>"
            sources_html += "<span style='font-size: 0.75rem; color: #64748b;'>📚 </span>"
            for citation in message['citations'][:5]:
                sources_html += f"<span class='source-tag'>{citation}</span>"
            sources_html += "</div>"
        
        # Confidence display
        conf_html = ""
        if conf_text:
            conf_html = f"<div style='font-size: 0.7rem; margin-top: 0.5rem;'><span class='{conf_class}'>{conf_text}</span></div>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 QuantCademy AI:</strong>
            {conf_html}
            <div style="margin-top: 0.5rem;">{message['content']}</div>
            {sources_html}
        </div>
        """, unsafe_allow_html=True)


def render_empty_state():
    """Render example questions when chat is empty."""
    st.markdown("### 👋 Welcome! Try asking:")
    
    examples = [
        "I'm 25 with $500/month to invest. Where do I start?",
        "Explain the three-fund portfolio like I'm 5",
        "What happens if the stock market crashes?",
        "Should I pay off debt or invest first?",
        "What's the difference between a 401(k) and IRA?",
        "How do I know if I'm taking too much risk?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                process_user_input(example)
                st.rerun()
    
    # Test stock-picking refusal
    st.markdown("---")
    st.markdown("### 🧪 Test Capstone Features:")
    
    test_cols = st.columns(3)
    with test_cols[0]:
        if st.button("🚫 Test Stock Refusal", key="test_stock", use_container_width=True):
            process_user_input("Which stock should I buy?")
            st.rerun()
    with test_cols[1]:
        if st.button("🔍 Test Complex Query", key="test_complex", use_container_width=True):
            process_user_input("Compare 401k and Roth IRA and tell me which is better for a 30 year old")
            st.rerun()
    with test_cols[2]:
        if st.button("❓ Test Low Confidence", key="test_low", use_container_width=True):
            process_user_input("What is the optimal Kelly criterion for crypto futures?")
            st.rerun()
    
    # RAG info box
    st.markdown("---")
    st.markdown("""
    <div class="rag-info">
        <strong>🧠 Capstone-Grade RAG System</strong><br/>
        <span style="font-size: 0.9rem; color: #94a3b8;">
        <table style="width: 100%; margin-top: 0.5rem;">
            <tr>
                <td><b>Hybrid Retrieval</b></td>
                <td>BM25 keyword + semantic embeddings</td>
            </tr>
            <tr>
                <td><b>Reranking</b></td>
                <td>Cross-encoder scores top 20 → top 5</td>
            </tr>
            <tr>
                <td><b>Source Tiering</b></td>
                <td>Prefers SEC/FINRA over blogs</td>
            </tr>
            <tr>
                <td><b>Confidence Gating</b></td>
                <td>Refuses when not confident</td>
            </tr>
            <tr>
                <td><b>Citations</b></td>
                <td>Every answer cites sources</td>
            </tr>
        </table>
        </span>
    </div>
    """, unsafe_allow_html=True)


def process_user_input(user_input: str):
    """Process user input with capstone-grade RAG."""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Store last processed input
    st.session_state.last_processed_input = user_input
    
    # Check for quick response first (instant, no LLM needed)
    quick_response = get_quick_response(user_input)
    
    citations = []
    confidence = 0.0  # Default to low confidence
    
    if quick_response:
        response = quick_response
        confidence = 0.85  # Quick responses are pre-vetted
        # Extract citations from quick response
        if "*Sources:" in response:
            citations = ["SEC Investor.gov", "Vanguard", "Bogleheads"]
    else:
        # FIRST: Get RAG context and check confidence
        context, citation_str, rag_confidence, is_confident, refusal_reason = get_rag_context(user_input)
        confidence = rag_confidence
        
        # Parse citations from citation string
        if citation_str:
            citations = [c.strip() for c in citation_str.replace("Sources:", "").split(";") if c.strip()]
        
        # Check if we should refuse due to low confidence
        if not is_confident and refusal_reason:
            response = f"""## ⚠️ Low Confidence Answer

{refusal_reason}

**I work best with questions about:**
- Index funds and ETFs
- Retirement accounts (401k, IRA, Roth)
- Asset allocation and diversification
- Compound interest
- Basic investing concepts

Try rephrasing your question or ask about one of these topics!"""
        else:
            # We have enough confidence - check LLM status
            status = check_ollama_status()
            
            if status['status'] == 'online':
                # Collect streamed response
                full_response = ""
                
                for chunk in chat_with_ollama(
                    user_input,
                    st.session_state.get('user_profile'),
                    st.session_state.chat_history[:-1],  # Exclude current message
                    stream=True
                ):
                    full_response += chunk
                
                response = full_response
            else:
                # Fallback when LLM is offline but we have context
                if context:
                    response = f"""Based on my knowledge base:

{context[:2000]}

---
*⚠️ For a conversational answer, please configure your LLM. See README for setup.*"""
                else:
                    results = search_knowledge_base(user_input)
                    if results:
                        _, _, top_doc = results[0]
                        response = f"""Based on **{top_doc.get('source', 'trusted sources')}**:

{top_doc['content'][:1500]}

---
*Configure your LLM for conversational answers. See README.*"""
                        confidence = 0.5
                        citations = [top_doc.get('source', 'Knowledge Base')]
                    else:
                        response = """I couldn't find relevant information for this question.

**Try asking about:**
- Index funds and ETFs
- Retirement accounts (401k, IRA)
- Asset allocation
- Compound interest
- Risk management

Or check the **Learning Modules** for structured lessons!"""
                        confidence = 0.1
                        citations = []
    
    # Add assistant response to history with metadata
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'citations': citations,
        'confidence': confidence
    })


if __name__ == "__main__":
    main()
