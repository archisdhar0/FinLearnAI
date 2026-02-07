"""
QuantCademy AI Tutor - RAG-powered chat interface
Uses Ollama + llama3 with semantic search over curated knowledge base.
Sources: SEC, Investopedia, Vanguard, Fidelity, Bogleheads, FINRA
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
        QUICK_RESPONSES
    )
    from rag.knowledge_base import search_knowledge_base, KNOWLEDGE_BASE
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Try to import vector store
try:
    from rag.vector_store import VECTOR_STORE_AVAILABLE, get_vector_store
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    get_vector_store = None

# Page config
st.set_page_config(
    page_title="AI Tutor | QuantCademy",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
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
    .status-online {
        color: #10b981;
    }
    .status-offline {
        color: #ef4444;
    }
    .status-warning {
        color: #f59e0b;
    }
    .rag-info {
        background: #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}


def main():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0;">🤖 AI Investing Tutor</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Ask me anything about investing! I use trusted sources to give you accurate, beginner-friendly answers.
        </p>
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
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📨", type="primary", use_container_width=True)
    
    if send_button and user_input:
        process_user_input(user_input)
        st.rerun()
    
    # Example questions for empty state
    if not st.session_state.chat_history:
        render_empty_state()


def show_status_indicators():
    """Show LLM and RAG status."""
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
    
    # Vector store status
    if VECTOR_STORE_AVAILABLE:
        try:
            store = get_vector_store(initialize=False) if get_vector_store else None
            if store and store.collection.count() > 0:
                rag_icon = "🟢"
                rag_text = f"RAG: {store.collection.count()} docs"
            else:
                rag_icon = "🟡"
                rag_text = "RAG: Not indexed"
        except Exception:
            rag_icon = "🟡"
            rag_text = "RAG: Keyword mode"
    else:
        rag_icon = "🟡"
        rag_text = "RAG: Keyword mode"
    
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.85rem;">
        <div>{llm_icon} {llm_text}</div>
        <div style="color: #64748b; font-size: 0.75rem;">{model_text}</div>
        <div style="margin-top: 0.25rem;">{rag_icon} {rag_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with topics and info."""
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
    
    # Knowledge base info
    st.markdown("### 📚 Knowledge Base")
    st.markdown(f"**{len(KNOWLEDGE_BASE)}** curated documents")
    
    categories = {}
    for doc in KNOWLEDGE_BASE.values():
        cat = doc.get('category', 'other')
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat in sorted(categories.keys()):
        count = categories[cat]
        st.markdown(f"- {cat.replace('_', ' ').title()}: {count}")
    
    st.markdown("---")
    
    # Sources
    st.markdown("### 📖 Sources")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94a3b8;">
    • SEC Investor.gov<br/>
    • Investopedia<br/>
    • Vanguard Research<br/>
    • Fidelity Learning Center<br/>
    • Bogleheads Wiki<br/>
    • Federal Reserve<br/>
    • FINRA
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Vector store initialization button
    if VECTOR_STORE_AVAILABLE and get_vector_store:
        st.markdown("---")
        st.markdown("### ⚙️ RAG Settings")
        if st.button("🔄 Initialize Vector Store", use_container_width=True):
            with st.spinner("Indexing knowledge base..."):
                try:
                    store = get_vector_store(initialize=True)
                    if store:
                        count = store.index_knowledge_base(force=True)
                        st.success(f"Indexed {count} documents!")
                except Exception as e:
                    st.error(f"Error: {e}")


def render_message(message):
    """Render a single chat message."""
    if message['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br/>{message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show sources if available
        sources_html = ""
        if message.get('sources'):
            sources_html = "<div style='margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #334155;'>"
            sources_html += "<span style='font-size: 0.75rem; color: #64748b;'>📚 Sources: </span>"
            for source in message['sources'][:3]:
                sources_html += f"<span class='source-tag'>{source}</span>"
            sources_html += "</div>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 QuantCademy AI:</strong><br/>
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
    
    # RAG info box
    st.markdown("---")
    st.markdown("""
    <div class="rag-info">
        <strong>🧠 How I Work</strong><br/>
        <span style="font-size: 0.9rem; color: #94a3b8;">
        I use <strong>Retrieval-Augmented Generation (RAG)</strong> to answer your questions:
        <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>Your question is analyzed using semantic search</li>
            <li>Relevant content is retrieved from my knowledge base</li>
            <li>The AI generates a personalized answer using this context</li>
        </ol>
        All information comes from trusted sources like the SEC, Vanguard, and Investopedia.
        </span>
    </div>
    """, unsafe_allow_html=True)


def process_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Check for quick response first (instant, no LLM needed)
    quick_response = get_quick_response(user_input)
    
    if quick_response:
        response = quick_response
        # Still get related sources for display
        results = search_knowledge_base(user_input)
        sources = [doc['title'] for _, _, doc in results[:3]]
    else:
        # Search knowledge base for relevant sources (using semantic if available)
        try:
            if VECTOR_STORE_AVAILABLE and get_vector_store:
                store = get_vector_store()
                if store:
                    results = store.search(user_input, n_results=3)
                    sources = [r['title'] for r in results]
                else:
                    results = search_knowledge_base(user_input)
                    sources = [doc['title'] for _, _, doc in results[:3]]
            else:
                results = search_knowledge_base(user_input)
                sources = [doc['title'] for _, _, doc in results[:3]]
        except Exception:
            results = search_knowledge_base(user_input)
            sources = [doc['title'] for _, _, doc in results[:3]]
        
        # Generate response with Ollama
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
            # Fallback response when Ollama is offline
            if results:
                # Use knowledge base directly
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], dict):
                        # Vector store results
                        top_doc = results[0]
                        response = f"""Based on my knowledge base (**{top_doc.get('source', 'Trusted Source')}**):

{top_doc.get('content', '')[:1500]}

---
*⚠️ For a more personalized, conversational answer, please start Ollama:*
1. Open a terminal
2. Run: `ollama serve`
3. Make sure you have llama3: `ollama pull llama3`
"""
                    else:
                        # Keyword search results
                        _, _, top_doc = results[0]
                        response = f"""Based on my knowledge base (**{top_doc['source']}**):

{top_doc['content'][:1500]}

---
*⚠️ For a more personalized answer, please start Ollama (`ollama serve`).*
"""
            else:
                response = """I apologize, but I can't connect to my AI backend right now. 

**To enable the full AI tutor:**
1. Open a terminal
2. Run: `ollama serve`
3. Make sure you have llama3: `ollama pull llama3`

In the meantime, check out the **Learning Modules** for structured lessons, or try these quick topics in the sidebar!"""
    
    # Add assistant response to history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'sources': sources if not quick_response else sources
    })


if __name__ == "__main__":
    main()
