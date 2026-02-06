"""
QuantCademy AI Tutor - RAG-powered chat interface
Uses Ollama + llama3 with knowledge base retrieval
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
        QUICK_RESPONSES
    )
    from rag.knowledge_base import search_knowledge_base, KNOWLEDGE_BASE
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    IMPORT_ERROR = str(e)

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
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background: #f1f5f9;
        color: #1e293b;
    }
    .source-tag {
        background: #e2e8f0;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }
    .status-online {
        color: #10b981;
    }
    .status-offline {
        color: #ef4444;
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
            Ask me anything about investing! I use trusted sources (SEC, Investopedia, Vanguard) 
            to give you accurate, beginner-friendly answers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if RAG module loaded correctly
    if not RAG_AVAILABLE:
        st.error(f"⚠️ Could not load AI module: {IMPORT_ERROR}")
        st.info("Make sure you're running from the quantcademy-app directory.")
        return
    
    # Check Ollama status
    col1, col2 = st.columns([3, 1])
    
    with col2:
        status = check_ollama_status()
        if status['status'] == 'online':
            st.markdown(f"""
            <div style="text-align: right;">
                <span class="status-online">● Ollama Online</span><br/>
                <span style="font-size: 0.8rem; color: #64748b;">
                    {'llama3 ✓' if status.get('has_llama3') else 'llama3 not found'}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: right;">
                <span class="status-offline">● Ollama Offline</span><br/>
                <span style="font-size: 0.8rem; color: #64748b;">
                    Run: ollama serve
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar with quick topics
    with st.sidebar:
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
            "How do I open a Roth IRA?",
            "What is diversification?"
        ]
        
        for topic in quick_topics:
            if st.button(topic, key=f"topic_{topic}", use_container_width=True):
                st.session_state.pending_question = topic
        
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        st.markdown(f"**{len(KNOWLEDGE_BASE)}** documents loaded")
        
        categories = set(doc.get('category', 'other') for doc in KNOWLEDGE_BASE.values())
        for cat in sorted(categories):
            count = sum(1 for doc in KNOWLEDGE_BASE.values() if doc.get('category') == cat)
            st.markdown(f"- {cat.replace('_', ' ').title()}: {count}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
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
                    sources_html = "<div style='margin-top: 0.5rem;'>"
                    for source in message['sources'][:3]:
                        sources_html += f"<span class='source-tag'>📄 {source}</span>"
                    sources_html += "</div>"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 QuantCademy AI:</strong><br/>{message['content']}
                    {sources_html}
                </div>
                """, unsafe_allow_html=True)
    
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
        st.markdown("### 👋 Try asking:")
        
        examples = [
            "I'm 25 with $500/month to invest. Where do I start?",
            "Explain the three-fund portfolio like I'm 5",
            "What happens if the stock market crashes?",
            "Should I pay off debt or invest first?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                    process_user_input(example)
                    st.rerun()


def process_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Check for quick response first
    quick_response = get_quick_response(user_input)
    
    if quick_response:
        response = quick_response
        sources = []
    else:
        # Search knowledge base for relevant sources
        search_results = search_knowledge_base(user_input)
        sources = [doc['title'] for _, _, doc in search_results[:3]]
        
        # Generate response with Ollama
        status = check_ollama_status()
        
        if status['status'] == 'online':
            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in chat_with_ollama(
                user_input,
                st.session_state.get('user_profile'),
                st.session_state.chat_history[:-1],  # Exclude current message
                stream=True
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.empty()
            response = full_response
        else:
            # Fallback response when Ollama is offline
            if search_results:
                # Use knowledge base directly
                top_doc = search_results[0][2]
                response = f"""Based on my knowledge base ({top_doc['source']}):

{top_doc['content'][:1000]}...

*Note: For a more personalized answer, please start Ollama (`ollama serve` in terminal).*
"""
            else:
                response = """I apologize, but I can't connect to my AI backend right now. 

**To enable the AI tutor:**
1. Open a terminal
2. Run: `ollama serve`
3. Make sure you have llama3: `ollama pull llama3`

In the meantime, check out the learning modules in the main app!"""
    
    # Add assistant response to history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'sources': sources if not quick_response else []
    })


if __name__ == "__main__":
    main()
