"""
QuantCademy LLM Agent using Ollama
Connects to local Ollama instance with llama3 for RAG-powered tutoring.
"""

import requests
import json
from typing import Generator, Union, Optional
from .knowledge_base import search_knowledge_base, format_context_for_llm, KNOWLEDGE_BASE

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"

# System prompt for the investing tutor
SYSTEM_PROMPT = """You are QuantCademy AI, a friendly and knowledgeable investing tutor designed to help beginners learn about investing.

YOUR ROLE:
- Teach investing concepts in simple, clear language
- Use analogies and examples beginners can understand
- Be encouraging and patient
- Never give specific financial advice or stock picks
- Always emphasize that investing involves risk

YOUR STYLE:
- Conversational but informative
- Break down complex topics into digestible parts
- Use bullet points and structure when helpful
- Relate concepts to the user's personal situation when possible
- Ask follow-up questions to ensure understanding

IMPORTANT GUIDELINES:
1. If asked about specific stocks to buy, politely decline and explain why diversified index funds are better for beginners
2. Always mention that you're an AI and users should consult a financial advisor for personal advice
3. When discussing risk, be honest about potential losses
4. Encourage long-term thinking over short-term speculation
5. Reference the educational content provided in your context

When answering questions, use the provided context from trusted sources (SEC, Investopedia, Vanguard, etc.) to support your explanations.
"""


def check_ollama_status() -> dict:
    """Check if Ollama is running and available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            has_llama3 = any('llama3' in name.lower() for name in model_names)
            return {
                "status": "online",
                "models": model_names,
                "has_llama3": has_llama3
            }
        return {"status": "error", "message": "Unexpected response"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "message": "Cannot connect to Ollama. Make sure it's running."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_rag_context(query: str, user_profile: dict = None) -> str:
    """
    Retrieve relevant context from knowledge base for the query.
    Also includes user profile for personalization.
    """
    # Search knowledge base
    results = search_knowledge_base(query)
    
    # Format retrieved documents
    if results:
        context = format_context_for_llm(results)
    else:
        context = "No specific documents found. Use general investing knowledge."
    
    # Add user profile context if available
    profile_context = ""
    if user_profile:
        profile_context = f"""

USER PROFILE:
- Investment horizon: {user_profile.get('horizon_years', 'Not specified')} years
- Monthly contribution: ${user_profile.get('monthly_contribution', 'Not specified')}
- Risk tolerance: {user_profile.get('risk_tolerance', 'Not specified')}/10
- Emergency fund: {user_profile.get('emergency_fund', 'Not specified')}

Personalize your response based on this profile.
"""
    
    return context + profile_context


def chat_with_ollama(
    message: str,
    user_profile: dict = None,
    conversation_history: list = None,
    model: str = DEFAULT_MODEL,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """
    Send a message to Ollama with RAG context.
    
    Args:
        message: User's question
        user_profile: User's investment profile for personalization
        conversation_history: Previous messages for context
        model: Ollama model to use
        stream: Whether to stream the response
    
    Yields/Returns:
        Response text (streamed or complete)
    """
    # Get RAG context
    context = get_rag_context(message, user_profile)
    
    # Build the prompt with context
    augmented_message = f"""CONTEXT FROM TRUSTED SOURCES:
{context}

USER QUESTION: {message}

Please answer the user's question using the context provided. If the context doesn't cover the topic, use your general knowledge about investing basics. Always be helpful, accurate, and beginner-friendly."""

    # Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": augmented_message})
    
    # Make request to Ollama
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }
    
    try:
        if stream:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
                    if data.get('done', False):
                        break
        else:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=60
            )
            data = response.json()
            return data.get('message', {}).get('content', 'No response generated.')
            
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Cannot connect to Ollama. Please make sure Ollama is running (`ollama serve`)."
        if stream:
            yield error_msg
        else:
            return error_msg
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        if stream:
            yield error_msg
        else:
            return error_msg


def generate_module_explanation(module_id: str, user_profile: dict = None) -> str:
    """
    Generate a personalized explanation for a learning module.
    """
    # Get documents related to this module
    from .knowledge_base import get_documents_for_module
    docs = get_documents_for_module(module_id)
    
    if not docs:
        return "No content available for this module yet."
    
    # Format context
    context = "\n\n".join([doc['content'] for _, doc in docs[:3]])
    
    prompt = f"""Based on the following educational content, create a personalized lesson summary for a user learning about investing.

CONTENT:
{context}

USER PROFILE:
- Horizon: {user_profile.get('horizon_years', 15) if user_profile else 15} years
- Monthly contribution: ${user_profile.get('monthly_contribution', 500) if user_profile else 500}
- Risk tolerance: {user_profile.get('risk_tolerance', 5) if user_profile else 5}/10

Create a 3-4 paragraph explanation that:
1. Explains the key concepts simply
2. Relates them to the user's specific situation
3. Provides one actionable takeaway
"""
    
    return chat_with_ollama(prompt, user_profile, stream=False)


def answer_quiz_explanation(question: str, user_answer: str, correct_answer: str, is_correct: bool) -> str:
    """
    Generate an explanation for a quiz answer.
    """
    prompt = f"""A user just answered a quiz question about investing.

Question: {question}
User's answer: {user_answer}
Correct answer: {correct_answer}
Was correct: {is_correct}

Provide a brief (2-3 sentence) explanation of why the correct answer is right, and if the user was wrong, gently explain their misconception. Be encouraging!"""

    return chat_with_ollama(prompt, stream=False)


# Pre-defined responses for common questions (faster than LLM)
QUICK_RESPONSES = {
    "what should i invest in": """
Great question! For beginners, I recommend starting with low-cost index funds like:

📈 **VTI** - Total US Stock Market
🌍 **VXUS** - International Stocks  
💵 **BND** - US Bonds

A simple 3-fund portfolio gives you instant diversification across thousands of companies. Start with your 401(k) if you have one (especially to get any employer match), then open a Roth IRA.

Would you like me to explain how to choose your allocation between stocks and bonds?
""",
    
    "what is an etf": """
An ETF (Exchange-Traded Fund) is like a basket that holds many investments at once! 🧺

Think of it this way: Instead of buying 500 individual stocks, you buy ONE share of an S&P 500 ETF and instantly own a tiny piece of all 500 companies.

**Benefits:**
- Instant diversification
- Very low fees (often 0.03%)
- Trade like stocks
- No minimum investment

Popular examples: VTI, VOO, VXUS, BND

Any questions about how they work?
""",
    
    "when should i start investing": """
The best time to start investing was yesterday. The second best time is TODAY! ⏰

Here's the priority order:
1. ✅ Build 3-6 month emergency fund first
2. ✅ Pay off high-interest debt (credit cards)
3. ✅ Invest in 401(k) up to employer match (free money!)
4. ✅ Open a Roth IRA

You don't need a lot to start - even $50/month makes a difference over time thanks to compound growth.

What's holding you back from starting?
"""
}


def get_quick_response(query: str) -> Optional[str]:
    """Check if there's a pre-defined quick response for common questions."""
    query_lower = query.lower().strip()
    
    for trigger, response in QUICK_RESPONSES.items():
        if trigger in query_lower:
            return response
    
    return None
