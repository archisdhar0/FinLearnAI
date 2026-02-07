"""
QuantCademy LLM Agent using Ollama
Connects to local Ollama instance with llama3 for RAG-powered tutoring.
Uses semantic search via vector store for better context retrieval.
"""

import requests
import json
from typing import Generator, Union, Optional, List, Dict

# Import knowledge base functions
from .knowledge_base import search_knowledge_base, format_context_for_llm, KNOWLEDGE_BASE

# Try to import vector store for semantic search
try:
    from .vector_store import (
        semantic_search, 
        get_context_for_query, 
        get_vector_store,
        VECTOR_STORE_AVAILABLE
    )
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    semantic_search = None
    get_context_for_query = None
    get_vector_store = None

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
- Use bullet points, tables, and structure when helpful
- Relate concepts to the user's personal situation when possible
- Ask follow-up questions to ensure understanding

YOUR KNOWLEDGE SOURCES:
- SEC Investor.gov
- Investopedia
- Vanguard Research
- Fidelity Learning Center
- Bogleheads Wiki
- Federal Reserve
- FINRA Investor Education

IMPORTANT GUIDELINES:
1. If asked about specific stocks to buy, politely decline and explain why diversified index funds are better for beginners
2. Always mention that you're an AI and users should consult a financial advisor for personal advice
3. When discussing risk, be honest about potential losses
4. Encourage long-term thinking over short-term speculation
5. Reference the educational content provided in your context
6. Cite your sources when possible (e.g., "According to Vanguard research...")
7. If you don't know something, say so rather than making things up

RESPONSE FORMAT:
- Use markdown formatting for clarity
- Use headers (##) for major sections
- Use bullet points for lists
- Use **bold** for important terms
- Keep responses focused and not too long (300-500 words usually)
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


def get_rag_context(query: str, user_profile: dict = None, use_semantic: bool = True) -> tuple:
    """
    Retrieve relevant context from knowledge base for the query.
    Uses semantic search if available, falls back to keyword search.
    
    Returns:
        Tuple of (context_string, source_list)
    """
    sources = []
    
    # Try semantic search first (much better results)
    if use_semantic and VECTOR_STORE_AVAILABLE and get_context_for_query:
        try:
            # Get vector store and ensure it's indexed
            store = get_vector_store()
            if store:
                results, context = store.search_with_context(query, n_results=3)
                sources = [r['title'] for r in results]
                
                if context:
                    # Add user profile context if available
                    if user_profile:
                        profile_context = format_user_profile(user_profile)
                        context = context + "\n" + profile_context
                    
                    return context, sources
        except Exception as e:
            print(f"Semantic search error, falling back to keyword: {e}")
    
    # Fallback to keyword search
    results = search_knowledge_base(query)
    
    if results:
        context = format_context_for_llm(results)
        sources = [doc['title'] for _, _, doc in results[:3]]
    else:
        context = "No specific documents found. Use general investing knowledge."
    
    # Add user profile context if available
    if user_profile:
        profile_context = format_user_profile(user_profile)
        context = context + "\n" + profile_context
    
    return context, sources


def format_user_profile(user_profile: dict) -> str:
    """Format user profile for context injection."""
    if not user_profile:
        return ""
    
    return f"""
USER PROFILE (personalize your response based on this):
- Investment time horizon: {user_profile.get('horizon_years', 'Not specified')} years
- Monthly contribution: ${user_profile.get('monthly_contribution', 'Not specified')}
- Risk tolerance: {user_profile.get('risk_tolerance', 'Not specified')}/10
- Has emergency fund: {user_profile.get('emergency_fund', 'Not specified')}
- Age: {user_profile.get('age', 'Not specified')}
- Investment goal: {user_profile.get('goal', 'Not specified')}

Tailor your advice to this person's specific situation.
"""


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
    # Get RAG context using semantic search
    context, sources = get_rag_context(message, user_profile)
    
    # Build the prompt with context
    augmented_message = f"""CONTEXT FROM TRUSTED FINANCIAL EDUCATION SOURCES:
{context}

USER QUESTION: {message}

Please answer the user's question using the context provided above. If the context covers the topic, reference the sources. If not, use your general knowledge about investing basics. Always be helpful, accurate, and beginner-friendly. Remember to:
1. Use simple language and analogies
2. Structure your response clearly
3. Mention if something requires professional advice
4. Be encouraging about their investing journey"""

    # Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided (for context continuity)
    if conversation_history:
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
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
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120
            )
            data = response.json()
            return data.get('message', {}).get('content', 'No response generated.')
            
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Cannot connect to Ollama. Please make sure Ollama is running (`ollama serve`)."
        if stream:
            yield error_msg
        else:
            return error_msg
    except requests.exceptions.Timeout:
        error_msg = "❌ Request timed out. Please try again with a shorter question."
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
    from .knowledge_base import get_documents_for_module
    docs = get_documents_for_module(module_id)
    
    if not docs:
        return "No content available for this module yet."
    
    # Format context from related documents
    context = "\n\n".join([doc['content'] for _, doc in docs[:3]])
    
    profile_str = ""
    if user_profile:
        profile_str = f"""
USER PROFILE:
- Horizon: {user_profile.get('horizon_years', 15)} years
- Monthly contribution: ${user_profile.get('monthly_contribution', 500)}
- Risk tolerance: {user_profile.get('risk_tolerance', 5)}/10
"""
    
    prompt = f"""Based on the following educational content, create a personalized lesson summary.

CONTENT:
{context}

{profile_str}

Create a 3-4 paragraph explanation that:
1. Explains the key concepts simply
2. Relates them to the user's specific situation (if profile provided)
3. Provides one actionable takeaway
4. Is encouraging and beginner-friendly

Keep it under 400 words."""
    
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

Provide a brief (2-3 sentence) explanation of why the correct answer is right. If the user was wrong, gently explain their misconception. Be encouraging!"""

    return chat_with_ollama(prompt, stream=False)


def get_related_topics(query: str) -> List[str]:
    """
    Get related topics the user might want to learn about.
    """
    if VECTOR_STORE_AVAILABLE and semantic_search:
        results = semantic_search(query, n_results=5)
        return [r['title'] for r in results if r['relevance_score'] > 0.3]
    else:
        results = search_knowledge_base(query)
        return [doc['title'] for _, _, doc in results]


# Pre-defined responses for common questions (faster than LLM)
QUICK_RESPONSES = {
    "what should i invest in": """
Great question! For beginners, I recommend starting with **low-cost index funds**:

📈 **For US Stocks**: VTI (Total US Market) or VOO (S&P 500)
🌍 **For International**: VXUS (Total International)
💵 **For Bonds**: BND (Total Bond Market)

**The Simple Approach**: Start with a target-date fund matched to your retirement year (like Vanguard Target Retirement 2055). It's diversified and rebalances automatically.

**Priority Order**:
1. Get your 401(k) employer match first (free money!)
2. Open a Roth IRA
3. Max out both over time

Would you like me to explain how to choose between stocks and bonds based on your age?

*Source: Vanguard, Bogleheads*
""",
    
    "what is an etf": """
An **ETF (Exchange-Traded Fund)** is like a basket that holds many investments at once! 🧺

**Think of it this way**: Instead of buying 500 individual stocks, you buy ONE share of an S&P 500 ETF and instantly own a tiny piece of all 500 companies.

**Key Benefits**:
- ✅ Instant diversification (own 500+ companies in one purchase)
- ✅ Very low fees (as low as 0.03% annually)
- ✅ Trade like stocks (buy/sell anytime)
- ✅ No minimum investment (buy 1 share)

**Popular ETFs for Beginners**:
- **VTI** - Total US Stock Market (~4,000 companies)
- **VOO** - S&P 500 (500 largest US companies)
- **VXUS** - International stocks
- **BND** - US bonds

An ETF is often the best way to start investing. One purchase = instant diversification!

*Source: Vanguard, Investopedia*
""",
    
    "when should i start investing": """
The best time to start investing was yesterday. The second best time is **TODAY**! ⏰

**But first, make sure you have**:
1. ✅ Emergency fund (3-6 months expenses)
2. ✅ High-interest debt paid off (credit cards)
3. ✅ Stable income to invest consistently

**Priority Order**:
1. **401(k) to employer match** - This is FREE MONEY (50-100% instant return)!
2. **Roth IRA** - Tax-free growth forever
3. **Max out 401(k)** - More tax-advantaged space
4. **Taxable brokerage** - After tax-advantaged is maxed

**You don't need much to start**:
- $50/month = $600/year
- Even that grows to ~$25,000 in 20 years at 7%

**The math favors starting early**: Someone who invests from 25-35 (10 years) often ends up with MORE than someone who invests from 35-65 (30 years), because compound interest had more time to work.

What's holding you back from starting?

*Source: SEC Investor.gov, Fidelity*
""",

    "how does compound interest work": """
**Compound interest** is earning interest on your interest - and it's the most powerful force in investing! 💰

**Simple Example**:
Start with $10,000 at 7% annual return:
- Year 1: $10,700 (+$700)
- Year 5: $14,026 (+$4,026)
- Year 10: $19,672 (+$9,672)
- Year 20: $38,697 (+$28,697)
- Year 30: **$76,123** (+$66,123!)

See how the growth accelerates? That's compounding at work.

**The Rule of 72**:
Divide 72 by your return rate to estimate years to double:
- At 7%: 72 ÷ 7 = ~10 years to double
- At 10%: 72 ÷ 10 = ~7 years to double

**Why Starting Early Matters**:
$5,000/year from age 25-35 (10 years) beats $5,000/year from age 35-65 (30 years).
The early investor contributed $100,000 LESS but ends up with MORE money!

**Key Takeaway**: Time is your most valuable asset. Start now, even with small amounts.

*Source: SEC Investor.gov*
""",

    "what is a 401k": """
A **401(k)** is an employer-sponsored retirement account with major tax advantages. It's one of the most powerful wealth-building tools available! 🏦

**How It Works**:
1. Money comes out of your paycheck before taxes
2. Reduces your taxable income now
3. Grows tax-deferred (no taxes until withdrawal)
4. You pay taxes when you withdraw in retirement

**2024 Limits**:
- Under 50: $23,000/year
- 50+: $30,500/year (extra catch-up)

**THE EMPLOYER MATCH** 🎉:
Many employers match your contributions. This is **FREE MONEY**!
- Example: They match 50% up to 6% of salary
- You put in 6%, they add 3%
- That's an instant 50% return!

**ALWAYS contribute at least enough to get the full match.**

**Traditional vs Roth 401(k)**:
- Traditional: Tax break now, taxed in retirement
- Roth: No tax break now, tax-FREE in retirement

**What to Invest In**:
Look for low-cost index funds or a target-date fund for your retirement year.

*Source: IRS, Fidelity*
""",

    "what is diversification": """
**Diversification** means spreading your investments across different assets so you don't have all your eggs in one basket 🥚🧺

**The Core Idea**:
If you own only one stock and it crashes, you lose everything.
If you own 500 stocks and one crashes, you barely notice.

**Types of Diversification**:
1. **Across stocks** - Own many companies (index funds do this automatically)
2. **Across sectors** - Tech, healthcare, finance, etc.
3. **Across geographies** - US, international, emerging markets
4. **Across asset classes** - Stocks, bonds, real estate

**The Magic**:
Different assets often move in opposite directions. When stocks fall, bonds often rise. This smooths out your returns.

**Simple Diversified Portfolio**:
- 60% VTI (US stocks - thousands of companies)
- 30% VXUS (International stocks)
- 10% BND (Bonds)

One purchase of each = instant diversification across the entire world economy!

**What Diversification WON'T Do**:
- Eliminate all risk (everything can fall together in crashes)
- Guarantee profits
- Beat concentrated bets that happen to win

But it WILL help you sleep at night and stay invested long-term.

*Source: Vanguard, Investopedia*
"""
}


def get_quick_response(query: str) -> Optional[str]:
    """Check if there's a pre-defined quick response for common questions."""
    query_lower = query.lower().strip()
    
    for trigger, response in QUICK_RESPONSES.items():
        if trigger in query_lower:
            return response
    
    return None
