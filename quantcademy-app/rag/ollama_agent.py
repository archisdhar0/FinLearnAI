"""
QuantCademy LLM Agent
Capstone-grade RAG agent with:
- Advanced retrieval (hybrid BM25 + semantic)
- Reranking
- Confidence gating
- Multi-query decomposition
- Citation-required answers
"""

import os
from typing import Generator, Union, Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import knowledge base functions (legacy fallback)
from .knowledge_base import search_knowledge_base, format_context_for_llm, KNOWLEDGE_BASE

# Try to import advanced retrieval system
ADVANCED_RAG_AVAILABLE = False
try:
    from .retrieval import (
        retrieve_with_citations,
        format_context_with_citations,
        RetrievalResponse,
        ADVANCED_RETRIEVAL_AVAILABLE
    )
    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Advanced retrieval not available: {e}")
    ADVANCED_RETRIEVAL_AVAILABLE = False

# Try to import LLM provider
try:
    from .llm_provider import (
        check_llm_status,
        chat_with_llm,
        LLM_PROVIDER,
        GEMINI_AVAILABLE,
        SYSTEM_PROMPT
    )
    LLM_PROVIDER_AVAILABLE = True
except ImportError:
    LLM_PROVIDER_AVAILABLE = False
    LLM_PROVIDER = "none"
    GEMINI_AVAILABLE = False


def check_ollama_status() -> dict:
    """Check LLM status (supports both Gemini and Ollama)."""
    if LLM_PROVIDER_AVAILABLE:
        status = check_llm_status()
        # Add advanced RAG status
        status["advanced_rag"] = ADVANCED_RAG_AVAILABLE
        status["advanced_retrieval"] = ADVANCED_RETRIEVAL_AVAILABLE
        # Map to expected format for backwards compatibility
        return {
            "status": status.get("status", "offline"),
            "message": status.get("message", ""),
            "provider": status.get("provider", "unknown"),
            "has_llama3": status.get("ollama_available", False),
            "models": [status.get("message", "")],
            "advanced_rag": ADVANCED_RAG_AVAILABLE,
            "advanced_retrieval": ADVANCED_RETRIEVAL_AVAILABLE
        }
    else:
        return {
            "status": "offline",
            "message": "LLM provider not configured",
            "provider": "none",
            "has_llama3": False,
            "models": [],
            "advanced_rag": False,
            "advanced_retrieval": False
        }


def get_rag_context(query: str, user_profile: dict = None) -> tuple:
    """
    Retrieve relevant context using advanced RAG (hybrid + reranking).
    Falls back to keyword search if advanced retrieval is unavailable.
    
    Returns:
        Tuple of (context_string, citations_string, confidence, is_confident, refusal_reason)
    """
    # Try advanced retrieval first
    if ADVANCED_RAG_AVAILABLE:
        try:
            response = retrieve_with_citations(query)
            
            context, citations = format_context_with_citations(response)
            
            # Add user profile to context
            if context and user_profile:
                profile_context = format_user_profile(user_profile)
                context = context + "\n" + profile_context
            
            return (
                context,
                response.get_citation_string(),
                response.confidence,
                response.is_confident,
                response.refusal_reason
            )
        except Exception as e:
            print(f"Advanced retrieval error, falling back to keyword: {e}")
    
    # Fallback to keyword search
    results = search_knowledge_base(query)
    
    if results:
        # Check if the top result has a reasonable relevance score
        top_score = results[0][1] if results else 0  # (doc_id, score, doc)
        
        # If score is too low, treat as no results
        if top_score < 30:  # Less than 30% relevance
            return (
                "",
                "",
                0.15,
                False,
                f"I couldn't find relevant information about this topic. (Match score: {top_score}%)"
            )
        
        context = format_context_for_llm(results)
        sources = [doc.get('source', 'Unknown') for _, _, doc in results[:3]]
        citations = "Sources: " + "; ".join(sources)
        
        # Scale confidence based on score
        confidence = min(0.4 + (top_score / 200), 0.8)
        
        if user_profile:
            profile_context = format_user_profile(user_profile)
            context = context + "\n" + profile_context
        
        return (context, citations, confidence, True, None)
    else:
        return (
            "",
            "",
            0.0,
            False,
            "No relevant information found in the knowledge base."
        )


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
    model: str = None,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """
    Send a message to the configured LLM with capstone-grade RAG.
    
    Features:
    - Hybrid retrieval (BM25 + semantic)
    - Reranking (top 20 → top 5)
    - Confidence gating (refuses if not confident)
    - Citation-required answers
    - Stock picking refusal
    
    Args:
        message: User's question
        user_profile: User's investment profile for personalization
        conversation_history: Previous messages for context
        model: Model to use (ignored, uses env config)
        stream: Whether to stream the response
    
    Yields/Returns:
        Response text (streamed or complete)
    """
    # Get RAG context using advanced retrieval
    context, citations, confidence, is_confident, refusal_reason = get_rag_context(message, user_profile)
    
    if LLM_PROVIDER_AVAILABLE:
        return chat_with_llm(
            message=message,
            context=context,
            citations=citations,
            confidence=confidence,
            is_confident=is_confident,
            refusal_reason=refusal_reason,
            conversation_history=conversation_history,
            stream=stream
        )
    else:
        # Fallback when no LLM available
        error_msg = """❌ No LLM provider configured. 

**To enable the AI tutor, create a `.env` file:**
```
GEMINI_API_KEY=your_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=models/gemini-1.5-flash-latest
```

Or start Ollama: `ollama serve`"""
        if stream:
            def error_gen():
                yield error_msg
            return error_gen()
        return error_msg


def answer_quiz_explanation(
    question: str,
    user_answer: str,
    correct_answer: str,
    is_correct: bool
) -> str:
    """Generate explanation for quiz answers using RAG."""
    prompt = f"""The user answered a quiz question:

Question: {question}
User's answer: {user_answer}
Correct answer: {correct_answer}
Result: {'Correct!' if is_correct else 'Incorrect'}

Provide a brief (2-3 sentence) explanation of why the correct answer is correct.
Use the context from your knowledge base to give an educational explanation."""

    result = chat_with_ollama(prompt, stream=False)
    if hasattr(result, '__iter__') and not isinstance(result, str):
        return "".join(result)
    return result


def generate_module_explanation(module_id: str, user_profile: dict = None) -> str:
    """Generate a personalized explanation for a learning module."""
    from .knowledge_base import get_documents_for_module
    docs = get_documents_for_module(module_id)
    
    if not docs:
        return "No content available for this module yet."
    
    context = "\n\n".join([doc['content'] for _, doc in docs[:3]])
    
    prompt = f"""Based on the following educational content, create a personalized lesson summary.

CONTENT:
{context}

Create a 3-4 paragraph explanation that:
1. Explains the key concepts simply
2. Provides one actionable takeaway
3. Is encouraging and beginner-friendly
4. Cites the sources

Keep it under 400 words."""
    
    result = chat_with_ollama(prompt, user_profile, stream=False)
    if hasattr(result, '__iter__') and not isinstance(result, str):
        return "".join(result)
    return result


def get_related_topics(query: str) -> List[str]:
    """Get related topics the user might want to learn about."""
    if ADVANCED_RAG_AVAILABLE:
        try:
            response = retrieve_with_citations(query, top_k=5, min_confidence=0.1)
            return [r.chunk.section for r in response.results]
        except Exception:
            pass
    
    # Fallback
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

*Sources: SEC Investor.gov (🏛️ Regulatory), Vanguard (🏦 Financial Institution), Bogleheads (🏦 Financial Institution)*
""",
    
    "what is an etf": """
An **ETF (Exchange-Traded Fund)** is like a basket that holds many investments at once! 🧺

**Definition (per SEC)**: An ETF is an investment fund that trades on exchanges like a stock, holding assets like stocks, commodities, or bonds.

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

*Sources: SEC Investor.gov (🏛️ Regulatory), Vanguard (🏦 Financial Institution)*
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

*Sources: SEC Investor.gov (🏛️ Regulatory), IRS (🏛️ Regulatory), Fidelity (🏦 Financial Institution)*
""",

    "how does compound interest work": """
**Compound interest** is earning interest on your interest - and it's the most powerful force in investing! 💰

**Definition (SEC)**: The process of generating earnings on an asset's reinvested earnings over time.

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

*Sources: SEC Investor.gov (🏛️ Regulatory)*
""",

    "what is a 401k": """
A **401(k)** is an employer-sponsored retirement account with major tax advantages. It's one of the most powerful wealth-building tools available! 🏦

**Definition (IRS)**: A qualified profit-sharing plan that allows employees to save for retirement on a tax-advantaged basis.

**How It Works**:
1. Money comes out of your paycheck before taxes
2. Reduces your taxable income now
3. Grows tax-deferred (no taxes until withdrawal)
4. You pay taxes when you withdraw in retirement

**2024 Limits (per IRS)**:
- Under 50: $23,000/year
- 50+: $30,500/year (extra catch-up)

**THE EMPLOYER MATCH** 🎉:
Many employers match your contributions. This is **FREE MONEY**!
- Example: They match 50% up to 6% of salary
- You put in 6%, they add 3%
- That's an instant 50% return!

**ALWAYS contribute at least enough to get the full match.**

*Sources: IRS (🏛️ Regulatory), FINRA (🏛️ Regulatory), Fidelity (🏦 Financial Institution)*
""",

    "what is diversification": """
**Diversification** means spreading your investments across different assets so you don't have all your eggs in one basket 🥚🧺

**Definition (SEC)**: A risk management strategy that mixes a wide variety of investments within a portfolio.

**The Core Idea**:
If you own only one stock and it crashes, you lose everything.
If you own 500 stocks and one crashes, you barely notice.

**Types of Diversification**:
1. **Across stocks** - Own many companies (index funds do this automatically)
2. **Across sectors** - Tech, healthcare, finance, etc.
3. **Across geographies** - US, international, emerging markets
4. **Across asset classes** - Stocks, bonds, real estate

**Simple Diversified Portfolio**:
- 60% VTI (US stocks - thousands of companies)
- 30% VXUS (International stocks)
- 10% BND (Bonds)

One purchase of each = instant diversification across the entire world economy!

*Sources: SEC Investor.gov (🏛️ Regulatory), FINRA (🏛️ Regulatory), Vanguard Research (🎓 Institutional)*
"""
}


def get_quick_response(query: str) -> Optional[str]:
    """Check if there's a pre-defined quick response for common questions."""
    query_lower = query.lower().strip()
    
    for trigger, response in QUICK_RESPONSES.items():
        if trigger in query_lower:
            return response
    
    return None
