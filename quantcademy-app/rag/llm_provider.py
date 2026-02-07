"""
QuantCademy LLM Provider
Supports multiple LLM backends: Gemini, Ollama, OpenAI
Configured via environment variables.
"""

import os
from typing import Generator, Union, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Import providers based on availability
GEMINI_AVAILABLE = False
OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
except ImportError:
    pass

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    pass


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
- Keep responses focused (300-500 words usually)
"""


def check_llm_status() -> dict:
    """Check which LLM providers are available."""
    status = {
        "provider": LLM_PROVIDER,
        "status": "offline",
        "message": "",
        "gemini_available": GEMINI_AVAILABLE,
        "ollama_available": False
    }
    
    if LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE:
        try:
            # Quick test to see if API key works
            model = genai.GenerativeModel(GEMINI_MODEL)
            status["status"] = "online"
            status["message"] = f"Gemini ({GEMINI_MODEL})"
            return status
        except Exception as e:
            status["message"] = f"Gemini error: {str(e)}"
    
    if LLM_PROVIDER == "ollama" or not GEMINI_AVAILABLE:
        try:
            import requests
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                status["ollama_available"] = True
                if any('llama3' in name.lower() for name in model_names):
                    status["status"] = "online"
                    status["message"] = "Ollama (llama3)"
                else:
                    status["message"] = "Ollama online but llama3 not found"
        except Exception:
            pass
    
    if status["status"] == "offline":
        if not GEMINI_API_KEY:
            status["message"] = "No GEMINI_API_KEY in .env"
        else:
            status["message"] = "No LLM provider available"
    
    return status


def chat_with_llm(
    message: str,
    context: str = "",
    conversation_history: list = None,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """
    Send a message to the configured LLM with RAG context.
    
    Args:
        message: User's question
        context: Retrieved context from knowledge base
        conversation_history: Previous messages
        stream: Whether to stream the response
    
    Yields/Returns:
        Response text
    """
    # Build the augmented prompt
    augmented_message = f"""CONTEXT FROM TRUSTED FINANCIAL EDUCATION SOURCES:
{context}

USER QUESTION: {message}

Please answer the user's question using the context provided above. If the context covers the topic, reference the sources. If not, use your general knowledge about investing basics. Always be helpful, accurate, and beginner-friendly."""

    # Try Gemini first
    if LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE:
        return _chat_with_gemini(augmented_message, conversation_history, stream)
    
    # Fall back to Ollama
    if OLLAMA_AVAILABLE:
        return _chat_with_ollama(augmented_message, conversation_history, stream)
    
    # No provider available
    error_msg = "❌ No LLM provider available. Please check your .env configuration."
    if stream:
        def error_generator():
            yield error_msg
        return error_generator()
    return error_msg


def _chat_with_gemini(
    message: str,
    conversation_history: list = None,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """Chat using Google Gemini."""
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT
        )
        
        # Build chat history
        history = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                history.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
        
        # Create chat session
        chat = model.start_chat(history=history)
        
        if stream:
            def generate():
                response = chat.send_message(message, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            return generate()
        else:
            response = chat.send_message(message)
            return response.text
            
    except Exception as e:
        error_msg = f"❌ Gemini error: {str(e)}"
        if stream:
            def error_gen():
                yield error_msg
            return error_gen()
        return error_msg


def _chat_with_ollama(
    message: str,
    conversation_history: list = None,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """Chat using Ollama."""
    import requests
    import json
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": OLLAMA_MODEL,
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
            def generate():
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
            return generate()
        else:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120
            )
            data = response.json()
            return data.get('message', {}).get('content', 'No response generated.')
            
    except Exception as e:
        error_msg = f"❌ Ollama error: {str(e)}"
        if stream:
            def error_gen():
                yield error_msg
            return error_gen()
        return error_msg
