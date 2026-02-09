# QuantCademy - Capstone-Grade AI Investing Education

> **Learn investing YOUR way** - Personalized, simulation-backed education with capstone-grade RAG.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![RAG](https://img.shields.io/badge/RAG-Capstone--Grade-gold)
![License](https://img.shields.io/badge/License-MIT-green)

## 🏆 Capstone-Grade RAG Features

This isn't a toy demo—it's production-quality RAG with:

| Feature | What It Does |
|---------|-------------|
| **Source Tiering** | SEC/FINRA (Tier 1) > Fed/CFA (Tier 2) > Fidelity (Tier 3) > Investopedia (Tier 4) |
| **Hybrid Retrieval** | BM25 keyword search + semantic embeddings |
| **Reranking** | Cross-encoder reranks top 20 → top 5 |
| **Confidence Gating** | Refuses when retrieval confidence is low |
| **Multi-Query** | Decomposes complex questions into 2-4 subqueries |
| **Citation-Required** | Every answer includes source citations |
| **Stock-Picking Refusal** | Politely declines to recommend individual stocks |

### ✅ RAG Checklist (All Implemented)

- [x] Curated sources with tiering
- [x] Structured chunking + metadata
- [x] Hybrid retrieval + reranking
- [x] Citation-required answers
- [x] Confidence gating + safe refusal
- [x] Multi-step retrieval for complex questions
- [x] Source attribution with tier badges

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to app directory
cd quantcademy-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AI Tutor (Required for Chat Feature)

Create a `.env` file in the `quantcademy-app` directory:

```bash
# Using your editor, create a .env file with:
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-1.5-flash-latest
```

**Get your free Gemini API key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google
3. Click "Create API Key"
4. Copy and paste into your `.env` file

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🤖 AI Tutor Architecture

### Source Tiers (Trust Hierarchy)

```
🏛️ Tier 1 (Highest): SEC, FINRA, IRS, Treasury
        ↓
🎓 Tier 2: Federal Reserve, CFA Institute, Vanguard Research
        ↓
🏦 Tier 3: Fidelity, Schwab, Bogleheads Wiki
        ↓
📚 Tier 4: Investopedia, NerdWallet
        ↓
📰 Tier 5: General sources
```

### Hybrid Retrieval Pipeline

```
User Query → [Multi-Query Decomposition (if complex)]
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
         BM25 Search                Semantic Search
        (keyword match)            (embedding similarity)
              ↓                           ↓
              └─────────────┬─────────────┘
                            ↓
                   Hybrid Merge (40% BM25 + 60% Semantic)
                            ↓
                   Tier Boost (prefer regulatory sources)
                            ↓
                   Cross-Encoder Rerank (top 20 → top 5)
                            ↓
                   Confidence Scoring
                            ↓
         ┌──────────────────┴──────────────────┐
         ↓                                     ↓
   Confident (>35%)                    Not Confident
         ↓                                     ↓
   Generate Answer                    Safe Refusal
   with Citations                     "Not enough sources..."
```

### Knowledge Base Stats

- **25+ semantic chunks** from authoritative sources
- **5 chunk types**: definitions, concepts, examples, procedures, tables
- **Full metadata**: source tier, category, difficulty, key terms

---

## 📊 Features by Capability

### 1. Source-Tiered Responses

When answering, the system:
- Prefers Tier 1 (SEC/FINRA) sources over Tier 4 (Investopedia)
- Applies a tier boost during retrieval
- Shows source tier badges in citations

### 2. Confidence Gating

```python
# If confidence < 35%, refuses to answer:
"I don't have enough reliable information in my sources to answer this confidently."
```

### 3. Stock-Picking Refusal

```python
# Triggers: "which stock", "should I buy", "is Tesla"
# Response: Educational explanation of why index funds are better
```

### 4. Multi-Query Decomposition

Complex queries like:
> "Compare 401k and Roth IRA and tell me which is better for a 30 year old"

Get decomposed into:
1. "What is a 401k?"
2. "What is a Roth IRA?"
3. "How does 401k compare to Roth IRA?"

---

## 🧪 Test the Capstone Features

Try these queries:

| Query | Expected Behavior |
|-------|------------------|
| "What is an ETF?" | High-confidence answer with SEC citations |
| "Which stock should I buy?" | Polite refusal + index fund explanation |
| "Compare 401k and IRA" | Multi-query decomposition |
| "Kelly criterion for crypto" | Low-confidence refusal |

---

## 🔧 Technical Architecture

```
quantcademy-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Dependencies
├── .env                      # Your API keys (create this, not committed)
├── pages/
│   ├── learning_modules.py   # Interactive learning content
│   ├── ai_tutor.py          # Capstone RAG chat interface
│   └── investor_insight.py   # Additional tools
├── rag/
│   ├── knowledge_base.py    # Legacy knowledge base
│   ├── knowledge_base_v2.py # Capstone: Chunked KB with source tiering
│   ├── retrieval.py         # Capstone: Hybrid + rerank + confidence
│   ├── vector_store.py      # Semantic search with ChromaDB
│   ├── llm_provider.py      # Gemini/Ollama with citations
│   └── ollama_agent.py      # RAG orchestration
├── data/
│   └── curriculum.py        # Learning paths, quizzes
└── simulations/
    └── portfolio_sim.py     # Monte Carlo, risk calculations
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | `gemini` (recommended) or `ollama` |
| `GEMINI_API_KEY` | For Gemini | Get from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | Optional | Default: `models/gemini-1.5-flash-latest` |
| `OLLAMA_BASE_URL` | For Ollama | Default: `http://localhost:11434` |
| `OLLAMA_MODEL` | For Ollama | Default: `llama3` |

### Using Ollama Instead (Local, Free)

```bash
# Install Ollama from https://ollama.ai
ollama serve  # In one terminal
ollama pull llama3  # In another terminal

# Update .env
LLM_PROVIDER=ollama
```

---

## 📚 Learning Modules

### Module 1: Market Mechanics (Beginner)
- What is a Stock & Equity?
- The Order Book (Bid/Ask/Spread)
- Market vs Limit vs Stop Orders
- Liquidity Providers vs Takers
- Exchanges vs Dark Pools

### Module 2: Macro Economics (Beginner)
- Interest Rates & The Fed
- Inflation & Purchasing Power
- GDP & Economic Cycles
- Currency & Exchange Rates
- Geopolitical Risk Factors

### Module 3: Technical Analysis (Intermediate)
- Candlestick Patterns
- Support & Resistance
- Moving Averages (SMA, EMA)
- RSI, MACD & Momentum
- Volume Analysis
- Chart Patterns

### Module 3.5: Fundamental Analysis (Intermediate)
- Reading Financial Statements
- P/E Ratio & Valuation Metrics
- Revenue & Earnings Growth
- Balance Sheet Analysis
- Cash Flow Analysis
- Competitive Moats

### Module 4: Quant Strategies (Advanced)
- Factor Investing
- Mean Reversion Strategies
- Momentum Strategies
- Statistical Arbitrage
- Backtesting & Validation
- Risk Management Systems

### Module 5: Advanced Options (Expert)
- Options Greeks Deep Dive
- Vertical Spreads
- Iron Condors & Butterflies
- Calendar & Diagonal Spreads
- Volatility Trading
- Portfolio Hedging

---

## 📊 Data Sources

### Knowledge Base Sources (by Tier)

**🏛️ Tier 1 - Regulatory:**
- SEC Investor.gov
- FINRA Investor Education
- IRS Retirement Plans
- TreasuryDirect.gov

**🎓 Tier 2 - Institutional:**
- Federal Reserve
- CFA Institute
- Vanguard Research
- SPIVA Research

**🏦 Tier 3 - Financial Institutions:**
- Fidelity Learning Center
- Schwab Education
- Bogleheads Wiki

**📚 Tier 4 - Educational:**
- Investopedia
- NerdWallet

---

## 🧪 Development

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Test retrieval system
python -m rag.retrieval "What is an ETF?"

# Check knowledge base stats
python -c "from rag.knowledge_base_v2 import get_knowledge_base_stats; print(get_knowledge_base_stats())"
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes
4. Submit a PR

---

## 📄 License

MIT License - feel free to use for educational purposes.

---

Built with ❤️ for breaking barriers to investing.
