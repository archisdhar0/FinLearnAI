# QuantCademy - AI-Powered Investing Education

> **Learn investing YOUR way** - Personalized, simulation-backed education that adapts to your goals.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 What Makes This Different

Unlike static content sites (Investopedia, etc.), QuantCademy teaches through:

| Feature | Investopedia | QuantCademy |
|---------|-------------|-------------|
| Content | Static articles | Interactive modules |
| Personalization | None | Adapts to YOUR numbers |
| Learning path | Random browsing | Sequenced curriculum |
| Risk explanation | Text definitions | YOUR portfolio simulations |
| Outcomes | Generic examples | Monte Carlo with YOUR inputs |
| AI Tutor | None | RAG-powered chat with trusted sources |

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
# Option 1: Using terminal
cat > .env << 'EOF'
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-1.5-flash-latest
EOF

# Option 2: Create manually with your editor
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

## 🤖 AI Tutor Features

The AI Tutor uses **Retrieval-Augmented Generation (RAG)** with:

- **20+ curated documents** from SEC, Investopedia, Vanguard, Fidelity, Bogleheads
- **Semantic search** using sentence-transformers for accurate retrieval
- **Gemini Flash** for fast, accurate responses
- **Source citations** for every answer

### Without API Key

The app still works! You can:
- ✅ Use all Learning Modules (6 modules, 30+ lessons)
- ✅ Get quick responses to common questions
- ✅ Access the knowledge base directly
- ❌ AI-generated personalized answers (needs API key)

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

## 🔧 Technical Architecture

```
quantcademy-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Dependencies
├── .env                      # Your API keys (create this, not committed)
├── pages/
│   ├── learning_modules.py   # Interactive learning content
│   ├── ai_tutor.py          # RAG-powered chat interface
│   └── investor_insight.py   # Additional tools
├── rag/
│   ├── knowledge_base.py    # 20+ curated documents
│   ├── vector_store.py      # Semantic search with ChromaDB
│   ├── llm_provider.py      # Gemini/Ollama integration
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

If you prefer running locally without an API key:

```bash
# Install Ollama from https://ollama.ai
ollama serve  # In one terminal
ollama pull llama3  # In another terminal

# Update .env
LLM_PROVIDER=ollama
```

---

## 📊 Data Sources

- **Knowledge Base**: SEC, Investopedia, Vanguard, Fidelity, Bogleheads, FINRA
- **Simulations**: Based on historical market parameters
- **EDA Insights**: SCF 2022 and Reddit community analysis

---

## 🧪 Development

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Initialize vector store (optional, improves search)
python -c "from rag.vector_store import get_vector_store; get_vector_store()"
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
