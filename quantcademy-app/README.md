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
| Decision support | None | Concrete portfolio recommendations |

## 🚀 Quick Start

```bash
# Navigate to app directory
cd quantcademy-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📚 MVP Modules

### Foundation Track (4 Modules)

1. **🎯 Your Goal + Timeline**
   - Define investment horizon
   - Create money buckets (emergency, near-term, long-term)
   - Personalized warnings based on your situation

2. **📊 Risk, Explained With Your Numbers**
   - Interactive volatility vs drawdown visualization
   - Historical crash examples
   - See probability of loss at YOUR horizon

3. **🏗️ Build Your First Portfolio**
   - 3-ETF strategy recommendation
   - Allocation sliders with real-time stats
   - Monthly contribution breakdown

4. **🔮 What Could Happen? (Simulator)**
   - Monte Carlo outcome bands
   - "What if I stop contributing?" toggle
   - Probability of loss over time

## 🔧 Technical Architecture

```
quantcademy-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── data/
│   ├── __init__.py
│   └── curriculum.py      # Learning paths, quizzes, personalization
├── simulations/
│   ├── __init__.py
│   └── portfolio_sim.py   # Monte Carlo, drawdown, risk calculations
├── components/            # (Future: reusable UI components)
└── assets/               # (Future: images, styles)
```

## 🎨 Key Features

### Personalization Engine
```python
# User profile drives all content
PERSONALIZATION = {
    "short_horizon": {
        "emphasis": "capital_preservation",
        "key_message": "With a shorter timeline, protecting principal matters more...",
        "recommended_allocation": {"stocks": 30, "bonds": 50, "cash": 20}
    },
    # ... adapts to user's situation
}
```

### Simulation Engine
```python
# Monte Carlo simulations with YOUR numbers
sim = monte_carlo_simulation(
    initial_investment=user_initial,
    monthly_contribution=user_monthly,
    weights=user_portfolio,
    years=user_horizon
)
# Returns: percentile bands, probability of loss, worst/best outcomes
```

### Misconception Detection
```python
# Quiz questions catch common mistakes
QUIZ_QUESTIONS = {
    "risk_explained": [{
        "question": "If your portfolio drops 20%...",
        "misconception_if_wrong": {
            0: "Selling during drops locks in losses..."
        }
    }]
}
```

## 📈 Roadmap

### Phase 1 (Current MVP)
- [x] 4 Foundation modules
- [x] Personalization by horizon & risk tolerance
- [x] Monte Carlo simulations
- [x] Interactive portfolio builder
- [x] Beautiful Streamlit UI

### Phase 2 (Next)
- [ ] Investor Insight track (3 modules)
- [ ] Quiz-based misconception routing
- [ ] Progress persistence (database)
- [ ] RAG-powered Q&A chatbot

### Phase 3 (Future)
- [ ] Applied Investing track
- [ ] Real market data integration
- [ ] Mobile-responsive design
- [ ] User accounts & saved portfolios

## 🧪 Development

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Run tests (future)
pytest tests/
```

## 📊 Data Sources

- **Simulations**: Based on historical market parameters
- **RAG Content**: See `/rag_sources.md` for educational content sources
- **EDA Insights**: Based on SCF 2022 and Reddit community analysis

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes
4. Submit a PR

## 📄 License

MIT License - feel free to use for educational purposes.

---

Built with ❤️ for breaking barriers to investing.
