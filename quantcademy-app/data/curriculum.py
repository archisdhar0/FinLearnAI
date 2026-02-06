"""
QuantCademy Curriculum Structure
This is your IP - the sequenced learning path with prerequisites and goals.
"""

CURRICULUM = {
    "foundations": {
        "title": "Foundations",
        "description": "Start your journey into investing with AI-supported guided lessons",
        "modules": [
            {
                "id": "goal_timeline",
                "title": "Your Goal + Timeline",
                "subtitle": "Start Here",
                "description": "Define your investment horizon, goals, and create your money buckets",
                "duration": "10 min",
                "prerequisites": [],
                "learning_goals": [
                    "Understand why you're investing",
                    "Define your time horizon",
                    "Separate emergency fund from investable assets",
                    "Set realistic contribution amounts"
                ],
                "common_pitfalls": [
                    "Investing emergency fund",
                    "Unrealistic timelines",
                    "Not accounting for near-term needs"
                ],
                "simulation": "money_buckets",
                "icon": "🎯"
            },
            {
                "id": "risk_explained",
                "title": "Risk, Explained",
                "subtitle": "With Your Numbers",
                "description": "Understand volatility, drawdowns, and find your risk profile",
                "duration": "15 min",
                "prerequisites": ["goal_timeline"],
                "learning_goals": [
                    "Distinguish volatility from permanent loss",
                    "Understand drawdowns and recovery time",
                    "Match risk tolerance to time horizon",
                    "See how YOUR portfolio could behave"
                ],
                "common_pitfalls": [
                    "Confusing short-term volatility with risk",
                    "Overestimating risk tolerance",
                    "Ignoring sequence of returns risk"
                ],
                "simulation": "risk_visualization",
                "icon": "📊"
            },
            {
                "id": "first_portfolio",
                "title": "Build Your First Portfolio",
                "subtitle": "3-ETF Strategy",
                "description": "Construct a diversified portfolio matched to your profile",
                "duration": "12 min",
                "prerequisites": ["goal_timeline", "risk_explained"],
                "learning_goals": [
                    "Understand asset allocation basics",
                    "Learn about low-cost index ETFs",
                    "Build a simple 3-fund portfolio",
                    "Know why this beats stock picking"
                ],
                "common_pitfalls": [
                    "Over-complicating with too many funds",
                    "Chasing past performance",
                    "Ignoring expense ratios"
                ],
                "simulation": "portfolio_builder",
                "icon": "🏗️"
            },
            {
                "id": "what_could_happen",
                "title": "What Could Happen?",
                "subtitle": "Outcome Simulator",
                "description": "See probability bands and stress-test your decisions",
                "duration": "10 min",
                "prerequisites": ["first_portfolio"],
                "learning_goals": [
                    "Visualize range of possible outcomes",
                    "Understand probability of loss by horizon",
                    "See impact of stopping contributions",
                    "Build confidence through understanding"
                ],
                "common_pitfalls": [
                    "Expecting average returns every year",
                    "Panic selling during drawdowns",
                    "Not understanding sequence risk"
                ],
                "simulation": "monte_carlo",
                "icon": "🔮"
            }
        ]
    },
    "investor_insight": {
        "title": "Investor Insight",
        "description": "Deepen your understanding of market mechanics",
        "modules": [
            {
                "id": "market_basics",
                "title": "Understanding the Market",
                "subtitle": "How Markets Work",
                "description": "Learn how stock markets function and price discovery works",
                "duration": "12 min",
                "prerequisites": ["first_portfolio"],
                "icon": "📈"
            },
            {
                "id": "etf_selection",
                "title": "Smart ETF Selection",
                "subtitle": "Beyond the Basics",
                "description": "Choose the right ETFs for your strategy",
                "duration": "15 min",
                "prerequisites": ["market_basics"],
                "icon": "🎯"
            },
            {
                "id": "passive_active",
                "title": "Passive vs Active",
                "subtitle": "The Evidence",
                "description": "Why passive investing wins for most people",
                "duration": "10 min",
                "prerequisites": ["etf_selection"],
                "icon": "⚖️"
            }
        ]
    },
    "applied_investing": {
        "title": "Applied Investing",
        "description": "Hands-on tools and real-world application",
        "modules": [
            {
                "id": "chart_reading",
                "title": "Chart Reading Basics",
                "subtitle": "Technical Analysis Intro",
                "description": "Learn to read price charts and understand trends",
                "duration": "20 min",
                "prerequisites": ["passive_active"],
                "icon": "📉"
            },
            {
                "id": "market_pulse",
                "title": "Market Pulse",
                "subtitle": "Sentiment & News",
                "description": "Track market sentiment and filter noise",
                "duration": "10 min",
                "prerequisites": ["chart_reading"],
                "icon": "💓"
            },
            {
                "id": "asset_allocation",
                "title": "Advanced Allocation",
                "subtitle": "Optimization & Rebalancing",
                "description": "Fine-tune your portfolio over time",
                "duration": "15 min",
                "prerequisites": ["market_pulse"],
                "icon": "⚙️"
            }
        ]
    }
}

# Quiz questions for misconception detection
QUIZ_QUESTIONS = {
    "goal_timeline": [
        {
            "question": "When should you start investing?",
            "options": [
                "After paying off ALL debt",
                "After building an emergency fund and paying high-interest debt",
                "Only when you have $10,000+ saved",
                "When the market is at a low point"
            ],
            "correct": 1,
            "misconception_if_wrong": {
                0: "Not all debt is bad - low-interest debt (mortgage) can coexist with investing",
                2: "You can start investing with any amount - $50/month is fine",
                3: "Timing the market is nearly impossible - time IN the market matters more"
            }
        },
        {
            "question": "What's the purpose of an emergency fund?",
            "options": [
                "To invest when the market drops",
                "To avoid selling investments during emergencies",
                "To maximize returns",
                "Only for people with unstable income"
            ],
            "correct": 1,
            "misconception_if_wrong": {
                0: "Emergency funds should stay liquid, not be invested",
                2: "Emergency funds prioritize safety over returns",
                3: "Everyone needs an emergency fund - unexpected expenses happen to all"
            }
        }
    ],
    "risk_explained": [
        {
            "question": "If your portfolio drops 20% in a year, you should:",
            "options": [
                "Sell immediately to prevent further losses",
                "Stay the course if your time horizon is long",
                "Move everything to bonds",
                "Stop contributing until it recovers"
            ],
            "correct": 1,
            "misconception_if_wrong": {
                0: "Selling during drops locks in losses - historically markets recover",
                2: "Panic-switching to bonds after a drop means buying high, selling low",
                3: "Stopping contributions during dips means missing the recovery"
            }
        },
        {
            "question": "Diversification means:",
            "options": [
                "Your portfolio will never lose money",
                "Owning many different stocks in the same sector",
                "Spreading investments across uncorrelated assets",
                "Buying whatever performed best last year"
            ],
            "correct": 2,
            "misconception_if_wrong": {
                0: "Diversification reduces risk but doesn't eliminate it - all assets can fall together",
                1: "Same-sector stocks often move together - true diversification crosses asset classes",
                3: "Past performance doesn't predict future returns"
            }
        }
    ],
    "first_portfolio": [
        {
            "question": "Why use index funds instead of picking stocks?",
            "options": [
                "Index funds always beat individual stocks",
                "90% of professional stock pickers underperform indexes long-term",
                "Individual stocks are illegal for beginners",
                "Index funds have no risk"
            ],
            "correct": 1,
            "misconception_if_wrong": {
                0: "Some stocks beat indexes, but consistently picking winners is nearly impossible",
                2: "Anyone can buy individual stocks - but it's usually not optimal",
                3: "Index funds still have market risk - they just eliminate single-stock risk"
            }
        }
    ]
}

# Personalization templates based on user profile
PERSONALIZATION = {
    "short_horizon": {
        "emphasis": "capital_preservation",
        "key_message": "With a shorter timeline, protecting your principal matters more than maximizing growth.",
        "recommended_allocation": {"stocks": 30, "bonds": 50, "cash": 20},
        "warnings": ["Stock volatility can hurt short-term", "You may not have time to recover from crashes"]
    },
    "medium_horizon": {
        "emphasis": "balanced_growth",
        "key_message": "You have time to ride out volatility while still building wealth.",
        "recommended_allocation": {"stocks": 60, "bonds": 30, "cash": 10},
        "warnings": ["Stay invested through downturns", "Don't panic during temporary drops"]
    },
    "long_horizon": {
        "emphasis": "growth_maximization",
        "key_message": "Time is your biggest advantage - staying invested matters more than timing.",
        "recommended_allocation": {"stocks": 80, "bonds": 15, "cash": 5},
        "warnings": ["Cash loses value to inflation over decades", "Don't let short-term drops scare you out"]
    },
    "high_loss_sensitivity": {
        "emphasis": "drawdown_focus",
        "key_message": "We'll focus on worst-case scenarios so you can stay invested when it matters.",
        "adjustment": "Show max drawdown prominently, reduce stock allocation by 10%"
    },
    "low_loss_sensitivity": {
        "emphasis": "return_focus", 
        "key_message": "You can handle volatility - let's optimize for long-term growth.",
        "adjustment": "Show expected returns prominently, standard allocation"
    }
}
