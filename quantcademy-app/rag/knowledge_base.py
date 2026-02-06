"""
QuantCademy RAG Knowledge Base
Structured content from trusted sources for retrieval-augmented generation.
"""

# This is your curated knowledge base - structured for RAG retrieval
# Each document has metadata for filtering and context injection

KNOWLEDGE_BASE = {
    # =========================================================
    # INVESTING BASICS (Source: SEC/Investor.gov + Investopedia)
    # =========================================================
    "what_is_investing": {
        "title": "What is Investing?",
        "source": "SEC Investor.gov",
        "url": "https://www.investor.gov/introduction-investing",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Investing is putting money to work to potentially grow over time. When you invest, 
you purchase assets like stocks, bonds, or funds with the expectation of earning 
a return.

KEY CONCEPTS:
- Principal: The amount you initially invest
- Return: The profit or loss from your investment
- Risk: The possibility of losing some or all of your investment
- Compound growth: Earning returns on your returns over time

WHY INVEST?
1. Beat inflation - Cash loses about 3% purchasing power per year
2. Build wealth - Historically, stocks return ~10% annually
3. Reach financial goals - Retirement, home purchase, education

IMPORTANT: Investing involves risk. You could lose money. Past performance 
doesn't guarantee future results.
        """,
        "key_terms": ["investing", "principal", "return", "risk", "compound interest"],
        "related_modules": ["goal_timeline"]
    },
    
    "compound_interest": {
        "title": "The Power of Compound Interest",
        "source": "SEC Investor.gov",
        "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Compound interest is when you earn interest on both your original investment AND 
on the interest you've already earned. It's often called "interest on interest."

THE MATH:
Future Value = Principal × (1 + rate)^years

EXAMPLE:
$10,000 invested at 7% annual return:
- Year 1: $10,700
- Year 5: $14,026
- Year 10: $19,672
- Year 20: $38,697
- Year 30: $76,123

THE RULE OF 72:
Divide 72 by your annual return to estimate years to double.
- At 7%: 72/7 = ~10 years to double
- At 10%: 72/10 = ~7 years to double

WHY START EARLY:
Someone who invests $5,000/year from age 25-35 (10 years, $50,000 total) 
will have MORE at age 65 than someone who invests $5,000/year from age 35-65 
(30 years, $150,000 total). That's the power of compound growth!
        """,
        "key_terms": ["compound interest", "rule of 72", "time value of money"],
        "related_modules": ["goal_timeline"]
    },

    # =========================================================
    # ASSET CLASSES (Source: Investopedia + Vanguard)
    # =========================================================
    "stocks_explained": {
        "title": "Stocks (Equities) Explained",
        "source": "Investopedia",
        "url": "https://www.investopedia.com/terms/s/stock.asp",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
A stock represents ownership in a company. When you buy a stock, you become a 
shareholder and own a small piece of that company.

HOW STOCKS MAKE MONEY:
1. Capital appreciation - Stock price increases over time
2. Dividends - Some companies pay regular cash distributions

HISTORICAL PERFORMANCE:
- US stocks have returned ~10% annually since 1926
- But individual years vary wildly: -37% (2008) to +33% (2013)
- Stocks are volatile short-term but historically rewarding long-term

TYPES OF STOCKS:
- Large-cap: Big established companies (Apple, Microsoft)
- Mid-cap: Medium-sized growing companies
- Small-cap: Smaller companies with more growth potential and risk
- International: Companies outside the US
- Emerging markets: Companies in developing countries

RISKS:
- Market risk: All stocks can fall during market downturns
- Company risk: Individual companies can fail (Enron, Lehman Brothers)
- Volatility: Prices can swing 20-30% in a single year

FOR BEGINNERS: Consider index funds that hold hundreds of stocks rather 
than trying to pick individual winners.
        """,
        "key_terms": ["stock", "equity", "dividend", "capital appreciation", "market cap"],
        "related_modules": ["first_portfolio", "risk_explained"]
    },

    "bonds_explained": {
        "title": "Bonds (Fixed Income) Explained",
        "source": "Investopedia",
        "url": "https://www.investopedia.com/terms/b/bond.asp",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
A bond is a loan you make to a company or government. In return, they pay you 
interest and return your principal at maturity.

HOW BONDS WORK:
1. You buy a bond for $1,000 (face value)
2. Issuer pays you interest (coupon) regularly, e.g., 4% = $40/year
3. At maturity (e.g., 10 years), you get your $1,000 back

TYPES OF BONDS:
- Treasury bonds: US government (safest, lower yield)
- Municipal bonds: State/local government (often tax-free)
- Corporate bonds: Companies (higher yield, more risk)
- High-yield (junk) bonds: Lower-rated companies (highest yield and risk)

HISTORICAL PERFORMANCE:
- US bonds have returned ~4-5% annually
- Much less volatile than stocks
- Often move opposite to stocks (diversification benefit)

KEY RELATIONSHIP:
When interest rates RISE, bond prices FALL (and vice versa).
This is because new bonds offer higher rates, making old bonds less attractive.

WHY OWN BONDS:
- Stability during stock market crashes
- Regular income from interest payments
- Capital preservation for shorter time horizons
        """,
        "key_terms": ["bond", "fixed income", "coupon", "yield", "maturity", "interest rate risk"],
        "related_modules": ["first_portfolio", "risk_explained"]
    },

    "etf_vs_mutual_fund": {
        "title": "ETFs vs Mutual Funds",
        "source": "Vanguard + Investopedia",
        "url": "https://www.investopedia.com/articles/exchangetradedfunds/08/etf-mutual-fund-difference.asp",
        "category": "investment_vehicles",
        "difficulty": "beginner",
        "content": """
Both ETFs and mutual funds pool money to buy a basket of investments. 
The main differences are in how they trade and their costs.

ETFs (Exchange-Traded Funds):
- Trade throughout the day like stocks
- No minimum investment (buy 1 share)
- Generally lower expense ratios
- More tax-efficient
- Examples: VTI, VOO, VXUS, BND

MUTUAL FUNDS:
- Trade once per day after market close
- Often have minimum investments ($1,000-$3,000)
- May have slightly higher fees
- Easier to set up automatic investments
- Examples: VTSAX, FXAIX, VBTLX

INDEX FUNDS (can be ETF or Mutual Fund):
- Track a market index (S&P 500, Total Stock Market)
- Passive management = very low fees (0.03-0.10%)
- Beat 90% of actively managed funds over 15+ years
- No stock picking, no market timing

EXPENSE RATIOS MATTER:
- 0.03% (index fund): $30/year per $100,000
- 1.00% (active fund): $1,000/year per $100,000
- Over 30 years, that 0.97% difference costs ~$200,000!

RECOMMENDATION FOR BEGINNERS:
Low-cost index ETFs or mutual funds. Don't pay for active management.
        """,
        "key_terms": ["ETF", "mutual fund", "index fund", "expense ratio", "passive investing"],
        "related_modules": ["first_portfolio", "etf_selection"]
    },

    # =========================================================
    # RISK & VOLATILITY (Source: Investopedia + Academic)
    # =========================================================
    "understanding_risk": {
        "title": "Understanding Investment Risk",
        "source": "Investopedia + FINRA",
        "url": "https://www.finra.org/investors/investing/investing-basics",
        "category": "risk",
        "difficulty": "intermediate",
        "content": """
Risk in investing isn't just "losing money." There are different types of risk 
that affect your portfolio in different ways.

TYPES OF INVESTMENT RISK:

1. MARKET RISK (Systematic)
- Affects all investments
- Can't be diversified away
- Example: 2008 financial crisis, COVID crash
- Mitigation: Time horizon, asset allocation

2. COMPANY RISK (Unsystematic)
- Affects individual companies
- CAN be diversified away
- Example: Enron bankruptcy, company fraud
- Mitigation: Own hundreds of stocks via index funds

3. INFLATION RISK
- Purchasing power decreases over time
- Cash loses ~3% value per year
- "Safe" investments may actually lose real value
- Mitigation: Own growth assets (stocks)

4. INTEREST RATE RISK
- Bond prices fall when rates rise
- Longer-term bonds more affected
- Mitigation: Shorter duration bonds, bond ladders

5. SEQUENCE OF RETURNS RISK
- Order of returns matters, especially in retirement
- Bad returns early can devastate a portfolio
- Mitigation: Bond tent, flexible withdrawal strategy

MEASURING RISK:
- Standard Deviation: How much returns vary (volatility)
- Beta: How much an investment moves vs the market
- Maximum Drawdown: Largest peak-to-trough decline
- Sharpe Ratio: Return per unit of risk (higher = better)

KEY INSIGHT:
Risk and return are linked. To earn higher returns, you must accept 
higher risk. There's no free lunch in investing.
        """,
        "key_terms": ["market risk", "systematic risk", "diversification", "volatility", "drawdown"],
        "related_modules": ["risk_explained"]
    },

    "drawdowns_explained": {
        "title": "Drawdowns and Recovery",
        "source": "Portfolio Visualizer + Historical Data",
        "url": "https://www.portfoliovisualizer.com/backtest-portfolio",
        "category": "risk",
        "difficulty": "intermediate",
        "content": """
A drawdown is the decline from a portfolio's peak to its lowest point before 
reaching a new high. It's the "pain" you experience as an investor.

HISTORICAL US STOCK DRAWDOWNS:
- 2008 Financial Crisis: -51%, recovered in 4 years
- Dot-com crash (2000-02): -49%, recovered in 7 years
- COVID crash (2020): -34%, recovered in 5 months
- 2022 bear market: -25%, recovered in ~1 year

WHY DRAWDOWNS MATTER MORE THAN VOLATILITY:
- Volatility is statistical (standard deviation)
- Drawdowns are what you FEEL
- A 50% loss requires a 100% gain to recover!

RECOVERY MATH:
| Loss | Gain Needed to Recover |
|------|------------------------|
| 10%  | 11%                    |
| 20%  | 25%                    |
| 30%  | 43%                    |
| 40%  | 67%                    |
| 50%  | 100%                   |

HOW TO SURVIVE DRAWDOWNS:
1. Know your risk tolerance BEFORE the crash
2. Have enough bonds to avoid forced selling
3. Don't check your portfolio daily
4. Remember: every past crash has recovered
5. Keep contributing (buy low!)

IMPORTANT CONTEXT:
Despite all these crashes, $10,000 invested in US stocks in 1980 
would be worth over $1,000,000 today. Staying invested matters.
        """,
        "key_terms": ["drawdown", "recovery", "bear market", "crash", "volatility"],
        "related_modules": ["risk_explained", "what_could_happen"]
    },

    # =========================================================
    # PORTFOLIO CONSTRUCTION (Source: Bogleheads + Vanguard)
    # =========================================================
    "three_fund_portfolio": {
        "title": "The Three-Fund Portfolio",
        "source": "Bogleheads Wiki",
        "url": "https://www.bogleheads.org/wiki/Three-fund_portfolio",
        "category": "portfolio",
        "difficulty": "beginner",
        "content": """
The three-fund portfolio is a simple, effective strategy recommended by 
Bogleheads and inspired by Vanguard founder John Bogle.

THE THREE FUNDS:
1. US Total Stock Market Index (e.g., VTI, VTSAX)
2. International Total Stock Market Index (e.g., VXUS, VTIAX)
3. US Total Bond Market Index (e.g., BND, VBTLX)

WHY IT WORKS:
- Holds thousands of stocks and bonds
- Extremely low cost (0.03-0.07% expense ratios)
- Automatic diversification
- Beats 90% of professional investors long-term
- Simple to understand and maintain

SAMPLE ALLOCATIONS BY AGE:
- Age 25: 70% US stocks, 20% Intl stocks, 10% bonds
- Age 40: 55% US stocks, 15% Intl stocks, 30% bonds
- Age 60: 35% US stocks, 10% Intl stocks, 55% bonds

THE MATH (Historical):
60/40 stock/bond portfolio has returned ~8.5% annually since 1926,
with much lower volatility than 100% stocks.

REBALANCING:
Once a year, sell what's grown too large and buy what's shrunk.
This forces "sell high, buy low" behavior.

VARIATIONS:
- Two-fund: Just US stocks + Bonds (simpler)
- Four-fund: Add REITs or TIPS
- Target-date fund: Single fund that adjusts automatically

"The simplest way to invest is often the best way to invest."
        """,
        "key_terms": ["three-fund portfolio", "Bogleheads", "asset allocation", "rebalancing"],
        "related_modules": ["first_portfolio"]
    },

    "asset_allocation": {
        "title": "Asset Allocation Principles",
        "source": "Vanguard Research",
        "url": "https://investor.vanguard.com/investor-resources-education/how-to-invest/asset-allocation",
        "category": "portfolio",
        "difficulty": "intermediate",
        "content": """
Asset allocation is how you divide your portfolio among different asset classes 
(stocks, bonds, cash). It's the most important investment decision you'll make.

WHY ALLOCATION MATTERS:
Studies show that asset allocation determines ~90% of portfolio returns variation.
Stock picking and market timing matter much less.

FACTORS TO CONSIDER:
1. Time horizon - Longer = more stocks
2. Risk tolerance - Lower tolerance = more bonds
3. Financial situation - Stable income = can take more risk
4. Goals - Retirement vs house down payment

STOCKS VS BONDS HISTORICALLY:
| Allocation | Avg Return | Worst Year | Best Year |
|------------|------------|------------|-----------|
| 100% stocks| 10.3%      | -43%       | +54%      |
| 80/20      | 9.4%       | -34%       | +45%      |
| 60/40      | 8.5%       | -27%       | +36%      |
| 40/60      | 7.5%       | -18%       | +28%      |
| 20/80      | 6.4%       | -10%       | +21%      |

RULE OF THUMB:
Your bond allocation = roughly your age
(More conservative: age + 10, More aggressive: age - 10)

DON'T FORGET INTERNATIONAL:
- US is only ~60% of global stock market
- International provides diversification
- Suggested: 20-40% of stocks in international

REBALANCING:
Check annually. If allocation drifts 5%+ from target, rebalance.
        """,
        "key_terms": ["asset allocation", "stocks", "bonds", "risk tolerance", "time horizon"],
        "related_modules": ["first_portfolio", "risk_explained"]
    },

    # =========================================================
    # BEHAVIORAL FINANCE (Source: Academic + Investopedia)
    # =========================================================
    "behavioral_mistakes": {
        "title": "Common Investor Mistakes",
        "source": "Behavioral Finance Research",
        "url": "https://www.investopedia.com/articles/basics/11/biggest-stock-market-myths.asp",
        "category": "behavioral",
        "difficulty": "intermediate",
        "content": """
The biggest enemy of investment returns isn't fees or bad picks—it's YOU.
Behavioral biases cause investors to underperform by 1-2% annually.

COMMON MISTAKES:

1. LOSS AVERSION
- Losses feel 2x worse than equivalent gains feel good
- Result: Hold losers too long, sell winners too soon
- Fix: Set rules in advance, automate decisions

2. RECENCY BIAS
- Overweight recent events
- "Stocks crashed, I'll never invest again" or "Tech always goes up!"
- Fix: Look at long-term historical data

3. FOMO (Fear of Missing Out)
- Chase hot stocks/sectors after they've risen
- Buy high, sell low (the opposite of good investing)
- Fix: Stick to your plan, ignore market noise

4. OVERCONFIDENCE
- "I can beat the market"
- Trade too frequently, incur costs
- Fix: Accept you probably can't beat indexes

5. HERD MENTALITY
- Follow the crowd (meme stocks, crypto hype)
- Often means buying at peaks
- Fix: Be skeptical of "everyone's doing it"

6. CHECKING TOO OFTEN
- Daily portfolio checks increase anxiety
- More likely to make emotional decisions
- Fix: Check quarterly at most

DALBAR STUDY:
Average investor earned 2.9% while S&P 500 earned 7.5% (20-year period).
The difference? Behavioral mistakes.

"The investor's chief problem—and even his worst enemy—is likely to be himself."
— Benjamin Graham
        """,
        "key_terms": ["behavioral finance", "loss aversion", "FOMO", "overconfidence"],
        "related_modules": ["risk_explained", "what_could_happen"]
    },

    # =========================================================
    # DOLLAR COST AVERAGING (Source: Investopedia + Vanguard)
    # =========================================================
    "dollar_cost_averaging": {
        "title": "Dollar Cost Averaging (DCA)",
        "source": "Investopedia + Vanguard",
        "url": "https://www.investopedia.com/terms/d/dollarcostaveraging.asp",
        "category": "strategies",
        "difficulty": "beginner",
        "content": """
Dollar cost averaging is investing a fixed amount at regular intervals, 
regardless of market conditions.

HOW IT WORKS:
$500/month invested in an index fund:
- January: Price $50 → Buy 10 shares
- February: Price $40 → Buy 12.5 shares
- March: Price $55 → Buy 9.1 shares
- Average cost: $47.62/share (lower than average price!)

WHY DCA WORKS:
1. Removes emotion from investing
2. No need to time the market (impossible anyway)
3. Automatically buy more when prices are low
4. Builds consistent investing habit
5. Reduces impact of volatility

DCA VS LUMP SUM:
- Lump sum wins ~67% of the time historically
- BUT DCA reduces regret if market drops after investing
- DCA is better than waiting for "the right time"

KEY INSIGHT:
"Time in the market beats timing the market."
Waiting for a "better entry point" often means missing gains.

HOW TO IMPLEMENT:
1. Set up automatic transfers from checking to brokerage
2. Choose your investment (total market index fund)
3. Set the amount and frequency (monthly, bi-weekly)
4. Don't look at it constantly
5. Increase amount when income rises

EXAMPLE:
$500/month for 30 years at 7% = $566,000
(You contributed only $180,000—the rest is growth!)
        """,
        "key_terms": ["dollar cost averaging", "DCA", "lump sum", "automation"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },

    # =========================================================
    # MARKET MECHANICS (Source: SEC + Investopedia)
    # =========================================================
    "how_markets_work": {
        "title": "How Stock Markets Work",
        "source": "SEC + Investopedia",
        "url": "https://www.investopedia.com/articles/investing/082614/how-stock-market-works.asp",
        "category": "markets",
        "difficulty": "intermediate",
        "content": """
Stock markets are where buyers and sellers come together to trade shares 
of publicly traded companies.

KEY CONCEPTS:

EXCHANGES:
- NYSE (New York Stock Exchange): Largest, most prestigious
- NASDAQ: Tech-heavy, electronic trading
- Other global: London, Tokyo, Hong Kong, etc.

PRICE DISCOVERY:
- Prices set by supply and demand
- If more buyers than sellers → price rises
- If more sellers than buyers → price falls
- Happens in milliseconds through electronic matching

MARKET INDICES:
- S&P 500: 500 largest US companies (~80% of market)
- Dow Jones: 30 large companies (less representative)
- Total Stock Market: ~4,000 US companies
- MSCI World: Global developed markets

MARKET HOURS:
- US markets: 9:30 AM - 4:00 PM Eastern
- Pre-market: 4:00 AM - 9:30 AM
- After-hours: 4:00 PM - 8:00 PM

ORDER TYPES:
- Market order: Buy/sell immediately at current price
- Limit order: Buy/sell only at specified price or better
- Stop-loss: Sell if price drops to specified level

IMPORTANT FOR BEGINNERS:
You don't need to understand market mechanics deeply.
Just buy index funds and hold long-term. The market works for you.
        """,
        "key_terms": ["stock market", "exchange", "NYSE", "NASDAQ", "S&P 500", "index"],
        "related_modules": ["market_basics"]
    },

    # =========================================================
    # RETIREMENT ACCOUNTS (Source: IRS + Investopedia)
    # =========================================================
    "retirement_accounts": {
        "title": "Retirement Accounts (401k, IRA)",
        "source": "IRS + Investopedia",
        "url": "https://www.irs.gov/retirement-plans",
        "category": "accounts",
        "difficulty": "beginner",
        "content": """
Tax-advantaged retirement accounts are the most powerful wealth-building tools.
Use them before taxable investing.

401(k) - EMPLOYER-SPONSORED:
- Contribution limit 2024: $23,000 ($30,500 if 50+)
- Often includes employer match (FREE MONEY!)
- Traditional: Pre-tax contributions, taxed at withdrawal
- Roth 401(k): After-tax contributions, tax-free withdrawal
- Always get the FULL employer match

IRA - INDIVIDUAL RETIREMENT ACCOUNT:
- Contribution limit 2024: $7,000 ($8,000 if 50+)
- Anyone with earned income can open one
- Traditional: May be tax-deductible, taxed at withdrawal
- Roth IRA: Not deductible, but grows and withdraws TAX-FREE

ROTH VS TRADITIONAL:
- Roth: Pay taxes now, never again (good if you expect higher taxes later)
- Traditional: Deduct now, pay taxes later (good if in high bracket now)
- When in doubt, Roth is often the better choice for younger investors

PRIORITY ORDER:
1. 401(k) up to employer match (100% instant return!)
2. Max out Roth IRA
3. Max out 401(k)
4. Taxable brokerage account

EARLY WITHDRAWAL:
- Generally 10% penalty before age 59½
- Roth contributions (not gains) can be withdrawn penalty-free
- Some exceptions: First home, medical, education

EXAMPLE:
$500/month to 401(k) with 3% match from age 25-65:
Your contributions: $240,000
Employer match: ~$100,000
Growth at 7%: ~$1.3 million total!
        """,
        "key_terms": ["401k", "IRA", "Roth", "traditional", "retirement", "employer match"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },
}

# =========================================================
# RETRIEVAL FUNCTIONS
# =========================================================

def search_knowledge_base(query: str, category: str = None, difficulty: str = None) -> list:
    """
    Search the knowledge base for relevant documents.
    Returns list of (doc_id, relevance_score, document) tuples.
    """
    query_lower = query.lower()
    results = []
    
    for doc_id, doc in KNOWLEDGE_BASE.items():
        score = 0
        
        # Check category filter
        if category and doc.get('category') != category:
            continue
        
        # Check difficulty filter
        if difficulty and doc.get('difficulty') != difficulty:
            continue
        
        # Title match (highest weight)
        if query_lower in doc['title'].lower():
            score += 10
        
        # Key terms match
        for term in doc.get('key_terms', []):
            if term.lower() in query_lower or query_lower in term.lower():
                score += 5
        
        # Content match
        content_lower = doc['content'].lower()
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in content_lower:
                score += 1
        
        if score > 0:
            results.append((doc_id, score, doc))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Return top 5


def get_document_by_id(doc_id: str) -> dict:
    """Get a specific document by ID."""
    return KNOWLEDGE_BASE.get(doc_id)


def get_documents_by_category(category: str) -> list:
    """Get all documents in a category."""
    return [
        (doc_id, doc) for doc_id, doc in KNOWLEDGE_BASE.items()
        if doc.get('category') == category
    ]


def get_documents_for_module(module_id: str) -> list:
    """Get documents related to a specific module."""
    return [
        (doc_id, doc) for doc_id, doc in KNOWLEDGE_BASE.items()
        if module_id in doc.get('related_modules', [])
    ]


def format_context_for_llm(documents: list, max_length: int = 4000) -> str:
    """Format retrieved documents as context for LLM."""
    context_parts = []
    total_length = 0
    
    for doc_id, score, doc in documents:
        doc_text = f"""
---
SOURCE: {doc['title']} ({doc['source']})
{doc['content'].strip()}
---
"""
        if total_length + len(doc_text) > max_length:
            break
        context_parts.append(doc_text)
        total_length += len(doc_text)
    
    return "\n".join(context_parts)
