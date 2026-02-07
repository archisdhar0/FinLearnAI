"""
QuantCademy RAG Knowledge Base
Comprehensive financial education content from trusted sources.
Sources: SEC, Investopedia, Vanguard, Fidelity, Bogleheads, FINRA, Federal Reserve
"""

from typing import List, Tuple, Dict, Optional

# =========================================================
# COMPREHENSIVE KNOWLEDGE BASE
# Each document has structured metadata for filtering and retrieval
# =========================================================

KNOWLEDGE_BASE = {
    # =========================================================
    # INVESTING BASICS
    # =========================================================
    "what_is_investing": {
        "title": "What is Investing?",
        "source": "SEC Investor.gov",
        "url": "https://www.investor.gov/introduction-investing",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Investing is putting money to work to potentially grow over time. When you invest, you purchase assets like stocks, bonds, or funds with the expectation of earning a return.

KEY CONCEPTS:
- Principal: The amount you initially invest
- Return: The profit or loss from your investment
- Risk: The possibility of losing some or all of your investment
- Compound growth: Earning returns on your returns over time

WHY INVEST?
1. Beat inflation - Cash loses about 3% purchasing power per year
2. Build wealth - Historically, stocks return ~10% annually
3. Reach financial goals - Retirement, home purchase, education

THE MATH OF INVESTING:
If you invest $500/month for 30 years at 7% average return:
- Total contributed: $180,000
- Ending balance: ~$567,000
- That's $387,000 in growth!

IMPORTANT: Investing involves risk. You could lose money. Past performance doesn't guarantee future results. But historically, staying invested in diversified portfolios has rewarded patient investors.
        """,
        "key_terms": ["investing", "principal", "return", "risk", "compound interest"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },
    
    "compound_interest": {
        "title": "The Power of Compound Interest",
        "source": "SEC Investor.gov",
        "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Compound interest is when you earn interest on both your original investment AND on the interest you've already earned. It's often called "interest on interest" and is the most powerful force in investing.

THE MATH:
Future Value = Principal × (1 + rate)^years

EXAMPLE:
$10,000 invested at 7% annual return:
- Year 1: $10,700 (you earned $700)
- Year 5: $14,026 (total growth: $4,026)
- Year 10: $19,672 (total growth: $9,672)
- Year 20: $38,697 (total growth: $28,697)
- Year 30: $76,123 (total growth: $66,123)

Notice how the growth accelerates over time? That's compounding at work!

THE RULE OF 72:
Divide 72 by your annual return to estimate years to double your money.
- At 7%: 72/7 = ~10 years to double
- At 10%: 72/10 = ~7 years to double
- At 12%: 72/12 = 6 years to double

WHY START EARLY MATTERS:
Someone who invests $5,000/year from age 25-35 (10 years, $50,000 total) will have MORE at age 65 than someone who invests $5,000/year from age 35-65 (30 years, $150,000 total). 

The early investor contributed $100,000 LESS but ends up with MORE money because they gave compound interest more time to work.

KEY TAKEAWAY: Time is your most valuable asset. Start investing as early as possible, even with small amounts.
        """,
        "key_terms": ["compound interest", "rule of 72", "time value of money", "exponential growth"],
        "related_modules": ["goal_timeline"]
    },

    "inflation_basics": {
        "title": "Understanding Inflation and Your Money",
        "source": "Federal Reserve, Bureau of Labor Statistics",
        "url": "https://www.federalreserve.gov/faqs/economy_14419.htm",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Inflation is the rate at which prices rise over time, reducing your money's purchasing power. Understanding inflation is crucial because it's the main reason you MUST invest - cash savings lose value every year.

WHAT IS INFLATION?
When inflation is 3%, something that costs $100 today will cost $103 next year. Your $100 bill still says $100, but it buys less stuff.

HISTORICAL INFLATION:
- US average inflation: ~3% per year
- 2021-2022 inflation spike: 7-9% (unusual)
- Federal Reserve target: 2%

THE SILENT WEALTH DESTROYER:
At 3% inflation:
- $100 today = $74 purchasing power in 10 years
- $100 today = $55 purchasing power in 20 years
- $100 today = $41 purchasing power in 30 years

This is why keeping money in a savings account (earning 0.5%) actually LOSES value over time in real terms.

HOW TO BEAT INFLATION:
1. Stocks historically return ~10%, beating inflation by ~7%
2. TIPS (Treasury Inflation-Protected Securities) adjust with CPI
3. I-Bonds (savings bonds that track inflation)
4. Real estate tends to appreciate with inflation

THE FED'S ROLE:
The Federal Reserve manages inflation through interest rates:
- High inflation → Fed raises rates → Slows economy/prices
- Low inflation/recession → Fed cuts rates → Stimulates growth

WHY 2% TARGET?
A little inflation is healthy - it encourages spending (prices will rise later) and makes debt easier to repay. Deflation (falling prices) is actually worse, as Japan learned.
        """,
        "key_terms": ["inflation", "purchasing power", "CPI", "Federal Reserve", "real returns"],
        "related_modules": ["macro_economics", "risk_explained"]
    },

    # =========================================================
    # STOCKS AND EQUITIES
    # =========================================================
    "stocks_explained": {
        "title": "Stocks (Equities) Explained",
        "source": "Investopedia, SEC",
        "url": "https://www.investopedia.com/terms/s/stock.asp",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
A stock represents ownership in a company. When you buy a stock, you become a shareholder and own a small piece of that company.

HOW STOCKS MAKE MONEY:
1. Capital appreciation - Stock price increases over time. Buy at $100, sell at $150 = $50 profit.
2. Dividends - Some companies pay regular cash distributions from their profits. Example: $1/share quarterly.

HISTORICAL PERFORMANCE (US Stocks):
- Average annual return since 1926: ~10%
- But individual years vary wildly:
  - Best year (1933): +54%
  - Worst year (2008): -37%
  - 2020 (COVID crash): -34% then +68%
  
TYPES OF STOCKS:
- Large-cap: Big established companies (Apple, Microsoft) - More stable
- Mid-cap: Medium-sized growing companies - Balance of growth and stability
- Small-cap: Smaller companies - Higher growth potential, higher risk
- International developed: Europe, Japan, Australia
- Emerging markets: China, India, Brazil - Higher growth, higher volatility

MARKET CAPITALIZATION:
Market Cap = Stock Price × Shares Outstanding
- Large-cap: >$10 billion
- Mid-cap: $2-10 billion
- Small-cap: <$2 billion

RISKS OF INDIVIDUAL STOCKS:
- Market risk: All stocks can fall during downturns
- Company risk: Individual companies can fail (Enron, Lehman Brothers, Blockbuster)
- Volatility: Prices can swing 30-50% in a single year
- Concentration: One bad stock can devastate your portfolio

FOR BEGINNERS: Consider index funds that hold hundreds or thousands of stocks rather than trying to pick individual winners. This diversifies away company-specific risk while capturing overall market growth.
        """,
        "key_terms": ["stock", "equity", "dividend", "capital appreciation", "market cap", "large cap", "small cap"],
        "related_modules": ["first_portfolio", "risk_explained", "market_mechanics"]
    },

    "dividends_explained": {
        "title": "Understanding Dividends",
        "source": "Investopedia, Fidelity",
        "url": "https://www.investopedia.com/terms/d/dividend.asp",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
A dividend is a portion of a company's profits distributed to shareholders. Not all companies pay dividends - many growth companies reinvest profits instead.

HOW DIVIDENDS WORK:
1. Company earns profit
2. Board of Directors declares dividend (e.g., $0.50/share)
3. On "record date," shareholders of record receive dividend
4. Paid quarterly (usually) to your brokerage account

KEY DATES:
- Declaration Date: Company announces dividend
- Ex-Dividend Date: Must own stock BEFORE this date to receive dividend
- Record Date: Company checks who owns shares
- Payment Date: Dividend hits your account

DIVIDEND YIELD:
Dividend Yield = Annual Dividend / Stock Price
Example: $2 annual dividend / $100 stock = 2% yield

DIVIDEND ARISTOCRATS:
Companies that have increased dividends for 25+ consecutive years:
- Johnson & Johnson, Coca-Cola, Procter & Gamble, 3M
- Shows financial stability and shareholder commitment

DIVIDEND REINVESTMENT (DRIP):
Automatically use dividends to buy more shares. This compounds your returns without any action required.

QUALIFIED VS ORDINARY DIVIDENDS:
- Qualified: Taxed at lower capital gains rates (0-20%)
- Ordinary: Taxed as regular income (up to 37%)
- Hold stocks 60+ days for qualified status

ARE HIGH DIVIDENDS BETTER?
Not always! Very high yields (8%+) may signal:
- Stock price crashed (yield calculated on old dividend)
- Dividend may be cut
- Company paying out more than it can afford

A sustainable 2-4% yield that grows annually is often better than an unsustainable 8% yield.
        """,
        "key_terms": ["dividend", "dividend yield", "DRIP", "ex-dividend date", "dividend aristocrats"],
        "related_modules": ["first_portfolio", "fundamental_analysis"]
    },

    # =========================================================
    # BONDS AND FIXED INCOME
    # =========================================================
    "bonds_explained": {
        "title": "Bonds (Fixed Income) Explained",
        "source": "Investopedia, Vanguard",
        "url": "https://www.investopedia.com/terms/b/bond.asp",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
A bond is a loan you make to a company or government. In return, they pay you interest (coupon) and return your principal at maturity. Bonds are generally less risky than stocks but offer lower returns.

HOW BONDS WORK:
1. You buy a bond for $1,000 (face/par value)
2. Issuer pays you interest (coupon) regularly, e.g., 4% = $40/year
3. At maturity (e.g., 10 years), you get your $1,000 back

TYPES OF BONDS:
- Treasury bonds (T-bonds): US government - Safest, lower yield (4-5%)
- Treasury bills (T-bills): Short-term government (weeks to 1 year)
- Municipal bonds: State/local government - Often tax-free
- Corporate bonds: Companies - Higher yield, more risk
- High-yield (junk) bonds: Lower-rated companies - Highest yield and risk

HISTORICAL PERFORMANCE:
- US bonds average return: ~4-5% annually
- Much less volatile than stocks
- Often move opposite to stocks (diversification benefit)

BOND RATINGS:
Credit rating agencies (Moody's, S&P) rate bond safety:
- AAA to BBB: Investment grade (safe)
- BB and below: High yield/junk (risky)

THE KEY RELATIONSHIP: PRICE VS YIELD
When interest rates RISE, bond prices FALL (and vice versa).

Why? If you own a bond paying 3% and new bonds pay 5%, no one wants your 3% bond unless you sell it at a discount.

This matters if you sell before maturity. If you hold to maturity, you get your full principal back regardless of price changes.

WHY OWN BONDS?
1. Stability during stock market crashes
2. Regular income from interest payments
3. Capital preservation for shorter time horizons
4. Diversification - different behavior than stocks

BOND DURATION:
Duration measures sensitivity to interest rate changes.
- Longer duration = more price volatility when rates change
- Short-term bonds (1-3 years): Lower risk, lower yield
- Long-term bonds (10-30 years): Higher risk, higher yield
        """,
        "key_terms": ["bond", "fixed income", "coupon", "yield", "maturity", "interest rate risk", "duration"],
        "related_modules": ["first_portfolio", "risk_explained", "asset_allocation"]
    },

    "treasury_securities": {
        "title": "Treasury Securities: The Safest Investments",
        "source": "TreasuryDirect.gov, Investopedia",
        "url": "https://www.treasurydirect.gov/",
        "category": "asset_classes",
        "difficulty": "beginner",
        "content": """
Treasury securities are debt instruments issued by the US government. They're considered the safest investments in the world because they're backed by the "full faith and credit" of the United States.

TYPES OF TREASURIES:
1. Treasury Bills (T-Bills): 4 weeks to 1 year maturity
   - Sold at discount, mature at face value
   - No coupon payments
   - Example: Buy for $980, get $1,000 at maturity

2. Treasury Notes (T-Notes): 2-10 year maturity
   - Pay interest every 6 months
   - Return face value at maturity

3. Treasury Bonds (T-Bonds): 20-30 year maturity
   - Pay interest every 6 months
   - Longest-term treasury

4. TIPS (Treasury Inflation-Protected Securities):
   - Principal adjusts with inflation (CPI)
   - Interest payments increase as principal grows
   - Protects against inflation risk

5. I-Bonds (Series I Savings Bonds):
   - Inflation-indexed savings bonds
   - Rate = fixed rate + inflation rate
   - Purchase limit: $10,000/year electronic + $5,000 paper
   - Must hold at least 1 year
   - Great for emergency funds or short-term savings

HOW TO BUY:
- TreasuryDirect.gov (directly from government)
- Brokerage account (treasury ETFs like SHY, IEF, TLT)
- Through your broker (individual bonds)

TAX TREATMENT:
- Federal tax: Yes
- State/local tax: NO (exempt!)
- This makes treasuries especially valuable in high-tax states

CURRENT YIELDS (check TreasuryDirect for current rates):
Treasury yields reflect the overall interest rate environment set by the Federal Reserve.
        """,
        "key_terms": ["treasury", "T-bills", "T-notes", "T-bonds", "TIPS", "I-bonds", "TreasuryDirect"],
        "related_modules": ["asset_allocation", "risk_explained"]
    },

    # =========================================================
    # ETFs AND MUTUAL FUNDS
    # =========================================================
    "etf_explained": {
        "title": "ETFs (Exchange-Traded Funds) Explained",
        "source": "Vanguard, Investopedia",
        "url": "https://investor.vanguard.com/etf/",
        "category": "investment_vehicles",
        "difficulty": "beginner",
        "content": """
An ETF (Exchange-Traded Fund) is a basket that holds many investments at once. Instead of buying individual stocks, you buy one share of an ETF and instantly own pieces of hundreds or thousands of companies.

THINK OF IT LIKE:
Instead of buying 500 individual stocks separately, you buy ONE share of an S&P 500 ETF and instantly own a tiny piece of all 500 companies. It's like buying a pre-made pizza instead of buying all the ingredients separately.

HOW ETFs WORK:
1. ETF company creates a fund tracking an index (e.g., S&P 500)
2. You buy shares of the ETF on the stock exchange
3. Your share value moves with the underlying holdings
4. Most ETFs pay dividends (passed through from holdings)

KEY ETF ADVANTAGES:
- Instant diversification (own 500+ companies in one purchase)
- Very low fees (as low as 0.03% annually)
- Trade like stocks (buy/sell anytime market is open)
- No minimum investment (buy 1 share or even fractional shares)
- Tax efficient (generally fewer capital gains distributions)
- Transparent (holdings disclosed daily)

POPULAR ETFs FOR BEGINNERS:
US Stocks:
- VTI (Vanguard Total Stock Market): ~4,000 US stocks, 0.03% fee
- VOO (Vanguard S&P 500): 500 largest US companies, 0.03% fee
- SPY (SPDR S&P 500): Oldest S&P 500 ETF, very liquid

International:
- VXUS (Vanguard Total International): All non-US stocks, 0.08% fee
- VEA (Vanguard Developed Markets): Europe, Japan, Australia
- VWO (Vanguard Emerging Markets): China, India, Brazil

Bonds:
- BND (Vanguard Total Bond): US investment-grade bonds, 0.03% fee
- BNDX (Vanguard International Bond): Non-US bonds

ETF VS INDIVIDUAL STOCKS:
| Individual Stocks | ETFs |
|-------------------|------|
| Pick winners yourself | Own the whole market |
| Can win big or lose big | Average market return |
| Requires research | Set and forget |
| Higher risk | Diversified risk |

90% of professional stock pickers fail to beat index ETFs over 15+ years. Save yourself the effort and just buy the index.
        """,
        "key_terms": ["ETF", "exchange-traded fund", "VTI", "VOO", "index fund", "diversification"],
        "related_modules": ["first_portfolio", "etf_selection"]
    },

    "mutual_funds_explained": {
        "title": "Mutual Funds Explained",
        "source": "Fidelity, Vanguard, SEC",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/mutual-funds-and-exchange-traded-1",
        "category": "investment_vehicles",
        "difficulty": "beginner",
        "content": """
A mutual fund pools money from many investors to buy a portfolio of stocks, bonds, or other assets. Unlike ETFs, mutual funds trade once per day after the market closes.

HOW MUTUAL FUNDS WORK:
1. Many investors pool their money together
2. Professional manager (or index) selects investments
3. You own shares of the fund
4. Value is calculated daily (NAV - Net Asset Value)
5. Buy/sell at end-of-day NAV price

TYPES OF MUTUAL FUNDS:
1. Index Funds: Track a market index passively
   - Lowest fees (0.03-0.20%)
   - No stock picking
   - Examples: VTSAX, FXAIX, SWPPX

2. Actively Managed: Manager tries to beat the market
   - Higher fees (0.50-1.50%)
   - 90% underperform indexes over 15 years
   - May outperform in short periods

3. Target-Date Funds: Automatically adjust allocation by retirement year
   - "Set and forget" option
   - Gets more conservative as you age
   - Example: Vanguard Target Retirement 2050

MUTUAL FUND VS ETF:
| Feature | Mutual Fund | ETF |
|---------|-------------|-----|
| Trading | Once daily | Anytime |
| Minimum | Often $1,000-3,000 | 1 share (~$50-400) |
| Auto-invest | Easy setup | Requires cash |
| Fees | Slightly higher | Slightly lower |
| Tax efficiency | Less efficient | More efficient |

FEES TO WATCH:
- Expense Ratio: Annual fee (aim for <0.20%)
- Load: Sales charge - AVOID funds with loads
- 12b-1 Fee: Marketing fee - adds to expense ratio

THE EXPENSE RATIO IMPACT:
$10,000 invested for 30 years at 7% return:
- 0.03% fee (VTSAX): $74,015
- 1.00% fee (average active fund): $57,435
- Difference: $16,580 lost to fees!

POPULAR LOW-COST INDEX FUNDS:
- VTSAX (Vanguard Total Stock Market): $3,000 min, 0.04%
- FXAIX (Fidelity 500 Index): No minimum, 0.015%
- SWTSX (Schwab Total Stock Market): No minimum, 0.03%
        """,
        "key_terms": ["mutual fund", "NAV", "expense ratio", "index fund", "actively managed", "target date fund"],
        "related_modules": ["first_portfolio", "etf_selection"]
    },

    "index_vs_active": {
        "title": "Index Funds vs Active Management: The Evidence",
        "source": "SPIVA Research, Vanguard, Academic Studies",
        "url": "https://www.spglobal.com/spdji/en/research-insights/spiva/",
        "category": "investment_vehicles",
        "difficulty": "intermediate",
        "content": """
Should you try to beat the market with active management, or just match it with index funds? The evidence overwhelmingly favors indexing for most investors.

THE SPIVA SCORECARD:
Standard & Poor's tracks active managers vs their benchmarks:

US Large-Cap funds underperforming S&P 500:
- 1 year: ~60% underperform
- 5 years: ~75% underperform
- 15 years: ~90% underperform
- 20 years: ~94% underperform

Even worse for small-cap and international categories.

WHY DO MOST MANAGERS FAIL?
1. Fees: 1% annual fee is hard to overcome
2. Transaction costs: Frequent trading costs money
3. Market efficiency: Information is priced in quickly
4. Reversion to mean: Hot managers usually cool off
5. Cash drag: Must hold some cash for redemptions

THE MATH:
If the market returns 10% and you pay 1% in fees:
- After 1 year: You keep 9% (10% - 1%)
- After 30 years: You've lost 30% of potential wealth to fees

BUT WHAT ABOUT STAR MANAGERS?
- Past performance doesn't predict future results
- Studies show top quartile managers are rarely top quartile the next period
- Warren Buffett bet $1M that an index would beat hedge funds over 10 years - he won

WHEN ACTIVE MIGHT MAKE SENSE:
- Tax-loss harvesting specialists
- Certain niche markets (small/micro-cap, emerging markets)
- Municipal bonds (local knowledge helps)
- You have a long track record of beating the market (be honest!)

THE BOTTOM LINE:
John Bogle (Vanguard founder): "Don't look for the needle in the haystack. Just buy the haystack."

Warren Buffett's advice for his heirs: Put 90% in a low-cost S&P 500 index fund.

For most investors, a simple portfolio of low-cost index funds will outperform 90%+ of alternatives over time.
        """,
        "key_terms": ["index fund", "active management", "SPIVA", "expense ratio", "market efficiency"],
        "related_modules": ["first_portfolio", "etf_selection", "passive_active"]
    },

    # =========================================================
    # PORTFOLIO CONSTRUCTION
    # =========================================================
    "three_fund_portfolio": {
        "title": "The Three-Fund Portfolio",
        "source": "Bogleheads Wiki, Vanguard",
        "url": "https://www.bogleheads.org/wiki/Three-fund_portfolio",
        "category": "portfolio",
        "difficulty": "beginner",
        "content": """
The three-fund portfolio is a simple, effective strategy recommended by Bogleheads (followers of Vanguard founder John Bogle). It's all you need for a diversified, low-cost portfolio.

THE THREE FUNDS:
1. US Total Stock Market Index
   - VTI (ETF) or VTSAX (mutual fund)
   - Covers ~4,000 US companies of all sizes
   - Expense ratio: 0.03-0.04%

2. International Total Stock Market Index
   - VXUS (ETF) or VTIAX (mutual fund)
   - Covers all non-US developed and emerging markets
   - Expense ratio: 0.07-0.11%

3. US Total Bond Market Index
   - BND (ETF) or VBTLX (mutual fund)
   - Investment-grade US bonds
   - Expense ratio: 0.03-0.05%

WHY IT WORKS:
- Holds thousands of stocks and bonds (instant diversification)
- Extremely low cost (weighted average ~0.05%)
- Captures global market returns
- Beats 90% of professional investors long-term
- Simple to understand and maintain
- No stock picking, no market timing

SAMPLE ALLOCATIONS BY AGE/RISK:
Aggressive (Age 25, high risk tolerance):
- 60% US stocks (VTI)
- 30% International stocks (VXUS)
- 10% Bonds (BND)

Moderate (Age 40, medium risk tolerance):
- 45% US stocks
- 15% International stocks
- 40% Bonds

Conservative (Age 60, low risk tolerance):
- 30% US stocks
- 10% International stocks
- 60% Bonds

RULE OF THUMB FOR BONDS:
Your bond allocation ≈ your age (or your age minus 10 for more aggressive)

REBALANCING:
Once a year, check if your allocation has drifted more than 5% from target.
If so, sell what's grown too large and buy what's shrunk.
This forces "sell high, buy low" behavior.

HISTORICAL PERFORMANCE:
A 60/40 stock/bond portfolio has returned ~8.5% annually since 1926, with much lower volatility than 100% stocks.

"The simplest way to invest is often the best way to invest."
        """,
        "key_terms": ["three-fund portfolio", "Bogleheads", "asset allocation", "VTI", "VXUS", "BND", "rebalancing"],
        "related_modules": ["first_portfolio", "asset_allocation"]
    },

    "asset_allocation": {
        "title": "Asset Allocation: The Most Important Decision",
        "source": "Vanguard Research, Fidelity",
        "url": "https://investor.vanguard.com/investor-resources-education/how-to-invest/asset-allocation",
        "category": "portfolio",
        "difficulty": "intermediate",
        "content": """
Asset allocation is how you divide your portfolio among different asset classes (stocks, bonds, cash). Studies show it determines ~90% of your portfolio's return variability - far more than stock picking or market timing.

WHY ALLOCATION MATTERS MORE THAN PICKING:
A famous study (Brinson, Hood, Beebower) found that asset allocation explained 91.5% of the variation in returns among pension funds. What you own matters much less than how much you own of each type.

FACTORS TO CONSIDER:
1. Time Horizon
   - Longer horizon → More stocks (time to recover from crashes)
   - Shorter horizon → More bonds (preserve capital)

2. Risk Tolerance
   - Can you sleep during a 40% crash?
   - Your emotional capacity for loss

3. Risk Capacity
   - Financial ability to take risk
   - Stable income, emergency fund, job security

4. Goals
   - Retirement in 30 years vs. house down payment in 3 years

HISTORICAL RETURNS BY ALLOCATION (1926-2023):
| Stocks/Bonds | Avg Return | Worst Year | Best Year | Max Drawdown |
|--------------|------------|------------|-----------|--------------|
| 100/0        | 10.3%      | -43%       | +54%      | -51%         |
| 80/20        | 9.4%       | -34%       | +45%      | -40%         |
| 60/40        | 8.5%       | -27%       | +36%      | -32%         |
| 40/60        | 7.5%       | -18%       | +28%      | -22%         |
| 20/80        | 6.4%       | -10%       | +21%      | -14%         |

Notice the tradeoff: Higher returns come with bigger potential losses.

AGE-BASED RULES OF THUMB:
- Aggressive: 120 minus your age in stocks
- Moderate: 110 minus your age in stocks
- Conservative: 100 minus your age in stocks

Example (age 30):
- Aggressive: 90% stocks, 10% bonds
- Moderate: 80% stocks, 20% bonds
- Conservative: 70% stocks, 30% bonds

DON'T FORGET INTERNATIONAL:
- US is only ~60% of global stock market
- International provides diversification
- Suggested: 20-40% of stocks in international
- Vanguard recommends: ~40% of stocks in international

REBALANCING:
Check annually. If any allocation drifts 5%+ from target, rebalance by selling high and buying low. This is one of the few "free lunches" in investing - it maintains your risk level while systematically buying low and selling high.
        """,
        "key_terms": ["asset allocation", "stocks", "bonds", "risk tolerance", "time horizon", "rebalancing"],
        "related_modules": ["first_portfolio", "risk_explained"]
    },

    # =========================================================
    # RISK AND VOLATILITY
    # =========================================================
    "understanding_risk": {
        "title": "Understanding Investment Risk",
        "source": "Investopedia, FINRA, Vanguard",
        "url": "https://www.finra.org/investors/investing/investing-basics",
        "category": "risk",
        "difficulty": "intermediate",
        "content": """
Risk in investing isn't just "losing money." There are different types of risk that affect your portfolio in different ways. Understanding them helps you make better decisions.

TYPES OF INVESTMENT RISK:

1. MARKET RISK (Systematic Risk)
- Affects all investments
- Cannot be diversified away
- Examples: 2008 financial crisis, COVID crash, 2022 bear market
- Mitigation: Time horizon, asset allocation (add bonds)

2. COMPANY RISK (Unsystematic Risk)
- Affects individual companies
- CAN be diversified away
- Examples: Enron bankruptcy, company fraud, bad products
- Mitigation: Own many stocks via index funds (problem solved!)

3. INFLATION RISK
- Purchasing power decreases over time
- Cash loses ~3% value per year to inflation
- "Safe" investments like CDs may lose real value
- Mitigation: Own growth assets (stocks, TIPS)

4. INTEREST RATE RISK
- Bond prices fall when interest rates rise
- Longer-term bonds more affected
- 2022: Bonds had worst year in 40+ years due to rate hikes
- Mitigation: Shorter-duration bonds, bond ladders

5. SEQUENCE OF RETURNS RISK
- The order of returns matters, especially near retirement
- Bad returns early in retirement can devastate portfolio
- Mitigation: Bond tent, flexible withdrawal strategy

6. CONCENTRATION RISK
- Too much in one stock, sector, or country
- What if your company stock crashes?
- Mitigation: Diversification across asset classes and geography

MEASURING RISK:
- Standard Deviation: How much returns vary (volatility)
- Beta: Movement relative to the market (1.0 = same as market)
- Maximum Drawdown: Largest peak-to-trough decline
- Sharpe Ratio: Return per unit of risk (higher = better)

THE RISK-RETURN TRADEOFF:
You cannot earn higher returns without accepting higher risk. There's no free lunch.

However, you CAN reduce risk without reducing return by:
- Diversifying across uncorrelated assets
- Maintaining appropriate time horizon
- Keeping costs low
        """,
        "key_terms": ["risk", "systematic risk", "unsystematic risk", "diversification", "volatility", "drawdown", "Sharpe ratio"],
        "related_modules": ["risk_explained", "asset_allocation"]
    },

    "drawdowns_recovery": {
        "title": "Drawdowns and Market Recovery",
        "source": "Portfolio Visualizer, Historical Data, Vanguard",
        "url": "https://www.portfoliovisualizer.com/",
        "category": "risk",
        "difficulty": "intermediate",
        "content": """
A drawdown is the decline from a portfolio's peak to its lowest point before reaching a new high. It's the "pain" you experience as an investor. Understanding drawdowns helps you stay invested during tough times.

HISTORICAL US STOCK DRAWDOWNS:
| Event | Decline | Duration to Bottom | Recovery Time |
|-------|---------|-------------------|---------------|
| Great Depression (1929-32) | -86% | 3 years | 25 years |
| 1973-74 Bear Market | -48% | 2 years | 7 years |
| Black Monday (1987) | -34% | 2 months | 2 years |
| Dot-com Crash (2000-02) | -49% | 2.5 years | 7 years |
| Financial Crisis (2008-09) | -51% | 1.5 years | 4 years |
| COVID Crash (2020) | -34% | 1 month | 5 months |
| 2022 Bear Market | -25% | 10 months | ~1 year |

WHY DRAWDOWNS MATTER MORE THAN VOLATILITY:
- Volatility is statistical (standard deviation)
- Drawdowns are what you FEEL
- A 50% loss requires a 100% gain just to break even!

THE MATH OF LOSSES:
| Loss | Gain Needed to Recover |
|------|------------------------|
| 10%  | 11%                    |
| 20%  | 25%                    |
| 30%  | 43%                    |
| 40%  | 67%                    |
| 50%  | 100%                   |

HOW TO SURVIVE DRAWDOWNS:
1. Know your risk tolerance BEFORE the crash (too late during)
2. Have enough bonds to avoid forced selling
3. Don't check your portfolio daily (quarterly is enough)
4. Remember: Every past crash has recovered
5. Keep contributing during crashes (buy low!)
6. Have an emergency fund so you don't sell investments

THE SILVER LINING:
Despite all these crashes, $10,000 invested in US stocks in 1970 would be worth over $2,000,000 today (with dividends reinvested). The key was staying invested through the scary times.

PRACTICAL ADVICE:
- If a 30% drop would make you sell, don't have 100% in stocks
- Your allocation should let you sleep at night
- Crashes are when future returns are made - stay the course
        """,
        "key_terms": ["drawdown", "recovery", "bear market", "crash", "volatility", "stay invested"],
        "related_modules": ["risk_explained", "what_could_happen"]
    },

    # =========================================================
    # RETIREMENT ACCOUNTS
    # =========================================================
    "401k_explained": {
        "title": "401(k) Complete Guide",
        "source": "IRS, Fidelity, Vanguard",
        "url": "https://www.irs.gov/retirement-plans/401k-plans",
        "category": "accounts",
        "difficulty": "beginner",
        "content": """
A 401(k) is an employer-sponsored retirement account with significant tax advantages. It's one of the most powerful wealth-building tools available.

HOW IT WORKS:
1. You contribute pre-tax money from your paycheck
2. Contributions reduce your taxable income
3. Money grows tax-deferred (no taxes on gains until withdrawal)
4. You pay income tax when you withdraw in retirement

2024 CONTRIBUTION LIMITS:
- Under 50: $23,000/year
- 50 and older: $30,500/year (extra $7,500 catch-up)
- Employer + employee total: $69,000

THE EMPLOYER MATCH (FREE MONEY!):
Many employers match your contributions. Common formulas:
- 50% match up to 6% of salary (you put in 6%, they add 3%)
- 100% match up to 3% (you put in 3%, they add 3%)
- Dollar-for-dollar up to 4%

ALWAYS CONTRIBUTE AT LEAST ENOUGH TO GET THE FULL MATCH.
A 50% match is an instant 50% return - you won't find that anywhere else!

TRADITIONAL VS ROTH 401(k):
| Feature | Traditional 401(k) | Roth 401(k) |
|---------|-------------------|-------------|
| Tax on contributions | Tax-deductible now | Pay taxes now |
| Tax on withdrawals | Taxed as income | Tax-free! |
| Best if | High tax bracket now | Low bracket now, expect higher later |

VESTING:
Your contributions are always 100% yours. Employer contributions may "vest" over time:
- Cliff vesting: 100% after X years (usually 3)
- Graded vesting: Gradually over 6 years

WHAT TO INVEST IN:
Most 401(k)s offer limited fund choices. Look for:
1. Target-date fund for your retirement year (simple, one-stop)
2. S&P 500 or Total Stock Market index fund
3. Total Bond index fund
4. International index fund

Avoid high-fee actively managed funds if low-cost index options exist.

COMMON MISTAKES:
- Not contributing enough to get full match
- Cashing out when changing jobs (10% penalty + taxes!)
- Being too conservative when young
- Not increasing contributions when you get raises
        """,
        "key_terms": ["401k", "employer match", "traditional", "Roth", "vesting", "contribution limit"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },

    "ira_explained": {
        "title": "IRA (Individual Retirement Account) Complete Guide",
        "source": "IRS, Fidelity, Schwab",
        "url": "https://www.irs.gov/retirement-plans/individual-retirement-arrangements-iras",
        "category": "accounts",
        "difficulty": "beginner",
        "content": """
An IRA (Individual Retirement Account) is a tax-advantaged account you open yourself (not through an employer). It offers more investment choices than most 401(k)s.

2024 CONTRIBUTION LIMITS:
- Under 50: $7,000/year
- 50 and older: $8,000/year
- Must have earned income to contribute
- Can contribute for previous year until April 15

TRADITIONAL IRA:
- Contributions may be tax-deductible (reduces taxable income)
- Money grows tax-deferred
- Withdrawals taxed as ordinary income
- Required Minimum Distributions (RMDs) start at age 73

Deductibility limits (if covered by workplace plan):
- Single: Full deduction if income <$77,000
- Married: Full deduction if income <$123,000
- Phase-out above these limits

ROTH IRA:
- Contributions are NOT tax-deductible
- Money grows tax-FREE
- Withdrawals in retirement are tax-FREE
- No RMDs during your lifetime
- Can withdraw contributions (not earnings) anytime, penalty-free

Income limits for Roth contributions (2024):
- Single: Can't contribute if income >$161,000
- Married: Can't contribute if income >$240,000
- "Backdoor Roth" can bypass this (contribute to Traditional, convert to Roth)

ROTH VS TRADITIONAL DECISION:
Choose Roth if:
- You're in a lower tax bracket now than you expect in retirement
- You want tax-free withdrawals
- You want no RMDs
- You're young (more time for tax-free growth)

Choose Traditional if:
- You're in a high tax bracket now and expect lower in retirement
- You need the tax deduction now
- You expect lower income in retirement

WHEN IN DOUBT: For young investors, Roth is usually better. Tax-free growth over decades is incredibly powerful.

WHERE TO OPEN AN IRA:
- Vanguard, Fidelity, Schwab (best for index funds)
- No account fees or minimums at these brokers
- Can hold any ETF or mutual fund

PRIORITY ORDER FOR RETIREMENT SAVINGS:
1. 401(k) up to employer match (free money!)
2. Max out Roth IRA ($7,000)
3. Max out 401(k) ($23,000)
4. HSA if available ($4,150 individual / $8,300 family)
5. Taxable brokerage account
        """,
        "key_terms": ["IRA", "Roth IRA", "traditional IRA", "tax-deferred", "tax-free", "backdoor Roth"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },

    "hsa_explained": {
        "title": "HSA: The Triple Tax-Advantaged Account",
        "source": "IRS, Fidelity, HealthCare.gov",
        "url": "https://www.irs.gov/publications/p969",
        "category": "accounts",
        "difficulty": "intermediate",
        "content": """
A Health Savings Account (HSA) is the only account with TRIPLE tax advantages - making it potentially better than a 401(k) or Roth IRA for long-term wealth building.

THE TRIPLE TAX ADVANTAGE:
1. Tax-deductible contributions (like Traditional 401k)
2. Tax-free growth (like Roth IRA)
3. Tax-free withdrawals for medical expenses (unique!)

No other account offers all three benefits.

ELIGIBILITY REQUIREMENTS:
- Must have a High Deductible Health Plan (HDHP)
- 2024 HDHP minimums: $1,600 deductible (individual), $3,200 (family)
- Cannot have other non-HDHP coverage
- Cannot be enrolled in Medicare

2024 CONTRIBUTION LIMITS:
- Individual: $4,150
- Family: $8,300
- Age 55+ catch-up: Additional $1,000

THE HSA HACK FOR WEALTH BUILDING:
Instead of using HSA for current medical expenses:
1. Pay medical expenses out-of-pocket now
2. Save receipts forever
3. Let HSA grow invested in index funds
4. Withdraw tax-free decades later with old receipts
5. After age 65, withdraw for ANY purpose (just pay income tax, like Traditional IRA)

WHY THIS WORKS:
- Medical expenses only grow over time
- You'll have plenty to reimburse yourself for later
- Meanwhile, your HSA compounds tax-free
- $100/month for 30 years at 7% = ~$122,000 tax-free

WHERE TO OPEN:
Many employer HSAs have high fees or bad investment options.
After building balance, transfer to:
- Fidelity HSA (no fees, all Fidelity funds)
- Lively + Schwab integration

IMPORTANT NOTES:
- HSA is yours forever (portable between jobs)
- No "use it or lose it" (that's FSA, not HSA)
- Funds roll over indefinitely
- After death, can transfer to spouse tax-free

STRATEGY:
If you can afford to pay medical expenses out-of-pocket, max out your HSA and invest it for the long term. It's the most tax-efficient account that exists.
        """,
        "key_terms": ["HSA", "health savings account", "triple tax advantage", "HDHP", "medical expenses"],
        "related_modules": ["goal_timeline"]
    },

    # =========================================================
    # BEHAVIORAL FINANCE
    # =========================================================
    "behavioral_mistakes": {
        "title": "Common Investor Mistakes and How to Avoid Them",
        "source": "Behavioral Finance Research, Dalbar Studies, Vanguard",
        "url": "https://www.investopedia.com/articles/basics/11/biggest-stock-market-myths.asp",
        "category": "behavioral",
        "difficulty": "intermediate",
        "content": """
The biggest enemy of investment returns isn't fees or bad picks - it's YOU. Behavioral biases cause average investors to underperform the market by 1-2% annually over their lifetimes.

THE DALBAR STUDY:
Over a 20-year period, the S&P 500 returned 7.5% annually, but the average stock fund investor earned only 2.9%. The difference? Behavioral mistakes - buying high and selling low.

COMMON MISTAKES:

1. LOSS AVERSION
- Losses feel 2x worse than equivalent gains feel good
- Result: Hold losers too long (hoping to break even), sell winners too soon
- Fix: Set rules in advance, automate decisions, don't check portfolio daily

2. RECENCY BIAS
- Overweight recent events in decision-making
- "Stocks crashed, I'll never invest again" or "Tech always goes up!"
- Fix: Look at 50+ year historical data, not the last 6 months

3. FOMO (Fear of Missing Out)
- Chase hot stocks/sectors AFTER they've already risen
- Buy high, then panic sell low when it crashes
- Fix: Stick to your plan, ignore market noise, avoid financial social media

4. OVERCONFIDENCE
- "I can beat the market" (90% of professionals can't)
- Trade too frequently, incur costs and taxes
- Fix: Accept you probably can't beat indexes, keep it simple

5. HERD MENTALITY
- Follow the crowd (meme stocks, crypto hype, gold rushes)
- Usually means buying at peaks
- Fix: Be skeptical of "everyone's doing it"

6. ANCHORING
- Fixate on purchase price: "I'll sell when I get back to even"
- The stock doesn't know what you paid for it
- Fix: Evaluate based on current value and future prospects, not your cost basis

7. CHECKING TOO OFTEN
- Daily portfolio checks increase anxiety and trading
- More likely to make emotional decisions during volatility
- Fix: Check quarterly at most, automate contributions

THE ANTIDOTE: AUTOMATION
- Automatic contributions (dollar-cost averaging)
- Automatic rebalancing
- Target-date funds
- Remove yourself from the decision process

"The investor's chief problem - and even his worst enemy - is likely to be himself." - Benjamin Graham
        """,
        "key_terms": ["behavioral finance", "loss aversion", "FOMO", "overconfidence", "recency bias"],
        "related_modules": ["risk_explained", "what_could_happen"]
    },

    "dollar_cost_averaging": {
        "title": "Dollar Cost Averaging (DCA)",
        "source": "Investopedia, Vanguard Research",
        "url": "https://www.investopedia.com/terms/d/dollarcostaveraging.asp",
        "category": "strategies",
        "difficulty": "beginner",
        "content": """
Dollar cost averaging is investing a fixed amount at regular intervals, regardless of market conditions. It's the opposite of trying to time the market.

HOW IT WORKS:
$500/month invested in an index fund:
- January: Price $50/share → Buy 10 shares
- February: Price $40/share → Buy 12.5 shares
- March: Price $55/share → Buy 9.1 shares
- Total: $1,500 invested, 31.6 shares
- Average cost: $47.47/share (lower than $48.33 average price!)

You automatically buy more shares when prices are low and fewer when high.

WHY DCA WORKS:
1. Removes emotion from investing (systematic, not reactive)
2. No need to time the market (impossible anyway)
3. Automatically buys more when prices are low
4. Builds consistent investing habit
5. Reduces impact of volatility
6. No regret if you invest and market drops

DCA VS LUMP SUM INVESTING:
If you have a large sum to invest, what's better?

Research shows lump sum wins ~67% of the time - money in the market longer usually beats waiting. BUT:
- DCA reduces regret if market drops after investing
- DCA is psychologically easier
- DCA is WAY better than waiting for "the right time" (which never comes)

The best investment strategy is the one you'll actually follow.

HOW TO IMPLEMENT:
1. Set up automatic transfers from checking to brokerage
2. Choose your investment (total market index fund)
3. Set the amount and frequency (monthly aligns with paychecks)
4. Don't look at it constantly
5. Increase amount when income rises

THE MATH OF CONSISTENCY:
$500/month for 30 years at 7% average return:
- Total contributed: $180,000
- Ending balance: $567,000
- Growth: $387,000 (more than 2x your contributions!)

$500/month for 40 years at 7%:
- Total contributed: $240,000
- Ending balance: $1,200,000
- Growth: $960,000 (4x your contributions!)

"Time in the market beats timing the market." - Every successful investor ever
        """,
        "key_terms": ["dollar cost averaging", "DCA", "lump sum", "automation", "systematic investing"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },

    # =========================================================
    # MARKET MECHANICS
    # =========================================================
    "how_markets_work": {
        "title": "How Stock Markets Work",
        "source": "SEC, NYSE, NASDAQ, Investopedia",
        "url": "https://www.investopedia.com/articles/investing/082614/how-stock-market-works.asp",
        "category": "markets",
        "difficulty": "intermediate",
        "content": """
Stock markets are where buyers and sellers come together to trade shares of publicly traded companies. Understanding how they work helps demystify investing.

MAJOR EXCHANGES:
- NYSE (New York Stock Exchange): Largest by market cap, 200+ years old
- NASDAQ: Electronic exchange, tech-heavy (Apple, Microsoft, Google)
- Other global: London Stock Exchange, Tokyo, Hong Kong, Shanghai

HOW TRADING WORKS:
1. You place an order through your broker
2. Order goes to exchange or market maker
3. Matched with opposite order (buyer finds seller)
4. Trade executes, ownership transfers
5. Settlement completes in T+1 (next business day)

PRICE DISCOVERY:
Prices are set by supply and demand:
- More buyers than sellers → Price rises
- More sellers than buyers → Price falls
- News/earnings → Orders flood one side → Rapid price change

The "price" you see is the last trade price. The actual price you get depends on the current bid/ask spread.

MARKET INDICES:
Indices track groups of stocks:
- S&P 500: 500 largest US companies (~80% of US market)
- Dow Jones: 30 large companies (price-weighted, less representative)
- NASDAQ Composite: All NASDAQ stocks (~3,000)
- Russell 2000: Small-cap stocks
- Total Stock Market: ~4,000 US stocks

When people say "the market," they usually mean the S&P 500 or Dow Jones.

MARKET HOURS (US Eastern Time):
- Pre-market: 4:00 AM - 9:30 AM
- Regular session: 9:30 AM - 4:00 PM
- After-hours: 4:00 PM - 8:00 PM

Most long-term investors should only trade during regular hours for best execution.

ORDER TYPES:
- Market order: Execute immediately at best available price
- Limit order: Only execute at specified price or better
- Stop order: Trigger market order when price hits threshold

FOR LONG-TERM INVESTORS:
You don't need to understand all the mechanics deeply. Just:
1. Buy diversified index funds
2. Hold for the long term
3. Don't try to time the market
4. Let the market work for you
        """,
        "key_terms": ["stock market", "exchange", "NYSE", "NASDAQ", "S&P 500", "index", "order types"],
        "related_modules": ["market_basics", "market_mechanics"]
    },

    # =========================================================
    # GETTING STARTED
    # =========================================================
    "how_to_start_investing": {
        "title": "How to Start Investing: A Step-by-Step Guide",
        "source": "Fidelity, Vanguard, NerdWallet",
        "url": "https://www.fidelity.com/learning-center/trading-investing/getting-started-investing",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
Ready to start investing? Here's exactly how to go from zero to invested, step by step.

STEP 1: BUILD AN EMERGENCY FUND FIRST
Before investing, save 3-6 months of expenses in a savings account.
Why? So you never have to sell investments during an emergency.

STEP 2: PAY OFF HIGH-INTEREST DEBT
Credit card debt at 20% interest? Pay that first.
You won't reliably beat 20% in the market.

Exception: Low-interest debt (mortgage, student loans under 5%) can coexist with investing.

STEP 3: GET FREE MONEY (401k MATCH)
If your employer offers a 401(k) match:
1. Enroll in the 401(k)
2. Contribute at least enough to get the full match
3. This is an instant 50-100% return!

STEP 4: OPEN A ROTH IRA
Best brokers for beginners (all free, no minimums):
- Fidelity: Great research, fractional shares, excellent app
- Vanguard: Lowest-cost funds, trusted name
- Schwab: Good all-around, strong customer service

Just pick one and open an account (takes 15 minutes).

STEP 5: CHOOSE YOUR INVESTMENTS
Simplest approach: A target-date fund for your retirement year.
- Vanguard Target Retirement 2055 (example)
- Automatically diversified
- Automatically rebalances
- One fund, done

Alternative: Three-fund portfolio
- 60% VTI (US stocks)
- 30% VXUS (international stocks)
- 10% BND (bonds)
Adjust percentages based on age/risk tolerance.

STEP 6: SET UP AUTOMATIC CONTRIBUTIONS
Link your bank account and set up recurring investments.
Even $50/week = $200/month = $2,600/year.

STEP 7: DON'T TOUCH IT
Seriously. Set it and forget it.
Check quarterly at most. Don't panic during drops.

COMMON BEGINNER QUESTIONS:

Q: How much do I need to start?
A: $1. Many brokers allow fractional shares and have no minimums.

Q: What if the market crashes right after I invest?
A: Keep investing. You're buying at lower prices. This is good for long-term investors.

Q: Should I wait for a better time to start?
A: No. Time in the market beats timing the market. Start now.

Q: What about crypto/meme stocks/gold?
A: Get your boring index fund portfolio set up first. Then if you want to speculate with 5% of your money, go ahead. But index funds should be your foundation.
        """,
        "key_terms": ["start investing", "beginner", "emergency fund", "Roth IRA", "target date fund"],
        "related_modules": ["goal_timeline", "first_portfolio"]
    },

    "emergency_fund": {
        "title": "Emergency Fund: Your Financial Foundation",
        "source": "FINRA, Fidelity, Consumer Finance Protection Bureau",
        "url": "https://www.consumerfinance.gov/start-small-save-up/",
        "category": "basics",
        "difficulty": "beginner",
        "content": """
An emergency fund is cash savings that covers unexpected expenses or income loss. It's the foundation of financial security and should be built BEFORE investing.

WHY YOU NEED IT:
1. Avoid selling investments during emergencies (locks in losses)
2. Prevent credit card debt during unexpected expenses
3. Peace of mind during job loss or medical issues
4. Freedom to take risks (career changes, starting business)

HOW MUCH TO SAVE:
- Minimum: $1,000 (starter fund for small emergencies)
- Target: 3-6 months of essential expenses
- More if: Self-employed, single income, unstable industry

Essential expenses include:
- Housing (rent/mortgage)
- Utilities
- Food
- Insurance
- Minimum debt payments
- Transportation

WHERE TO KEEP IT:
NOT invested - this needs to be safe and liquid:
- High-yield savings account (currently 4-5% APY at online banks)
- Money market account
- Treasury bills (T-bills) or I-Bonds
- NOT in CDs (penalties for early withdrawal)

GOOD HIGH-YIELD SAVINGS OPTIONS:
- Marcus by Goldman Sachs
- Ally Bank
- Discover
- Capital One 360
- American Express Savings

HOW TO BUILD IT:
1. Open a separate savings account (out of sight, out of mind)
2. Set up automatic transfers from each paycheck
3. Start with $100/month, increase when possible
4. Direct deposit tax refunds here
5. Put windfalls (bonuses, gifts) here until full

THE MATH:
If monthly expenses are $4,000:
- 3 months = $12,000 target
- 6 months = $24,000 target

Even $50/week = $2,600/year toward your goal.

WHAT COUNTS AS AN EMERGENCY:
✅ Job loss
✅ Medical emergency
✅ Major car repair
✅ Critical home repair
✅ Unexpected travel for family emergency

NOT EMERGENCIES:
❌ Vacation
❌ New phone
❌ Holiday shopping
❌ Investment "opportunity"
❌ Anything predictable (save separately)

ONCE YOU HAVE 3-6 MONTHS SAVED:
Stop contributing and redirect that money to investing.
Your emergency fund doesn't need to grow beyond 6 months.
        """,
        "key_terms": ["emergency fund", "savings", "high-yield savings", "financial foundation", "3-6 months"],
        "related_modules": ["goal_timeline"]
    },

    # =========================================================
    # TAXES
    # =========================================================
    "tax_efficiency": {
        "title": "Tax-Efficient Investing",
        "source": "Bogleheads Wiki, Vanguard, IRS",
        "url": "https://www.bogleheads.org/wiki/Tax-efficient_fund_placement",
        "category": "advanced",
        "difficulty": "intermediate",
        "content": """
Tax efficiency can add 0.5-1% to your annual returns over time. Here's how to keep more of what you earn.

ACCOUNT TYPES BY TAX TREATMENT:
1. Taxable brokerage: Dividends and gains taxed yearly
2. Tax-deferred (Traditional 401k/IRA): Tax on withdrawal
3. Tax-free (Roth): No tax on growth or withdrawal

ASSET LOCATION STRATEGY:
Put tax-inefficient investments in tax-advantaged accounts:

TAX-ADVANTAGED ACCOUNTS (401k, IRA):
- Bonds (interest taxed as income)
- REITs (dividends taxed as income)
- Actively managed funds (more turnover = more taxes)

TAXABLE ACCOUNTS:
- US stock index funds (tax-efficient, low turnover)
- International stock funds (foreign tax credit)
- Municipal bonds (tax-free interest)
- ETFs over mutual funds (more tax-efficient)

TAX-LOSS HARVESTING:
Sell investments at a loss to offset gains:
1. Sell losing investment
2. Use loss to offset capital gains (or $3,000 of income)
3. Buy similar (not identical) investment to maintain exposure
4. Carry forward unused losses to future years

Example: You sell stock A at $5,000 loss, buy similar stock B immediately. You maintain market exposure but harvest the tax loss.

WASH SALE RULE:
Can't buy "substantially identical" security within 30 days before or after selling at a loss. VTI and VOO are different enough. VTI and VTSAX are NOT.

CAPITAL GAINS RATES (2024):
| Tax Bracket | Short-term (<1 year) | Long-term (>1 year) |
|-------------|---------------------|---------------------|
| 10-12% | 10-12% | 0% |
| 22-35% | 22-35% | 15% |
| 37% | 37% | 20% |

Hold investments >1 year for lower long-term rates!

QUALIFIED DIVIDENDS:
Most stock dividends are "qualified" and taxed at lower long-term capital gains rates. Bond interest is taxed as ordinary income (higher rate).

SIMPLE RULES:
1. Max out tax-advantaged accounts first
2. Use tax-efficient index funds in taxable accounts
3. Hold investments >1 year when possible
4. Consider tax-loss harvesting in down markets
5. Be mindful of dividend taxes in taxable accounts
        """,
        "key_terms": ["tax efficiency", "asset location", "tax-loss harvesting", "capital gains", "qualified dividends"],
        "related_modules": ["asset_allocation", "advanced_strategies"]
    },

    # =========================================================
    # ADDITIONAL TOPICS
    # =========================================================
    "rebalancing": {
        "title": "Portfolio Rebalancing: When and How",
        "source": "Vanguard Research, Bogleheads",
        "url": "https://investor.vanguard.com/investor-resources-education/education/why-rebalancing",
        "category": "portfolio",
        "difficulty": "intermediate",
        "content": """
Rebalancing means adjusting your portfolio back to your target allocation after market movements cause drift. It's a disciplined way to "buy low, sell high."

WHY REBALANCE:
Without rebalancing, a 60/40 stock/bond portfolio might become 75/25 after a bull market, increasing your risk beyond what you intended.

Example:
- Start: 60% stocks, 40% bonds
- After bull market: 75% stocks, 25% bonds
- Rebalance: Sell stocks, buy bonds to return to 60/40

REBALANCING METHODS:

1. CALENDAR-BASED:
Rebalance on a set schedule (quarterly, semi-annually, annually).
- Pros: Simple, systematic
- Cons: May rebalance when not needed

2. THRESHOLD-BASED:
Rebalance when allocation drifts 5%+ from target.
- Example: Stocks go from 60% to 66%? Rebalance.
- Pros: Only act when needed
- Cons: Requires monitoring

3. HYBRID:
Check quarterly, only rebalance if threshold exceeded.
Best of both worlds.

HOW TO REBALANCE:
In tax-advantaged accounts (401k, IRA):
- Sell high, buy low without tax consequences
- Simple!

In taxable accounts (to minimize taxes):
- Use new contributions to buy underweight assets
- Direct dividends to underweight assets
- Only sell as last resort (triggers capital gains)

FREQUENCY:
Research shows annual rebalancing is sufficient.
More frequent rebalancing doesn't improve returns and may increase costs/taxes.

REBALANCING BONUS:
Rebalancing forces you to buy low and sell high systematically.
During 2009 crash, rebalancing meant buying stocks at low prices.
During 2021 bull market, rebalancing meant selling stocks at high prices.

DON'T OVER-REBALANCE:
- Transaction costs add up
- In taxable accounts, selling triggers capital gains
- Small drifts (2-3%) don't need action

SIMPLE RULE:
Check once a year. If any asset is 5%+ away from target, rebalance. Otherwise, leave it alone.
        """,
        "key_terms": ["rebalancing", "portfolio drift", "asset allocation", "buy low sell high"],
        "related_modules": ["asset_allocation", "three_fund_portfolio"]
    },
}


# =========================================================
# RETRIEVAL FUNCTIONS
# =========================================================

def search_knowledge_base(query: str, category: str = None, difficulty: str = None) -> List[Tuple[str, int, Dict]]:
    """
    Search the knowledge base for relevant documents using keyword matching.
    Returns list of (doc_id, relevance_score, document) tuples.
    
    For better results, use the vector store in vector_store.py
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
            score += 15
        
        # Key terms match (high weight)
        for term in doc.get('key_terms', []):
            if term.lower() in query_lower or query_lower in term.lower():
                score += 8
            # Partial match
            for query_word in query_lower.split():
                if len(query_word) > 3 and query_word in term.lower():
                    score += 3
        
        # Content match (medium weight)
        content_lower = doc['content'].lower()
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in content_lower:
                score += 1
                # Bonus for multiple occurrences
                if content_lower.count(word) > 3:
                    score += 1
        
        # Category relevance
        if doc.get('category', '').lower() in query_lower:
            score += 5
        
        if score > 0:
            results.append((doc_id, score, doc))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Return top 5


def get_document_by_id(doc_id: str) -> Optional[Dict]:
    """Get a specific document by ID."""
    return KNOWLEDGE_BASE.get(doc_id)


def get_documents_by_category(category: str) -> List[Tuple[str, Dict]]:
    """Get all documents in a category."""
    return [
        (doc_id, doc) for doc_id, doc in KNOWLEDGE_BASE.items()
        if doc.get('category') == category
    ]


def get_documents_for_module(module_id: str) -> List[Tuple[str, Dict]]:
    """Get documents related to a specific module."""
    return [
        (doc_id, doc) for doc_id, doc in KNOWLEDGE_BASE.items()
        if module_id in doc.get('related_modules', [])
    ]


def format_context_for_llm(documents: List[Tuple[str, int, Dict]], max_length: int = 6000) -> str:
    """Format retrieved documents as context for LLM."""
    context_parts = []
    total_length = 0
    
    for doc_id, score, doc in documents:
        doc_text = f"""
---
SOURCE: {doc['title']} ({doc['source']})
CATEGORY: {doc.get('category', 'general')} | DIFFICULTY: {doc.get('difficulty', 'beginner')}

{doc['content'].strip()}
---
"""
        if total_length + len(doc_text) > max_length:
            break
        context_parts.append(doc_text)
        total_length += len(doc_text)
    
    return "\n".join(context_parts)


def get_all_documents() -> List[Dict]:
    """Get all documents for indexing."""
    return [
        {
            "id": doc_id,
            "title": doc["title"],
            "source": doc["source"],
            "category": doc.get("category", "general"),
            "difficulty": doc.get("difficulty", "beginner"),
            "content": doc["content"],
            "key_terms": doc.get("key_terms", [])
        }
        for doc_id, doc in KNOWLEDGE_BASE.items()
    ]
