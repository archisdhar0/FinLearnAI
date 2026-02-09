"""
QuantCademy RAG Knowledge Base v2
Capstone-grade knowledge base with:
- Source tiering (regulatory > institutional > educational)
- Structured chunking by section
- Rich metadata for retrieval
"""

from typing import List, Dict, Optional
from enum import IntEnum
from dataclasses import dataclass, field
import hashlib


class SourceTier(IntEnum):
    """Source trust hierarchy - higher = more authoritative."""
    TIER_1_REGULATORY = 5      # SEC, FINRA, IRS, Treasury
    TIER_2_INSTITUTIONAL = 4   # Federal Reserve, CFA Institute, Vanguard Research
    TIER_3_MAJOR_FINANCIAL = 3 # Fidelity, Schwab, Bogleheads
    TIER_4_EDUCATIONAL = 2     # Investopedia, NerdWallet
    TIER_5_GENERAL = 1         # Blogs, forums, general sources


# Source to tier mapping
SOURCE_TIERS = {
    # Tier 1: Regulatory/Government
    "SEC": SourceTier.TIER_1_REGULATORY,
    "SEC Investor.gov": SourceTier.TIER_1_REGULATORY,
    "FINRA": SourceTier.TIER_1_REGULATORY,
    "IRS": SourceTier.TIER_1_REGULATORY,
    "TreasuryDirect.gov": SourceTier.TIER_1_REGULATORY,
    "Treasury": SourceTier.TIER_1_REGULATORY,
    "Bureau of Labor Statistics": SourceTier.TIER_1_REGULATORY,
    
    # Tier 2: Institutional/Academic
    "Federal Reserve": SourceTier.TIER_2_INSTITUTIONAL,
    "CFA Institute": SourceTier.TIER_2_INSTITUTIONAL,
    "Vanguard Research": SourceTier.TIER_2_INSTITUTIONAL,
    "SPIVA Research": SourceTier.TIER_2_INSTITUTIONAL,
    "Academic Studies": SourceTier.TIER_2_INSTITUTIONAL,
    "Dalbar Studies": SourceTier.TIER_2_INSTITUTIONAL,
    "Behavioral Finance Research": SourceTier.TIER_2_INSTITUTIONAL,
    
    # Tier 3: Major Financial Institutions
    "Vanguard": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "Fidelity": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "Schwab": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "Bogleheads": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "Bogleheads Wiki": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "NYSE": SourceTier.TIER_3_MAJOR_FINANCIAL,
    "NASDAQ": SourceTier.TIER_3_MAJOR_FINANCIAL,
    
    # Tier 4: Educational
    "Investopedia": SourceTier.TIER_4_EDUCATIONAL,
    "NerdWallet": SourceTier.TIER_4_EDUCATIONAL,
    "Consumer Finance Protection Bureau": SourceTier.TIER_4_EDUCATIONAL,
    "HealthCare.gov": SourceTier.TIER_4_EDUCATIONAL,
    
    # Tier 5: General
    "Portfolio Visualizer": SourceTier.TIER_5_GENERAL,
    "Historical Data": SourceTier.TIER_5_GENERAL,
}


def get_source_tier(source_string: str) -> SourceTier:
    """Get the highest tier from a source string (may contain multiple sources)."""
    best_tier = SourceTier.TIER_5_GENERAL
    for source_name, tier in SOURCE_TIERS.items():
        if source_name.lower() in source_string.lower():
            if tier > best_tier:
                best_tier = tier
    return best_tier


def get_tier_label(tier: SourceTier) -> str:
    """Human-readable tier label."""
    labels = {
        SourceTier.TIER_1_REGULATORY: "🏛️ Regulatory",
        SourceTier.TIER_2_INSTITUTIONAL: "🎓 Institutional",
        SourceTier.TIER_3_MAJOR_FINANCIAL: "🏦 Financial Institution",
        SourceTier.TIER_4_EDUCATIONAL: "📚 Educational",
        SourceTier.TIER_5_GENERAL: "📰 General",
    }
    return labels.get(tier, "📰 General")


@dataclass
class Chunk:
    """A single retrievable chunk with metadata."""
    id: str
    document_id: str
    content: str
    chunk_type: str  # "definition", "concept", "example", "procedure", "table"
    section: str
    source: str
    source_tier: SourceTier
    url: str
    category: str
    difficulty: str
    key_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "section": self.section,
            "source": self.source,
            "source_tier": int(self.source_tier),
            "source_tier_label": get_tier_label(self.source_tier),
            "url": self.url,
            "category": self.category,
            "difficulty": self.difficulty,
            "key_terms": self.key_terms,
        }
    
    def get_citation(self) -> str:
        """Generate a citation string."""
        tier_label = get_tier_label(self.source_tier)
        return f"{self.source} ({tier_label})"


def generate_chunk_id(doc_id: str, section: str, idx: int) -> str:
    """Generate unique chunk ID."""
    content = f"{doc_id}:{section}:{idx}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


# =========================================================
# CHUNKED KNOWLEDGE BASE
# Each document is split into semantic chunks
# =========================================================

CHUNKED_KNOWLEDGE_BASE: List[Chunk] = []


def build_chunked_knowledge_base():
    """Build the chunked knowledge base from raw documents."""
    global CHUNKED_KNOWLEDGE_BASE
    CHUNKED_KNOWLEDGE_BASE = []
    
    # =========================================================
    # INVESTING BASICS
    # =========================================================
    
    # What is Investing - DEFINITIONS
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("investing_basics", "definition", 0),
        document_id="investing_basics",
        content="""DEFINITION: Investing
Investing is putting money to work to potentially grow over time. When you invest, you purchase assets like stocks, bonds, or funds with the expectation of earning a return.

Key terms:
- Principal: The amount you initially invest
- Return: The profit or loss from your investment  
- Risk: The possibility of losing some or all of your investment
- Compound growth: Earning returns on your returns over time""",
        chunk_type="definition",
        section="What is Investing",
        source="SEC Investor.gov",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/introduction-investing",
        category="basics",
        difficulty="beginner",
        key_terms=["investing", "principal", "return", "risk", "compound growth"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("investing_basics", "why_invest", 1),
        document_id="investing_basics",
        content="""WHY INVEST?
According to SEC Investor.gov, the main reasons to invest are:

1. Beat inflation - Cash loses about 3% purchasing power per year
2. Build wealth - Historically, stocks return approximately 10% annually
3. Reach financial goals - Retirement, home purchase, education

IMPORTANT WARNING: Investing involves risk. You could lose money. Past performance does not guarantee future results. However, historically, staying invested in diversified portfolios has rewarded patient investors.""",
        chunk_type="concept",
        section="Why Invest",
        source="SEC Investor.gov",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/introduction-investing",
        category="basics",
        difficulty="beginner",
        key_terms=["inflation", "wealth building", "financial goals", "risk"]
    ))
    
    # Compound Interest - DEFINITION
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("compound_interest", "definition", 0),
        document_id="compound_interest",
        content="""DEFINITION: Compound Interest
Compound interest is when you earn interest on both your original investment AND on the interest you've already earned. It's often called "interest on interest" and is considered the most powerful force in investing.

Mathematical formula:
Future Value = Principal × (1 + rate)^years

THE RULE OF 72:
Divide 72 by your annual return to estimate years to double your money.
- At 7%: 72/7 ≈ 10 years to double
- At 10%: 72/10 ≈ 7 years to double
- At 12%: 72/12 = 6 years to double""",
        chunk_type="definition",
        section="Compound Interest Definition",
        source="SEC Investor.gov",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
        category="basics",
        difficulty="beginner",
        key_terms=["compound interest", "rule of 72", "time value of money", "exponential growth"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("compound_interest", "example", 1),
        document_id="compound_interest",
        content="""EXAMPLE: Compound Interest Growth
$10,000 invested at 7% annual return:
- Year 1: $10,700 (earned $700)
- Year 5: $14,026 (total growth: $4,026)
- Year 10: $19,672 (total growth: $9,672)
- Year 20: $38,697 (total growth: $28,697)
- Year 30: $76,123 (total growth: $66,123)

Notice how the growth accelerates over time - that's compounding at work.

WHY STARTING EARLY MATTERS:
Someone who invests $5,000/year from age 25-35 (10 years, $50,000 total) will have MORE at age 65 than someone who invests $5,000/year from age 35-65 (30 years, $150,000 total). The early investor contributed $100,000 LESS but ends up with MORE money because compound interest had more time to work.""",
        chunk_type="example",
        section="Compound Interest Examples",
        source="SEC Investor.gov",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
        category="basics",
        difficulty="beginner",
        key_terms=["compound interest", "early investing", "growth"]
    ))
    
    # =========================================================
    # STOCKS AND EQUITIES
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("stocks", "definition", 0),
        document_id="stocks",
        content="""DEFINITION: Stock (Equity)
A stock represents ownership in a company. When you buy a stock, you become a shareholder and own a small piece of that company.

HOW STOCKS MAKE MONEY:
1. Capital appreciation - Stock price increases over time. Buy at $100, sell at $150 = $50 profit per share.
2. Dividends - Some companies pay regular cash distributions from their profits.

MARKET CAPITALIZATION:
Market Cap = Stock Price × Shares Outstanding
- Large-cap: Greater than $10 billion (more stable)
- Mid-cap: $2-10 billion (balance of growth and stability)
- Small-cap: Less than $2 billion (higher growth potential, higher risk)""",
        chunk_type="definition",
        section="Stock Definition",
        source="SEC, Investopedia",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
        category="asset_classes",
        difficulty="beginner",
        key_terms=["stock", "equity", "shareholder", "dividend", "capital appreciation", "market cap"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("stocks", "historical", 1),
        document_id="stocks",
        content="""HISTORICAL STOCK PERFORMANCE (US Markets)
According to long-term market data:
- Average annual return since 1926: approximately 10%
- Individual years vary significantly:
  - Best year (1933): +54%
  - Worst year (2008): -37%
  - 2020 COVID crash: -34% then +68% recovery

TYPES OF STOCKS:
- Large-cap: Big established companies (Apple, Microsoft) - More stable
- Mid-cap: Medium-sized growing companies
- Small-cap: Smaller companies - Higher growth potential, higher risk
- International developed: Europe, Japan, Australia
- Emerging markets: China, India, Brazil - Higher growth, higher volatility""",
        chunk_type="concept",
        section="Stock Performance History",
        source="Vanguard Research, Historical Data",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education",
        category="asset_classes",
        difficulty="beginner",
        key_terms=["stock returns", "historical performance", "large-cap", "small-cap", "international"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("stocks", "risks", 2),
        document_id="stocks",
        content="""RISKS OF INDIVIDUAL STOCKS
According to FINRA and SEC guidelines, key risks include:

1. Market Risk: All stocks can fall during market downturns - this cannot be diversified away
2. Company Risk: Individual companies can fail (examples: Enron, Lehman Brothers, Blockbuster)
3. Volatility Risk: Prices can swing 30-50% in a single year
4. Concentration Risk: One bad stock can devastate your portfolio

RECOMMENDATION FOR BEGINNERS:
Consider index funds that hold hundreds or thousands of stocks rather than trying to pick individual winners. This diversifies away company-specific risk while capturing overall market growth.""",
        chunk_type="concept",
        section="Stock Risks",
        source="FINRA, SEC",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.finra.org/investors/investing/investment-products/stocks",
        category="risk",
        difficulty="beginner",
        key_terms=["market risk", "company risk", "volatility", "diversification", "index funds"]
    ))
    
    # =========================================================
    # BONDS
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("bonds", "definition", 0),
        document_id="bonds",
        content="""DEFINITION: Bond (Fixed Income)
A bond is a loan you make to a company or government. In return, they pay you interest (coupon) and return your principal at maturity. Bonds are generally less risky than stocks but offer lower returns.

HOW BONDS WORK:
1. You buy a bond for $1,000 (face/par value)
2. Issuer pays you interest (coupon) regularly, e.g., 4% = $40/year
3. At maturity (e.g., 10 years), you get your $1,000 back

TYPES OF BONDS:
- Treasury bonds (T-bonds): US government - Safest, lower yield
- Treasury bills (T-bills): Short-term government (weeks to 1 year)
- Municipal bonds: State/local government - Often tax-free
- Corporate bonds: Companies - Higher yield, more risk
- High-yield (junk) bonds: Lower-rated companies - Highest yield and risk""",
        chunk_type="definition",
        section="Bond Definition",
        source="SEC, TreasuryDirect.gov",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/bonds-or-fixed-income-products",
        category="asset_classes",
        difficulty="beginner",
        key_terms=["bond", "fixed income", "coupon", "yield", "maturity", "treasury", "corporate bond"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("bonds", "interest_rate_risk", 1),
        document_id="bonds",
        content="""CRITICAL CONCEPT: Interest Rate Risk for Bonds
When interest rates RISE, bond prices FALL (and vice versa).

WHY THIS HAPPENS:
If you own a bond paying 3% and new bonds pay 5%, no one wants your 3% bond unless you sell it at a discount.

IMPORTANT DISTINCTION:
- If you sell BEFORE maturity: Price changes matter
- If you hold TO maturity: You get your full principal back regardless of price changes

BOND DURATION:
Duration measures sensitivity to interest rate changes.
- Short-term bonds (1-3 years): Lower risk, lower yield
- Long-term bonds (10-30 years): Higher risk, higher yield
- Longer duration = more price volatility when rates change""",
        chunk_type="concept",
        section="Interest Rate Risk",
        source="FINRA, Federal Reserve",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.finra.org/investors/investing/investment-products/bonds",
        category="risk",
        difficulty="intermediate",
        key_terms=["interest rate risk", "duration", "bond prices", "yield"]
    ))
    
    # =========================================================
    # ETFs AND INDEX FUNDS
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("etf", "definition", 0),
        document_id="etf",
        content="""DEFINITION: ETF (Exchange-Traded Fund)
An ETF is a basket that holds many investments at once. Instead of buying individual stocks, you buy one share of an ETF and instantly own pieces of hundreds or thousands of companies.

KEY CHARACTERISTICS (per SEC regulations):
- Trade on exchanges like stocks (buy/sell anytime market is open)
- Price fluctuates throughout the day
- Generally very low expense ratios (as low as 0.03%)
- No minimum investment (buy 1 share or fractional shares)
- Tax efficient (generally fewer capital gains distributions)
- Holdings disclosed daily (transparent)""",
        chunk_type="definition",
        section="ETF Definition",
        source="SEC, Vanguard",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/mutual-funds-and-exchange-traded-1",
        category="investment_vehicles",
        difficulty="beginner",
        key_terms=["ETF", "exchange-traded fund", "index fund", "diversification", "expense ratio"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("etf", "popular", 1),
        document_id="etf",
        content="""POPULAR ETFs FOR BEGINNERS
US Stock Market:
- VTI (Vanguard Total Stock Market): ~4,000 US stocks, 0.03% expense ratio
- VOO (Vanguard S&P 500): 500 largest US companies, 0.03% expense ratio
- SPY (SPDR S&P 500): Oldest S&P 500 ETF, very liquid

International:
- VXUS (Vanguard Total International): All non-US stocks, 0.08% expense ratio
- VEA (Vanguard Developed Markets): Europe, Japan, Australia
- VWO (Vanguard Emerging Markets): China, India, Brazil

Bonds:
- BND (Vanguard Total Bond): US investment-grade bonds, 0.03% expense ratio
- BNDX (Vanguard International Bond): Non-US bonds""",
        chunk_type="reference",
        section="Popular ETFs",
        source="Vanguard, Fidelity",
        source_tier=SourceTier.TIER_3_MAJOR_FINANCIAL,
        url="https://investor.vanguard.com/etf/",
        category="investment_vehicles",
        difficulty="beginner",
        key_terms=["VTI", "VOO", "SPY", "VXUS", "BND", "index fund", "ETF"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("index_vs_active", "evidence", 0),
        document_id="index_vs_active",
        content="""INDEX FUNDS VS ACTIVE MANAGEMENT: THE EVIDENCE
According to the SPIVA Scorecard (Standard & Poor's):

US Large-Cap funds underperforming S&P 500:
- 1 year: ~60% underperform
- 5 years: ~75% underperform  
- 15 years: ~90% underperform
- 20 years: ~94% underperform

WHY DO MOST MANAGERS FAIL?
1. Fees: 1% annual fee is difficult to overcome
2. Transaction costs: Frequent trading costs money
3. Market efficiency: Information is priced in quickly
4. Reversion to mean: Hot managers usually cool off
5. Cash drag: Must hold some cash for redemptions""",
        chunk_type="concept",
        section="Active vs Passive Evidence",
        source="SPIVA Research, Vanguard Research",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://www.spglobal.com/spdji/en/research-insights/spiva/",
        category="investment_vehicles",
        difficulty="intermediate",
        key_terms=["index fund", "active management", "SPIVA", "expense ratio", "market efficiency"]
    ))
    
    # =========================================================
    # ASSET ALLOCATION
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("asset_allocation", "definition", 0),
        document_id="asset_allocation",
        content="""DEFINITION: Asset Allocation
Asset allocation is how you divide your portfolio among different asset classes (stocks, bonds, cash). According to academic research, it determines approximately 90% of your portfolio's return variability.

THE BRINSON STUDY:
A landmark study (Brinson, Hood, Beebower) found that asset allocation explained 91.5% of the variation in returns among pension funds. What specific investments you own matters much less than how much you own of each asset type.

FACTORS TO CONSIDER:
1. Time Horizon - Longer horizon → More stocks (time to recover from crashes)
2. Risk Tolerance - Your emotional capacity for loss
3. Risk Capacity - Your financial ability to take risk
4. Goals - Retirement in 30 years vs. house down payment in 3 years""",
        chunk_type="definition",
        section="Asset Allocation Definition",
        source="Vanguard Research, CFA Institute",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education/how-to-invest/asset-allocation",
        category="portfolio",
        difficulty="intermediate",
        key_terms=["asset allocation", "stocks", "bonds", "risk tolerance", "time horizon"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("asset_allocation", "historical", 1),
        document_id="asset_allocation",
        content="""HISTORICAL RETURNS BY ALLOCATION (1926-2023)
Based on Vanguard Research:

| Stocks/Bonds | Avg Return | Worst Year | Best Year | Max Drawdown |
|--------------|------------|------------|-----------|--------------|
| 100/0        | 10.3%      | -43%       | +54%      | -51%         |
| 80/20        | 9.4%       | -34%       | +45%      | -40%         |
| 60/40        | 8.5%       | -27%       | +36%      | -32%         |
| 40/60        | 7.5%       | -18%       | +28%      | -22%         |
| 20/80        | 6.4%       | -10%       | +21%      | -14%         |

NOTICE: The tradeoff - Higher returns come with bigger potential losses.""",
        chunk_type="table",
        section="Historical Allocation Returns",
        source="Vanguard Research",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education/how-to-invest/asset-allocation",
        category="portfolio",
        difficulty="intermediate",
        key_terms=["asset allocation", "historical returns", "drawdown", "stocks", "bonds"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("three_fund", "definition", 0),
        document_id="three_fund",
        content="""DEFINITION: Three-Fund Portfolio
The three-fund portfolio is a simple, effective strategy recommended by Bogleheads (followers of Vanguard founder John Bogle).

THE THREE FUNDS:
1. US Total Stock Market Index - VTI (ETF) or VTSAX (mutual fund)
   Covers ~4,000 US companies of all sizes, expense ratio: 0.03-0.04%

2. International Total Stock Market Index - VXUS (ETF) or VTIAX (mutual fund)
   Covers all non-US developed and emerging markets, expense ratio: 0.07-0.11%

3. US Total Bond Market Index - BND (ETF) or VBTLX (mutual fund)
   Investment-grade US bonds, expense ratio: 0.03-0.05%

WHY IT WORKS:
- Holds thousands of stocks and bonds (instant diversification)
- Extremely low cost (weighted average ~0.05%)
- Captures global market returns
- Beats 90% of professional investors long-term
- Simple to understand and maintain""",
        chunk_type="definition",
        section="Three-Fund Portfolio",
        source="Bogleheads Wiki, Vanguard",
        source_tier=SourceTier.TIER_3_MAJOR_FINANCIAL,
        url="https://www.bogleheads.org/wiki/Three-fund_portfolio",
        category="portfolio",
        difficulty="beginner",
        key_terms=["three-fund portfolio", "Bogleheads", "VTI", "VXUS", "BND", "asset allocation"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("three_fund", "allocations", 1),
        document_id="three_fund",
        content="""THREE-FUND PORTFOLIO: SAMPLE ALLOCATIONS BY AGE/RISK

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
Once a year, check if your allocation has drifted more than 5% from target. If so, sell what's grown too large and buy what's shrunk. This forces "sell high, buy low" behavior.""",
        chunk_type="procedure",
        section="Three-Fund Allocations",
        source="Bogleheads Wiki, Vanguard Research",
        source_tier=SourceTier.TIER_3_MAJOR_FINANCIAL,
        url="https://www.bogleheads.org/wiki/Three-fund_portfolio",
        category="portfolio",
        difficulty="beginner",
        key_terms=["asset allocation", "age-based allocation", "rebalancing", "three-fund"]
    ))
    
    # =========================================================
    # RISK AND DRAWDOWNS
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("risk", "types", 0),
        document_id="risk",
        content="""TYPES OF INVESTMENT RISK
According to FINRA and SEC educational materials:

1. MARKET RISK (Systematic Risk)
- Affects all investments simultaneously
- Cannot be diversified away
- Examples: 2008 financial crisis, COVID crash, 2022 bear market
- Mitigation: Time horizon, asset allocation (add bonds)

2. COMPANY RISK (Unsystematic Risk)
- Affects individual companies
- CAN be diversified away
- Examples: Enron bankruptcy, company fraud, bad products
- Mitigation: Own many stocks via index funds

3. INFLATION RISK
- Purchasing power decreases over time
- Cash loses ~3% value per year to inflation
- Mitigation: Own growth assets (stocks, TIPS)

4. INTEREST RATE RISK
- Bond prices fall when interest rates rise
- Longer-term bonds more affected
- Mitigation: Shorter-duration bonds, bond ladders""",
        chunk_type="concept",
        section="Types of Risk",
        source="FINRA, SEC",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.finra.org/investors/investing/investing-basics",
        category="risk",
        difficulty="intermediate",
        key_terms=["market risk", "systematic risk", "unsystematic risk", "inflation risk", "interest rate risk"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("drawdowns", "definition", 0),
        document_id="drawdowns",
        content="""DEFINITION: Drawdown
A drawdown is the decline from a portfolio's peak to its lowest point before reaching a new high. It measures the "pain" you experience as an investor.

CRITICAL MATH:
A 50% loss requires a 100% gain just to break even.

| Loss | Gain Needed to Recover |
|------|------------------------|
| 10%  | 11%                    |
| 20%  | 25%                    |
| 30%  | 43%                    |
| 40%  | 67%                    |
| 50%  | 100%                   |

This asymmetry is why managing downside risk matters so much.""",
        chunk_type="definition",
        section="Drawdown Definition",
        source="Vanguard Research, CFA Institute",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education",
        category="risk",
        difficulty="intermediate",
        key_terms=["drawdown", "recovery", "loss", "volatility"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("drawdowns", "historical", 1),
        document_id="drawdowns",
        content="""HISTORICAL US STOCK DRAWDOWNS

| Event                      | Decline | Duration to Bottom | Recovery Time |
|----------------------------|---------|-------------------|---------------|
| Great Depression (1929-32) | -86%    | 3 years           | 25 years      |
| 1973-74 Bear Market        | -48%    | 2 years           | 7 years       |
| Black Monday (1987)        | -34%    | 2 months          | 2 years       |
| Dot-com Crash (2000-02)    | -49%    | 2.5 years         | 7 years       |
| Financial Crisis (2008-09) | -51%    | 1.5 years         | 4 years       |
| COVID Crash (2020)         | -34%    | 1 month           | 5 months      |
| 2022 Bear Market           | -25%    | 10 months         | ~1 year       |

IMPORTANT CONTEXT: Despite all these crashes, $10,000 invested in US stocks in 1970 would be worth over $2,000,000 today (with dividends reinvested). The key was staying invested through the scary times.""",
        chunk_type="table",
        section="Historical Drawdowns",
        source="Vanguard Research, Historical Data",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education",
        category="risk",
        difficulty="intermediate",
        key_terms=["drawdown", "crash", "bear market", "recovery", "historical"]
    ))
    
    # =========================================================
    # RETIREMENT ACCOUNTS
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("401k", "definition", 0),
        document_id="401k",
        content="""DEFINITION: 401(k) Retirement Account
A 401(k) is an employer-sponsored retirement account with significant tax advantages, as defined by IRS regulations.

HOW IT WORKS:
1. Money comes out of your paycheck before taxes (Traditional) or after taxes (Roth)
2. Traditional: Contributions reduce your taxable income now; taxed at withdrawal
3. Roth 401(k): Pay taxes now; withdrawals in retirement are tax-free
4. Money grows tax-deferred (no taxes on gains until withdrawal)

2024 CONTRIBUTION LIMITS (per IRS):
- Under 50: $23,000/year
- 50 and older: $30,500/year (extra $7,500 catch-up)
- Employer + employee total: $69,000""",
        chunk_type="definition",
        section="401(k) Definition",
        source="IRS",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.irs.gov/retirement-plans/401k-plans",
        category="accounts",
        difficulty="beginner",
        key_terms=["401k", "retirement", "tax-deferred", "contribution limit", "traditional", "Roth"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("401k", "match", 1),
        document_id="401k",
        content="""401(k) EMPLOYER MATCH: FREE MONEY
Many employers match your contributions. This is FREE MONEY and an instant return.

Common match formulas:
- 50% match up to 6% of salary (you put in 6%, they add 3%)
- 100% match up to 3% (you put in 3%, they add 3%)
- Dollar-for-dollar up to 4%

CRITICAL RULE: ALWAYS CONTRIBUTE AT LEAST ENOUGH TO GET THE FULL MATCH.
A 50% match is an instant 50% return - you won't find that anywhere else!

VESTING:
Your contributions are always 100% yours. Employer contributions may "vest" over time:
- Cliff vesting: 100% after X years (usually 3)
- Graded vesting: Gradually over 6 years""",
        chunk_type="concept",
        section="401(k) Employer Match",
        source="IRS, Fidelity",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.irs.gov/retirement-plans/401k-plans",
        category="accounts",
        difficulty="beginner",
        key_terms=["401k", "employer match", "vesting", "free money"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("ira", "definition", 0),
        document_id="ira",
        content="""DEFINITION: IRA (Individual Retirement Account)
An IRA is a tax-advantaged account you open yourself (not through an employer), as defined by IRS regulations.

2024 CONTRIBUTION LIMITS (per IRS):
- Under 50: $7,000/year
- 50 and older: $8,000/year
- Must have earned income to contribute

TRADITIONAL IRA:
- Contributions may be tax-deductible
- Money grows tax-deferred
- Withdrawals taxed as ordinary income
- Required Minimum Distributions (RMDs) start at age 73

ROTH IRA:
- Contributions are NOT tax-deductible
- Money grows tax-FREE
- Withdrawals in retirement are tax-FREE
- No RMDs during your lifetime
- Can withdraw contributions (not earnings) anytime, penalty-free""",
        chunk_type="definition",
        section="IRA Definition",
        source="IRS",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.irs.gov/retirement-plans/individual-retirement-arrangements-iras",
        category="accounts",
        difficulty="beginner",
        key_terms=["IRA", "Roth IRA", "traditional IRA", "tax-deferred", "tax-free", "RMD"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("ira", "roth_vs_traditional", 1),
        document_id="ira",
        content="""ROTH VS TRADITIONAL IRA: DECISION FRAMEWORK

CHOOSE ROTH IRA IF:
- You're in a lower tax bracket now than you expect in retirement
- You want tax-free withdrawals in retirement
- You want no Required Minimum Distributions
- You're young (more time for tax-free growth)

CHOOSE TRADITIONAL IRA IF:
- You're in a high tax bracket now and expect lower in retirement
- You need the tax deduction now
- You expect lower income in retirement

GENERAL GUIDANCE:
For young investors, Roth is usually better. Tax-free growth over decades is incredibly powerful.

INCOME LIMITS FOR ROTH (2024):
- Single: Can't contribute directly if income > $161,000
- Married: Can't contribute directly if income > $240,000
- "Backdoor Roth" can bypass this limit""",
        chunk_type="procedure",
        section="Roth vs Traditional Decision",
        source="IRS, Fidelity",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.irs.gov/retirement-plans/individual-retirement-arrangements-iras",
        category="accounts",
        difficulty="beginner",
        key_terms=["Roth IRA", "traditional IRA", "tax bracket", "income limits", "backdoor Roth"]
    ))
    
    # =========================================================
    # BEHAVIORAL FINANCE
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("behavioral", "dalbar", 0),
        document_id="behavioral",
        content="""BEHAVIORAL INVESTING: THE DALBAR STUDY
According to Dalbar's Quantitative Analysis of Investor Behavior:

Over a 20-year period:
- S&P 500 returned: 7.5% annually
- Average stock fund investor earned: 2.9% annually

THE GAP: 4.6% annually lost to behavioral mistakes

WHY INVESTORS UNDERPERFORM:
1. Buying high and selling low (chasing performance)
2. Panic selling during downturns
3. Market timing attempts
4. Emotional decision-making
5. Checking portfolios too frequently""",
        chunk_type="concept",
        section="Dalbar Study",
        source="Dalbar Studies, Behavioral Finance Research",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://www.dalbar.com/",
        category="behavioral",
        difficulty="intermediate",
        key_terms=["behavioral finance", "Dalbar", "investor behavior", "underperformance"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("behavioral", "mistakes", 1),
        document_id="behavioral",
        content="""COMMON INVESTOR MISTAKES (Per FINRA/SEC Educational Materials)

1. LOSS AVERSION
- Losses feel 2x worse than equivalent gains feel good
- Result: Hold losers too long, sell winners too soon
- Fix: Set rules in advance, automate decisions

2. RECENCY BIAS
- Overweight recent events in decision-making
- Fix: Look at 50+ year historical data

3. FOMO (Fear of Missing Out)
- Chase hot stocks/sectors AFTER they've already risen
- Fix: Stick to your plan, ignore market noise

4. OVERCONFIDENCE
- "I can beat the market" (90% of professionals can't)
- Fix: Accept you probably can't beat indexes

5. HERD MENTALITY
- Follow the crowd (meme stocks, crypto hype)
- Fix: Be skeptical of "everyone's doing it"

"The investor's chief problem - and even his worst enemy - is likely to be himself." - Benjamin Graham""",
        chunk_type="concept",
        section="Common Investor Mistakes",
        source="FINRA, SEC, Behavioral Finance Research",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.finra.org/investors/investing/investing-basics",
        category="behavioral",
        difficulty="intermediate",
        key_terms=["behavioral finance", "loss aversion", "FOMO", "overconfidence", "herd mentality"]
    ))
    
    # =========================================================
    # DOLLAR COST AVERAGING
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("dca", "definition", 0),
        document_id="dca",
        content="""DEFINITION: Dollar Cost Averaging (DCA)
Dollar cost averaging is investing a fixed amount at regular intervals, regardless of market conditions.

HOW IT WORKS:
$500/month invested in an index fund:
- January: Price $50/share → Buy 10 shares
- February: Price $40/share → Buy 12.5 shares
- March: Price $55/share → Buy 9.1 shares
- Total: $1,500 invested, 31.6 shares
- Average cost: $47.47/share (lower than $48.33 average price)

You automatically buy more shares when prices are low and fewer when high.

WHY DCA WORKS:
1. Removes emotion from investing
2. No need to time the market
3. Automatically buys more when prices are low
4. Builds consistent investing habit
5. Reduces regret if market drops after investing""",
        chunk_type="definition",
        section="Dollar Cost Averaging Definition",
        source="Vanguard Research, Fidelity",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education",
        category="strategies",
        difficulty="beginner",
        key_terms=["dollar cost averaging", "DCA", "systematic investing", "automation"]
    ))
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("dca", "vs_lump_sum", 1),
        document_id="dca",
        content="""DCA VS LUMP SUM INVESTING
According to Vanguard Research:

Lump sum investing wins approximately 67% of the time historically (money in the market longer usually beats waiting).

HOWEVER, DCA advantages:
- Reduces regret if market drops after investing
- Psychologically easier for many investors
- Much better than waiting for "the right time" (which never comes)

KEY INSIGHT: "Time in the market beats timing the market."

THE MATH OF CONSISTENCY:
$500/month for 30 years at 7% average return:
- Total contributed: $180,000
- Ending balance: ~$567,000
- Growth: $387,000 (more than 2x your contributions)""",
        chunk_type="concept",
        section="DCA vs Lump Sum",
        source="Vanguard Research",
        source_tier=SourceTier.TIER_2_INSTITUTIONAL,
        url="https://investor.vanguard.com/investor-resources-education",
        category="strategies",
        difficulty="beginner",
        key_terms=["dollar cost averaging", "lump sum", "market timing"]
    ))
    
    # =========================================================
    # STOCK PICKING REFUSAL
    # =========================================================
    
    CHUNKED_KNOWLEDGE_BASE.append(Chunk(
        id=generate_chunk_id("stock_picking", "warning", 0),
        document_id="stock_picking",
        content="""WHY WE DON'T RECOMMEND INDIVIDUAL STOCKS
Based on SEC and FINRA investor education guidelines:

THE EVIDENCE AGAINST STOCK PICKING:
1. 90%+ of professional fund managers underperform index funds over 15+ years (SPIVA data)
2. Individual investors perform even worse due to behavioral biases
3. A single company can go to zero (Enron, Lehman, etc.)
4. Transaction costs and taxes erode returns
5. Requires significant time and expertise

WHAT HAPPENS TO STOCK PICKERS:
- Individual stocks have extreme outcome distributions
- A few stocks drive most market returns
- Missing just a few of the best performers devastates returns
- Concentration risk can wipe out years of gains

REGULATORY PERSPECTIVE:
FINRA explicitly warns investors about the risks of concentrated stock positions and the difficulty of outperforming diversified index approaches.

RECOMMENDATION: For beginners and most investors, diversified index funds (like VTI or a target-date fund) provide better risk-adjusted returns with far less effort.""",
        chunk_type="concept",
        section="Stock Picking Warning",
        source="SEC, FINRA, SPIVA Research",
        source_tier=SourceTier.TIER_1_REGULATORY,
        url="https://www.finra.org/investors/investing/investment-products/stocks",
        category="risk",
        difficulty="beginner",
        key_terms=["stock picking", "individual stocks", "diversification", "index funds"]
    ))


# Initialize on import
build_chunked_knowledge_base()


def get_all_chunks() -> List[Chunk]:
    """Get all chunks for indexing."""
    return CHUNKED_KNOWLEDGE_BASE


def get_chunks_by_tier(min_tier: SourceTier = SourceTier.TIER_5_GENERAL) -> List[Chunk]:
    """Get chunks at or above a minimum tier."""
    return [c for c in CHUNKED_KNOWLEDGE_BASE if c.source_tier >= min_tier]


def get_chunks_by_category(category: str) -> List[Chunk]:
    """Get chunks by category."""
    return [c for c in CHUNKED_KNOWLEDGE_BASE if c.category == category]


def get_chunk_by_id(chunk_id: str) -> Optional[Chunk]:
    """Get a specific chunk by ID."""
    for chunk in CHUNKED_KNOWLEDGE_BASE:
        if chunk.id == chunk_id:
            return chunk
    return None


# Statistics
def get_knowledge_base_stats() -> Dict:
    """Get statistics about the knowledge base."""
    chunks = CHUNKED_KNOWLEDGE_BASE
    
    tier_counts = {}
    category_counts = {}
    type_counts = {}
    
    for chunk in chunks:
        tier_label = get_tier_label(chunk.source_tier)
        tier_counts[tier_label] = tier_counts.get(tier_label, 0) + 1
        category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
    
    return {
        "total_chunks": len(chunks),
        "by_tier": tier_counts,
        "by_category": category_counts,
        "by_type": type_counts
    }
