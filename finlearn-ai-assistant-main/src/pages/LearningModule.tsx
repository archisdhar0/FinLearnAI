import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { ChatPanel } from "@/components/ChatPanel";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle,
  Circle,
  Search,
  Settings,
} from "lucide-react";

// Module content matching quantcademy-app
const MODULES: Record<string, {
  title: string;
  icon: any;
  lessons: { id: string; title: string; content: string }[];
}> = {
  "foundations": {
    title: "The Foundation",
    icon: BookOpen,
    lessons: [
      {
        id: "what_is_investing",
        title: "What is Investing?",
        content: `
You've probably heard people talk about investing. Your friend who won't stop talking about their portfolio. That coworker who's always checking stock prices. Your parents telling you to "start investing early."

But what does investing actually mean? And why should you care?

Let me put it simply: investing is just using your money to make more money over time. Instead of letting your cash sit in a savings account earning next to nothing, you're putting it to work. You're buying pieces of companies, or lending money to governments, or owning a slice of the entire stock market.

Think of it like this: if you stash $100 under your mattress, you'll still have $100 in ten years (minus whatever the mice eat). But if you invest that $100 and it grows at 7% per year, you'll have about $200. Your money made money while you slept.

---

## The Savings Account Trap

Here's something that might surprise you: keeping all your money in a savings account is actually risky. Not risky like "you might lose it" risky, but risky like "it's definitely going to lose value" risky.

See, there's this thing called inflation. Every year, stuff costs more. A gallon of milk that cost $3 last year might cost $3.10 this year. Your $100 can buy less stuff over time, even though it's still $100.

If inflation is 3% per year (which is pretty normal), that $100 under your mattress will only be worth about $74 in purchasing power after ten years. You didn't lose the money, but you lost what the money can actually buy.

---

## Saving vs Investing: What's the Difference?

**Saving** is like your safety net. It's the money you keep in a regular bank account for emergencies. Your car breaks down? Your savings account has you covered.

**Investing** is for the long game. This is money you're putting away for retirement, or a house down payment in five years, or your kid's college fund. You're okay with this money going up and down in value because you're not touching it for a while.

The key difference? Time. If you need the money in the next year or two, keep it in savings. If you don't need it for five, ten, or thirty years, that's when investing makes sense.

---

## Why Time Matters So Much

The earlier you start investing, the easier it is. Not because you have more money (you probably don't), but because time does most of the heavy lifting.

Let's say you invest $100 when you're 25. By the time you're 65, that $100 could be worth $1,500 or more (assuming 7% average returns). But if you wait until you're 35 to invest that same $100, by 65 it might only be worth $750.

Same $100. Same return rate. But starting ten years earlier means you end up with twice as much money.

That's the power of compounding. Your money makes money, and then that money makes money, and it keeps going.
        `,
      },
      {
        id: "what_youre_buying",
        title: "What You're Actually Buying",
        content: `
When you "invest," you're actually buying something. Not something you can hold in your hands, but something very real. Let's break down the main things investors buy.

---

## Stocks: Owning a Piece of a Company

When you buy a stock, you're buying a tiny piece of a company. If you own one share of Apple, you literally own a small fraction of Apple Inc.

What does that mean practically?
- If Apple makes money, you might get dividends (cash payments to shareholders)
- If Apple's value goes up, your shares are worth more
- If Apple tanks, your shares lose value

Stocks are considered "riskier" because their prices bounce around a lot. But historically, they've also provided the best long-term returns.

---

## Bonds: Lending Your Money

When you buy a bond, you're lending money to someone (usually a company or the government). They promise to pay you back with interest.

Think of it like this: you give the government $1,000. They say "thanks, we'll pay you 4% interest every year, and in 10 years we'll give you your $1,000 back."

Bonds are considered "safer" because you know exactly what you're getting. But the trade-off is lower returns.

---

## Funds: Buying a Basket

Instead of picking individual stocks or bonds, you can buy a fund that holds hundreds or thousands of them.

**Index Funds**: Track a market index like the S&P 500. You're basically buying a little bit of the 500 largest U.S. companies all at once.

**ETFs (Exchange-Traded Funds)**: Similar to index funds, but they trade like stocks throughout the day.

**Mutual Funds**: Professionally managed pools of investments. Usually more expensive.

For most beginners, index funds are the way to go. Low cost, instant diversification, and you don't have to pick winners.
        `,
      },
      {
        id: "how_markets_work",
        title: "How Markets Function",
        content: `
The stock market can seem mysterious, but it's really just a place where buyers and sellers meet to trade.

---

## The Basics

**Stock Exchanges** are like marketplaces. The big ones in the U.S. are:
- NYSE (New York Stock Exchange)
- NASDAQ

When you buy a stock, you're buying it from someone who wants to sell. When you sell, someone else is buying from you.

**Market Hours**: U.S. markets are open 9:30 AM - 4:00 PM Eastern, Monday through Friday.

---

## Supply and Demand

Stock prices work just like any other price: supply and demand.

- More buyers than sellers → price goes up
- More sellers than buyers → price goes down

That's it. All the complicated-sounding stuff (P/E ratios, earnings reports, Fed announcements) just affects how many people want to buy or sell.

---

## What Moves Prices?

- **Company news**: Good earnings? Stock goes up. Scandal? Stock goes down.
- **Economic data**: Jobs report, inflation numbers, GDP growth
- **Interest rates**: When the Fed raises rates, stocks often fall
- **Sentiment**: Sometimes the market just gets scared or excited

The key insight: in the short term, prices are driven by emotions and news. In the long term, prices follow actual business performance.

---

## You Don't Need to Predict the Market

Here's the good news: you don't need to understand every market move. Most successful long-term investors just buy regularly and hold for years. They don't try to time the market.
        `,
      },
      {
        id: "time_and_compounding",
        title: "Time and Compounding",
        content: `
Compound interest is the most powerful force in investing. Einstein supposedly called it the "eighth wonder of the world." Whether he actually said that or not, the math is real.

---

## How Compounding Works

Simple interest: You earn interest only on your original investment.
Compound interest: You earn interest on your investment PLUS all the interest you've already earned.

**Example:**
- Year 1: You invest $1,000, earn 10% = $1,100
- Year 2: You earn 10% on $1,100 = $1,210
- Year 3: You earn 10% on $1,210 = $1,331

See how each year you're earning more? That's compounding.

---

## The Rule of 72

Want a quick way to estimate how long it takes to double your money? Divide 72 by your interest rate.

- At 7% return: 72 ÷ 7 = about 10 years to double
- At 10% return: 72 ÷ 10 = about 7 years to double

---

## Why Starting Early Matters

Let's compare two people:

**Sarah** starts investing $200/month at age 25, stops at 35 (10 years, $24,000 total invested)

**Mike** starts investing $200/month at age 35, continues until 65 (30 years, $72,000 total invested)

At 7% returns, by age 65:
- Sarah has about $240,000
- Mike has about $240,000

Sarah invested 1/3 the money but ended up with the same amount. That's the power of starting early.

---

## The Takeaway

You can't control the market. You can't control interest rates. But you CAN control when you start. And starting now, even with small amounts, gives you the biggest advantage of all: time.
        `,
      },
      {
        id: "basics_of_risk",
        title: "The Basics of Risk",
        content: `
Risk in investing doesn't mean "chance of losing everything." It means "how much your investments might go up or down in value."

---

## Risk and Reward Are Linked

Higher potential returns = higher risk. Lower risk = lower potential returns.

- **Savings account**: Very low risk, very low return (0.5%)
- **Bonds**: Low risk, low-moderate return (3-5%)
- **Stocks**: Higher risk, higher return (7-10% historically)
- **Individual stocks**: Highest risk, highest potential return (or loss)

There's no free lunch. Anyone promising high returns with no risk is lying.

---

## Types of Risk

**Market Risk**: The whole market drops (like in 2008 or 2020). Even good companies fall.

**Company Risk**: One specific company fails (like Enron or Blockbuster). This is why you diversify.

**Inflation Risk**: Your returns don't keep up with rising prices. This is the hidden risk of keeping everything in cash.

---

## How to Manage Risk

1. **Diversify**: Don't put all your eggs in one basket. Own many different investments.

2. **Match risk to timeline**: 
   - Money you need soon → low risk (savings, bonds)
   - Money you won't touch for decades → can handle more risk (stocks)

3. **Don't panic sell**: Market drops are normal. Selling during a crash locks in losses.

---

## Your Risk Tolerance

Everyone's different. Some people can watch their portfolio drop 30% and sleep fine. Others panic at a 5% dip.

Be honest with yourself. If you can't handle volatility, it's okay to invest more conservatively. A boring portfolio you stick with beats an aggressive one you abandon.
        `,
      },
      {
        id: "accounts_and_setup",
        title: "Accounts and Setup",
        content: `
Before you can invest, you need somewhere to do it. Here's a quick guide to the accounts you'll encounter.

---

## Brokerage Accounts

A brokerage account is where you buy and sell investments. Think of it like a bank account, but for stocks and funds.

**Popular brokerages**:
- Fidelity
- Vanguard
- Charles Schwab
- Robinhood

Most have no minimums and no trading fees now. Pick one with a good app and customer service.

---

## Retirement Accounts

These have special tax advantages but restrictions on when you can withdraw.

**401(k)**: Offered through employers
- Pre-tax contributions (reduces your taxable income)
- Often includes employer match (free money!)
- Can't withdraw before 59½ without penalty

**IRA (Individual Retirement Account)**: You open this yourself
- Traditional IRA: Tax deduction now, pay taxes later
- Roth IRA: Pay taxes now, withdrawals are tax-free in retirement

---

## Which Account First?

1. **401(k) up to employer match**: If your employer matches contributions, always get the full match. It's free money.

2. **Emergency fund**: Keep 3-6 months of expenses in savings.

3. **Roth IRA**: After the match, a Roth IRA is great for most people.

4. **More 401(k)**: Max it out if you can.

5. **Taxable brokerage**: After retirement accounts are maxed.

---

## Getting Started

Opening an account takes about 15 minutes online. You'll need:
- Social Security number
- Bank account for transfers
- Basic personal info

Start with whatever amount you're comfortable with. Even $50/month is a great start.
        `,
      },
      {
        id: "first_time_mindset",
        title: "First Time Investor Mindset",
        content: `
The hardest part of investing isn't picking stocks or understanding markets. It's managing your own psychology.

---

## Common Beginner Mistakes

**1. Waiting for the "right time"**
There's never a perfect time to start. The market might drop after you invest. It might also go up. No one knows. The best time to start was yesterday. The second best time is today.

**2. Checking too often**
Looking at your portfolio daily is a recipe for anxiety. The market goes up and down constantly. If you're investing for 20+ years, daily movements don't matter.

**3. Panic selling**
When the market drops, your instinct screams "sell everything!" This is exactly wrong. Drops are when stocks are on sale. Selling locks in losses.

**4. Chasing hot tips**
Your cousin's crypto pick. That Reddit stock. The "next big thing." Most hot tips lose money. Boring, diversified investing wins.

---

## The Right Mindset

**Think in decades, not days**. You're building wealth over 20, 30, 40 years. A bad week or month is noise.

**Automate everything**. Set up automatic transfers so you invest without thinking about it.

**Ignore the news**. Financial media needs drama to get clicks. "Markets do normal thing" doesn't make headlines.

**Stay the course**. The investors who do best are often the ones who do the least. Buy, hold, don't panic.

---

## You've Got This

Millions of regular people build wealth through investing. You don't need to be a genius or have a finance degree. You just need to start, stay consistent, and give it time.

The fact that you're learning this stuff puts you ahead of most people. Now it's time to take action.
        `,
      },
    ],
  },
  "investor-insight": {
    title: "Investor Insight",
    icon: Search,
    lessons: [
      {
        id: "market_psychology",
        title: "Market Psychology",
        content: `
Markets are driven by millions of human decisions, which means they're driven by human emotions. Understanding psychology helps you avoid common traps.

---

## Fear and Greed

The market swings between two emotions:

**Greed**: When prices are rising, everyone wants in. People buy more, pushing prices higher. Eventually, prices get disconnected from reality.

**Fear**: When prices fall, everyone panics. People sell, pushing prices lower. Eventually, good companies become bargains.

Warren Buffett's famous advice: "Be fearful when others are greedy, and greedy when others are fearful."

---

## Common Psychological Traps

**Loss Aversion**: Losses hurt about twice as much as gains feel good. This makes people hold losing investments too long (hoping to break even) and sell winners too early (locking in gains).

**Recency Bias**: We assume recent trends will continue. If the market's been going up, we think it'll keep going up. If it's been falling, we think it'll keep falling.

**Herd Mentality**: It feels safer to do what everyone else is doing. But by the time "everyone" is buying something, it's usually too late.

---

## How to Beat Your Brain

1. **Have a plan before you invest**. Decide when you'll buy and sell before emotions take over.

2. **Automate**: Regular automatic investments remove emotion from the equation.

3. **Zoom out**: Look at 10-year charts, not daily moves.

4. **Remember history**: Every crash has eventually recovered. Every bubble has eventually popped.
        `,
      },
      {
        id: "economic_indicators",
        title: "Economic Indicators",
        content: `
Economic indicators help you understand the broader environment your investments exist in. You don't need to obsess over them, but knowing the basics helps.

---

## Key Indicators

**GDP (Gross Domestic Product)**: The total value of goods and services produced. Growing GDP = healthy economy. Two consecutive quarters of decline = recession.

**Unemployment Rate**: Percentage of people looking for work who can't find it. Low unemployment = strong economy, but can also mean inflation pressure.

**Inflation (CPI)**: How fast prices are rising. Some inflation (2-3%) is normal. High inflation erodes your purchasing power.

**Interest Rates**: Set by the Federal Reserve. Higher rates make borrowing expensive, which slows the economy and often hurts stocks. Lower rates stimulate growth.

---

## How They Affect Investments

**Stocks**: Generally do well when the economy is growing and interest rates are low.

**Bonds**: Do well when interest rates fall, poorly when rates rise.

**Cash**: Loses value during high inflation.

---

## Don't Overthink It

Economic indicators matter, but they're also already priced into the market by the time you hear about them. 

For long-term investors, trying to time investments based on economic data usually backfires. Consistent investing through all conditions tends to work better.
        `,
      },
      {
        id: "valuation_basics",
        title: "Valuation Basics",
        content: `
How do you know if a stock is cheap or expensive? Valuation metrics help, but they're not perfect.

---

## P/E Ratio (Price to Earnings)

The most common metric. Stock price divided by earnings per share.

- P/E of 15 means you're paying $15 for every $1 of earnings
- Lower P/E = potentially cheaper
- Higher P/E = potentially expensive (or high growth expected)

**S&P 500 average P/E**: Around 15-17 historically.

---

## Other Useful Metrics

**P/B (Price to Book)**: Price vs. company's assets. Useful for banks and asset-heavy companies.

**Dividend Yield**: Annual dividends / stock price. Higher yield = more income.

**PEG Ratio**: P/E divided by growth rate. Accounts for how fast the company is growing.

---

## The Limits of Valuation

A "cheap" stock can get cheaper. An "expensive" stock can keep rising.

Amazon looked expensive for years (high P/E) but kept growing and rewarded investors. Value stocks sometimes stay cheap forever.

Valuation is one input, not the whole answer. A company's quality, competitive position, and growth prospects matter too.

---

## For Index Fund Investors

If you're buying the whole market through index funds, you don't need to worry much about individual stock valuation. You're buying everything at market prices.
        `,
      },
      {
        id: "news_and_noise",
        title: "News vs. Noise",
        content: `
Financial media exists to get your attention. That doesn't mean it's useful for making investment decisions.

---

## The Problem with Financial News

**It's designed to create urgency**: "Markets PLUNGE!" "This stock is about to EXPLODE!" Calm, long-term thinking doesn't get clicks.

**It's backward-looking**: By the time news breaks, the market has usually already reacted.

**It changes constantly**: Monday's "buy" recommendation becomes Wednesday's "sell." Analysts are often wrong.

---

## What Actually Matters

For long-term investors, very little news matters. What matters:
- Your time horizon
- Your asset allocation
- Your savings rate
- Your costs (fees)

A company's quarterly earnings? Noise. The Fed's latest statement? Usually noise. Political drama? Definitely noise.

---

## How to Filter

**Ignore daily market commentary**. The market went up 0.5% today because... who cares?

**Be skeptical of predictions**. No one consistently predicts the market. No one.

**Focus on your plan**. Are you saving enough? Are your costs low? Is your allocation right for your goals?

---

## The Paradox

The investors who pay the least attention to news often do the best. They buy, hold, and let time work. They're not smarter—they're just not distracted.
        `,
      },
      {
        id: "behavioral_biases",
        title: "Behavioral Biases",
        content: `
Your brain evolved to keep you alive on the savannah, not to make good investment decisions. Understanding your biases helps you work around them.

---

## Confirmation Bias

We seek out information that confirms what we already believe and ignore information that contradicts it.

**In investing**: You buy a stock, then only read positive news about it. You dismiss warning signs because you want to be right.

**Fix**: Actively seek out opposing viewpoints. Ask "what could go wrong?"

---

## Overconfidence

We think we're better than average. 90% of drivers think they're above average. This is mathematically impossible.

**In investing**: We think we can pick winners, time the market, or beat the pros. Most can't.

**Fix**: Humility. Accept that you probably can't beat the market consistently. Index funds exist for a reason.

---

## Anchoring

We fixate on the first number we see and judge everything relative to it.

**In investing**: "I bought at $100, so I won't sell until it gets back to $100." The stock doesn't know or care what you paid.

**Fix**: Evaluate investments based on current value and future prospects, not your purchase price.

---

## The Endowment Effect

We value things more just because we own them.

**In investing**: We hold onto stocks we own even when we wouldn't buy them today.

**Fix**: Ask yourself: "If I had cash instead of this investment, would I buy it right now?"
        `,
      },
    ],
  },
  "applied-investing": {
    title: "Applied Investing",
    icon: Settings,
    lessons: [
      {
        id: "asset_allocation",
        title: "Asset Allocation",
        content: `
Asset allocation—how you divide your money between stocks, bonds, and other assets—is the most important investment decision you'll make.

---

## Why It Matters

Studies show that asset allocation explains about 90% of portfolio returns over time. Stock picking and market timing? Much less important.

Your allocation determines:
- How much your portfolio might grow
- How much it might drop in a bad year
- How bumpy the ride will be

---

## The Basic Framework

**Stocks**: Higher growth, more volatility. Good for long time horizons.

**Bonds**: Lower growth, more stability. Good for shorter horizons or reducing risk.

**Cash**: Safest, but loses to inflation. Keep enough for emergencies.

---

## Common Allocations

**Aggressive (80/20 stocks/bonds)**: For young investors with 20+ years.

**Moderate (60/40)**: Classic balanced portfolio. Less volatility.

**Conservative (40/60)**: For those near retirement or risk-averse.

---

## The Simple Approach

A common rule of thumb: Your age in bonds, the rest in stocks.
- Age 30: 30% bonds, 70% stocks
- Age 50: 50% bonds, 50% stocks

It's not perfect, but it's a reasonable starting point.

---

## Target Date Funds

Don't want to think about allocation? Target date funds do it for you. Pick the fund closest to your retirement year (like "Target 2055") and it automatically adjusts over time.
        `,
      },
      {
        id: "rebalancing",
        title: "Rebalancing",
        content: `
Over time, your portfolio drifts from your target allocation. Rebalancing brings it back.

---

## Why Portfolios Drift

Say you start with 70% stocks, 30% bonds. After a great year for stocks:
- Stocks grow to 80% of your portfolio
- Bonds shrink to 20%

Now you have more risk than you planned.

---

## How to Rebalance

**Option 1: Sell high, buy low**
Sell some of what's grown (stocks) and buy more of what's lagged (bonds).

**Option 2: Direct new money**
Put new contributions into whatever's underweight.

---

## When to Rebalance

**Calendar-based**: Check once or twice a year (January and July, for example).

**Threshold-based**: Rebalance when any asset class drifts more than 5% from target.

Don't rebalance too often. You'll rack up taxes and fees.

---

## The Psychological Benefit

Rebalancing forces you to sell winners and buy losers—the opposite of what your emotions want. This discipline often improves returns.
        `,
      },
      {
        id: "tax_efficiency",
        title: "Tax-Efficient Investing",
        content: `
Taxes can take a big bite out of your returns. Smart tax planning keeps more money working for you.

---

## Tax-Advantaged Accounts

**401(k) / Traditional IRA**: Contributions reduce taxable income now. Pay taxes when you withdraw in retirement.

**Roth IRA / Roth 401(k)**: Pay taxes now. Withdrawals in retirement are tax-free.

**HSA (Health Savings Account)**: Triple tax advantage if used for medical expenses.

---

## Asset Location

Put tax-inefficient investments in tax-advantaged accounts:
- **In 401(k)/IRA**: Bonds, REITs (generate taxable income)
- **In taxable accounts**: Stock index funds (more tax-efficient)

---

## Tax-Loss Harvesting

If an investment drops below what you paid, you can sell it to realize a loss. That loss offsets gains elsewhere, reducing your tax bill.

Then you can buy a similar (but not identical) investment to maintain your allocation.

---

## Hold for Long-Term Gains

Investments held over a year qualify for lower long-term capital gains rates (0%, 15%, or 20% depending on income) vs. short-term rates (taxed as regular income).

---

## Don't Let Taxes Drive Everything

Tax efficiency matters, but don't let it override good investment decisions. A tax-efficient bad investment is still a bad investment.
        `,
      },
      {
        id: "building_portfolio",
        title: "Building Your Portfolio",
        content: `
Let's put it all together. Here's how to actually build a portfolio.

---

## The Simple 3-Fund Portfolio

Many investors use just three funds:

1. **U.S. Total Stock Market Index Fund** (e.g., VTI, FSKAX)
2. **International Stock Index Fund** (e.g., VXUS, FTIHX)
3. **U.S. Bond Index Fund** (e.g., BND, FXNAX)

That's it. You own the entire global stock market and a diversified bond portfolio.

---

## Sample Allocations

**Young investor (20s-30s)**:
- 60% U.S. stocks
- 30% International stocks
- 10% Bonds

**Mid-career (40s-50s)**:
- 50% U.S. stocks
- 20% International stocks
- 30% Bonds

**Near retirement (60s)**:
- 35% U.S. stocks
- 15% International stocks
- 50% Bonds

---

## The Even Simpler Option

Just buy a target date fund. One fund, done. It holds a mix of stocks and bonds appropriate for your timeline and adjusts automatically.

---

## What NOT to Do

- Don't buy individual stocks until you have a solid foundation
- Don't chase last year's hot fund
- Don't pay high fees (keep expense ratios under 0.2%)
- Don't try to time the market
        `,
      },
      {
        id: "staying_the_course",
        title: "Staying the Course",
        content: `
The hardest part of investing isn't getting started. It's staying invested when things get scary.

---

## Markets Will Drop

Since 1950, the S&P 500 has dropped:
- 10% or more: about once a year
- 20% or more: about every 3-4 years
- 30% or more: about every 10 years

Drops are normal. They're the price of admission for long-term returns.

---

## The Danger of Panic Selling

The worst thing you can do is sell during a crash. You lock in losses and miss the recovery.

From 2000-2020, if you missed just the 10 best days in the market, your returns would be cut in half. Most of those best days came right after the worst days.

---

## How to Stay the Course

**1. Have a plan**: Write down your strategy. When you're tempted to sell, read it.

**2. Automate**: Keep investing automatically, even during downturns.

**3. Don't look**: Seriously. Check your portfolio quarterly at most.

**4. Remember history**: Every crash has recovered. Every single one.

**5. Talk to someone**: A financial advisor or even a trusted friend can talk you off the ledge.

---

## The Long View

If you invested $10,000 in the S&P 500 in 1980 and just left it alone, you'd have over $1,000,000 today. Through crashes, recessions, wars, and pandemics.

The secret? Doing nothing. Just staying invested.

That's the real skill in investing. Not picking stocks. Not timing markets. Just having the patience to let time work.
        `,
      },
      {
        id: "common_mistakes",
        title: "Common Mistakes to Avoid",
        content: `
Learn from others' mistakes so you don't have to make them yourself.

---

## Mistake #1: Not Starting

The biggest mistake is never starting at all. Waiting for the "right time" or "more money" costs you years of compounding.

Start with whatever you have. $50/month is fine. Just start.

---

## Mistake #2: Trying to Time the Market

"I'll wait for the dip." "I'll sell before the crash." No one does this consistently. Not professionals, not algorithms, not your uncle who "called" the last crash.

Time IN the market beats timing the market.

---

## Mistake #3: Chasing Performance

Last year's best fund is often this year's worst. Investors who chase hot performance usually buy high and sell low.

Stick with your plan regardless of what's "hot."

---

## Mistake #4: Paying High Fees

A 1% fee doesn't sound like much, but over 30 years it can cost you hundreds of thousands of dollars.

Keep expense ratios under 0.2%. Index funds make this easy.

---

## Mistake #5: Checking Too Often

Looking at your portfolio daily creates anxiety and tempts you to tinker. Every change is an opportunity to make a mistake.

Check quarterly. Rebalance annually. Otherwise, leave it alone.

---

## Mistake #6: Not Diversifying

Putting everything in one stock, one sector, or one country is gambling, not investing.

Own the whole market through index funds. Let diversification protect you.

---

## The Pattern

Notice something? Most mistakes come from DOING too much:
- Trading too much
- Checking too much
- Reacting too much

Successful investing is mostly about what you DON'T do.
        `,
      },
    ],
  },
};

export default function LearningModule() {
  const navigate = useNavigate();
  const { moduleId } = useParams();
  const [currentLessonIndex, setCurrentLessonIndex] = useState(0);
  const [completedLessons, setCompletedLessons] = useState<string[]>([]);

  const module = moduleId ? MODULES[moduleId] : null;

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) navigate("/");
    });
  }, [navigate]);

  if (!module) {
    return (
      <div className="min-h-screen flex items-center justify-center flex-col gap-4">
        <p className="text-xl">Module not found</p>
        <button
          onClick={() => navigate("/dashboard")}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg"
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

  const currentLesson = module.lessons[currentLessonIndex];
  const Icon = module.icon;

  const markComplete = () => {
    if (!completedLessons.includes(currentLesson.id)) {
      setCompletedLessons([...completedLessons, currentLesson.id]);
    }
    if (currentLessonIndex < module.lessons.length - 1) {
      setCurrentLessonIndex(currentLessonIndex + 1);
    }
  };

  // Simple markdown renderer
  const renderContent = (content: string) => {
    return content.split("\n").map((line, i) => {
      const trimmed = line.trim();
      
      if (trimmed.startsWith("## ")) {
        return <h2 key={i} className="text-2xl font-bold mt-8 mb-4 text-foreground">{trimmed.replace("## ", "")}</h2>;
      }
      if (trimmed.startsWith("### ")) {
        return <h3 key={i} className="text-xl font-semibold mt-6 mb-3 text-foreground">{trimmed.replace("### ", "")}</h3>;
      }
      if (trimmed.startsWith("---")) {
        return <hr key={i} className="my-6 border-border" />;
      }
      if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
        return <p key={i} className="font-bold mt-4 text-foreground">{trimmed.replace(/\*\*/g, "")}</p>;
      }
      if (trimmed.startsWith("- ")) {
        return <li key={i} className="ml-6 text-muted-foreground">{trimmed.replace("- ", "")}</li>;
      }
      if (trimmed.match(/^\d+\./)) {
        return <li key={i} className="ml-6 text-muted-foreground list-decimal">{trimmed.replace(/^\d+\.\s*/, "")}</li>;
      }
      if (trimmed) {
        // Handle inline bold
        const parts = trimmed.split(/(\*\*[^*]+\*\*)/g);
        return (
          <p key={i} className="text-muted-foreground leading-relaxed my-2">
            {parts.map((part, j) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                return <strong key={j} className="text-foreground">{part.replace(/\*\*/g, "")}</strong>;
              }
              return part;
            })}
          </p>
        );
      }
      return null;
    });
  };

  return (
    <div className="flex min-h-screen">
      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-md">
          <div className="max-w-4xl mx-auto px-6 h-14 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/dashboard")}
                className="p-2 rounded-lg hover:bg-muted transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-primary/10">
                  <Icon className="w-4 h-4 text-primary" />
                </div>
                <span className="font-display text-sm font-semibold">{module.title}</span>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              Lesson {currentLessonIndex + 1} of {module.lessons.length}
            </div>
          </div>
        </header>

        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Lesson Navigation */}
          <div className="flex gap-2 mb-8 overflow-x-auto pb-2">
            {module.lessons.map((lesson, index) => (
              <button
                key={lesson.id}
                onClick={() => setCurrentLessonIndex(index)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
                  index === currentLessonIndex
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted hover:bg-muted/80"
                }`}
              >
                {completedLessons.includes(lesson.id) ? (
                  <CheckCircle className="w-4 h-4 text-success" />
                ) : (
                  <Circle className="w-4 h-4" />
                )}
                {lesson.title}
              </button>
            ))}
          </div>

          {/* Lesson Content */}
          <div className="glass-card rounded-xl p-8">
            <h1 className="font-display text-3xl font-bold mb-6">{currentLesson.title}</h1>
            <div className="prose prose-invert max-w-none">
              {renderContent(currentLesson.content)}
            </div>
          </div>

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8">
            <button
              onClick={() => setCurrentLessonIndex(Math.max(0, currentLessonIndex - 1))}
              disabled={currentLessonIndex === 0}
              className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg text-sm hover:bg-muted/80 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowLeft className="w-4 h-4" />
              Previous
            </button>
            <button
              onClick={markComplete}
              className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground rounded-lg text-sm hover:opacity-90"
            >
              {currentLessonIndex === module.lessons.length - 1 ? (
                <>
                  <CheckCircle className="w-4 h-4" />
                  Complete Module
                </>
              ) : (
                <>
                  Mark Complete & Continue
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* AI Chat Panel - Always visible during lessons */}
      <div className="hidden lg:flex w-[380px] flex-shrink-0 h-screen sticky top-0">
        <ChatPanel />
      </div>
    </div>
  );
}
