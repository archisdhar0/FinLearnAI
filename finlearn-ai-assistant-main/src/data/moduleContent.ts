// Complete module content with quizzes and interactive elements
// Extracted from quantcademy-app/pages/learning_modules.py

export interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation?: string;
}

export interface Lesson {
  id: string;
  title: string;
  content: string;
  quiz?: QuizQuestion[];
  interactiveElements?: string[]; // Component names to render
}

export interface Module {
  id: string;
  title: string;
  description: string;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  lessons: Lesson[];
  finalQuiz: QuizQuestion[];
}

export const MODULES: Record<string, Module> = {
  "foundations": {
    id: "foundations",
    title: "The Foundation",
    description: "Core concepts every first-time investor should know.",
    level: "Beginner",
    lessons: [
      {
        id: "what_is_investing",
        title: "What is Investing?",
        content: `
You've probably heard people talk about investing. Your friend who won't stop talking about their portfolio. That coworker who's always checking stock prices. Your parents telling you to "start investing early."

But what does investing actually mean? And why should you care?

Let me put it simply: **investing is just using your money to make more money over time**. Instead of letting your cash sit in a savings account earning next to nothing, you're putting it to work. You're buying pieces of companies, or lending money to governments, or owning a slice of the entire stock market.

Think of it like this: if you stash $100 under your mattress, you'll still have $100 in ten years (minus whatever the mice eat). But if you invest that $100 and it grows at 7% per year, you'll have about $200. Your money made money while you slept.

---

## The Savings Account Trap

Here's something that might surprise you: keeping all your money in a savings account is actually risky. Not risky like "you might lose it" risky, but risky like "it's definitely going to lose value" risky.

See, there's this thing called **inflation**. Every year, stuff costs more. A gallon of milk that cost $3 last year might cost $3.10 this year. Your $100 can buy less stuff over time, even though it's still $100.

If inflation is 3% per year (which is pretty normal), that $100 under your mattress will only be worth about $74 in purchasing power after ten years. You didn't lose the money, but you lost what the money can actually buy.

---

## Saving vs Investing: What's the Difference?

**Saving** is like your safety net. It's the money you keep in a regular bank account for emergencies. Your car breaks down? Your savings account has you covered.

**Investing** is for the long game. This is money you're putting away for retirement, or a house down payment in five years, or your kid's college fund. You're okay with this money going up and down in value because you're not touching it for a while.

The key difference? **Time**. If you need the money in the next year or two, keep it in savings. If you don't need it for five, ten, or thirty years, that's when investing makes sense.

---

## Why Time Matters So Much

The earlier you start investing, the easier it is. Not because you have more money (you probably don't), but because time does most of the heavy lifting.

Let's say you invest $100 when you're 25. By the time you're 65, that $100 could be worth $1,500 or more (assuming 7% average returns). But if you wait until you're 35 to invest that same $100, by 65 it might only be worth $750.

Same $100. Same return rate. But starting ten years earlier means you end up with **twice as much money**.

That's the power of compounding. Your money makes money, and then that money makes money, and it keeps going.
        `,
        interactiveElements: ['InflationCalculator'],
        quiz: [
          {
            question: "What's the main difference between saving and investing?",
            options: [
              "Saving is riskier than investing",
              "Investing is for long-term goals, saving is for short-term needs",
              "They're the same thing",
              "Only rich people can invest"
            ],
            correct: 1,
            explanation: "Investing is for money you won't need for years, while saving is for emergencies and short-term goals."
          },
          {
            question: "Why does starting to invest early matter so much?",
            options: [
              "You have more money when you're young",
              "Time allows compound interest to work its magic",
              "Stocks always go up when you're young",
              "It doesn't really matter"
            ],
            correct: 1,
            explanation: "Starting early gives your money more time to compound, which can double or triple your returns over decades."
          },
          {
            question: "What is inflation's effect on cash savings?",
            options: [
              "It makes your money worth more over time",
              "It has no effect on savings",
              "It reduces the purchasing power of your money",
              "It only affects stocks"
            ],
            correct: 2,
            explanation: "Inflation means prices rise over time, so the same amount of money buys less in the future."
          }
        ]
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
- If Apple makes money, you might get **dividends** (cash payments to shareholders)
- If Apple's value goes up, your shares are worth more
- If Apple tanks, your shares lose value

Stocks are considered "riskier" because their prices bounce around a lot. But historically, they've also provided the **best long-term returns**.

---

## Bonds: Lending Your Money

When you buy a bond, you're lending money to someone (usually a company or the government). They promise to pay you back with interest.

Think of it like this: you give the government $1,000. They say "thanks, we'll pay you 4% interest every year, and in 10 years we'll give you your $1,000 back."

Bonds are considered "safer" because you know exactly what you're getting. But the trade-off is **lower returns**.

---

## Funds: Buying a Basket

Instead of picking individual stocks or bonds, you can buy a fund that holds hundreds or thousands of them.

**Index Funds**: Track a market index like the S&P 500. You're basically buying a little bit of the 500 largest U.S. companies all at once.

**ETFs (Exchange-Traded Funds)**: Similar to index funds, but they trade like stocks throughout the day.

**Mutual Funds**: Professionally managed pools of investments. Usually more expensive.

For most beginners, **index funds are the way to go**. Low cost, instant diversification, and you don't have to pick winners.
        `,
        quiz: [
          {
            question: "When you buy a stock, what are you actually buying?",
            options: [
              "A loan to a company",
              "A small ownership stake in a company",
              "A guarantee of future profits",
              "A government bond"
            ],
            correct: 1,
            explanation: "Stocks represent partial ownership of a company. You become a shareholder."
          },
          {
            question: "What's the main advantage of index funds for beginners?",
            options: [
              "They guarantee profits",
              "They're managed by experts who pick winners",
              "They provide instant diversification at low cost",
              "They have no risk"
            ],
            correct: 2,
            explanation: "Index funds give you exposure to hundreds of companies at once, reducing risk through diversification."
          }
        ]
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

Stock prices work just like any other price: **supply and demand**.

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
        quiz: [
          {
            question: "What primarily determines stock prices?",
            options: [
              "The government sets prices",
              "Supply and demand from buyers and sellers",
              "The company's CEO decides",
              "Random chance"
            ],
            correct: 1,
            explanation: "Stock prices are determined by supply and demand - more buyers push prices up, more sellers push prices down."
          }
        ]
      },
      {
        id: "time_and_compounding",
        title: "Time and Compounding",
        content: `
Compound interest is the most powerful force in investing. Einstein supposedly called it the "eighth wonder of the world." Whether he actually said that or not, the math is real.

---

## How Compounding Works

**Simple interest**: You earn interest only on your original investment.
**Compound interest**: You earn interest on your investment PLUS all the interest you've already earned.

Example:
- Year 1: You invest $1,000, earn 10% = $1,100
- Year 2: You earn 10% on $1,100 = $1,210
- Year 3: You earn 10% on $1,210 = $1,331

See how each year you're earning more? That's compounding.

---

## The Rule of 72

Want a quick way to estimate how long it takes to double your money? Divide 72 by your interest rate.

- At 7% return: 72 ÷ 7 = about **10 years to double**
- At 10% return: 72 ÷ 10 = about **7 years to double**

---

## Why Starting Early Matters

Let's compare two people:

**Sarah** starts investing $200/month at age 25, stops at 35 (10 years, $24,000 total invested)

**Mike** starts investing $200/month at age 35, continues until 65 (30 years, $72,000 total invested)

At 7% returns, by age 65:
- Sarah has about $240,000
- Mike has about $240,000

Sarah invested **1/3 the money** but ended up with the same amount. That's the power of starting early.
        `,
        interactiveElements: ['CompoundCalculator'],
        quiz: [
          {
            question: "Using the Rule of 72, how long does it take to double your money at 6% annual return?",
            options: [
              "6 years",
              "12 years",
              "18 years",
              "72 years"
            ],
            correct: 1,
            explanation: "72 ÷ 6 = 12 years. The Rule of 72 gives you a quick estimate of doubling time."
          },
          {
            question: "What makes compound interest so powerful?",
            options: [
              "You earn interest only on your original investment",
              "You earn interest on your interest, creating exponential growth",
              "The government adds extra money",
              "It only works for rich people"
            ],
            correct: 1,
            explanation: "Compound interest means your earnings generate their own earnings, leading to exponential growth over time."
          }
        ]
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
        interactiveElements: ['RiskReturnChart', 'DiversificationDemo'],
        quiz: [
          {
            question: "What does 'risk' mean in investing?",
            options: [
              "The chance of losing all your money",
              "How much your investments might go up or down in value",
              "The fees you pay",
              "How complicated the investment is"
            ],
            correct: 1,
            explanation: "Risk refers to volatility - how much the value of your investment fluctuates over time."
          },
          {
            question: "Why is diversification important?",
            options: [
              "It guarantees profits",
              "It reduces the impact of any single investment failing",
              "It makes investing more exciting",
              "It's required by law"
            ],
            correct: 1,
            explanation: "Diversification spreads your money across many investments, so one failure doesn't devastate your portfolio."
          }
        ]
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
        `,
        quiz: [
          {
            question: "What's the first investment priority if your employer offers a 401(k) match?",
            options: [
              "Max out a Roth IRA first",
              "Contribute enough to get the full employer match",
              "Buy individual stocks",
              "Pay off all debt first"
            ],
            correct: 1,
            explanation: "An employer match is essentially free money - you should always capture the full match before other investments."
          }
        ]
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
        `,
        quiz: [
          {
            question: "What's the biggest mistake new investors make during market drops?",
            options: [
              "Buying more at lower prices",
              "Panic selling and locking in losses",
              "Staying invested",
              "Checking their portfolio"
            ],
            correct: 1,
            explanation: "Panic selling during drops locks in your losses and means you miss the recovery. Stay the course."
          },
          {
            question: "How often should long-term investors check their portfolio?",
            options: [
              "Multiple times per day",
              "Daily",
              "Weekly",
              "Quarterly or less"
            ],
            correct: 3,
            explanation: "Checking too often leads to anxiety and poor decisions. Long-term investors benefit from checking quarterly or less."
          }
        ]
      }
    ],
    finalQuiz: [
      {
        question: "What is the primary benefit of compound interest?",
        options: [
          "It provides guaranteed returns",
          "You earn interest on your interest, creating exponential growth",
          "It eliminates all investment risk",
          "It allows you to withdraw money anytime"
        ],
        correct: 1,
        explanation: "Compound interest means your earnings generate their own earnings over time."
      },
      {
        question: "Which investment typically offers the highest long-term returns but also the highest volatility?",
        options: [
          "Savings accounts",
          "Government bonds",
          "Stocks",
          "CDs (Certificates of Deposit)"
        ],
        correct: 2,
        explanation: "Stocks historically provide the best long-term returns but with more short-term volatility."
      },
      {
        question: "What is diversification?",
        options: [
          "Putting all your money in one high-performing stock",
          "Spreading investments across many different assets to reduce risk",
          "Only investing in bonds",
          "Timing the market perfectly"
        ],
        correct: 1,
        explanation: "Diversification means not putting all your eggs in one basket."
      },
      {
        question: "Using the Rule of 72, approximately how long does it take to double your money at 8% annual return?",
        options: [
          "5 years",
          "9 years",
          "12 years",
          "15 years"
        ],
        correct: 1,
        explanation: "72 ÷ 8 = 9 years to double your money."
      },
      {
        question: "What's the recommended order for investing priorities?",
        options: [
          "Individual stocks → 401(k) → Savings",
          "401(k) match → Emergency fund → Roth IRA → More 401(k)",
          "Crypto → Bonds → Stocks",
          "Savings only until you're 40"
        ],
        correct: 1,
        explanation: "Start with employer match (free money), then emergency fund, then tax-advantaged accounts."
      },
      {
        question: "What should you do when the market drops 20%?",
        options: [
          "Sell everything immediately",
          "Stop investing until it recovers",
          "Stay the course and continue your investment plan",
          "Move everything to cash"
        ],
        correct: 2,
        explanation: "Market drops are normal. Staying invested and continuing your plan typically leads to better outcomes."
      },
      {
        question: "What is inflation risk?",
        options: [
          "The risk that stocks will crash",
          "The risk that your returns won't keep up with rising prices",
          "The risk of losing your job",
          "The risk of paying too much in fees"
        ],
        correct: 1,
        explanation: "Inflation risk is the hidden danger of keeping too much in cash - your purchasing power erodes over time."
      },
      {
        question: "Why is starting to invest early so important?",
        options: [
          "You have more money when young",
          "Time allows compound interest to work, potentially doubling your returns",
          "Stocks are cheaper when you're young",
          "Banks give better rates to young people"
        ],
        correct: 1,
        explanation: "Time is your biggest advantage. Starting 10 years earlier can double your final wealth."
      }
    ]
  },
  // Add more modules here...
  "investor-insight": {
    id: "investor-insight",
    title: "Investor Insight",
    description: "Understand market psychology, indicators, and valuation.",
    level: "Intermediate",
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
        quiz: [
          {
            question: "What does Warren Buffett mean by 'Be fearful when others are greedy'?",
            options: [
              "Always be afraid of the market",
              "When everyone is buying aggressively, be cautious as prices may be inflated",
              "Never invest when others are investing",
              "Greed is always bad"
            ],
            correct: 1,
            explanation: "When markets are euphoric, prices often become overvalued. Caution is warranted."
          }
        ]
      }
    ],
    finalQuiz: [
      {
        question: "What is loss aversion?",
        options: [
          "Avoiding all risky investments",
          "The tendency for losses to feel worse than equivalent gains feel good",
          "Selling investments at a loss",
          "Being afraid of the stock market"
        ],
        correct: 1,
        explanation: "Loss aversion is a cognitive bias where we feel losses more intensely than gains of the same size."
      },
      {
        question: "What is recency bias in investing?",
        options: [
          "Only investing in recent IPOs",
          "Assuming recent trends will continue indefinitely",
          "Preferring recently founded companies",
          "Checking your portfolio recently"
        ],
        correct: 1,
        explanation: "Recency bias leads us to overweight recent events when making predictions about the future."
      }
    ]
  },
  "applied-investing": {
    id: "applied-investing",
    title: "Applied Investing",
    description: "Put knowledge into practice with portfolios, taxes, and strategies.",
    level: "Advanced",
    lessons: [
      {
        id: "asset_allocation",
        title: "Asset Allocation",
        content: `
Asset allocation—how you divide your money between stocks, bonds, and other assets—is the most important investment decision you'll make.

---

## Why It Matters

Studies show that asset allocation explains about **90% of portfolio returns** over time. Stock picking and market timing? Much less important.

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
        `,
        quiz: [
          {
            question: "What percentage of portfolio returns does asset allocation explain?",
            options: [
              "About 10%",
              "About 50%",
              "About 90%",
              "100%"
            ],
            correct: 2,
            explanation: "Research shows asset allocation is the primary driver of long-term returns."
          }
        ]
      }
    ],
    finalQuiz: [
      {
        question: "What is rebalancing?",
        options: [
          "Selling all your investments",
          "Adjusting your portfolio back to your target allocation",
          "Only buying bonds",
          "Checking your balance"
        ],
        correct: 1,
        explanation: "Rebalancing means periodically adjusting your holdings to maintain your desired asset mix."
      },
      {
        question: "Which account type offers tax-free withdrawals in retirement?",
        options: [
          "Traditional IRA",
          "Regular brokerage account",
          "Roth IRA",
          "401(k)"
        ],
        correct: 2,
        explanation: "Roth IRA contributions are made with after-tax dollars, but qualified withdrawals are tax-free."
      }
    ]
  }
};

export default MODULES;
