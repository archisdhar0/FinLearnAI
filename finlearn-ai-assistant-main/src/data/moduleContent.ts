// Complete module content with quizzes and interactive elements
// Extracted from quantcademy-app/pages/learning_modules.py

export interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation?: string;
}

export interface ToolLink {
  text: string;
  route: string;
  description: string;
}

export interface Lesson {
  id: string;
  title: string;
  content: string;
  quiz?: QuizQuestion[];
  interactiveElements?: string[]; // Component names to render
  toolLinks?: ToolLink[]; // Links to AI tools
  goalReminder?: string; // Goal reminder at end of lesson
}

export interface Module {
  id: string;
  title: string;
  description: string;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  goal: string; // Module goal statement
  lessons: Lesson[];
  finalQuiz: QuizQuestion[];
}

export const MODULES: Record<string, Module> = {
  "foundations": {
    id: "foundations",
    title: "The Foundation",
    description: "Core concepts every first-time investor should know.",
    level: "Beginner",
    goal: "Understand investing well enough to confidently open an account and make your first investment.",
    lessons: [
      {
        id: "what_is_investing",
        title: "What is Investing?",
        goalReminder: "After this lesson, you'll understand WHY investing matters and why starting now is so important.",
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
        goalReminder: "You're learning what you can actually invest in. Soon you'll pick your first investment.",
        toolLinks: [
          {
            text: "See Real Stocks in Action",
            route: "/screener",
            description: "Check out our AI Stock Screener to see real stocks with support/resistance levels and trend analysis."
          }
        ],
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
        goalReminder: "Understanding how markets work helps you stay calm when prices move. You're almost ready to invest!",
        toolLinks: [
          {
            text: "Analyze Real Charts",
            route: "/analyzer",
            description: "Try our Chart Analyzer to see how AI detects support/resistance levels and trends in real stock charts."
          },
          {
            text: "Watch Live Market Data",
            route: "/screener",
            description: "See how real stocks are moving right now with our AI Stock Screener."
          }
        ],
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
  "investor-insight": {
    id: "investor-insight",
    title: "Investor Insight",
    description: "Understand market psychology, indicators, and valuation.",
    level: "Intermediate",
    goal: "Develop the mindset to stay calm during market swings and avoid costly emotional decisions.",
    lessons: [
      {
        id: "what_moves_markets",
        title: "What Moves Markets",
        content: `
Markets don't move randomly. They respond to a mix of economic forces, news, and human behavior. Understanding these drivers helps you interpret market movements without overreacting.

---

## Economic Indicators

**GDP (Gross Domestic Product)**: Measures the total value of goods and services produced. Growing GDP usually means a healthy economy.

**Unemployment Rate**: High unemployment signals economic weakness. Low unemployment suggests strength but can lead to inflation.

**Inflation**: Rising prices erode purchasing power. Central banks raise interest rates to combat inflation, which affects stock and bond prices.

**Interest Rates**: When the Federal Reserve raises rates, borrowing becomes more expensive. This can slow economic growth and often causes stock prices to fall.

---

## Corporate Earnings

At the end of the day, stock prices reflect what companies earn. When companies report strong earnings, stock prices tend to rise. When earnings disappoint, prices fall.

**Earnings Season**: Four times a year, companies report quarterly results. These reports can cause significant price swings.

**Forward Guidance**: What companies say about the future often matters more than past results.

---

## News and Events

- **Geopolitical events**: Wars, elections, trade disputes
- **Industry news**: Regulatory changes, technological breakthroughs
- **Company-specific news**: Product launches, scandals, leadership changes

---

## Market Sentiment

Sometimes markets move based on mood rather than facts. Fear and greed drive short-term price swings that may not reflect underlying value.

**Key insight**: In the short term, markets are voting machines (popularity contests). In the long term, they're weighing machines (measuring actual value).
        `,
        quiz: [
          {
            question: "What typically happens to stock prices when the Federal Reserve raises interest rates?",
            options: [
              "They always go up",
              "They often decline as borrowing becomes more expensive",
              "Interest rates don't affect stocks",
              "Only bond prices are affected"
            ],
            correct: 1,
            explanation: "Higher interest rates increase borrowing costs for companies and make bonds more attractive relative to stocks."
          },
          {
            question: "What does 'forward guidance' refer to in earnings reports?",
            options: [
              "Past financial results",
              "What the company expects for future performance",
              "The stock's historical price",
              "Dividend payments"
            ],
            correct: 1,
            explanation: "Forward guidance is management's outlook for future quarters, which often moves stock prices more than past results."
          }
        ]
      },
      {
        id: "investor_psychology",
        title: "Investor Psychology",
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

**Confirmation Bias**: We seek out information that confirms what we already believe and ignore contradicting evidence.

---

## How to Beat Your Brain

1. **Have a plan before you invest**. Decide when you'll buy and sell before emotions take over.

2. **Automate**: Regular automatic investments remove emotion from the equation.

3. **Zoom out**: Look at 10-year charts, not daily moves.

4. **Remember history**: Every crash has eventually recovered. Every bubble has eventually popped.

5. **Write down your reasons**: Before buying, write why. Before selling in a panic, re-read your original thesis.
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
          },
          {
            question: "What is loss aversion?",
            options: [
              "Avoiding all investments",
              "The tendency for losses to feel worse than equivalent gains feel good",
              "Only investing in safe assets",
              "Selling at a loss"
            ],
            correct: 1,
            explanation: "Loss aversion is a cognitive bias where we feel losses about twice as intensely as gains of the same size."
          }
        ]
      },
      {
        id: "hype_vs_fundamentals",
        title: "Hype vs Fundamentals",
        content: `
One of the most important skills in investing is distinguishing between genuine value and hype. Many investors lose money chasing trends that have no substance.

---

## What Are Fundamentals?

Fundamentals are the actual financial health and performance of a company:

- **Revenue**: How much money the company brings in
- **Earnings**: How much profit it makes
- **Cash Flow**: Actual cash generated by operations
- **Debt**: How much the company owes
- **Assets**: What the company owns

These numbers tell you if a company is actually making money or just telling a good story.

---

## What Is Hype?

Hype is excitement based on stories, trends, or speculation rather than proven results:

- "This will be the next Amazon!"
- "Revolutionary technology will change everything!"
- "Everyone is buying it!"

Hype isn't always wrong—some hyped companies do become giants. But most don't.

---

## Red Flags

**No revenue or path to profitability**: The company has a cool idea but no business model.

**Valuation disconnected from reality**: A company worth billions but making no money.

**"This time is different"**: Every bubble uses this phrase.

**FOMO-driven buying**: You're buying because you're afraid of missing out, not because you've done research.

---

## How to Evaluate

1. **Look at the numbers**: Revenue growth, profit margins, cash flow
2. **Understand the business model**: How does the company actually make money?
3. **Compare to peers**: Is the valuation reasonable compared to similar companies?
4. **Ask: Would I buy this if no one else was talking about it?**

---

## The Dot-Com Lesson

In the late 1990s, companies with ".com" in their name skyrocketed regardless of whether they made money. Most went to zero. The few that survived (Amazon, eBay) had real businesses underneath the hype.
        `,
        quiz: [
          {
            question: "What's a key difference between fundamentals and hype?",
            options: [
              "There is no difference",
              "Fundamentals are based on actual financial performance; hype is based on stories and speculation",
              "Hype is always more accurate",
              "Fundamentals only matter for bonds"
            ],
            correct: 1,
            explanation: "Fundamentals reflect real business performance, while hype is often driven by excitement and speculation."
          },
          {
            question: "What's a red flag when evaluating a hyped investment?",
            options: [
              "Strong revenue growth",
              "Valuation disconnected from actual earnings or revenue",
              "Positive cash flow",
              "Low debt levels"
            ],
            correct: 1,
            explanation: "When a company's valuation doesn't match its financial reality, that's a warning sign."
          }
        ]
      },
      {
        id: "types_of_investing",
        title: "Types of Investing",
        content: `
There's no single "right" way to invest. Different approaches suit different goals, time horizons, and personalities.

---

## Passive Investing

**What it is**: Buy and hold a diversified portfolio, usually through index funds. Don't try to beat the market—just match it.

**Pros**:
- Low fees
- Less time required
- Historically outperforms most active strategies

**Best for**: Most people, especially beginners

---

## Active Investing

**What it is**: Try to beat the market by picking individual stocks or timing when to buy and sell.

**Pros**:
- Potential for higher returns
- More engaging for those who enjoy research

**Cons**:
- Higher fees
- Most active managers underperform indexes
- Requires significant time and skill

---

## Value Investing

**What it is**: Buy stocks that appear undervalued based on fundamentals. Look for companies trading below their intrinsic worth.

**Famous practitioners**: Warren Buffett, Benjamin Graham

**Key metrics**: P/E ratio, P/B ratio, dividend yield

---

## Growth Investing

**What it is**: Buy companies expected to grow faster than average, even if they're expensive now.

**Focus**: Revenue growth, market opportunity, innovation

**Risk**: High valuations can lead to big losses if growth disappoints

---

## Income Investing

**What it is**: Focus on investments that generate regular income through dividends or interest.

**Typical investments**: Dividend stocks, bonds, REITs

**Best for**: Retirees or those seeking steady cash flow

---

## Which Is Right for You?

Most beginners should start with passive investing through index funds. As you learn more, you might incorporate other strategies. Many successful investors use a combination.
        `,
        quiz: [
          {
            question: "What is passive investing?",
            options: [
              "Actively trading stocks daily",
              "Buying and holding diversified index funds to match market returns",
              "Only investing in bonds",
              "Trying to beat the market through stock picking"
            ],
            correct: 1,
            explanation: "Passive investing means matching the market through index funds rather than trying to beat it."
          },
          {
            question: "Who is value investing associated with?",
            options: [
              "Day traders",
              "Warren Buffett and Benjamin Graham",
              "Only hedge funds",
              "Cryptocurrency investors"
            ],
            correct: 1,
            explanation: "Value investing was pioneered by Benjamin Graham and made famous by Warren Buffett."
          }
        ]
      },
      {
        id: "risk_portfolio_thinking",
        title: "Risk and Portfolio Thinking",
        content: `
Understanding risk isn't just about avoiding losses—it's about building a portfolio that matches your goals and lets you sleep at night.

---

## Types of Risk

**Market Risk**: The entire market declines (like 2008 or 2020). Even good companies fall.

**Company Risk**: A specific company fails or underperforms. This is why you diversify.

**Inflation Risk**: Your returns don't keep up with rising prices.

**Interest Rate Risk**: Rising rates hurt bond prices and can affect stocks.

**Concentration Risk**: Too much money in one stock, sector, or asset class.

---

## Diversification

Don't put all your eggs in one basket. Spread investments across:

- **Asset classes**: Stocks, bonds, real estate
- **Sectors**: Technology, healthcare, finance, etc.
- **Geographies**: US, international, emerging markets
- **Company sizes**: Large-cap, mid-cap, small-cap

---

## Asset Allocation

Your mix of stocks, bonds, and other assets is the most important investment decision. It determines:

- How much your portfolio might grow
- How much it might drop in a bad year
- How volatile the ride will be

**Common allocations**:
- Aggressive (80% stocks, 20% bonds): For young investors with decades ahead
- Moderate (60% stocks, 40% bonds): Balanced approach
- Conservative (40% stocks, 60% bonds): For those near retirement

---

## Rebalancing

Over time, your allocation drifts as different assets perform differently. Rebalancing means periodically adjusting back to your target mix.

**Example**: If stocks surge and your 60/40 portfolio becomes 70/30, you'd sell some stocks and buy bonds to get back to 60/40.
        `,
        interactiveElements: ['RiskReturnChart', 'DiversificationDemo'],
        quiz: [
          {
            question: "What is the main purpose of diversification?",
            options: [
              "To guarantee profits",
              "To reduce the impact of any single investment failing",
              "To maximize returns",
              "To avoid all risk"
            ],
            correct: 1,
            explanation: "Diversification spreads risk so that one bad investment doesn't devastate your portfolio."
          },
          {
            question: "What is rebalancing?",
            options: [
              "Selling all your investments",
              "Adjusting your portfolio back to your target allocation",
              "Only buying more stocks",
              "Checking your balance"
            ],
            correct: 1,
            explanation: "Rebalancing means periodically adjusting your holdings to maintain your desired asset mix."
          }
        ]
      },
      {
        id: "reading_market_signals",
        title: "Reading Market Signals",
        content: `
Markets constantly send signals about direction, risk, and investor behavior. Learning to read these patterns helps you respond strategically instead of emotionally.

---

## Trends

A trend describes the general direction of market movement over time.

**Uptrend**: Prices make higher highs and higher lows. Buyers are in control.

**Downtrend**: Prices make lower highs and lower lows. Sellers are in control.

**Sideways**: Prices move within a range. Neither buyers nor sellers dominate.

---

## Volatility

Volatility measures how much prices fluctuate.

**Low volatility**: Calm markets, small price changes

**High volatility**: Uncertainty, large rapid price swings

Volatility isn't necessarily bad—it means movement and opportunity. But it also means risk.

---

## Volume

Volume shows how many shares are being traded.

**High volume**: Strong conviction behind price moves

**Low volume**: Less conviction, moves may not be sustainable

Volume confirms trends. A price rise on high volume is more meaningful than one on low volume.

---

## Support and Resistance

**Support**: A price level where buying pressure tends to emerge, stopping declines.

**Resistance**: A price level where selling pressure tends to emerge, stopping advances.

These levels represent psychological barriers where many investors make decisions.

---

## Key Insight

You don't need to predict every market move. Understanding these signals helps you:
- Avoid panic selling during normal corrections
- Recognize when trends might be changing
- Make more informed decisions about when to buy or sell
        `,
        quiz: [
          {
            question: "What does an uptrend typically show?",
            options: [
              "Lower highs and lower lows",
              "Higher highs and higher lows",
              "No clear pattern",
              "Only volume matters"
            ],
            correct: 1,
            explanation: "An uptrend is defined by a series of higher highs and higher lows, showing buyers are in control."
          },
          {
            question: "What does high volume during a price move indicate?",
            options: [
              "The move is likely weak",
              "Strong conviction behind the move",
              "Volume doesn't matter",
              "The market is closed"
            ],
            correct: 1,
            explanation: "High volume suggests many investors agree with the direction, making the move more significant."
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
      },
      {
        question: "What typically happens when the Federal Reserve raises interest rates?",
        options: [
          "Stock prices always go up",
          "Borrowing becomes more expensive, often causing stock prices to decline",
          "Nothing changes",
          "Only bonds are affected"
        ],
        correct: 1,
        explanation: "Higher interest rates increase costs for companies and make bonds more attractive relative to stocks."
      },
      {
        question: "What is the main advantage of passive investing?",
        options: [
          "It guarantees beating the market",
          "Low fees and historically outperforms most active strategies",
          "It requires constant monitoring",
          "It only works in bull markets"
        ],
        correct: 1,
        explanation: "Passive investing through index funds has low fees and historically beats most active managers."
      },
      {
        question: "What does support level mean in technical analysis?",
        options: [
          "A price where selling pressure emerges",
          "A price level where buying pressure tends to stop declines",
          "The highest price ever reached",
          "The company's customer support"
        ],
        correct: 1,
        explanation: "Support is a price level where buying interest tends to emerge, preventing further declines."
      }
    ]
  },
  "applied-investing": {
    id: "applied-investing",
    title: "Applied Investing",
    description: "Put knowledge into practice with portfolios, taxes, and strategies.",
    level: "Advanced",
    goal: "Build a complete, diversified portfolio strategy you can execute this week.",
    lessons: [
      {
        id: "costs_fees_taxes",
        title: "Costs, Fees, and Taxes",
        content: `
Every investment comes with costs—and those costs eat into your returns over time. Understanding them is critical to building wealth efficiently.

---

## Expense Ratios

Expense ratios are annual fees charged by funds to manage your money.

- Expressed as a percentage of assets
- Paid automatically, reducing returns gradually
- Lower expense ratios mean more of your money stays invested

**Example**: $10,000 invested
- Fund A (0.05% fee): $5/year
- Fund B (1.00% fee): $100/year

Over 30 years at 7% returns, a 1% fee can cost you 25% of your potential wealth.

---

## Trading Fees

Buying and selling investments often carries fees:
- Brokerage commissions
- Transaction fees
- Bid-ask spreads

High trading frequency amplifies these costs—another reason to think long-term.

---

## Taxes on Gains

Capital gains taxes apply when you sell assets at a profit.

**Short-term gains** (held < 1 year): Taxed at ordinary income rates (higher)

**Long-term gains** (held > 1 year): Taxed at preferential rates (lower)

**Key insight**: Holding investments longer can significantly reduce your tax bill.

---

## Tax-Advantaged Accounts

Accounts like IRAs, 401(k)s, and HSAs shelter investments from taxes.

**Traditional accounts**: Tax deduction now, pay taxes later

**Roth accounts**: Pay taxes now, withdrawals are tax-free in retirement

**HSA**: Triple tax advantage for healthcare expenses
        `,
        quiz: [
          {
            question: "Why does a 1% expense ratio matter over 30 years?",
            options: [
              "It doesn't matter much",
              "It compounds and can cost a large portion of your growth over decades",
              "Expense ratios are tax-deductible",
              "Only bonds have expense ratios"
            ],
            correct: 1,
            explanation: "Fees compound against you every year, potentially costing 25% or more of your ending wealth."
          },
          {
            question: "What's the advantage of long-term capital gains over short-term?",
            options: [
              "There is no difference",
              "Long-term gains are typically taxed at lower rates",
              "Short-term gains are always better",
              "Only dividends are taxed"
            ],
            correct: 1,
            explanation: "Holding investments for more than one year qualifies for lower long-term capital gains rates."
          }
        ]
      },
      {
        id: "what_to_do_in_crash",
        title: "What to Do in a Market Crash",
        content: `
Market crashes are scary. Your portfolio drops 20%, 30%, maybe more. Headlines scream disaster. Your instincts tell you to sell everything.

Here's the thing: what you do during a crash often determines your long-term investing success.

---

## First: Don't Panic Sell

This is the most important rule. Selling during a crash locks in your losses and means you miss the recovery.

**Historical fact**: Every major crash in history has eventually recovered. The investors who stayed invested came out ahead.

- 1987 crash: Recovered in 2 years
- 2008 financial crisis: Recovered in 4 years
- 2020 COVID crash: Recovered in 5 months

---

## Second: Remember Your Timeline

If you're investing for retirement 20+ years away, a crash today is just noise. You have time to recover.

Ask yourself: "Am I going to need this money in the next 5 years?" If no, stay the course.

---

## Third: Consider Buying More

Crashes mean stocks are "on sale." If you have cash available and a long time horizon, buying during crashes can significantly boost your long-term returns.

**Dollar-cost averaging**: Continue your regular investments through the crash. You'll buy more shares at lower prices.

---

## Fourth: Review Your Allocation

A crash is a good time to check if your portfolio matches your risk tolerance. If you can't sleep at night, you might be too aggressive.

But don't make changes during the panic. Wait until markets stabilize, then reassess calmly.

---

## What NOT to Do

- Don't check your portfolio daily
- Don't watch financial news constantly
- Don't try to time the bottom
- Don't make permanent decisions based on temporary emotions
        `,
        quiz: [
          {
            question: "What's the most important rule during a market crash?",
            options: [
              "Sell everything immediately",
              "Don't panic sell—stay invested",
              "Check your portfolio hourly",
              "Move everything to cash"
            ],
            correct: 1,
            explanation: "Panic selling locks in losses and means you miss the recovery. Stay the course."
          },
          {
            question: "What does history show about major market crashes?",
            options: [
              "Markets never recover",
              "Every major crash has eventually recovered",
              "Only some crashes recover",
              "Recovery takes 50+ years"
            ],
            correct: 1,
            explanation: "Every major crash in history has recovered, though recovery times vary."
          }
        ]
      },
      {
        id: "setting_long_term_structure",
        title: "Setting a Long-Term Structure",
        content: `
Successful investing isn't about making perfect decisions. It's about setting up a system that works over decades.

---

## The Three-Fund Portfolio

A simple, effective structure used by millions:

1. **US Total Stock Market Index Fund**: Captures the entire US market
2. **International Stock Index Fund**: Diversifies globally
3. **Total Bond Market Index Fund**: Provides stability

That's it. Three funds can give you a complete, diversified portfolio.

---

## Automation Is Your Friend

Set up automatic investments so you don't have to think about it:

- Automatic 401(k) contributions from your paycheck
- Automatic transfers to your IRA each month
- Automatic reinvestment of dividends

When investing is automatic, you remove emotion and ensure consistency.

---

## Rebalancing Schedule

Pick a schedule and stick to it:

- **Annual rebalancing**: Check once a year, adjust if needed
- **Threshold rebalancing**: Rebalance when allocation drifts more than 5-10%

Don't rebalance too often—it can trigger taxes and fees.

---

## Emergency Fund First

Before investing aggressively, ensure you have 3-6 months of expenses in cash. This prevents you from having to sell investments during emergencies.

---

## The Investment Policy Statement

Write down your plan:
- Your goals and timeline
- Your target allocation
- When you'll rebalance
- What you'll do during crashes

Having a written plan helps you stay disciplined when emotions run high.
        `,
        quiz: [
          {
            question: "What is the three-fund portfolio?",
            options: [
              "Three individual stocks",
              "US stocks, international stocks, and bonds—a simple diversified approach",
              "Three different brokerages",
              "Only for professional investors"
            ],
            correct: 1,
            explanation: "The three-fund portfolio provides complete diversification with just US stocks, international stocks, and bonds."
          },
          {
            question: "Why is automation important in investing?",
            options: [
              "It guarantees returns",
              "It removes emotion and ensures consistent investing",
              "It's required by law",
              "It only works for large amounts"
            ],
            correct: 1,
            explanation: "Automatic investing removes the temptation to time the market and ensures you invest consistently."
          }
        ]
      },
      {
        id: "realistic_expectations",
        title: "Realistic Expectations About Returns",
        content: `
One of the biggest mistakes investors make is having unrealistic expectations. Let's set some grounded expectations based on history.

---

## Historical Stock Returns

The S&P 500 has returned about **10% annually** on average over the long term (before inflation).

After inflation, that's about **7% real returns**.

**Important**: This is an average. Individual years vary wildly:
- Some years: +30%
- Some years: -30%
- Most years: Somewhere in between

---

## What 7% Real Returns Means

Using the Rule of 72: Your money doubles roughly every 10 years.

- $10,000 at age 25 → ~$80,000 at age 55
- $500/month for 30 years → ~$500,000

This is life-changing wealth, but it takes time.

---

## What's NOT Realistic

- Doubling your money in a year (without extreme risk)
- Consistent 20%+ annual returns
- Never having a down year
- Timing the market perfectly

Anyone promising these is either lying or taking enormous risks.

---

## Sequence of Returns Risk

The order of returns matters, especially near retirement. A crash right before you retire hurts more than a crash early in your career.

This is why you gradually shift to more conservative allocations as you approach your goal.

---

## The Power of Consistency

Slow and steady wins the race. A boring portfolio of index funds, held for decades, beats most "exciting" strategies.

**The real secret**: Start early, invest consistently, keep costs low, stay the course.
        `,
        interactiveElements: ['CompoundCalculator'],
        quiz: [
          {
            question: "What is the historical average annual return of the S&P 500?",
            options: [
              "About 3%",
              "About 10% (7% after inflation)",
              "About 20%",
              "It varies too much to say"
            ],
            correct: 1,
            explanation: "Historically, the S&P 500 has returned about 10% annually, or 7% after adjusting for inflation."
          },
          {
            question: "What's the 'secret' to successful long-term investing?",
            options: [
              "Finding the next hot stock",
              "Timing the market perfectly",
              "Start early, invest consistently, keep costs low, stay the course",
              "Only investing during bull markets"
            ],
            correct: 2,
            explanation: "Consistency and patience beat trying to outsmart the market."
          }
        ]
      },
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

## Target-Date Funds

If this feels overwhelming, consider target-date funds. Pick the fund matching your retirement year, and it automatically adjusts allocation as you age.

Example: "Target 2055 Fund" starts aggressive and gradually becomes conservative.

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
          },
          {
            question: "What is a target-date fund?",
            options: [
              "A fund that only invests on certain dates",
              "A fund that automatically adjusts allocation as you approach retirement",
              "A fund for day traders",
              "A fund that guarantees returns by a target date"
            ],
            correct: 1,
            explanation: "Target-date funds automatically shift from aggressive to conservative as you near your target retirement year."
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
      },
      {
        question: "What should you do during a market crash?",
        options: [
          "Sell everything immediately",
          "Stay invested and possibly buy more if you have a long time horizon",
          "Check your portfolio every hour",
          "Move everything to cryptocurrency"
        ],
        correct: 1,
        explanation: "Staying invested through crashes and continuing to invest captures the recovery."
      },
      {
        question: "What is the three-fund portfolio?",
        options: [
          "Three individual stocks",
          "US stocks, international stocks, and bonds",
          "Three different brokerages",
          "A complex hedge fund strategy"
        ],
        correct: 1,
        explanation: "The three-fund portfolio is a simple, diversified approach using US stocks, international stocks, and bonds."
      },
      {
        question: "Why is a 1% expense ratio significant over 30 years?",
        options: [
          "It's not significant",
          "It can cost you 25% or more of your potential wealth due to compounding",
          "It only affects bonds",
          "Expense ratios are refundable"
        ],
        correct: 1,
        explanation: "Fees compound against you every year, significantly reducing your ending wealth over decades."
      }
    ]
  }
};

export default MODULES;
