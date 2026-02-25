"""
QuantCademy - Complete Learning Modules
Comprehensive financial education from beginner to expert.
Content sourced from: Investopedia, Vanguard, Fidelity, SEC, FINRA, Bogleheads
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Learning Modules | QuantCademy",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .module-btn {
        background: #16213e;
        border: none;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
        cursor: pointer;
        border-left: 4px solid #4ade80;
    }
    .module-btn:hover {
        background: #1e3a5f;
    }
    .lesson-btn {
        background: #1a1a2e;
        border-left: 2px solid #4b5563;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0 0.25rem 1rem;
    }
    .level-beginner { border-left-color: #4ade80 !important; }
    .level-intermediate { border-left-color: #fbbf24 !important; }
    .level-advanced { border-left-color: #f97316 !important; }
    .level-expert { border-left-color: #ef4444 !important; }
    .badge-beginner { color: #4ade80; }
    .badge-intermediate { color: #fbbf24; }
    .badge-advanced { color: #f97316; }
    .badge-expert { color: #ef4444; }
    .source-tag {
        background: #1e3a5f;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .key-concept {
        background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #6366f1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state.current_module = None
if 'current_lesson' not in st.session_state:
    st.session_state.current_lesson = None
if 'completed_lessons' not in st.session_state:
    st.session_state.completed_lessons = set()

# Module definitions
MODULES = {
    "foundations": {
        "number": "1",
        "title": "The Foundation",
        "level": "Beginner",
        "icon": "📖",
        "description": "Core concepts every first-time investor should know.",
        "source": "Various",
        "lessons": [
            {
  "id": "what_is_investing",
  "title": "What is Investing?",
  "content": 
"""
You've probably heard people talk about investing. Your friend who won't stop talking about their portfolio. That coworker who's always checking stock prices. Your parents telling you to "start investing early."

But what does investing actually mean? And why should you care?

Let me put it simply: investing is just using your money to make more money over time. Instead of letting your cash sit in a savings account earning next to nothing, you're putting it to work. You're buying pieces of companies, or lending money to governments, or owning a slice of the entire stock market.

Think of it like this: if you stash $100 under your mattress, you'll still have $100 in ten years (minus whatever the mice eat). But if you invest that $100 and it grows at 7% per year, you'll have about $200. Your money made money while you slept.

That's the whole point of investing. You're not gambling or day trading or trying to get rich quick. You're just letting your money work for you instead of the other way around.

---

## The Savings Account Trap

Here's something that might surprise you: keeping all your money in a savings account is actually risky. Not risky like "you might lose it" risky, but risky like "it's definitely going to lose value" risky.

See, there's this thing called inflation. Every year, stuff costs more. A gallon of milk that cost $3 last year might cost $3.10 this year. Your $100 can buy less stuff over time, even though it's still $100.

If inflation is 3% per year (which is pretty normal), that $100 under your mattress will only be worth about $74 in purchasing power after ten years. You didn't lose the money, but you lost what the money can actually buy.

Investing is how you fight back. Instead of watching your money slowly shrink, you're putting it somewhere that (historically, at least) grows faster than inflation. You're not trying to beat Warren Buffett or time the market perfectly. You're just trying to keep up with the rising cost of everything.

---

## Saving vs Investing: What's the Difference?

Okay, so saving and investing are different things, and you need both.

**Saving** is like your safety net. It's the money you keep in a regular bank account for emergencies. Your car breaks down? Your savings account has you covered. You lose your job? Your savings buys you time to find a new one. This money needs to be safe and accessible, even if it's not growing much.

**Investing** is for the long game. This is money you're putting away for retirement, or a house down payment in five years, or your kid's college fund. You're okay with this money going up and down in value because you're not touching it for a while. You're playing the long game.

The key difference? Time. If you need the money in the next year or two, keep it in savings. If you don't need it for five, ten, or thirty years, that's when investing makes sense.

---

## Why Time Matters So Much

Here's where it gets interesting. The earlier you start investing, the easier it is. Not because you have more money (you probably don't), but because time does most of the heavy lifting.

Let's say you invest $100 when you're 25. By the time you're 65, that $100 could be worth $1,500 or more (assuming 7% average returns). But if you wait until you're 35 to invest that same $100, by 65 it might only be worth $750.

Same $100. Same return rate. But starting ten years earlier means you end up with twice as much money.

That's the power of compounding. Your money makes money, and then that money makes money, and it keeps going. The longer you let it run, the more it multiplies.

This is why people get so excited about starting early. It's not about having a lot of money to start with. It's about giving your money as much time as possible to grow.

---

## The Reality Check

Now, let's be honest. Investing isn't magic. Your money can go down. Sometimes it goes down a lot. The stock market has crashed before, and it'll crash again. That's just how it works.

But here's the thing: if you're investing for the long term (like retirement), those crashes don't matter as much. Yeah, your portfolio might drop 20% or 30% in a bad year. But if you're not touching that money for decades, you have time to wait for it to recover. And historically, it always has.

The people who get hurt are the ones who panic and sell when things drop. They lock in their losses and miss the recovery. But if you can just... not do that? If you can just leave your money alone and let time do its thing? You'll probably be fine.

That's really what investing is about. It's not about being smart enough to pick the right stocks or time the market perfectly. It's about being patient enough to let your money grow over time, even when things get scary.

---

## So What Does This Mean for You?

If you're reading this, you're probably thinking about starting to invest. Maybe you've got some money saved up. Maybe you're just curious. Either way, you're in the right place.

The most important thing to understand right now is this: investing isn't complicated. You don't need to become a financial expert. You don't need to watch CNBC every day. You just need to understand the basics, start small, and be patient.

You're not trying to beat the market. You're just trying to participate in it. And that's a lot simpler than most people make it sound.

""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "position": "after_paragraph_1",
      "question": "What's the main difference between saving and investing?",
      "options": [
        "Saving is riskier than investing",
        "Investing is for long-term goals, saving is for short-term needs",
        "They're the same thing",
        "Only rich people can invest"
      ],
      "correct_index": 1,
      "explanation": "Investing is for money you won't need for years, while saving is for emergencies and short-term goals."
    },
    {
      "type": "inflation_calculator",
      "position": "after_inflation_section",
      "initial_amount": 100,
      "years": 10,
      "inflation_rate": 3.0
    },
    {
      "type": "concept_check",
      "position": "after_time_section",
      "question": "Why does starting to invest early matter so much?",
      "options": [
        "You have more money when you're young",
        "Time allows compound interest to work its magic",
        "Stocks always go up when you're young",
        "It doesn't really matter"
      ],
      "correct_index": 1,
      "explanation": "Starting early gives your money more time to compound, which can double or triple your returns over decades."
    },
    {
      "type": "saving_vs_investing",
      "position": "end",
      "initial": 10000,
      "monthly": 500,
      "years": 20
    }
  ],
  "quiz": [
    {
      "question": "Which of the following best describes investing?",
      "options": [
        "Keeping money in a savings account to avoid risk",
        "Using money to grow purchasing power over time while accepting some risk",
        "Hoarding cash to protect against inflation",
        "Spending money quickly to achieve immediate gains"
      ],
      "correct": 1
    },
    {
      "question": "Why is keeping all your money in cash risky over the long run?",
      "options": [
        "Cash is illegal for long-term holding",
        "Inflation erodes purchasing power; your money buys less over time",
        "Banks charge huge fees for cash",
        "Cash always loses value immediately"
      ],
      "correct": 1
    },
    {
      "question": "What is the main benefit of starting to invest early?",
      "options": [
        "You can take more risks with no consequences",
        "Compound growth has more time to work, so the same amount can grow much more",
        "Young people always pick better stocks",
        "It has no special benefit"
      ],
      "correct": 1
    }
  ]
}
,
           {"id": "what_youre_actually_buying", "title": "What You're Actually Buying", "content": 
"""
When you invest, what are you actually buying? This is probably the most confusing part for beginners. You hear words like "stocks" and "bonds" and "ETFs" and it all sounds like financial jargon designed to confuse you.

Let me break it down in plain English.

---

## Stocks: You Own a Tiny Piece of a Company

When you buy a stock, you're buying a tiny piece of a company. That's it. If Apple has a billion shares and you own one, you own one billionth of Apple. You're a part-owner.

Now, you don't get to make decisions or walk into the headquarters and demand a tour. But you do get to benefit if the company does well. If Apple's value goes up, your tiny piece becomes more valuable. If Apple pays dividends (which is just a fancy word for sharing profits with owners), you get a tiny cut.

The flip side? If Apple struggles, your piece becomes less valuable. That's the risk. You're betting that the company will be worth more in the future than it is today.

Most people think of stocks as these abstract things that go up and down on a screen. But really, you're just buying a small ownership stake in a real business. That business makes products, hires people, tries to grow. Your investment rises or falls based on how well that business does.

---

## ETFs: The Easy Way to Own Everything

Here's where it gets interesting. Instead of buying one stock, what if you could buy a little bit of hundreds of stocks at once? That's what an ETF (Exchange-Traded Fund) does.

Think of it like a basket. Instead of buying individual apples, oranges, and bananas, you buy a fruit basket that has a little bit of everything. If one fruit goes bad, you've still got the others. That's diversification, and it's one of the smartest things you can do as an investor.

An ETF might own shares of 500 different companies. When you buy one share of that ETF, you're buying a tiny piece of all 500 companies. You don't have to research each one. You don't have to decide which ones to pick. You just buy the basket and you're instantly diversified.

The best part? ETFs trade just like stocks. You can buy and sell them throughout the day. They're simple, cheap, and they're probably what you should start with as a beginner.

---

## Index Funds: Following the Market

An index fund is basically a special type of ETF. Instead of a fund manager picking which stocks to include, an index fund just copies a market index like the S&P 500.

The S&P 500 is just a list of the 500 biggest companies in America. An S&P 500 index fund owns all 500 of them, in the same proportions. When the S&P 500 goes up 10%, your index fund goes up 10%. When it goes down, your fund goes down. You're not trying to beat the market. You're just trying to match it.

This might sound boring, but here's the thing: most professional investors can't beat the market over the long term. So if you can just match it, you're doing better than most people who do this for a living.

Index funds are also really cheap. Some charge as little as 0.03% per year. That means if you invest $10,000, you pay $3 in fees. Compare that to actively managed funds that might charge 1% or more, and you're saving hundreds of dollars a year.

---

## Bonds: You're the Bank

Stocks are about ownership. Bonds are about lending.

When you buy a bond, you're lending money to a company or government. They promise to pay you back with interest. It's like being a bank, except you're the customer.

Let's say you buy a $1,000 bond that pays 5% interest. The company borrows your $1,000 and promises to pay you $50 per year in interest, then give you your $1,000 back when the bond "matures" (usually in 5, 10, or 30 years).

Bonds are generally safer than stocks because you have a contract. The company has to pay you back. But they also usually grow slower. You're trading potential upside for more stability.

Most people should have some bonds in their portfolio, especially as they get older. They smooth out the ride when stocks are volatile.

---

## Cash: The Safest, Slowest Option

Cash is just... cash. Or cash-like things like money market funds. It's the safest option because it's not going anywhere. But it's also the slowest. Your money just sits there, maybe earning a tiny bit of interest, but mostly just... sitting.

You need some cash for emergencies and short-term goals. But if you're investing for the long term, too much cash is actually risky because inflation will eat away at its value.

---

## The Big Picture

Here's what you need to understand: when you invest, you're not gambling. You're not buying lottery tickets. You're buying real things. A piece of a company. A loan to a government. A basket of investments.

These things have value because they represent real economic activity. Companies make products and services. Governments build infrastructure. The economy grows over time, and your investments grow with it.

That's really what investing is. You're participating in the economy's growth. You're not trying to outsmart anyone or time the market perfectly. You're just putting your money where the economy is going, and letting it grow over time.

""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is the main advantage of an ETF over buying a single stock?",
      "options": [
        "ETFs never lose value",
        "ETFs give you instant diversification across many companies",
        "ETFs are only for experts",
        "ETFs avoid all taxes"
      ],
      "correct_index": 1,
      "explanation": "ETFs bundle many securities in one trade, so you get diversification without picking each stock yourself."
    },
    {
      "type": "concept_check",
      "question": "When you buy a bond, what are you doing?",
      "options": [
        "Buying a piece of a company",
        "Lending money in exchange for interest and eventual repayment",
        "Betting on short-term price movement only",
        "Avoiding all risk"
      ],
      "correct_index": 1,
      "explanation": "Bonds are loans: you lend money to an issuer who pays you interest and repays principal at maturity."
    }
  ],
  "quiz": [
    {"question": "Which investment gives you direct ownership in a company?", "options": ["Bond", "Stock", "ETF", "Cash"], "correct": 1},
    {"question": "Which type of fund typically has the lowest ongoing fees?", "options": ["Actively managed mutual fund", "S&P 500 index fund or ETF", "Hedge fund", "Fund that trades daily"], "correct": 1},
    {"question": "Why might an investor hold both stocks and bonds?", "options": ["Bonds always outperform stocks", "To balance growth potential with some stability", "To avoid paying any taxes", "Stocks are illegal in retirement accounts"], "correct": 1}
  ]},
            {"id": "how_markets_function", "title": "How Markets Function", "content": 
"""
The stock market can seem like this mysterious force that moves up and down for no reason. One day everything's up, the next day everything's down. It feels random, chaotic, like there's no logic to it.

But there is logic. It's actually pretty simple once you understand what's happening.

---

## It's Just People Buying and Selling

At its core, the stock market is just a place where people buy and sell stocks. That's it. There's no magic. No mysterious forces. Just millions of people making decisions about what they think something is worth.

When you want to buy a stock, you place an order. Someone else who wants to sell places an order. When your prices match, a trade happens. The stock market is just a giant matching system, connecting buyers with sellers.

Think of it like a farmers market, but instead of vegetables, people are trading pieces of companies. Some people think a company is worth more, so they're willing to pay more. Some people think it's worth less, so they want to sell. When they agree on a price, they make a deal.

---

## Supply and Demand: The Only Rule That Matters

You've probably heard of supply and demand. It's the only rule that really matters in markets.

If more people want to buy something than sell it, the price goes up. If more people want to sell than buy, the price goes down. That's it. That's the whole game.

Let's say a company announces great earnings. More people want to buy the stock because they think it's going to do well. But there aren't enough sellers at the current price. So buyers start offering more money. The price goes up until enough sellers are willing to part with their shares at that new price.

Or the opposite happens. Bad news comes out. People want to sell. But there aren't enough buyers at the current price. So sellers start accepting lower offers. The price goes down until enough buyers are willing to step in.

This happens millions of times per day, across thousands of stocks. It's just people reacting to information and adjusting what they're willing to pay.

---

## Why Prices Move (It's Not Random)

Prices don't move randomly. They move because something changed. Maybe it's news about the company. Maybe it's news about the economy. Maybe it's just that more people woke up today and decided they wanted to buy or sell.

The key thing to understand is that prices reflect what people think something is worth right now. Not what it's actually worth. Not what it will be worth in the future. Just what people are willing to pay for it today.

And people's opinions change constantly. New information comes out. People change their minds. More buyers or sellers show up. Prices adjust.

This is why the market is so volatile. It's not that companies are actually worth 20% more or less from one day to the next. It's that people's opinions about what they're worth changed. And when millions of people are making these decisions, prices can swing pretty dramatically.

---

## News Doesn't Move Prices. People's Reactions to News Do

Here's something important: news doesn't directly move stock prices. People's reactions to news do.

Let's say a company announces they're going to make a new product. The news itself doesn't change the stock price. What changes the price is how people react. Do they think this product will be successful? Do they think it will make the company more valuable? If enough people think yes, they start buying. That drives the price up.

But here's the thing: people can be wrong. They can overreact. They can underreact. They can get caught up in hype or panic. So sometimes prices move way more than they should based on the actual news. And sometimes they barely move at all, even when the news seems significant.

This is why trying to trade based on news is so hard. You're not just trying to understand the news. You're trying to understand how everyone else will react to the news. And that's nearly impossible.

---

## The Big Picture

Here's what you need to know: the stock market is just a reflection of what millions of people think companies are worth. Those opinions change constantly. Prices adjust constantly. It's messy, it's noisy, and it's often irrational in the short term.

But over the long term, prices tend to reflect reality. Good companies that grow their earnings tend to see their stock prices go up. Bad companies that struggle tend to see their prices go down. The market isn't perfectly efficient, but it's efficient enough that trying to outsmart it is really, really hard.

For most investors, the best strategy is to just accept that prices will go up and down. Don't try to predict the movements. Don't try to time the market. Just invest in good companies (or better yet, index funds that own lots of companies) and let time do its thing.

The market will do what it's going to do. Your job is just to stay invested and not panic when things get scary.

""", 
"interactive_elements": [
  {
    "type": "stock_market_demo"
  },
  {
    "type": "supply_demand_viz"
  },
  {
    "type": "concept_check",
    "question": "What causes stock prices to move?",
    "options": [
      "Random chance",
      "Changes in supply and demand",
      "The time of day",
      "The weather"
    ],
    "correct_index": 1,
    "explanation": "Prices move when the balance between buyers (demand) and sellers (supply) changes. More buyers = price goes up. More sellers = price goes down."
  }
],
"quiz": [{"question": "When does price tend to rise in a market?", "options": ["When supply increases", "When demand decreases", "When demand exceeds supply", "When news mentions the word 'growth'"], "correct": 2}]},
            {"id": "time_compounding", "title": "Time and Compounding", "content": 
"""
Here's a secret that most people don't realize: when it comes to investing, time is more powerful than skill. You don't need to be a genius. You don't need to pick the perfect stocks. You just need to start early and be patient.

Let me explain why.

---

## Compounding: The Eighth Wonder of the World

Albert Einstein supposedly called compound interest the eighth wonder of the world. Whether he actually said that or not, the sentiment is true. Compounding is magical.

Here's how it works: when you invest money, it earns returns. Those returns then earn their own returns. Your money makes money, and then that money makes money. It's like a snowball rolling downhill, getting bigger and bigger.

Let's say you invest $1,000 and it grows 10% in the first year. You now have $1,100. In the second year, that 10% growth is on $1,100, not $1,000. So you earn $110 instead of $100. In year three, you earn $121. The growth accelerates.

This doesn't sound like much at first. But give it time. After 20 years, that $1,000 could be worth over $6,700. After 30 years, it could be worth over $17,000. After 40 years, it could be worth over $45,000.

Same $1,000. Same 10% return. Just time doing its thing.

---

## Why Starting Early is Everything

The earlier you start, the less you actually need to invest. That's the crazy part.

Let's say you want to have $1 million by age 65. If you start at 25 and earn 7% per year, you only need to invest about $300 per month. But if you wait until 35 to start, you need to invest about $700 per month. Wait until 45, and you need over $2,000 per month.

Same goal. Same return. But starting ten years earlier means you only need to invest half as much per month.

This is why people get so excited about starting young. It's not about having a lot of money. It's about having a lot of time. Time is the secret ingredient that makes investing work.

---

## The Time Horizon Game

Your time horizon is just how long you plan to keep your money invested before you need it. This matters because it determines how much risk you can take.

If you need the money in a year, you can't afford to take big risks. What if the market crashes right before you need it? You'd be screwed. So you keep it safe, even if that means lower returns.

But if you don't need the money for 30 years? You can handle some volatility. The market might crash next year, but you've got 29 more years for it to recover. History shows it will. So you can take more risk, which means higher potential returns.

This is why retirement accounts are so powerful. You're locking that money away for decades. You can invest aggressively because you have time on your side. Short-term crashes don't matter when you're not touching the money for 30 years.

---

## Consistency Beats Everything

Here's another thing most people get wrong: they think they need to invest big chunks of money. They wait until they have $5,000 or $10,000 saved up before they start investing.

But that's backwards. Small, consistent investments are often more powerful than big, occasional ones.

Let's say you invest $100 every month for 30 years. That's $36,000 total. But at 7% returns, it could grow to over $120,000. You turned $36,000 into $120,000 just by being consistent.

Now let's say you wait 10 years, then invest $200 per month for 20 years. That's $48,000 total (more than the first example). But it only grows to about $100,000. Less money, even though you invested more total.

The lesson? Start small. Start now. Be consistent. Time and consistency will do the heavy lifting.

---

## The Reality Check

I know what you're thinking. "But I don't have much money. What's the point of investing $50 a month?"

The point is that $50 a month, invested for 30 years at 7% returns, becomes about $60,000. That's not life-changing money, but it's real money. And if you can do $100 a month, that becomes $120,000. $200 a month becomes $240,000.

You're not trying to get rich overnight. You're just trying to build something over time. And the best part? As you make more money, you can invest more. That $50 a month might become $100, then $200, then $500. And each increase compounds on top of what you've already built.

The people who get wealthy from investing aren't the ones who made one brilliant trade. They're the ones who invested consistently for decades and let time do its thing.

That's the real secret. It's not complicated. It's just patient.

""", 
"interactive_elements": [
  {
    "type": "compound_interest",
    "principal": 100,
    "monthly": 50,
    "rate": 7.0,
    "years": 30
  }
],
"quiz": [{"question": "Why is starting to invest early so important?", "options": ["You can take more risks in the short term", "Compounding has more time to grow your money", "You will always pick better stocks", "It avoids paying taxes"], "correct": 1}]},
           {"id": "basics_of_risk", "title": "The Basics of Risk", "content": 
"""
Let's talk about risk. This is probably the scariest part of investing for most people. The idea that you could lose money. That your hard-earned cash could just... disappear.

I'm not going to sugarcoat it. Yes, you can lose money when you invest. But understanding risk is the key to managing it. And once you understand it, it's not as scary as it seems.

---

## Volatility: The Price of Admission

First, let's talk about volatility. This is just a fancy word for how much prices go up and down.

Some investments are really volatile. Their prices swing wildly. One day they're up 5%, the next day they're down 8%. It's a roller coaster.

Other investments are more stable. Their prices move slowly, predictably. It's more like a gentle boat ride.

Here's the thing: volatility isn't necessarily bad. It's just... volatility. High volatility usually means higher potential returns. But it also means bigger swings. Your investment might drop 30% in a bad year, then gain 40% the next year. If you can handle that ride, you might be rewarded. If you can't, you might panic and sell at the worst time.

The key is understanding that short-term volatility is normal. Prices go up and down all the time. That's just how markets work. It doesn't mean you're losing money permanently. It just means prices are moving.

---

## The Difference Between Losing Money and Actually Losing Money

This is important: there's a difference between seeing your portfolio value drop and actually losing money.

When the market crashes and your $10,000 investment drops to $7,000, you haven't actually lost $3,000 yet. You've only lost it if you sell. If you hold on, the value might come back. In fact, historically, it almost always does.

This is why time horizon matters so much. If you need the money next year and the market crashes, you're in trouble. You might have to sell at a loss. But if you don't need the money for 20 years? Who cares if it drops this year? You've got 19 more years for it to recover.

The people who actually lose money are usually the ones who panic and sell when things get scary. They lock in their losses and miss the recovery. But if you can just... not do that? If you can just wait it out? You'll probably be fine.

---

## Diversification: Don't Put All Your Eggs in One Basket

This is probably the most important concept in investing: diversification.

Diversification just means spreading your money across different investments. Instead of putting all your money in one stock, you put it in hundreds of stocks. Or thousands. That way, if one company goes bankrupt, you don't lose everything.

Think of it like this: if you own one stock and that company goes under, you've lost everything. But if you own 500 stocks and one goes under, you've lost 0.2% of your money. Big difference.

Most people achieve diversification by buying index funds or ETFs. One purchase gives you exposure to hundreds or thousands of companies. You don't have to pick individual stocks. You just buy the whole market.

This is why index funds are so popular. They're automatically diversified. You're not betting on one company. You're betting on the entire economy. And the entire economy is a lot less likely to disappear than one company.

---

**See the difference for yourself:**

---

## Risk Tolerance: Know Thyself

Your risk tolerance is basically how much volatility you can handle without panicking. This is more about psychology than math.

Some people can watch their portfolio drop 30% and just shrug. They know it'll come back. They're in it for the long haul. These people have high risk tolerance.

Other people see their portfolio drop 10% and start having panic attacks. They can't sleep. They check their phone constantly. They're tempted to sell everything. These people have low risk tolerance.

Neither is wrong. But you need to be honest with yourself about which one you are.

If you have low risk tolerance, that's okay. You can still invest. You just need to invest more conservatively. More bonds, fewer stocks. Lower potential returns, but also lower stress. And honestly? Lower stress might be worth the lower returns if it means you'll actually stick with your plan.

The worst thing you can do is invest aggressively when you have low risk tolerance. You'll panic at the first crash and sell everything, locking in losses. It's better to invest conservatively and actually stick with it.

---

## The Reality of Risk

Here's the truth: you can't eliminate risk. Every investment has some risk. Even "safe" investments like bonds can lose value. Even cash loses value to inflation.

The question isn't "how do I avoid risk?" The question is "how do I manage risk?"

And the answer is: diversify, invest for the long term, invest according to your risk tolerance, and don't panic when things get scary.

The market will crash. It always does. But it also always recovers. If you can just wait it out, you'll probably be fine. That's really all there is to it.

""", 
"interactive_elements": [
  {
    "type": "diversification_chart"
  },
  {
    "type": "concept_check",
    "question": "Why is diversification important?",
    "options": [
      "It guarantees higher returns",
      "It reduces risk by spreading investments across many companies",
      "It's required by law",
      "It makes investing more complicated"
    ],
    "correct_index": 1,
    "explanation": "Diversification doesn't guarantee higher returns, but it significantly reduces risk. If one company fails, you don't lose everything."
  }
],
"quiz": [{"question": "Which strategy helps reduce the risk of losing all your money in one investment?", "options": ["Holding only one stock you trust", "Diversifying across multiple assets", "Checking the news daily", "Investing only in volatile stocks"], "correct": 1}]},
            {"id": "accounts_setup", "title": "Accounts and Setup", "content": 
"""
Before you can invest, you need somewhere to put your money. That's what accounts are for. But there are different types of accounts, and they work differently. Let me break it down.

---

## Brokerage Accounts: Your Basic Investing Account

A brokerage account is just a regular account where you can buy and sell investments. It's like a bank account, but instead of just holding cash, it holds stocks, bonds, ETFs, and other investments.

You open one at a brokerage firm (like Fidelity, Vanguard, or Schwab). You transfer money in from your bank account. Then you use that money to buy investments. When you want to sell, the money goes back into the account, and you can transfer it back to your bank.

The thing about brokerage accounts is they don't have special tax advantages. You pay taxes on dividends and interest as you earn them. When you sell investments for a profit, you pay taxes on those gains. It's straightforward, but not particularly tax-efficient.

The upside? You can take money out anytime, no penalties, no restrictions. It's your money, and you can do whatever you want with it.

---

## Roth IRAs: Tax-Free Growth

A Roth IRA is a special type of retirement account. The key difference? Your money grows tax-free.

Here's how it works: you put money in after you've already paid taxes on it. So if you earn $100 and pay $20 in taxes, you can put that remaining $80 into a Roth IRA. Then, when you're 60 years old and ready to retire, you can take that money out completely tax-free.

This is huge. If you invest $10,000 in a Roth IRA and it grows to $100,000 over 30 years, you get all $100,000 tax-free. In a regular brokerage account, you'd pay taxes on the $90,000 in gains.

The catch? There are rules. You can only contribute so much per year (around $7,000 for most people). And you generally can't take the money out until you're 59 and a half without penalties. But you can always take out the money you originally put in, just not the earnings.

For most young people investing for retirement, a Roth IRA is the best place to start. You're probably in a lower tax bracket now than you will be in retirement, so paying taxes now and getting tax-free growth later is a great deal.

---

## How Money Actually Moves

When you invest, your money doesn't just disappear into the void. Here's what actually happens:

You transfer money from your bank account to your brokerage account. The brokerage holds it as cash. Then, when you decide to buy a stock or fund, you place an order. The brokerage finds someone who wants to sell at your price (or close to it), and the trade happens. Your cash becomes a stock or fund share. The seller's stock becomes cash.

It's all just ownership changing hands. You're not creating money out of thin air. You're buying something that someone else is selling. The price is just what you both agree it's worth at that moment.

When millions of people are doing this every day, prices adjust constantly based on what people are willing to pay. That's how markets work. It's just supply and demand, happening millions of times per second.

---

## Automation: The Secret Weapon

Here's the thing about investing: the hardest part isn't picking the right investments. It's actually doing it consistently, month after month, year after year.

This is where automation comes in. Instead of having to remember to invest every month, you just set it up once. Link your bank account, set an amount, pick a date, and forget about it. Every month, the money automatically transfers and gets invested.

This is powerful for a few reasons. First, you can't forget. Second, you can't talk yourself out of it when the market is scary. Third, you're buying consistently, which means you're buying at different prices over time. Sometimes you'll buy high, sometimes low. Over the long term, this averages out and works in your favor.

Most successful investors automate everything. They set up automatic contributions, automatic dividend reinvestment, automatic rebalancing. They remove themselves from the decision-making process as much as possible. Because the more decisions you have to make, the more opportunities you have to make bad ones.

---

## The Big Picture

Here's what you need to know: you'll probably want both types of accounts eventually. Start with a Roth IRA for retirement savings. Then, if you have more money to invest beyond the Roth IRA limits, open a regular brokerage account.

The accounts themselves aren't complicated. They're just containers. The important part is what you put in them and how consistently you contribute. Get that right, and the accounts will take care of themselves.

""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is a key advantage of a Roth IRA for many young investors?",
      "options": [
        "You can withdraw unlimited amounts anytime with no penalty",
        "You pay taxes now; growth and qualified withdrawals can be tax-free later",
        "It replaces the need for an emergency fund",
        "It has no contribution limits"
      ],
      "correct_index": 1,
      "explanation": "Roth contributions are after-tax; in return, growth and qualified withdrawals in retirement can be tax-free."
    },
    {
      "type": "concept_check",
      "question": "When might you use a regular brokerage account instead of only a retirement account?",
      "options": [
        "You never should; retirement accounts are always better",
        "When you want to invest beyond IRA limits or for goals before retirement",
        "Only when you have already maxed a 401(k)",
        "Brokerage accounts are only for professional traders"
      ],
      "correct_index": 1,
      "explanation": "Brokerage accounts are flexible: no contribution caps and no early-withdrawal rules, so they suit non-retirement goals too."
    }
  ],
  "quiz": [
    {"question": "Which account lets your investments grow tax‑free for retirement?", "options": ["Brokerage account", "Roth IRA", "Savings account", "Checking account"], "correct": 1},
    {"question": "Why is it important to have an emergency fund before investing heavily?", "options": ["So you can invest the emergency fund in stocks", "To avoid selling investments at a loss when life happens", "Emergency funds are not recommended", "To get the highest possible return"], "correct": 1},
    {"question": "What typically happens when you withdraw from a Roth IRA before age 59½?", "options": ["You always pay a penalty on any amount", "Contributions can often be withdrawn without penalty; earnings may be taxed/penalized", "The account is closed permanently", "You get a tax refund"], "correct": 1}
  ]},
            {"id": "first_time_mindset", "title": "First Time Investor Mindset", "content": 
"""
Investing isn't just about numbers and strategies. It's also about how you think. Your mindset matters more than you might realize.

Most people think investing is about being smart. But really, it's about being patient. It's about not panicking when things get scary. It's about sticking to a plan even when your emotions are telling you to do something else.

Let me tell you about the psychological traps that catch most investors, and how to avoid them.

---

## Fear: The Enemy of Good Decisions

Fear is normal. Everyone feels it. The market crashes, your portfolio drops 20%, and your first instinct is to sell everything and run. That's human nature.

But here's the thing: fear makes you do stupid things. It makes you sell at the bottom, right when you should be buying. It makes you wait on the sidelines, missing out on recoveries. It makes you second-guess every decision.

The key is to recognize fear for what it is: an emotion, not a signal. When you feel scared, that's not the market telling you something. That's your brain's ancient survival instincts kicking in. Your brain doesn't understand that a 20% drop in your portfolio isn't the same as a bear attacking you.

So what do you do? You acknowledge the fear. You feel it. But you don't act on it. You stick to your plan. You remember why you're investing in the first place. You remember that markets have always recovered, eventually.

The investors who succeed aren't the ones who never feel fear. They're the ones who feel it and do the right thing anyway.

---

## FOMO: The Siren Song of Hot Stocks

FOMO is short for "fear of missing out," and it's probably responsible for more bad investment decisions than anything else.

You see a stock going up. Everyone's talking about it. Your friends are making money. You feel like you're missing out. So you buy in, right at the top. Then it crashes, and you're left holding the bag.

This happens over and over. People chase whatever's hot. They buy high because they're afraid of missing out. Then they sell low because they're afraid of losing more. It's a recipe for disaster.

The antidote? Have a plan. Know what you're investing in and why. Don't invest in something just because everyone else is. Don't invest in something you don't understand. And most importantly, don't check your portfolio constantly and compare it to what other people are doing.

Your investment journey is yours. It doesn't matter what other people are doing. What matters is whether you're sticking to your plan and making progress toward your goals.

---

## Patience: The Superpower

Patience is the most underrated skill in investing. Most people don't have it. They want results now. They want to see their money grow immediately. When it doesn't, they get frustrated and start making bad decisions.

But investing doesn't work like that. It's slow. It's boring. It's years of small gains, occasional losses, and gradual growth. The people who get wealthy from investing aren't the ones who made one brilliant trade. They're the ones who invested consistently for decades and let time do its thing.

Patience means not checking your portfolio every day. It means not panicking when the market drops. It means sticking with your plan even when it feels like nothing is happening. It means trusting the process, even when you can't see immediate results.

This is hard. We're wired for instant gratification. But investing rewards patience more than anything else. The longer you can wait, the more time your money has to grow. It's that simple.

---

## Setting Realistic Expectations

Here's something that trips up a lot of beginners: they expect investing to be exciting. They expect to beat the market. They expect to get rich quick.

But that's not how it works. Investing is boring. You're not going to beat the market. You're probably going to match it, and that's actually really good. You're not going to get rich quick. You're going to get rich slowly, over decades.

Setting realistic expectations is crucial. If you expect 20% returns every year, you're going to be disappointed. If you expect to beat the market, you're going to take unnecessary risks. If you expect to get rich in a year, you're going to make bad decisions.

The reality? If you can earn 7-10% per year over the long term, you're doing great. That might not sound exciting, but over 30 years, that turns $100,000 into $700,000 or more. That's real money.

So set your expectations accordingly. You're not trying to get rich quick. You're trying to build wealth slowly, steadily, over time. And that's actually a lot more achievable than trying to beat the market.

---

## The Bottom Line

Investing is as much about psychology as it is about finance. The math is simple. The hard part is managing your emotions, staying patient, and sticking to your plan when everything in you wants to do something else.

The good news? You can learn this. You can get better at it. It just takes practice. Start small. Make mistakes. Learn from them. And most importantly, be patient with yourself.

You're going to feel fear. You're going to feel FOMO. You're going to want to check your portfolio constantly. That's all normal. The key is recognizing these feelings for what they are, and not letting them drive your decisions.

Investing is a marathon, not a sprint. The people who finish aren't the fastest. They're the ones who keep going, even when it's hard.

""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is the best response when you feel FOMO about a hot stock?",
      "options": [
        "Buy immediately so you don't miss out",
        "Stick to your plan and only invest in what you understand and have researched",
        "Sell your current investments to free up cash",
        "Check social media for more tips"
      ],
      "correct_index": 1,
      "explanation": "FOMO leads to buying high. A plan and discipline help you avoid chasing trends and making emotional decisions."
    },
    {
      "type": "concept_check",
      "question": "Why is patience called a 'superpower' in investing?",
      "options": [
        "It guarantees higher returns every year",
        "Time and consistency do most of the work; impatience leads to costly mistakes",
        "Only patient people are allowed to invest",
        "Patience replaces the need for diversification"
      ],
      "correct_index": 1,
      "explanation": "Compounding and recovery from downturns need time. Impatient investors often sell low and buy high."
    }
  ],
  "quiz": [
    {"question": "Which mindset habit can help you avoid impulsive investment decisions?", "options": ["Reacting to every market headline", "Focusing on your long‑term plan", "Following every social media trend", "Trying to time the market"], "correct": 1},
    {"question": "What is a realistic long-term return expectation for a diversified stock portfolio?", "options": ["20% or more every year", "7–10% per year over decades", "Zero; stocks don't grow", "Exactly the inflation rate"], "correct": 1},
    {"question": "When you feel fear during a market crash, what is usually the best course of action?", "options": ["Sell everything immediately", "Stick to your plan and avoid locking in losses", "Wait until the news is positive to invest again", "Move everything to the hottest sector"], "correct": 1}
  ]}
        ]
    },
    "investor_insight": {
        "number": "2",
        "title": "Investor Insight",
        "level": "intermediate",
        "icon": "🔍",
        "description": "Deeper thinking about markets and behavior.",
        "source": "Various",
        "lessons": [
            {
  "id": "what_moves_markets",
  "title": "What Moves Markets",
  "content": 
"""
# 📊 What Moves Markets?

Financial markets are constantly shifting, but most of that movement is driven by a small set of powerful forces. Understanding these helps you move from reacting emotionally to thinking strategically.

---

## 1. 📈 Earnings — The Core Signal

**Earnings** reflect how much profit a company generates. They act as a scoreboard for business performance.

### Why they matter:
- Strong earnings often increase investor confidence.
- Weak or surprising results can trigger rapid price drops.
- Markets care more about **changes and expectations** than raw numbers.

### Visual:
Revenue → Costs → Earnings → Investor Reaction → Stock Price


💡 *Think of earnings as the “report card” investors use to judge companies.*

---

## 2. 💵 Interest Rates — The Price of Money

Interest rates influence nearly every asset class.

### When rates rise:
- Borrowing becomes more expensive.
- Stock valuations tend to fall.
- Bonds become more attractive.

### When rates fall:
- Businesses invest more.
- Consumers spend more.
- Stocks usually benefit.

### Visual:
Low Rates → Cheap Loans → More Growth → Higher Asset Prices
High Rates → Expensive Loans → Slower Growth → Lower Prices


---

## 3. 🔥 Inflation — The Silent Force

Inflation measures how quickly prices rise over time.

### Why investors care:
- High inflation reduces purchasing power.
- It often leads to higher interest rates.
- It changes which assets perform best.

### Common market reactions:
| Inflation Environment | Typical Winners |
|----------------------|-----------------|
| Low & Stable         | Stocks, Growth  |
| High & Rising        | Commodities, Real Assets |

---

## 4. 📰 News Cycles — Short-Term Volatility

Markets don’t wait for perfect information — they move on headlines.

### Examples of market-moving news:
- Central bank announcements  
- Corporate scandals  
- Geopolitical conflicts  
- New regulations  

### Visual:
News → Investor Emotion → Rapid Trading → Price Swings


⚠️ News often drives **short-term noise**, not long-term value.

---

## 5. 🧠 Economic Expectations — Beliefs Shape Reality

Markets respond more to **what people think will happen** than what already happened.

### Why expectations matter:
- If investors expect a recession, they sell before it happens.
- If growth is expected, prices rise early.
- Perception often becomes self-fulfilling.

### Visual:
Expectation → Behavior → Market Movement → Economic Outcome


---

## Putting It All Together

Markets move due to a mix of:
- **Fundamentals** (earnings, inflation)
- **Policy** (interest rates)
- **Psychology** (news and expectations)

Successful investors learn to:
- Filter noise from signal  
- Think in probabilities, not certainties  
- Focus on long-term drivers  

---

## 🧪 Try This Reflection

Think about a recent market event:
- Was it driven by **real data** or **investor emotion**?
- Did prices move because something changed — or because people *thought* it would?

This habit builds true market intuition.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "When interest rates rise, what typically happens to stock valuations?",
      "options": [
        "They always rise because the economy is strong",
        "They often fall because borrowing costs increase and future earnings are discounted more",
        "Rates have no effect on stocks",
        "Only bond prices are affected"
      ],
      "correct_index": 1,
      "explanation": "Higher rates make borrowing costlier and reduce the present value of future earnings, which can pressure stock valuations."
    },
    {
      "type": "concept_check",
      "question": "Why do markets often move on news before the full data is known?",
      "options": [
        "Because all news is fake",
        "Because prices reflect expectations and reactions, not just the news itself",
        "Only central banks move markets",
        "Earnings are the only driver"
      ],
      "correct_index": 1,
      "explanation": "Markets discount expectations; when news hits, investors reassess and trade, so prices move on reaction as much as on the event."
    }
  ],
  "quiz": [
    {
      "question": "Which factor most directly influences investor behavior before an economic event actually occurs?",
      "options": [
        "Corporate earnings",
        "Inflation rate",
        "Economic expectations",
        "Interest rates"
      ],
      "correct": 2
    },
    {
      "question": "What is usually the effect of high and rising inflation on many growth-oriented stocks?",
      "options": [
        "They always outperform",
        "They often face pressure as rates rise and discount rates increase",
        "Inflation has no effect",
        "Only bonds are affected"
      ],
      "correct": 1
    },
    {
      "question": "Why is it hard to profit from trading on news alone?",
      "options": [
        "News is always wrong",
        "Prices adjust quickly and you're competing with many others; timing is very difficult",
        "Only professionals can trade on news",
        "News doesn't move markets"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "investor_psychology",
  "title": "Investor Psychology",
  "content":
"""
# 🧠 Investor Psychology

Markets are not driven by numbers alone — they are driven by **human behavior**. Emotions, biases, and shared stories often explain why prices move more dramatically than fundamentals justify.

---

## 1. 🐑 Herd Behavior — Following the Crowd

Herd behavior occurs when investors copy what others are doing instead of thinking independently.

### Why it happens:
- Fear of missing out (FOMO)
- Belief that “everyone else knows something”
- Social proof feels safer than analysis

### Visual:
One buyer → More buyers → Media attention → Mass participation


⚠️ Herds amplify both **bubbles** and **crashes**.

---

## 2. 🚨 Panic Selling — Fear Takes Control

Panic selling is emotionally driven selling during sharp market drops.

### Common triggers:
- Sudden bad news  
- Rapid price declines  
- Loss of confidence  

### Emotional cycle:
Shock → Fear → Selling → Regret → Reflection


Most panic selling locks in losses that might have recovered.

---

## 3. 🧠 Overconfidence — The Invisible Risk

Overconfidence makes investors believe they are more skilled than they really are.

### Typical symptoms:
- Trading too frequently  
- Ignoring risk  
- Underestimating uncertainty  

### Visual:
Early success → Confidence spike → Larger risks → Bigger losses


Overconfidence is dangerous because it **feels like intelligence**.

---

## 4. 📖 Market Narratives — Stories That Move Money

Narratives are simple explanations people use to justify complex market behavior.

Examples:
- “Tech will always outperform”
- “This time is different”
- “The economy is about to collapse”

### Narrative feedback loop:
Story → Belief → Investment behavior → Price movement → Stronger story


Markets often move on **stories first, data second**.

---

## Putting It All Together

These forces interact constantly:

| Bias | Risk |
|------|------|
| Herd behavior | Buying near peaks |
| Panic selling | Selling near bottoms |
| Overconfidence | Excessive risk |
| Narratives | Ignoring evidence |

Great investors focus on:
- Self-awareness  
- Long-term thinking  
- Rules over emotions  

---

## 🎯 Self-Check Exercise

Next time you feel the urge to trade, ask:
- Am I reacting to **information or emotion**?
- Would I make this decision if no one else were watching?

Building this habit protects you from your own brain.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is a common result of panic selling during a sharp market drop?",
      "options": [
        "Investors usually lock in losses and miss the eventual recovery",
        "Panic selling always improves long-term returns",
        "Only inexperienced investors panic",
        "Markets never recover after panic"
      ],
      "correct_index": 0,
      "explanation": "Selling in a panic crystallizes losses and often means missing the rebound. Staying the course or adding per plan is usually better for long-term outcomes."
    },
    {
      "type": "concept_check",
      "question": "How do market narratives often affect prices?",
      "options": [
        "Narratives have no effect",
        "Stories can drive behavior and prices before data confirms them",
        "Only earnings matter",
        "Narratives are always correct"
      ],
      "correct_index": 1,
      "explanation": "Narratives shape beliefs and trading; prices can move on stories first and data later, which is why discipline and a plan matter."
    }
  ],
  "quiz": [
    {
      "question": "Which psychological factor most often causes investors to buy at market peaks?",
      "options": [
        "Overconfidence",
        "Herd behavior",
        "Panic selling",
        "Risk management"
      ],
      "correct": 1
    },
    {
      "question": "What is the main risk of overconfidence in investing?",
      "options": [
        "It has no real effect",
        "It leads to taking excessive risk and trading too much",
        "It only affects beginners",
        "It guarantees losses"
      ],
      "correct": 1
    },
    {
      "question": "Why is 'following the crowd' dangerous in markets?",
      "options": [
        "The crowd is always wrong",
        "Herds can push prices to extremes in both directions, leading to buying high and selling low",
        "Only professionals should follow the crowd",
        "Crowds never move markets"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "hype_vs_fundamentals",
  "title": "Hype vs Fundamentals",
  "content":
"""
# 🔊 Hype vs Fundamentals

Modern markets are louder than ever. With constant headlines, trending posts, and viral opinions, investors must learn to separate **signal from noise**.

---

## 1. 📢 Media Noise — Information Overload

Media noise refers to the flood of short-term news that creates urgency without lasting importance.

### Why it’s dangerous:
- Encourages frequent trading  
- Amplifies fear and excitement  
- Distracts from long-term goals  

### Visual:
Headlines → Emotion → Action → Regret


Not all information is useful — some of it is just **distraction in disguise**.

---

## 2. 📱 Social Media — Crowd Emotion at Scale

Social platforms turn opinions into market-moving forces.

### Common effects:
- Viral stock picks  
- Echo chambers  
- Fear of missing out  

### Feedback loop:
Post → Hype → Buying → Price spike → More posts


Social media accelerates herd behavior faster than any tool in history.

---

## 3. 🎲 Speculation — Betting on Uncertainty

Speculation focuses on short-term price movement rather than business value.

### Speculators often rely on:
- Timing the market  
- Rumors or trends  
- Quick profits  

### Visual:
Guess → Trade → Volatility → Stress


Speculation feels exciting, but it replaces strategy with luck.

---

## 4. 🌱 Long-Term Value — Quiet but Powerful

Long-term investing focuses on steady growth over time.

### Core principles:
- Compounding  
- Patience  
- Fundamentals over hype  

### Visual:
Time + Discipline → Compounding → Wealth


Long-term value usually looks boring — until you zoom out.

---

## The Big Contrast

| Hype-Based Thinking | Fundamentals-Based Thinking |
|---------------------|-----------------------------|
| React to news       | Focus on data               |
| Chase trends        | Hold quality                |
| Short-term gains    | Long-term growth            |
| Emotional decisions | Rational decisions          |

---

## 🎯 Reflection Exercise

Before acting on any market tip, ask:
- Will this matter in **5 years**?
- Or just **5 minutes**?

That single question filters 90% of bad decisions.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is the main risk of making investment decisions based on social media hype?",
      "options": [
        "Social media has no effect on markets",
        "You may buy high when everyone is excited and sell low when fear spreads",
        "Only young investors use social media",
        "It is the best source of stock picks"
      ],
      "correct_index": 1,
      "explanation": "Hype and FOMO on social media often coincide with peaks; by the time something is viral, a lot of the move may already have happened."
    },
    {
      "type": "concept_check",
      "question": "How can you tell if information is 'signal' vs 'noise'?",
      "options": [
        "All news is signal",
        "Ask whether it will matter for your strategy in 5 years, not just 5 days",
        "Only earnings reports are signal",
        "Noise is always from social media"
      ],
      "correct_index": 1,
      "explanation": "Signal supports long-term decisions; noise creates short-term emotion. A long-term lens helps filter."
    }
  ],
  "quiz": [
    {
      "question": "Which behavior best represents long-term value investing?",
      "options": [
        "Buying trending stocks on social media",
        "Trading frequently based on headlines",
        "Holding quality assets through volatility",
        "Timing short-term price movements"
      ],
      "correct": 2
    },
    {
      "question": "What does 'speculation' typically emphasize?",
      "options": [
        "Long-term business value",
        "Short-term price movement and timing",
        "Only dividends",
        "Diversification"
      ],
      "correct": 1
    },
    {
      "question": "Why might long-term value investing look 'boring'?",
      "options": [
        "It is always unprofitable",
        "It relies on patience and compounding rather than excitement; results show over years",
        "Only bonds are boring",
        "It requires no discipline"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "types_of_investing",
  "title": "Types of Investing",
  "content":
"""
# 🧭 Types of Investing

Not all investors play the same game. The way you invest depends on **how involved you want to be** and **how long you plan to stay invested**.

---

## 1. 🤖 Passive Investing — Letting the Market Work

Passive investing focuses on matching the market rather than trying to beat it.

### Key traits:
- Low effort  
- Broad diversification  
- Long-term mindset  

### Visual:
Market Growth → Index Fund → Time → Compounding


Passive strategies are built for consistency, not excitement.

---

## 2. 🎯 Active Investing — Trying to Outperform

Active investors attempt to beat the market through research, timing, and selection.

### Key traits:
- Frequent decisions  
- Higher risk  
- Higher costs  

### Visual:
Analysis → Trade → Monitor → Repeat


Active investing demands skill — and even then, results vary widely.

---

## 3. ⏳ Long-Term Investing — The Patience Advantage

Long-term investors stay invested for years, focusing on growth and compounding.

### Benefits:
- Smoother returns  
- Lower stress  
- Tax efficiency  

### Visual:
Small Gains + Time = Large Results


Time is the most powerful force in investing.

---

## 4. ⚡ Short-Term Investing — Speed and Speculation

Short-term investing targets quick profits over days, weeks, or months.

### Common approaches:
- Trading  
- Momentum strategies  
- Event-based moves  

### Visual:
Volatility → Opportunity → Risk


Short-term strategies require constant attention and emotional control.

---

## The Core Trade-Offs

| Style | Risk | Effort | Typical Outcome |
|------|------|--------|-----------------|
| Passive | Low | Low | Market-level returns |
| Active | High | High | Uncertain |
| Long-term | Lower | Lower | Compounding growth |
| Short-term | Higher | Higher | Volatile results |

---

## 🎯 Choose Your Time Horizon

Ask yourself:
- How long can I leave this money untouched?
- Do I enjoy managing investments — or would I rather automate?

Your strategy should fit **your lifestyle**, not just your goals.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is a key advantage of passive investing for most people?",
      "options": [
        "It guarantees beating the market",
        "Low effort, broad diversification, and typically lower costs",
        "It is only for experts",
        "It avoids all risk"
      ],
      "correct_index": 1,
      "explanation": "Passive strategies aim to match the market at low cost with minimal ongoing decisions, which suits many long-term investors."
    },
    {
      "type": "concept_check",
      "question": "Why does short-term investing usually require more emotional control?",
      "options": [
        "Short-term gains are guaranteed",
        "Volatility and quick decisions can trigger fear and greed",
        "Only long-term investing is emotional",
        "There are no emotions in investing"
      ],
      "correct_index": 1,
      "explanation": "Short-term moves are noisy and stressful; staying disciplined is harder when you're focused on daily or weekly swings."
    }
  ],
  "quiz": [
    {
      "question": "Which investing style relies most on patience and compounding over time?",
      "options": [
        "Active short-term trading",
        "Passive long-term investing",
        "Speculative investing",
        "Momentum trading"
      ],
      "correct": 1
    },
    {
      "question": "What typically characterizes active investing?",
      "options": [
        "No decisions required",
        "More decisions, higher costs, and uncertain ability to beat the market",
        "Zero risk",
        "Only index funds"
      ],
      "correct": 1
    },
    {
      "question": "Who is long-term investing best suited for?",
      "options": [
        "Only retirees",
        "Investors who can leave money invested for years and tolerate volatility",
        "Only day traders",
        "People who need the money next month"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "risk_portfolio_thinking",
  "title": "Risk and Portfolio Thinking",
  "content":
"""
# 🛡️ Risk and Portfolio Thinking

Investing isn’t about picking winners — it’s about **structuring your decisions** so you understand risk, build resilience, and support your long-term goals. Concepts like diversification, asset classes, and allocation are foundational tools every investor should master. :contentReference[oaicite:0]{index=0}

---

## 🌈 1. Diversification — Safety in Variety

### What it is:
Diversification means spreading money **across different types of investments** so one loss doesn’t dominate your returns. :contentReference[oaicite:1]{index=1}

### Why it matters:
- Different asset types seldom rise and fall together. :contentReference[oaicite:2]{index=2}
- When one asset underperforms, another might soften the blow. :contentReference[oaicite:3]{index=3}

Single Investment → Big Risk
Diversified Investments → Smoother Ride


💡 True diversification happens **between asset classes** (stocks, bonds, cash) and **within them** (multiple sectors, sizes, regions). :contentReference[oaicite:4]{index=4}

---

## 🧱 2. Asset Classes — The Building Blocks

Asset classes are broad categories of investments with similar risk and return behaviors:

- **Stocks** → ownership & growth potential  
- **Bonds** → income and often lower volatility  
- **Cash & equivalents** → liquidity and safety :contentReference[oaicite:5]{index=5}

Stocks ↔ Growth
Bonds ↔ Stability
Cash ↔ Safety


Each reacts differently under changing market conditions — that’s what gives diversification its power. :contentReference[oaicite:6]{index=6}

---

## ⚖️ 3. Asset Allocation — The Strategic Mix

### What it is:
Asset allocation is how you **divide your investable money** among different asset classes based on your goals, risk tolerance, and timeline. :contentReference[oaicite:7]{index=7}

### Why it’s powerful:
Your *mix* matters more than each individual investment — research shows allocation drives most of a portfolio’s return and volatility. :contentReference[oaicite:8]{index=8}

Aggressive → More Stocks
Balanced → Stocks + Bonds
Conservative → More Bonds/Cash


This mix determines how much risk you take and how your portfolio behaves over time. :contentReference[oaicite:9]{index=9}

---

## 🗂️ 4. Why Portfolios Exist — More Than Just a Collection

A portfolio is simply **the sum of your investments**, structured to reflect your financial plan. :contentReference[oaicite:10]{index=10}

Portfolios help you:

- Define risk levels  
- Break large financial goals into manageable parts  
- Avoid overexposure to any single trend or event :contentReference[oaicite:11]{index=11}

Proper planning and periodic **rebalancing** ensure your allocation stays aligned with your goals even as the market shifts. :contentReference[oaicite:12]{index=12}

---

## 🧠 Quick Visual Summary

Diversify → Spread Risks
Asset Classes → Variety of Behaviors
Allocation → Customized Risk
Portfolio → Strategy + Discipline


> **Remember:** Diversification does not guarantee profit, but it helps *manage* risk and reduces emotional stress during volatile markets. :contentReference[oaicite:13]{index=13}

---

## 🎯 Reflection Exercise

Ask yourself:
- If one part of my portfolio fell sharply, how much would it matter?
- Am I diversified across different risks — not just different investments?
- Does my allocation fit my long-term financial timeline?

These questions help you think like a strategist, not a short-term trader.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What does asset allocation primarily determine?",
      "options": [
        "Which individual stocks to pick",
        "Your mix of stocks, bonds, and cash — and thus your risk and return profile",
        "Only your tax rate",
        "The exact date you will retire"
      ],
      "correct_index": 1,
      "explanation": "Allocation is how you divide money across asset classes; it drives most of your portfolio's risk and return behavior."
    },
    {
      "type": "concept_check",
      "question": "Why do different asset classes (stocks, bonds, cash) behave differently?",
      "options": [
        "They don't; they always move together",
        "They have different roles: growth vs stability vs liquidity; they react differently to economic conditions",
        "Only stocks matter",
        "Bonds are the same as stocks"
      ],
      "correct_index": 1,
      "explanation": "Stocks, bonds, and cash have different return and risk characteristics, which is why combining them can smooth results."
    },
    {
      "type": "diversification_chart"
    }
  ],
  "quiz": [
    {
      "question": "What is the main purpose of diversification in investing?",
      "options": [
        "To eliminate all investment risk",
        "To reduce the impact of a single investment losing value",
        "To guarantee the highest possible returns",
        "To predict exact market movements"
      ],
      "correct": 1
    },
    {
      "question": "What is a portfolio in this context?",
      "options": [
        "A single stock",
        "Your collection of investments, structured to match your plan and goals",
        "Only retirement accounts",
        "A type of bond"
      ],
      "correct": 1
    },
    {
      "question": "Why might you rebalance your portfolio periodically?",
      "options": [
        "To maximize short-term gains every month",
        "To bring your mix back in line with your target allocation and manage risk",
        "Rebalancing is never recommended",
        "To avoid all taxes"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "reading_market_signals",
  "title": "Reading Basic Market Signals",
  "content":
"""
# 📈 Reading Basic Market Signals

Markets constantly send signals about direction, risk, and investor behavior. Learning to read these patterns helps you **respond strategically instead of emotionally**.

---

## 1. 📊 Trends — Uptrend vs Downtrend

A **trend** describes the general direction of market movement over time.

### Two primary types:
- **Uptrend**: prices make higher highs and higher lows  
- **Downtrend**: prices make lower highs and lower lows  

### Visual:
Uptrend: / / /
Downtrend: \ \ \


Trends reveal whether optimism or pessimism currently dominates the market.

---

## 2. 🌪️ Volatility — How Wild Are the Swings?

Volatility measures how much prices fluctuate.

### Low volatility:
- Calm markets  
- Small price changes  

### High volatility:
- Uncertainty  
- Large, rapid price swings  

### Visual:
Low Volatility: ~~~~~
High Volatility: ////\


Volatility doesn’t mean “bad” — it means **movement and risk**.

---

## 3. 🚀 Momentum — Strength of Movement

Momentum reflects how strongly prices are moving in one direction.

### High momentum:
- Strong buying or selling pressure  
- Trends accelerate  

### Weak momentum:
- Trend losing energy  
- Possible reversal  

### Visual:
Slow climb → Faster climb → Steep surge


Momentum shows not just *where* prices are going, but **how forcefully**.

---

## 4. 🔄 Cycles — Markets Move in Phases

Markets don’t move in straight lines — they move in **repeating cycles**.

### Typical phases:
1. Expansion (growth)
2. Peak (optimism)
3. Contraction (decline)
4. Trough (pessimism)

### Visual:
Growth → Peak → Decline → Recovery → Repeat


Understanding cycles helps investors avoid thinking “this time is different”.

---

## Putting It Together

| Signal | What It Tells You |
|--------|-------------------|
| Trend | Direction |
| Volatility | Risk level |
| Momentum | Strength |
| Cycles | Long-term context |

Great investors observe all four — not just price.

---

## 🎯 Reflection Exercise

Look at a recent market chart and ask:
- Is the trend rising or falling?
- Are price swings calm or chaotic?
- Is momentum building or fading?

This turns charts into **decision tools**, not just pictures.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What does an uptrend typically show on a chart?",
      "options": [
        "Lower highs and lower lows",
        "Higher highs and higher lows",
        "No clear pattern",
        "Only volume matters"
      ],
      "correct_index": 1,
      "explanation": "An uptrend is defined by a series of higher highs and higher lows, showing buyers are in control."
    },
    {
      "type": "concept_check",
      "question": "Why do investors watch volatility?",
      "options": [
        "Volatility guarantees returns",
        "It reflects how much prices swing and thus risk and potential for large moves",
        "Only professionals care about volatility",
        "Low volatility is always bad"
      ],
      "correct_index": 1,
      "explanation": "Volatility measures the size and speed of price changes, which helps set expectations for risk and behavior."
    }
  ],
  "quiz": [
    {
      "question": "Which signal best describes the strength behind a price movement?",
      "options": [
        "Trend",
        "Volatility",
        "Momentum",
        "Cycle"
      ],
      "correct": 2
    },
    {
      "question": "What do market cycles typically include?",
      "options": [
        "Only growth with no declines",
        "Phases such as expansion, peak, contraction, and trough that tend to repeat",
        "Cycles are random and unpredictable",
        "Only one cycle per decade"
      ],
      "correct": 1
    },
    {
      "question": "How can momentum help an investor?",
      "options": [
        "It guarantees future returns",
        "It shows how strongly prices are moving in one direction, which can confirm or question a trend",
        "Momentum is irrelevant",
        "Only day traders use momentum"
      ],
      "correct": 1
    }
  ]
},
        ]
    },
    "applied_investing": {
        "number": "3",
        "title": "Applied Investing",
        "level": "advanced",
        "icon": "⚙️",
        "description": "Practical steps to build and manage your portfolio.",
        "source": "Various",
        "lessons": [
           {
  "id": "costs_fees_taxes",
  "title": "Costs, Fees, and Taxes",
  "content":
"""
# 💰 Costs, Fees, and Taxes in Investing

Every investment comes with costs — and those costs **eat into your returns over time**. Understanding them is critical to building wealth efficiently.

---

## 1. 📊 Expense Ratios — The Ongoing Cost

Expense ratios are annual fees charged by funds to manage your money.

### Key points:
- Expressed as a percentage of assets  
- Paid automatically, reducing returns gradually  
- Lower expense ratios mean more of your money stays invested

### Visual:
$10,000 investment
Fund A 0.05% → $5/year
Fund B 1% → $100/year


Even small differences compound over decades.

### Example: The real cost of a 1% fee
Suppose you invest $10,000 and earn 7% per year for 30 years. With **no fee**, you'd end up with about $76,000. With a **1% annual expense ratio**, the same investment grows at only 6% net, so you'd have about $57,000 — a difference of **$19,000** lost to fees. That's roughly 25% of your potential wealth. Choosing a low-cost fund (e.g., 0.05%) keeps almost all of that growth in your pocket.

---

## 2. 💸 Trading Fees — Paying for Activity

Buying and selling investments often carries fees.

### Examples:
- Brokerage commissions  
- Transaction fees  
- Bid-ask spreads

### Visual:
Buy → Fee → Investment
Sell → Fee → Investment


High trading frequency amplifies these costs — another reason to **think long-term**.

---

## 3. 🏦 Taxes on Gains — Uncle Sam Wants a Cut

Capital gains taxes apply when you sell assets at a profit.

### Key points:
- **Short-term gains** are taxed higher (ordinary income rate)  
- **Long-term gains** enjoy lower rates  
- Reinvesting dividends may also trigger taxes

### Visual:
Purchase → Hold → Sell → Pay tax on gain


Tax planning can preserve wealth significantly.

---

## 4. ⚡ Low-Cost Funds — Keep More of What You Earn

Index funds and ETFs often have **minimal expense ratios**.

### Benefits:
- Reduced drag on returns  
- Efficient way to diversify  
- Ideal for long-term growth

### Visual:
High-cost fund → Lower net growth
Low-cost fund → Higher net growth


---

## 5. 🏦 Tax-Advantaged Accounts — Grow Without Penalty

Accounts like IRAs, 401(k)s, and HSAs shelter investments from taxes.

### Advantages:
- Tax-deferred growth  
- Tax-free withdrawals in some cases  
- Encourages long-term saving

### Visual:
Taxable account → Growth taxed yearly
Tax-advantaged account → Growth tax-deferred or free


---

## 💡 Key Takeaways

| Cost Type | Effect |
|-----------|--------|
| Expense ratio | Reduces compounding returns |
| Trading fees | Frequent trading magnifies costs |
| Taxes | Long-term strategies and tax-advantaged accounts mitigate |
| Low-cost funds | Keep more of your returns |

---

## 🎯 Reflection Exercise

Before investing, ask yourself:
- How much am I paying each year in fees?  
- Can I use tax-advantaged accounts?  
- Will lower-cost funds achieve similar results?

Being mindful of costs and taxes can **add years of growth** to your portfolio.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "Why does a 1% expense ratio matter over 30 years?",
      "options": [
        "It doesn't; only trading fees matter",
        "It compounds and can cost a large portion of your growth over decades",
        "Expense ratios are tax-deductible",
        "Only bonds have expense ratios"
      ],
      "correct_index": 1,
      "explanation": "Fees are deducted every year from your balance, so they compound against you. A 1% fee can significantly reduce ending wealth over 30 years."
    },
    {
      "type": "concept_check",
      "question": "What is a key advantage of long-term capital gains over short-term?",
      "options": [
        "There is no difference",
        "Long-term gains are typically taxed at lower rates",
        "Short-term gains are always better",
        "Only dividends are taxed"
      ],
      "correct_index": 1,
      "explanation": "Holding investments for more than one year usually qualifies for lower long-term capital gains rates, which preserves more of your return."
    }
  ],
  "quiz": [
    {
      "question": "Which strategy helps maximize long-term investment growth?",
      "options": [
        "Choosing high-expense funds for active management",
        "Minimizing fees and using tax-advantaged accounts",
        "Trading frequently to capture short-term gains",
        "Ignoring tax implications on dividends and gains"
      ],
      "correct": 1
    },
    {
      "question": "What is an expense ratio?",
      "options": [
        "A one-time fee when you open an account",
        "An annual fee charged by a fund, expressed as a percentage of assets",
        "The tax rate on your gains",
        "A fee only for selling stocks"
      ],
      "correct": 1
    },
    {
      "question": "How do tax-advantaged accounts like IRAs help investors?",
      "options": [
        "They eliminate all fees",
        "They allow growth to compound with reduced or deferred taxes",
        "They guarantee higher returns",
        "They are only for the wealthy"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "what_do_in_crash",
  "title": "What to do in a Market Crash",
  "content":
"""
# ⚠️ What to Do in a Market Crash

Market downturns are **normal and temporary**. Understanding how to respond can protect your portfolio and preserve long-term growth.

---

## 1. 📉 Market Downturns Are Normal

Stock markets experience declines regularly, and recoveries are historically consistent.

### Key points:
- Drawdowns happen in all markets  
- Historical data shows markets typically recover over time  
- Downturns are opportunities for disciplined investors

### Visual:
Peak → Decline → Trough → Recovery → New Peak


Knowing that downturns are expected reduces panic.

### A brief historical perspective
Major indices have experienced sharp drops many times (e.g., 2008–2009, March 2020, and others). In each case, investors who **stayed invested** and continued contributing saw their portfolios eventually recover and often reach new highs. Those who sold near the bottom locked in losses and missed the rebound. The pattern doesn't guarantee the future, but it illustrates why discipline and a long-term plan matter more than reacting to short-term fear.

---

## 2. 🚫 Avoid Panic Selling

Panic selling locks in losses and often results in **missing the recovery**.

### Common mistakes:
- Selling when prices are falling  
- Reacting to short-term headlines  
- Letting emotions override strategy

### Visual:
Market falls → Investor sells → Price rebounds → Missed gains


Patience and discipline outperform emotional reactions.

---

## 3. ⏱️ Historical Recoveries

Even severe bear markets eventually rebound.

### Observations:
- Short-term drops are sharper than recoveries  
- Long-term investors benefit from staying invested  
- Rebalancing ensures portfolio stays aligned with goals

### Visual:
Crash → Hold → Recovery → Compound growth


Rebalancing can be done strategically rather than reacting impulsively.

---

## 4. 🧠 Emotional Discipline

Investing is as much **psychology** as analysis.

### Tips:
- Recognize emotional triggers  
- Follow a pre-defined investment plan  
- Avoid checking portfolios constantly during volatility

### Visual:
Emotion → Impulse → Mistake
Discipline → Strategy → Growth


Controlling emotions ensures decisions are based on **strategy, not fear**.

---

## 💡 Quick Takeaways

| Principle | Action |
|-----------|--------|
| Market downturns | Expect them |
| Panic selling | Avoid |
| Historical recoveries | Remain invested |
| Emotional discipline | Stick to your plan |

---

## 🎯 Reflection Exercise

Ask yourself:
- Am I reacting to emotion or long-term goals?  
- Could staying invested benefit me in the future?  
- Is my portfolio aligned with my risk tolerance?

Taking a calm, disciplined approach turns market drops into opportunities.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "Why does panic selling often hurt long-term results?",
      "options": [
        "It doesn't; selling always protects you",
        "It locks in losses and can cause you to miss the eventual recovery",
        "Only beginners panic",
        "Markets never recover"
      ],
      "correct_index": 1,
      "explanation": "Selling after a drop turns paper losses into real ones and often means missing the rebound. Staying invested has historically rewarded disciplined investors."
    },
    {
      "type": "concept_check",
      "question": "What can rebalancing during a crash help you do?",
      "options": [
        "Time the exact bottom",
        "Restore your target allocation by buying assets that have fallen (e.g., stocks) with proceeds from those that held up (e.g., bonds)",
        "Avoid all losses",
        "Only sell bonds"
      ],
      "correct_index": 1,
      "explanation": "Rebalancing forces a disciplined process: trim what's gone up and add to what's down, which can improve long-term outcomes."
    }
  ],
  "quiz": [
    {
      "question": "During a market downturn, what is generally considered the best course of action?",
      "options": [
        "Sell all investments immediately",
        "Panic and buy new high-risk stocks",
        "Stay invested and maintain discipline",
        "Ignore portfolio allocation completely"
      ],
      "correct": 2
    },
    {
      "question": "Why are market downturns called 'normal'?",
      "options": [
        "They only happen once a decade",
        "History shows they occur regularly and have typically been followed by recoveries",
        "They don't affect long-term investors",
        "They are always mild"
      ],
      "correct": 1
    },
    {
      "question": "What is a key benefit of having a written investment plan before a crash?",
      "options": [
        "It guarantees no losses",
        "It gives you clear rules to follow so emotion doesn't drive decisions",
        "Plans are only for professionals",
        "You should change your plan during a crash"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "setting_long_term_structure",
  "title": "Setting a Long-Term Structure",
  "content":
"""
# 🏗️ Setting a Long-Term Investment Structure

Successful investing is less about timing the market and more about **consistent, disciplined execution**. Building a long-term structure helps you stay on track through ups and downs.

---

## 1. 💵 Recurring Investments — Dollar-Cost Averaging

Investing a fixed amount regularly reduces risk from market timing.

### Benefits:
- Smooths out purchase prices over time  
- Encourages discipline  
- Reduces stress from short-term fluctuations

### Visual:
Invest $100/month
Month 1 → Price $10 → 10 shares
Month 2 → Price $20 → 5 shares
Average cost → Balanced over time


---

## 2. 📅 Contribution Schedules — Consistency Matters

Decide **how often** to contribute based on your income and goals.

### Options:
- Weekly, bi-weekly, or monthly contributions  
- Automated transfers ensure habit formation  
- Align schedule with paycheck or budgeting cycles

### Visual:
Weekly → Small, frequent contributions
Monthly → Larger, scheduled contributions


Consistency compounds over time.

---

## 3. ⚖️ Rebalancing Annually — Keep Risk in Check

Portfolios drift as assets perform differently. Annual rebalancing restores your target allocation.

### Key points:
- Sell overweight assets  
- Buy underweight assets  
- Maintains intended risk profile

### Visual:
Stocks 60% → 65% after growth
Bonds 40% → 35%
Rebalance → Stocks 60%, Bonds 40%


---

## 4. 🔇 Ignoring Daily Noise — Stay Focused on Goals

Short-term market swings are normal. Avoid reacting to every headline.

### Tips:
- Limit portfolio checks  
- Focus on long-term objectives  
- Avoid frequent trading

### Visual:
Daily news → Emotional reaction → Mistake
Focus on long-term → Strategy → Growth


---

## 5. 📝 Reviewing Goals Annually — Course Correction

Check your plan each year to ensure it aligns with goals and risk tolerance.

### Steps:
- Evaluate asset allocation  
- Adjust contributions if needed  
- Consider life changes

### Visual:
Annual review → Adjustments → Continue disciplined investing


---

## 6. 🔑 Staying Consistent — The Power of Discipline

The combination of regular investing, ignoring noise, and periodic rebalancing produces **steady long-term growth**.

### Visual:
Recurring investments + Rebalancing + Patience → Compounded returns


---

## 🎯 Reflection Exercise

Ask yourself:
- Am I investing regularly, regardless of market movements?  
- Do I review and rebalance my portfolio systematically?  
- Am I staying consistent even when the market is volatile?

Building a structured approach reduces stress and improves long-term outcomes.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is dollar-cost averaging?",
      "options": [
        "Investing only when the market is down",
        "Investing a fixed amount on a regular schedule regardless of price",
        "Selling the same amount every month",
        "Only for bonds"
      ],
      "correct_index": 1,
      "explanation": "Dollar-cost averaging means investing the same amount on a schedule (e.g., monthly), which can smooth out the average purchase price over time."
    },
    {
      "type": "concept_check",
      "question": "Why is rebalancing annually useful?",
      "options": [
        "It guarantees higher returns",
        "It restores your target allocation so risk doesn't drift over time",
        "It is only for day traders",
        "It eliminates all volatility"
      ],
      "correct_index": 1,
      "explanation": "Over time, some assets grow more than others. Rebalancing brings your mix back to your target and helps control risk."
    },
    {
      "type": "compound_interest",
      "principal": 5000,
      "monthly": 200,
      "rate": 7.0,
      "years": 20
    }
  ],
  "quiz": [
    {
      "question": "Which practice best supports long-term investing success?",
      "options": [
        "Timing the market for daily gains",
        "Making large, irregular contributions",
        "Investing regularly and rebalancing annually",
        "Reacting to daily market news"
      ],
      "correct": 2
    },
    {
      "question": "What is a benefit of automating your contributions?",
      "options": [
        "It guarantees the best price every time",
        "It builds habit and reduces the chance of skipping investments when you feel nervous",
        "Automation is only for experts",
        "It avoids all fees"
      ],
      "correct": 1
    },
    {
      "question": "Why might you limit how often you check your portfolio?",
      "options": [
        "Checking frequently always improves returns",
        "Less frequent checking can reduce emotional reactions to short-term noise",
        "You should check every day",
        "Portfolios don't change"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "realistic_expectations",
  "title": "Realistic Expectations About Returns",
  "content":
"""
# 📊 Realistic Expectations About Returns

Investing doesn’t follow a straight upward line. Markets can be unpredictable in the short term, but over longer time horizons, risk and return tend to **smooth into more reliable patterns**.

---

## 📉 1. Markets Don’t Go Up Every Year

Calendar-year performance varies — some years are negative.

- Markets experience routine volatility and corrections.  
- Losses in some years are part of normal market behavior.  
- Even frequent drops don’t erase long-term growth trends.

↓ -15% (bad year)
↑ +20% (good year)
⬆️ Long-term average still positive


Short-term ups and downs are expected components of investing.

---

## 📈 2. Average Long-Term Returns

Over decades, broad indexes have tended to rise in value.

- Long-term averages encompass both positive and negative years.  
- A multi-year holding period increases the odds of positive outcomes. :contentReference[oaicite:0]{index=0}

5‑Year average → varied returns
10‑Year average → more consistency
20‑Year average → historically positive


The *average* is not the *every year* experience.

---

## 🔄 3. Volatility — The Price of Risk

Volatility measures how widely returns swing from year to year.

- Large drops and rallies happen regularly.  
- Volatility isn’t a failure — it’s a feature of risk assets.  
- Calm markets and turbulent markets are both normal.

Low volatility → smoother path
High volatility → wild swings but higher long-term rewards


Volatility exists because markets react to new information constantly.

---

## ⏳ 4. Time Horizon — Your Most Powerful Tool

The longer you stay invested, the more likely your returns are positive.

- Short-term horizons show wide variability.  
- Longer horizons tend to smooth outcomes and reduce the impact of negative years. :contentReference[oaicite:1]{index=1}

1 Year → high chance of negative return
10 Years → higher chance of positive return
20+ Years → consistently positive historically


Time in the market beats timing the market.

---

## 🧠 5. Why Patience Wins

Patience reduces the emotional impact of short-term losses.

- Trying to time peaks and troughs often leads to **locking in losses**.  
- Staying invested allows markets to recover and compound returns. :contentReference[oaicite:2]{index=2}

Sell low out of fear → miss the next rebound
Stay invested → capture best days and recover


Discipline and time together are powerful.

---

## 🌟 Key Takeaways

| Concept | What It Means |
|---------|----------------|
| Market volatility | Normal and recurring |
| Negative years | Expected in short term |
| Average returns | Long-run oriented |
| Time horizon | Longer = higher odds of gains |
| Patience | Rewards long-term investors |

Investing isn’t about avoiding volatility — it’s about **putting volatility to work for you**.

---

## 🎯 Reflection Exercise

Look up the annual returns of a major stock index (like the S&P 500) for the past 30 years:
- How many individual years were negative?
- How did the overall trajectory look?

This exercise emphasizes that **short-term losses don’t erase long-term gains**.

""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "Why might a negative year in the market still be consistent with long-term investing?",
      "options": [
        "Negative years never happen",
        "Short-term volatility is normal; long-term outcomes have historically included both up and down years",
        "Only bonds have negative years",
        "You should sell after any negative year"
      ],
      "correct_index": 1,
      "explanation": "Single years can be down; over many years, the average has tended positive. Staying invested through down years has historically been part of earning long-term returns."
    },
    {
      "type": "concept_check",
      "question": "What does 'time in the market' mean in practice?",
      "options": [
        "Checking the market every day",
        "Staying invested over long periods rather than trying to time entries and exits",
        "Only investing when the market is up",
        "Selling as soon as you have a gain"
      ],
      "correct_index": 1,
      "explanation": "Time in the market means holding through cycles so you capture compounding and recoveries; timing the market is notoriously difficult."
    }
  ],
  "quiz": [
    {
      "question": "Which statement best reflects market behavior over time?",
      "options": [
        "Markets go up every single year without fail",
        "Short-term returns can be negative, but long-term returns tend to average positive",
        "Average returns guarantee future results",
        "Volatility is a market failure"
      ],
      "correct": 1
    },
    {
      "question": "How does a longer time horizon generally affect the likelihood of positive returns?",
      "options": [
        "It has no effect",
        "Longer horizons tend to increase the chance of positive outcomes and smooth volatility",
        "Short horizons are always better",
        "Only 1-year horizons matter"
      ],
      "correct": 1
    },
    {
      "question": "Why is volatility sometimes called 'the price of risk'?",
      "options": [
        "Volatility is always bad",
        "Higher-return assets tend to have more short-term volatility; you accept swings for growth potential",
        "Only stocks have volatility",
        "Volatility guarantees losses"
      ],
      "correct": 1
    }
  ]
},
            {
  "id": "charts_101",
  "title": "Charts 101",
  "content":
"""
# 📈 Charts 101 — Reading Stock Charts Basics

Stock charts are visual tools that show how prices move over time. Learning to read them helps you understand **direction, momentum, and investor behavior**.

---

## 📊 1. What a Stock Chart Is

A stock chart plots the price of a security over a chosen timeframe.

### Key elements:
- **Price axis** (vertical): value of the stock  
- **Time axis** (horizontal): days, weeks, months, or years  
- **Volume**: how much was traded

Price ↑
● ● ●
● ●
Time —————————>


Charts turn data into stories about market behavior.

---

## 🕯️ 2. Types of Charts

There are three common chart types:

### • Line Chart
- Plots closing prices  
- Good for overall direction

### • Bar Chart
- Shows high, low, open, and close for each period

### • Candlestick Chart
- Similar to bar charts but easier to read visually  
- Represents open/close as a body and extremes as wicks

Candlestick Visual:
▲ bullish (close > open)
▼ bearish (close < open)


Candlesticks help spot patterns at a glance.

---

## 📈 3. Trends — Direction Matters

A **trend** is the overall direction of price movement.

- **Uptrend** → higher highs & higher lows  
- **Downtrend** → lower lows & lower highs  
- **Sideways** → no clear direction

Uptrend: / / /
Downtrend: \ \ \


Identifying trends helps decide whether to follow the momentum.

---

## 🔄 4. Volatility — How Wild Are the Moves?

Volatility refers to how much and how quickly prices change.

- Larger swings mean higher volatility  
- Calm trading suggests lower volatility

Low Volatility: ~~~~
High Volatility: ///\


Volatility helps set expectations for risk and potential price action.

---

## 🛑 5. Support & Resistance — Price Anchors

- **Support**: a price level where buyers historically step in  
- **Resistance**: a price level where sellers typically appear

Resistance
———————
●
●
Support
———————


These levels often act like invisible barriers to price movement.

---

## 🔊 6. Volume — Confirming Strength

Volume shows how many shares traded during a period.

- Rising prices with high volume → stronger move  
- Rising prices with low volume → weaker conviction

Volume bars below the main chart:
|| ||| ||||| |


Volume confirms *how serious* market participants are about a move.

---

## ⏱️ 7. Timeframes & Perspectives

Charts can be viewed over different timeframes:

- **Intraday (minutes/hours)** — short-term trading
- **Daily/Weekly** — swing & trend analysis
- **Monthly/Yearly** — long-term investing trends

Short view: noisy swings
Long view: broader trend


Different timeframes give different insights — always consider context.

---

## 📌 Putting It Together

| Element | What It Tells You |
|---------|------------------|
| Chart Type | How clearly price data is shown |
| Trend | Direction of market sentiment |
| Volatility | Degree of price fluctuation |
| Support/Resistance | Key price levels |
| Volume | Strength of moves |
| Timeframe | Depth of perspective |

---

## 🎯 Reflection Exercise

Open a stock chart right now:
- Can you identify the current **trend**?
- Where do you see support or resistance?
- Is volume increasing with price moves?

This hands-on look helps you translate theory into real-world chart reading.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What does 'support' mean on a chart?",
      "options": [
        "The highest price ever reached",
        "A price level where buying has historically stepped in and limited further decline",
        "Only volume",
        "The chart type"
      ],
      "correct_index": 1,
      "explanation": "Support is a level where demand has historically been strong enough to slow or halt declines."
    },
    {
      "type": "concept_check",
      "question": "Why is volume often shown below the price chart?",
      "options": [
        "It has no relationship to price",
        "Volume helps confirm whether a price move is backed by strong participation",
        "Only for decoration",
        "Volume is the same as price"
      ],
      "correct_index": 1,
      "explanation": "High volume on a price move suggests conviction; low volume can suggest a weaker or less reliable move."
    },
    {
      "type": "stock_market_demo"
    }
  ],
  "quiz": [
    {
      "question": "Which chart feature helps confirm the *strength* of a price move?",
      "options": [
        "Chart title",
        "Volume",
        "Support levels",
        "Candlestick color"
      ],
      "correct": 1
    },
    {
      "question": "What does a candlestick's body represent?",
      "options": [
        "Only volume",
        "The range between the open and close price for that period",
        "The high and low only",
        "Future prices"
      ],
      "correct": 1
    },
    {
      "question": "What is an uptrend typically characterized by?",
      "options": [
        "Lower highs and lower lows",
        "Higher highs and higher lows",
        "Flat volume only",
        "Only green candlesticks"
      ],
      "correct": 1
    }
  ]
},
            {"id": "market_pulse", "title": "Market Pulse", "content": """
# 📡 Market Pulse — Tracking Sentiment, News, and Key Indicators

Understanding the market's "pulse" means paying attention to **sentiment**, **news flow**, and **key indicators** — without letting short-term noise drive your long-term decisions. This lesson gives you a practical framework for reading the environment while staying focused on your plan.

---

## 1. 🧠 What Is Market Sentiment?

**Market sentiment** is the overall mood of investors: how optimistic or pessimistic they feel about the economy and asset prices.

### Why it matters
- **Extreme optimism** often coincides with high valuations and potential for pullbacks.
- **Extreme pessimism** often coincides with fear, selling, and potential opportunities for long-term buyers.
- Sentiment is a **contrarian indicator** when it reaches extremes: when everyone is fearful, opportunities can appear; when everyone is greedy, risk is often elevated.

### Common sentiment gauges
- **Surveys** (e.g., investor bullish/bearish percentages)
- **Put/call ratios** (options activity showing hedging vs speculation)
- **Volatility indices** (e.g., VIX — "fear gauge")
- **Headlines and social buzz** (qualitative; easy to overstate)

Sentiment does **not** tell you exactly when to buy or sell. It helps you understand **context** and avoid getting swept up in the crowd.

---

## 2. 📰 News Flow — Signal vs Noise

Markets react to news constantly. Not all news is equal.

### Types of news that move markets
- **Central bank decisions** (interest rates, forward guidance)
- **Economic data** (jobs, inflation, GDP)
- **Corporate earnings** and guidance
- **Geopolitical events** (trade, conflict, elections)
- **Regulatory or tax changes**

### How to use news without being ruled by it
- **Prioritize** information that affects long-term earnings and rates over one-day headlines.
- **Avoid** making big portfolio changes based on a single report or tweet.
- **Ask**: "Will this matter in 5 years, or only for 5 days?"

Short-term news creates **volatility**; your job is to keep your strategy intact and use dislocations only if they align with your plan.

---

## 3. 📊 Key Indicators Worth Watching

Some indicators help you gauge economic health and market context. You don't need to trade on them — they help you **stay oriented**.

### Economic indicators
- **Employment** (e.g., monthly jobs report) — strength or weakness in the labor market
- **Inflation** (CPI, PCE) — influences interest rates and Fed policy
- **GDP growth** — broad picture of economic expansion or contraction

### Market-based indicators
- **Yield curve** (short- vs long-term rates) — often watched for recession signals
- **Credit spreads** — how much extra yield investors demand for risk; widening can signal stress
- **Breadth** — how many stocks are participating in a rally or selloff (narrow rallies can be fragile)

### A word of caution
Indicators are **lagging** or **coincident** much of the time. They rarely give you a clear "buy now" or "sell now" signal. Use them to understand the **backdrop**, not to time the market.

---

## 4. 🎯 Putting It Into Practice

### For long-term investors
- **Do** check in on sentiment and news periodically so you're not surprised by big moves.
- **Do** use extreme fear as a reminder that selling into panic is usually costly.
- **Don't** change your allocation every time sentiment shifts or a new headline hits.
- **Don't** assume you can consistently time entries and exits using sentiment or news.

### Simple habit
When sentiment is very negative: ask yourself whether your **goals** and **time horizon** have changed. If not, stick to your plan. History shows that **fear often creates opportunities** for those who can stay disciplined — not because you can time the bottom, but because you avoid selling at the worst time.

---

## 5. 📌 Summary

| Concept | Takeaway |
|--------|----------|
| Sentiment | Gauges crowd mood; extremes can be contrarian signals. |
| News | Creates short-term volatility; focus on long-term impact. |
| Indicators | Provide context, not perfect timing. |
| Your role | Stay disciplined; use pulse to inform, not dictate, decisions. |

When market sentiment is very negative, long-term investors often find that **fear has created better entry points** — not because the bottom is predictable, but because panic selling has already happened. Your edge is discipline and a plan, not predicting the next headline.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "When sentiment is extremely fearful, what is a common mistake long-term investors make?",
      "options": [
        "Staying invested and rebalancing according to their plan",
        "Selling everything to avoid further losses",
        "Adding a small amount to their regular contributions",
        "Reviewing their goals and time horizon"
      ],
      "correct_index": 1,
      "explanation": "Selling in a panic locks in losses and often misses the recovery. Staying invested or adding according to a plan is usually better for long-term outcomes."
    },
    {
      "type": "concept_check",
      "question": "What is the main benefit of following news and sentiment for a long-term investor?",
      "options": [
        "To time exactly when to buy and sell",
        "To stay informed and avoid being surprised by big moves",
        "To trade on every headline",
        "To predict recessions precisely"
      ],
      "correct_index": 1,
      "explanation": "The goal is context and discipline — not timing every move. Knowing the backdrop helps you stick to your plan when others panic."
    }
  ],
  "quiz": [
    {"question": "When market sentiment is very negative, what does this typically signal for long-term investors?", "options": ["Don't invest until sentiment improves", "Fear may create buying opportunities", "The market is permanently broken", "Sell everything"], "correct": 1, "misconception_if_wrong": {0: "Extreme fear is often associated with market bottoms.", 2: "Markets recover historically; sentiment swings are normal.", 3: "Panicked selling often marks the worst time to sell."}},
    {"question": "Which of these is best used as context rather than a precise timing signal?", "options": ["A single jobs report", "Investor sentiment surveys", "Your own asset allocation plan", "Put/call ratios"], "correct": 1, "misconception_if_wrong": {0: "One report can move markets short-term but doesn't define your plan.", 2: "Sentiment gives context; it doesn't tell you exactly when to act.", 3: "Your plan should drive decisions; sentiment informs context."}},
    {"question": "What should you prioritize when big news hits the market?", "options": ["Selling immediately to avoid losses", "Whether the news changes your long-term goals or time horizon", "Copying what other investors are doing", "Checking sentiment surveys first"], "correct": 1}
  ]},
            {"id": "asset_allocation", "title": "Asset Allocation", "content": """
# ⚖️ Asset Allocation — Building Diversified Portfolios Across Asset Classes

**Asset allocation** is how you divide your investable money among **stocks**, **bonds**, **cash**, and sometimes other assets. It is one of the most important decisions you make as an investor — often more impactful than picking individual securities.

---

## 1. 🧱 Why Asset Classes Matter

Different **asset classes** behave differently over time.

### Stocks (equities)
- **Role**: Long-term growth; ownership in companies.
- **Typical behavior**: Higher return potential, higher volatility.
- **Best for**: Long time horizons (e.g., 10+ years), growth of purchasing power.

### Bonds (fixed income)
- **Role**: Income and relative stability.
- **Typical behavior**: Lower volatility than stocks; interest payments; can fall when rates rise.
- **Best for**: Shorter time horizons, dampening portfolio swings, income.

### Cash and cash equivalents
- **Role**: Liquidity and safety.
- **Typical behavior**: Minimal return; loses purchasing power to inflation over time.
- **Best for**: Emergency fund, short-term goals, and "dry powder" you might deploy.

No single asset class is "best" — the right **mix** depends on your goals, time horizon, and risk tolerance.

---

## 2. 🎯 What Is Asset Allocation?

**Asset allocation** is the decision of what **percentage** of your portfolio to hold in each asset class.

### Examples (conceptual)
- **Aggressive**: 90% stocks, 10% bonds — for long horizons and high risk tolerance.
- **Moderate/Balanced**: 60% stocks, 40% bonds — middle ground.
- **Conservative**: 30% stocks, 70% bonds/cash — for shorter horizons or lower risk tolerance.

Your allocation should **match your ability and willingness** to ride out market downturns. A portfolio that keeps you up at night will tempt you to sell at the wrong time.

---

## 3. 📐 How to Choose Your Allocation

### Step 1: Define your goals and time horizon
- **Retirement in 30 years** → more room for stocks.
- **House down payment in 3 years** → more bonds and cash.
- **Already retired, drawing income** → often more bonds/cash for stability.

### Step 2: Assess your risk tolerance
- How much can your portfolio fall in a bad year without you abandoning your plan?
- Be honest: if a 30% drop would make you sell, don't allocate like you can handle 50% stocks.

### Step 3: Choose a starting mix
- Use simple rules of thumb only as a **starting point** (e.g., "100 minus age" in stocks).
- Prefer **broad, low-cost funds** (e.g., total stock, total bond) to implement your mix.

### Step 4: Write it down and revisit periodically
- Put your target allocation in writing.
- Rebalance when your actual mix drifts too far from the target (e.g., once a year or when bands are exceeded).

---

## 4. 🔄 Rebalancing — Keeping Your Mix on Track

Over time, gains and losses will **shift** your percentages (e.g., stocks outperform and you end up with 70% stocks instead of 60%).

**Rebalancing** means bringing your portfolio back toward your target (e.g., selling some stocks and buying bonds, or directing new money to the underweight asset).

### Benefits
- Controls risk (stops you from becoming too concentrated in the winner).
- Encourages "buy low, sell high" in a disciplined way (you sell what went up and buy what went down relative to target).

You don't need to rebalance daily — **periodic** rebalancing (e.g., annually or when bands are hit) is usually enough.

---

## 5. 📌 What Asset Allocation Does *Not* Do

- It does **not** eliminate risk — it **manages** it.
- It does **not** guarantee maximum returns — it balances return potential with your comfort level.
- It does **not** replace the need for an emergency fund or a clear plan — those come first.

---

## 6. Summary Table

| Idea | Takeaway |
|------|----------|
| Asset classes | Stocks (growth), bonds (stability/income), cash (liquidity). |
| Allocation | Your chosen mix of those classes. |
| Goal | Match your risk tolerance and time horizon. |
| Rebalancing | Periodically restore your target mix. |

The goal of asset allocation is to **match your portfolio to your risk tolerance and time horizon** — not to eliminate risk or chase the highest possible short-term return.
""",
  "interactive_elements": [
    {
      "type": "concept_check",
      "question": "What is the main purpose of rebalancing a portfolio?",
      "options": [
        "To maximize short-term returns every year",
        "To bring the portfolio back toward your target allocation and control risk",
        "To sell all bonds and buy only stocks",
        "To avoid paying any taxes"
      ],
      "correct_index": 1,
      "explanation": "Rebalancing restores your intended mix and helps you avoid becoming too concentrated in one asset class."
    },
    {
      "type": "concept_check",
      "question": "Who typically benefits from a higher allocation to stocks?",
      "options": [
        "Someone who needs the money in 2 years",
        "Someone with a long time horizon and higher risk tolerance",
        "Someone who cannot tolerate any loss",
        "Someone who only holds cash"
      ],
      "correct_index": 1,
      "explanation": "Stocks are volatile; they suit long horizons and investors who can stay the course during downturns."
    },
    {
      "type": "diversification_chart"
    }
  ],
  "quiz": [
    {"question": "What is the goal of asset allocation?", "options": ["Eliminate all risk", "Maximize short-term gains", "Match your risk tolerance and time horizon", "Own only growth stocks"], "correct": 2, "misconception_if_wrong": {0: "No allocation eliminates risk; allocation manages it.", 1: "Asset allocation focuses on long-term goals, not short-term gains.", 3: "Growth stocks alone create concentration risk."}},
    {"question": "Which action best describes rebalancing?", "options": ["Selling everything and moving to cash", "Adjusting your holdings back toward your target mix", "Buying only the best-performing fund", "Avoiding bonds entirely"], "correct": 1},
    {"question": "Why might a conservative investor hold more bonds than stocks?", "options": ["Bonds always outperform stocks", "To reduce volatility and preserve capital", "To maximize growth over 30 years", "Because stocks are illegal in some accounts"], "correct": 1}
  ]}
        ]
    }
}


def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📚 Learning Modules</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Master investing from the fundamentals to advanced strategies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check what view to show
    if st.session_state.current_lesson:
        render_lesson_view()
    elif st.session_state.current_module:
        render_module_view()
    else:
        render_modules_list()


def render_modules_list():
    """Show all modules as clickable cards."""
    st.markdown("### Select a Module to Begin")
    
    # Progress
    total_lessons = sum(len(m['lessons']) for m in MODULES.values())
    completed = len(st.session_state.completed_lessons)
    st.progress(completed / total_lessons if total_lessons > 0 else 0)
    st.caption(f"Progress: {completed}/{total_lessons} lessons completed")
    
    st.markdown("---")
    
    # Module grid
    cols = st.columns(2)
    
    for i, (mod_id, mod) in enumerate(MODULES.items()):
        with cols[i % 2]:
            level_color = {
                'beginner': '#4ade80',
                'intermediate': '#fbbf24', 
                'advanced': '#f97316',
                'expert': '#ef4444'
            }.get(mod['level'], '#6366f1')
            
            # Count completed lessons in this module
            mod_completed = sum(1 for lesson in mod['lessons'] 
                              if f"{mod_id}_{lesson['id']}" in st.session_state.completed_lessons)
            
            st.markdown(f"""
            <div style="background: #16213e; border-radius: 12px; padding: 1.5rem; 
                        margin-bottom: 1rem; border-left: 4px solid {level_color};">
                <span style="color: {level_color}; font-size: 0.75rem; font-weight: 600; 
                             text-transform: uppercase;">{mod['level']}</span>
                <h3 style="color: white; margin: 0.5rem 0;">{mod['icon']} {mod['number']}. {mod['title']}</h3>
                <p style="color: #9ca3af; font-size: 0.9rem; margin: 0.5rem 0;">{mod['description']}</p>
                <p style="color: #6b7280; font-size: 0.8rem; margin: 0;">
                    📝 {len(mod['lessons'])} lessons | ✅ {mod_completed} completed
                </p>
                <p style="color: #4b5563; font-size: 0.7rem; margin-top: 0.5rem;">
                    📖 Sources: {mod.get('source', 'Various')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"▶️ Start {mod['title']}", key=f"start_{mod_id}", use_container_width=True):
                st.session_state.current_module = mod_id
                st.rerun()


def render_module_view():
    """Show lessons within a module."""
    mod_id = st.session_state.current_module
    mod = MODULES[mod_id]
    
    level_color = {
        'beginner': '#4ade80',
        'intermediate': '#fbbf24',
        'advanced': '#f97316',
        'expert': '#ef4444'
    }.get(mod['level'], '#6366f1')
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_module = None
            st.rerun()
    
    # Module header
    st.markdown(f"""
    <div style="background: #16213e; border-radius: 12px; padding: 2rem; margin: 1rem 0;
                border-left: 4px solid {level_color};">
        <span style="color: {level_color}; font-size: 0.85rem; font-weight: 600; 
                     text-transform: uppercase;">{mod['level']}</span>
        <h2 style="color: white; margin: 0.5rem 0;">{mod['icon']} {mod['number']}. {mod['title']}</h2>
        <p style="color: #9ca3af;">{mod['description']}</p>
        <p style="color: #4b5563; font-size: 0.8rem;">📖 Sources: {mod.get('source', 'Various')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Lessons")
    st.markdown("Click on a lesson to start learning:")
    
    for i, lesson in enumerate(mod['lessons']):
        lesson_key = f"{mod_id}_{lesson['id']}"
        is_completed = lesson_key in st.session_state.completed_lessons
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            status = "✅" if is_completed else f"{i+1}."
            if st.button(
                f"{status} {lesson['title']}", 
                key=f"lesson_{lesson_key}",
                use_container_width=True
            ):
                st.session_state.current_lesson = lesson['id']
                st.rerun()
        
        with col2:
            if is_completed:
                st.markdown("✅")


def render_lesson_view():
    """Show individual lesson content."""
    mod_id = st.session_state.current_module
    lesson_id = st.session_state.current_lesson
    mod = MODULES[mod_id]
    
    # Find current lesson
    lesson = next((l for l in mod['lessons'] if l['id'] == lesson_id), None)
    if not lesson:
        st.error("Lesson not found")
        return
    
    lesson_idx = next(i for i, l in enumerate(mod['lessons']) if l['id'] == lesson_id)
    lesson_key = f"{mod_id}_{lesson_id}"
    
    # Navigation header
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("← Back to Module", use_container_width=True):
            st.session_state.current_lesson = None
            st.rerun()
    
    with col3:
        if lesson_idx < len(mod['lessons']) - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.current_lesson = mod['lessons'][lesson_idx + 1]['id']
                st.rerun()
    
    # Lesson header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
                border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <p style="color: #6366f1; margin: 0;">{mod['icon']} {mod['title']} / Lesson {lesson_idx + 1} of {len(mod['lessons'])}</p>
        <h2 style="color: white; margin: 0.5rem 0;">{lesson['title']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Per-lesson progress + self-assessment (generic interactive layer)
    lesson_key = f"{mod_id}_{lesson_id}"
    progress_key = f"confidence_{lesson_key}"
    reflection_key = f"reflection_{lesson_key}"
    
    st.markdown("#### 🧭 How confident do you feel about this lesson?")
    confidence = st.slider(
        "Confidence (1 = still confused, 5 = very confident)",
        min_value=1,
        max_value=5,
        value=st.session_state.get(progress_key, 3),
        key=progress_key
    )
    st.caption("This is just for you – it helps you track how your understanding grows over time.")
    
    with st.expander("🧠 Jot down a quick takeaway or question"):
        st.text_area(
            "Write a brief note, key insight, or question to revisit later.",
            key=reflection_key,
            height=120,
        )
    
    # Render lesson content (markdown)
    render_lesson_content(mod_id, lesson_id)
    
    # Render interactive elements (after content, before quiz)
    render_lesson_interactive_elements(mod_id, lesson_id)
    
    # Render quiz if available
    render_lesson_quiz(mod_id, lesson_id)
    
    # Completion button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if lesson_key in st.session_state.completed_lessons:
            st.success("✅ Lesson completed!")
            if st.button("📖 Review Again", use_container_width=True):
                pass  # Already showing content
        else:
            if st.button("✅ Mark as Complete", type="primary", use_container_width=True):
                st.session_state.completed_lessons.add(lesson_key)
                st.balloons()
                st.success("Lesson completed! 🎉")
                
                # Auto-advance to next lesson
                if lesson_idx < len(mod['lessons']) - 1:
                    st.session_state.current_lesson = mod['lessons'][lesson_idx + 1]['id']
                    st.rerun()


def render_lesson_content(mod_id, lesson_id):
    """Render the actual lesson content (markdown only)."""
    
    # Try rendering from MODULES data first (markdown stored in lesson['content']).
    mod = MODULES.get(mod_id)
    if mod:
        lesson = next((l for l in mod['lessons'] if l['id'] == lesson_id), None)
        if lesson and lesson.get('content'):
            st.markdown(lesson['content'], unsafe_allow_html=True)
            return
    
    # Fallback to existing per-module render functions for compatibility
    if mod_id == "market_mechanics":
        render_market_mechanics_content(lesson_id)
    elif mod_id == "macro_economics":
        render_macro_economics_content(lesson_id)
    elif mod_id == "technical_analysis":
        render_technical_analysis_content(lesson_id)
    elif mod_id == "fundamental_analysis":
        render_fundamental_analysis_content(lesson_id)
    elif mod_id == "quant_strategies":
        render_quant_strategies_content(lesson_id)
    elif mod_id == "advanced_options":
        render_advanced_options_content(lesson_id)


def render_lesson_interactive_elements(mod_id, lesson_id):
    """Render interactive elements after lesson content but before quiz."""
    try:
        from pages.components.interactive_elements import (
            inflation_calculator,
            saving_vs_investing_comparison,
            compound_interest_calculator,
            concept_check,
            ai_tutor_sidebar,
            stock_market_movement_demo,
            supply_demand_visualization,
            portfolio_diversification_chart
        )
        
        # Look up lesson metadata if available
        mod = MODULES.get(mod_id)
        lesson = None
        interactive_elements = []
        lesson_title = ""
        if mod:
            lesson = next((l for l in mod['lessons'] if l['id'] == lesson_id), None)
            if lesson:
                lesson_title = lesson.get('title', '')
                interactive_elements = lesson.get('interactive_elements', [])
        
        # Render AI tutor sidebar for every lesson, even if no extra tools are defined
        ai_tutor_sidebar(lesson_id, lesson_title)
        
        # If this lesson doesn't define additional interactive tools, we're done
        if not interactive_elements:
            return
        
        # Add separator before interactive elements
        st.markdown("---")
        st.markdown("### Interactive Learning Tools")
        
        # Render other interactive elements
        for element in interactive_elements:
            element_type = element.get('type')
            if element_type == 'inflation_calculator':
                inflation_calculator(
                    initial_amount=element.get('initial_amount', 100),
                    years=element.get('years', 10),
                    default_inflation=element.get('inflation_rate', 3.0)
                )
            elif element_type == 'saving_vs_investing':
                saving_vs_investing_comparison(
                    initial=element.get('initial', 10000),
                    monthly=element.get('monthly', 500),
                    years=element.get('years', 20)
                )
            elif element_type == 'compound_interest':
                compound_interest_calculator(
                    principal=element.get('principal', 10000),
                    monthly=element.get('monthly', 500),
                    rate=element.get('rate', 7.0),
                    years=element.get('years', 30)
                )
            elif element_type == 'concept_check':
                concept_check(
                    question=element.get('question', ''),
                    options=element.get('options', []),
                    correct_index=element.get('correct_index', 0),
                    explanation=element.get('explanation', '')
                )
            elif element_type == 'stock_market_demo':
                stock_market_movement_demo()
            elif element_type == 'supply_demand_viz':
                supply_demand_visualization()
            elif element_type == 'diversification_chart':
                portfolio_diversification_chart()
    except ImportError as e:
        st.warning(f"Interactive components not available: {e}")


def render_lesson_quiz(mod_id, lesson_id):
    """Render a quiz for the lesson if available."""
    mod = MODULES.get(mod_id)
    if mod:
        lesson = next((l for l in mod['lessons'] if l['id'] == lesson_id), None)
        if lesson and lesson.get('quiz'):
            st.markdown("---")
            st.markdown("### ✅ Quick Check: Test Your Understanding")
            
            quiz_questions = lesson['quiz']
            quiz_key = f"quiz_{mod_id}_{lesson_id}"
            
            # Initialize quiz state
            if quiz_key not in st.session_state:
                st.session_state[quiz_key] = {
                    "answered": [False] * len(quiz_questions),
                    "selected": [None] * len(quiz_questions),
                    "submitted": False
                }
            
            quiz_state = st.session_state[quiz_key]
            
            for i, q in enumerate(quiz_questions):
                with st.container():
                    st.markdown(f"**Question {i+1}: {q['question']}**")
                    
                    # Radio button for options
                    selected_idx = st.radio(
                        label="hidden",
                        options=range(len(q['options'])),
                        format_func=lambda x: q['options'][x],
                        key=f"{quiz_key}_q{i}",
                        label_visibility="collapsed"
                    )
                    
                    quiz_state['selected'][i] = selected_idx
                    
                    # Check answer
                    if st.button("Check Answer", key=f"{quiz_key}_check_{i}"):
                        quiz_state['answered'][i] = True
                        if selected_idx == q['correct']:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. The correct answer is: **{q['options'][q['correct']]}**")
                            if selected_idx in q.get('misconception_if_wrong', {}):
                                st.info(f"💡 {q['misconception_if_wrong'][selected_idx]}")
                    
                    st.markdown("---")


# ==================== MODULE 1: MARKET MECHANICS ====================
def render_market_mechanics_content(lesson_id):
    """Render Market Mechanics module content."""
    
    if lesson_id == "stock_equity":
        st.markdown("*📖 Source: SEC Investor.gov, Investopedia*")
        
        st.markdown("""
        ## What is a Stock?
        
        A **stock** (also called equity or share) represents **ownership** in a company.
        When you buy a stock, you become a **shareholder** — you literally own a tiny piece of that company.
        
        Think of it this way: If a company is a pizza, each stock is a slice. When you buy shares,
        you're buying slices of that pizza.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Key Terms
            
            | Term | Definition |
            |------|------------|
            | **Share** | A single unit of ownership |
            | **Shareholder** | Someone who owns shares |
            | **Market Cap** | Total value (Price × Shares Outstanding) |
            | **Float** | Shares available to trade publicly |
            | **Outstanding Shares** | Total shares that exist |
            """)
        
        with col2:
            fig = go.Figure(go.Pie(
                values=[30, 25, 20, 15, 10],
                labels=['Institutions', 'Mutual Funds', 'Retail', 'Insiders', 'ETFs'],
                hole=0.4,
                marker_colors=['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe']
            ))
            fig.update_layout(
                title="Typical Stock Ownership Breakdown",
                height=300,
                margin=dict(t=50, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 💰 How Shareholders Make Money
        
        **1. Capital Appreciation**
        - The stock price goes up over time
        - You sell for more than you paid
        - Example: Buy at $100, sell at $150 = $50 profit per share
        
        **2. Dividends**
        - Company pays you a share of its profits
        - Usually paid quarterly (4x per year)
        - Not all companies pay dividends (growth companies often don't)
        - Example: $1.00 dividend × 100 shares = $100 per quarter
        """)
        
        st.info("""
        💡 **Key Insight from Vanguard**: When you buy a total stock market index fund like VTI,
        you're buying tiny pieces of ~4,000 companies at once! This provides instant diversification.
        """)
        
        st.markdown("""
        ### 🏢 Why Companies Issue Stock
        
        1. **Raise capital without debt** - No interest payments required
        2. **Reward employees** - Stock options attract top talent
        3. **Make acquisitions** - Use stock as currency to buy other companies
        4. **Provide liquidity for founders** - Early investors can sell shares
        
        ### ⚠️ Shareholder Rights
        
        As a shareholder, you typically have:
        - **Voting rights** on major company decisions
        - **Right to dividends** if the company pays them
        - **Claim on assets** if the company is liquidated (after creditors)
        - **Right to information** via annual reports and SEC filings
        """)
        
    elif lesson_id == "order_book":
        st.markdown("*📖 Source: SEC, FINRA, Investopedia*")
        
        st.markdown("""
        ## The Order Book
        
        The **order book** is where all buy and sell orders are collected. It shows
        supply and demand in real-time and is the mechanism that determines stock prices.
        
        ### 🔑 Key Terms
        
        - **Bid**: The highest price buyers are willing to pay
        - **Ask** (or Offer): The lowest price sellers will accept  
        - **Spread**: The gap between bid and ask
        - **Depth**: Number of shares at each price level
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Bids (Buyers)")
            bids = pd.DataFrame({
                'Price': ['$99.95', '$99.90', '$99.85', '$99.80', '$99.75'],
                'Size': [500, 1200, 800, 2000, 1500],
                'Orders': [3, 8, 5, 12, 7]
            })
            st.dataframe(bids, hide_index=True, use_container_width=True)
            st.caption("Buyers waiting to purchase at these prices")
        
        with col2:
            st.markdown("### 🔴 Asks (Sellers)")
            asks = pd.DataFrame({
                'Price': ['$100.00', '$100.05', '$100.10', '$100.15', '$100.20'],
                'Size': [300, 900, 1500, 600, 2200],
                'Orders': [2, 6, 9, 4, 11]
            })
            st.dataframe(asks, hide_index=True, use_container_width=True)
            st.caption("Sellers waiting to sell at these prices")
        
        st.info("**Spread** = $100.00 - $99.95 = **$0.05** (0.05%)")
        
        st.markdown("""
        ### 📊 Why the Spread Matters
        
        The spread is effectively a **transaction cost**. Every time you trade, you "pay" the spread:
        
        | Spread Type | Example | What It Means |
        |-------------|---------|---------------|
        | **Tight** (good) | $0.01 on Apple | High liquidity, easy to trade |
        | **Wide** (costly) | $0.50 on penny stock | Low liquidity, harder to get fair price |
        
        ### 🎯 How Prices Move
        
        1. **More buyers than sellers** → Buyers bid higher to get filled → Price rises
        2. **More sellers than buyers** → Sellers ask lower to get filled → Price falls
        3. **Big news** → Orders flood one side → Rapid price movement
        
        > 💡 **Pro Tip from Fidelity**: For most long-term investors, the spread is negligible.
        > Focus on transaction costs when trading illiquid securities or in large sizes.
        """)
        
    elif lesson_id == "order_types":
        st.markdown("*📖 Source: Fidelity, Charles Schwab, Investopedia*")
        
        st.markdown("""
        ## Order Types Explained
        
        Different order types give you different levels of control over price and execution speed.
        Understanding when to use each is crucial for effective trading.
        """)
        
        st.markdown("""
        ### 📋 Order Type Comparison
        
        | Order Type | How It Works | Guaranteed Fill? | Best For |
        |------------|--------------|------------------|----------|
        | **Market** | Buy/sell immediately at best available price | ✅ Yes | Speed, small orders |
        | **Limit** | Buy/sell only at your specified price or better | ❌ No | Price control |
        | **Stop** | Becomes market order when price hits trigger | ✅ Once triggered | Limiting losses |
        | **Stop-Limit** | Becomes limit order when price hits trigger | ❌ No | Precise exit points |
        | **Trailing Stop** | Stop price trails the market price | ✅ Once triggered | Protecting gains |
        """)
        
        st.markdown("### 📈 Example: Stock Trading at $100")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Market Buy
            *"Buy now at whatever price!"*
            
            - Fills immediately at ~$100.05 (the ask)
            - ✅ Guaranteed to fill
            - ⚠️ Might pay more in volatile markets
            - 👍 Best for: Urgent trades, liquid stocks
            
            ---
            
            #### Limit Buy at $98
            *"Only buy if price drops to $98 or less"*
            
            - Order waits until price reaches $98
            - ✅ You control your entry price
            - ⚠️ Might never fill if price doesn't drop
            - 👍 Best for: Patient buyers, volatile markets
            """)
        
        with col2:
            st.markdown("""
            #### Stop Sell at $95
            *"If price drops to $95, sell immediately!"*
            
            - Triggers when price hits $95
            - Then executes as market order
            - ✅ Automatic loss protection
            - ⚠️ Can trigger on temporary dips
            - 👍 Best for: Downside protection
            
            ---
            
            #### Stop-Limit Sell at $95/$94
            *"If price drops to $95, sell but only at $94 or better"*
            
            - Triggers at $95, becomes limit at $94
            - ✅ More price control
            - ⚠️ Might not fill if price gaps down
            - 👍 Best for: Avoiding forced sales at bad prices
            """)
        
        st.warning("""
        ⚠️ **Important Note from Fidelity**: During volatile markets or gaps (overnight moves), 
        stop orders can execute at prices significantly different from your stop price. 
        This is called "slippage."
        """)
        
        st.markdown("""
        ### 🎯 Which Order Should You Use?
        
        | Scenario | Recommended Order |
        |----------|-------------------|
        | Long-term investing, buying ETFs | **Market** (simple, fast) |
        | Want to buy on a dip | **Limit** (set your price) |
        | Already own stock, want protection | **Stop** or **Trailing Stop** |
        | Very volatile stock | **Limit** (avoid slippage) |
        | Exiting a position precisely | **Stop-Limit** |
        """)
        
    elif lesson_id == "liquidity":
        st.markdown("*📖 Source: SEC, FINRA, Investopedia*")
        
        st.markdown("""
        ## Liquidity: Makers vs Takers
        
        **Liquidity** = How easily you can buy or sell an asset without significantly moving its price.
        
        High liquidity means you can trade large amounts quickly at fair prices.
        Low liquidity means your trades may move the market or you may struggle to exit positions.
        """)
        
        st.markdown("""
        ### 🎭 Two Roles in Every Trade
        
        | Role | What They Do | Order Type | Exchange Treatment |
        |------|--------------|------------|-------------------|
        | **Market Maker** | Adds orders to the book | Limit orders | Often receives rebates |
        | **Market Taker** | Removes orders from book | Market orders | Pays fees |
        
        **Makers provide liquidity** - They put orders in the book for others to trade against.
        
        **Takers consume liquidity** - They fill existing orders immediately.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ High Liquidity (Ideal)
            
            **Examples**: Apple, Microsoft, S&P 500 ETFs (SPY, VOO)
            
            - Tight spreads ($0.01)
            - Deep order books (millions of shares)
            - Can trade large amounts easily
            - Price accurately reflects value
            - Low transaction costs
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Low Liquidity (Risky)
            
            **Examples**: Penny stocks, small-cap stocks, exotic ETFs
            
            - Wide spreads (5-10%+)
            - Thin order books
            - Your order moves the price
            - Difficult to exit positions
            - High hidden costs
            """)
        
        st.markdown("""
        ### 📊 Measuring Liquidity
        
        | Metric | What It Shows | Good Sign |
        |--------|---------------|-----------|
        | **Average Volume** | Shares traded daily | Higher = better |
        | **Bid-Ask Spread** | Gap between buy/sell | Smaller = better |
        | **Market Depth** | Orders at each price | Deeper = better |
        | **Price Impact** | How much your order moves price | Lower = better |
        """)
        
        # Liquidity comparison chart
        fig = go.Figure()
        categories = ['Apple (AAPL)', 'Mid-Cap Stock', 'Small Cap', 'Penny Stock']
        spreads = [0.01, 0.05, 0.25, 2.00]
        colors = ['#4ade80', '#fbbf24', '#f97316', '#ef4444']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=spreads,
            marker_color=colors,
            text=[f'{s:.2f}%' for s in spreads],
            textposition='outside'
        ))
        fig.update_layout(
            title="Typical Bid-Ask Spreads by Stock Type",
            yaxis_title="Spread (%)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        💡 **Beginner Advice from Vanguard**: Stick to liquid investments like VTI, VOO, or SPY. 
        You'll always get fair prices and can buy/sell easily. Avoid penny stocks and 
        thinly-traded securities until you understand liquidity risk.
        """)
        
    elif lesson_id == "exchanges":
        st.markdown("*📖 Source: SEC, NYSE, NASDAQ, Investopedia*")
        
        st.markdown("""
        ## Exchanges vs Dark Pools
        
        Not all trades happen on the public exchanges you see on TV!
        Understanding market structure helps you know where your orders go.
        
        ### 🏛️ Public Exchanges
        
        These are the traditional, regulated marketplaces:
        
        | Exchange | Description | Notable Listings |
        |----------|-------------|------------------|
        | **NYSE** | New York Stock Exchange, largest by market cap | Berkshire, JPMorgan, Walmart |
        | **NASDAQ** | Electronic exchange, tech-heavy | Apple, Microsoft, Google, Amazon |
        | **CBOE** | Chicago Board Options Exchange | Options, VIX |
        | **IEX** | "Investors Exchange" - designed for fairness | Various |
        
        **Key Features:**
        - Transparent order books
        - Regulated by SEC
        - Fair access rules
        - Public price discovery
        """)
        
        st.markdown("""
        ### 🌑 Dark Pools (~40% of US Trading Volume!)
        
        Dark pools are **private trading venues** where orders are hidden until executed.
        
        **Why They Exist:**
        
        Imagine you're a pension fund buying 1 million shares of Apple:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### On Public Exchange ❌
            1. You start buying
            2. Everyone sees huge buyer
            3. Traders front-run you
            4. Price jumps before you finish
            5. You pay more than necessary
            """)
        
        with col2:
            st.markdown("""
            #### In Dark Pool ✅
            1. Your order is hidden
            2. No one knows you're buying
            3. You match with hidden sellers
            4. Price doesn't move
            5. Better execution price
            """)
        
        st.markdown("""
        ### 🔄 Payment for Order Flow (PFOF)
        
        When you place an order at Robinhood, Schwab, or TD Ameritrade:
        
        1. Your broker **sells your order** to a market maker (Citadel, Virtu)
        2. The market maker fills your order (often at slight improvement)
        3. Market maker profits from the spread
        4. You get "free" trading
        
        **Is this bad?** It's debated:
        - ✅ You often get price improvement over the public quote
        - ⚠️ Critics argue you might get even better prices elsewhere
        - 📊 SEC requires brokers to disclose execution quality
        """)
        
        st.info("""
        💡 **For Retail Investors**: The venue your order goes to rarely matters for small trades.
        Focus on keeping costs low and investing consistently. Market structure is mainly 
        important for institutional traders moving large amounts.
        """)


# ==================== MODULE 2: MACRO ECONOMICS ====================
def render_macro_economics_content(lesson_id):
    """Render Macro Economics module content."""
    
    if lesson_id == "interest_rates":
        st.markdown("*📖 Source: Federal Reserve, Investopedia, Fidelity*")
        
        st.markdown("""
        ## Interest Rates & The Federal Reserve
        
        The **Federal Reserve** (the Fed) is the central bank of the United States.
        Its decisions on interest rates affect everything from your savings account to the stock market.
        
        ### 🏛️ What Does the Fed Do?
        
        The Fed has a **dual mandate**:
        1. **Maximum employment** - Keep unemployment low
        2. **Price stability** - Keep inflation around 2%
        
        Their main tool? **The Federal Funds Rate** - the rate banks charge each other for overnight loans.
        """)
        
        # Rate history chart
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        rates = [0.25, 0.50, 1.00, 1.75, 2.50, 0.25, 0.25, 4.50, 5.25, 5.50]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=rates,
            mode='lines+markers',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title="Federal Funds Rate History (2015-2024)",
            height=350,
            xaxis_title="Year",
            yaxis_title="Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 How Rate Changes Affect Markets
        
        | Asset Class | Rates **UP** ⬆️ | Rates **DOWN** ⬇️ |
        |-------------|-----------------|-------------------|
        | **Stocks** | Often fall (borrowing costs more) | Often rise (cheaper to grow) |
        | **Bonds** | Prices fall, yields rise | Prices rise, yields fall |
        | **Real Estate** | Mortgages expensive → fewer buyers | More affordable → demand up |
        | **Savings** | Higher interest on deposits | Lower interest earned |
        | **Dollar** | Strengthens vs other currencies | Weakens |
        
        ### 🔄 The Rate Cycle
        
        1. **Economy overheating** → Fed raises rates → Slows borrowing → Cools economy
        2. **Economy weak** → Fed cuts rates → Encourages borrowing → Stimulates growth
        3. **Crisis** → Fed cuts to near zero → Emergency stimulus
        """)
        
        st.warning("""
        ⚠️ **"Don't Fight the Fed"** - This classic Wall Street saying means: 
        When the Fed is cutting rates, it's usually good for stocks. 
        When they're raising aggressively, be cautious.
        """)
        
        st.markdown("""
        ### 🎯 Key Takeaways for Investors
        
        - **Higher rates** = Growth stocks (tech) often struggle more than value stocks
        - **Lower rates** = Bonds are less attractive, stocks benefit
        - **Rate expectations** matter as much as actual changes
        - Watch Fed meeting dates and Jerome Powell's speeches
        """)
        
    elif lesson_id == "inflation":
        st.markdown("*📖 Source: Bureau of Labor Statistics, Federal Reserve, Investopedia*")
        
        st.markdown("""
        ## Inflation & Purchasing Power
        
        **Inflation** = The rate at which prices rise over time, reducing your money's purchasing power.
        
        If you keep $100 in cash and inflation is 3%, after one year that $100 only buys 
        what $97 could buy before. Over time, this compounds dramatically.
        
        ### 📊 The Fed's Target: 2% Inflation
        
        Why not 0%? A little inflation is actually healthy:
        - Encourages spending (prices will rise later)
        - Makes debt easier to pay off over time
        - **Deflation** (falling prices) is actually worse - see Japan's "lost decades"
        """)
        
        # Inflation impact visualization
        years = np.arange(0, 31)
        value_2pct = 100 * (0.98) ** years
        value_3pct = 100 * (0.97) ** years
        value_5pct = 100 * (0.95) ** years
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=value_2pct, name='2% inflation', 
                                 line=dict(color='#4ade80', width=2)))
        fig.add_trace(go.Scatter(x=years, y=value_3pct, name='3% inflation', 
                                 line=dict(color='#fbbf24', width=2)))
        fig.add_trace(go.Scatter(x=years, y=value_5pct, name='5% inflation', 
                                 line=dict(color='#ef4444', width=2)))
        fig.update_layout(
            title="$100 Purchasing Power Over Time",
            height=350,
            xaxis_title="Years",
            yaxis_title="Real Value ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("""
        ⚠️ **The Silent Wealth Destroyer**: At 3% inflation, your cash loses HALF its purchasing 
        power in about 24 years. This is why you MUST invest - cash savings alone won't preserve wealth.
        """)
        
        st.markdown("""
        ### 📈 Measuring Inflation
        
        | Index | What It Measures | Used By |
        |-------|------------------|---------|
        | **CPI** (Consumer Price Index) | Basket of consumer goods | Most common measure |
        | **Core CPI** | CPI excluding food & energy | Fed's preferred view |
        | **PCE** (Personal Consumption Expenditures) | Broader measure | Fed's official target |
        | **PPI** (Producer Price Index) | Wholesale prices | Leading indicator |
        
        ### 🛡️ Inflation-Hedging Investments
        
        | Investment | How It Helps |
        |------------|--------------|
        | **Stocks** | Companies can raise prices → earnings grow |
        | **TIPS** | Treasury bonds indexed to CPI |
        | **Real Estate** | Property values and rents rise with inflation |
        | **Commodities** | Raw materials rise with inflation |
        | **I-Bonds** | Government savings bonds indexed to CPI |
        """)
        
        st.success("""
        💡 **Key Insight**: Over the long term, stocks have been the best inflation hedge, 
        returning ~10% annually vs ~3% inflation. A diversified portfolio protects your 
        purchasing power far better than cash.
        """)
        
    elif lesson_id == "gdp":
        st.markdown("*📖 Source: Bureau of Economic Analysis, Investopedia, Federal Reserve*")
        
        st.markdown("""
        ## GDP & Economic Cycles
        
        **GDP (Gross Domestic Product)** = The total value of all goods and services produced 
        in a country over a specific period. It's the primary measure of economic health.
        
        ### 📊 GDP Components
        
        GDP = **C + I + G + (X - M)**
        
        | Component | Description | % of US GDP |
        |-----------|-------------|-------------|
        | **C** - Consumer Spending | Households buying stuff | ~68% |
        | **I** - Business Investment | Companies investing | ~18% |
        | **G** - Government Spending | Federal + state + local | ~17% |
        | **X-M** - Net Exports | Exports minus imports | ~-3% (deficit) |
        """)
        
        # GDP growth chart
        years = list(range(2015, 2025))
        gdp_growth = [2.9, 1.6, 2.4, 2.9, 2.3, -2.8, 5.9, 2.1, 2.5, 2.8]
        colors = ['#4ade80' if g > 0 else '#ef4444' for g in gdp_growth]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=gdp_growth, marker_color=colors))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="US Real GDP Growth Rate (%)",
            height=300,
            xaxis_title="Year",
            yaxis_title="Growth Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔄 The Business Cycle
        
        Economies move through predictable phases:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📈 Expansion
            - GDP growing
            - Unemployment falling
            - Consumer confidence high
            - Stocks typically rise
            
            #### 🔝 Peak
            - Economy at maximum output
            - Inflation may be rising
            - Fed may raise rates
            - Stocks may be overvalued
            """)
        
        with col2:
            st.markdown("""
            #### 📉 Contraction (Recession)
            - GDP declining for 2+ quarters
            - Unemployment rising
            - Businesses cutting costs
            - Stocks typically fall
            
            #### 🔻 Trough
            - Economy at lowest point
            - Unemployment peaks
            - Fed cuts rates
            - Stocks often turn around here
            """)
        
        st.info("""
        💡 **Investment Tip from Fidelity**: Different sectors perform better at different 
        cycle stages. Cyclical stocks (consumer discretionary, industrials) outperform 
        in early expansion. Defensive stocks (utilities, healthcare) outperform in recession.
        """)
        
    elif lesson_id == "currency":
        st.markdown("*📖 Source: Federal Reserve, IMF, Investopedia*")
        
        st.markdown("""
        ## Currency & Exchange Rates
        
        Exchange rates affect international investments, imports/exports, and the global economy.
        Understanding currency dynamics is essential for diversified portfolios.
        
        ### 💱 What Determines Exchange Rates?
        
        | Factor | Effect on Dollar |
        |--------|------------------|
        | **Higher US interest rates** | Dollar strengthens (attracts capital) |
        | **Strong US economy** | Dollar strengthens (investment flows in) |
        | **Higher US inflation** | Dollar weakens (purchasing power falls) |
        | **Trade deficit** | Dollar weakens (more dollars going abroad) |
        | **Safe haven demand** | Dollar strengthens (crisis = buy USD) |
        """)
        
        st.markdown("""
        ### 🌍 Impact on Your Investments
        
        **International Stocks (VXUS, VEU, etc.)**
        
        When you own international stocks, you're exposed to **currency risk**:
        
        | Scenario | Impact on Your International Holdings |
        |----------|---------------------------------------|
        | Dollar strengthens | Foreign stocks worth less in USD (hurts returns) |
        | Dollar weakens | Foreign stocks worth more in USD (boosts returns) |
        
        **Example**: If European stocks rise 10% in euros, but the euro falls 5% vs dollar,
        your return is only about 5% in USD terms.
        """)
        
        st.markdown("""
        ### 🏭 Currency & Corporate Earnings
        
        US multinational companies are also affected:
        
        - **Strong dollar** = Foreign revenue worth less when converted → Hurts earnings
        - **Weak dollar** = Foreign revenue worth more → Boosts earnings
        
        Companies like Apple, Coca-Cola, and McDonald's earn 50%+ of revenue abroad!
        
        ### 🛡️ Should You Hedge Currency?
        
        | Approach | Pros | Cons |
        |----------|------|------|
        | **Unhedged** (most common) | Simpler, natural diversification | Currency volatility |
        | **Hedged** | Removes currency risk | Costs money, removes potential gains |
        
        **Vanguard's view**: For long-term investors, currency movements tend to even out. 
        Hedging adds costs and complexity that usually aren't worth it for most investors.
        """)
        
    elif lesson_id == "geopolitics":
        st.markdown("*📖 Source: Council on Foreign Relations, IMF, Investopedia*")
        
        st.markdown("""
        ## Geopolitical Risk Factors
        
        Geopolitical events can cause sudden market volatility. Understanding these risks 
        helps you stay calm during turbulent times.
        
        ### 🌍 Types of Geopolitical Risk
        
        | Risk Type | Examples | Market Impact |
        |-----------|----------|---------------|
        | **War/Conflict** | Russia-Ukraine, Middle East | Energy spikes, flight to safety |
        | **Trade Wars** | US-China tariffs | Supply chain disruption |
        | **Sanctions** | Russia sanctions | Sector-specific impacts |
        | **Elections** | Major policy shifts | Uncertainty, volatility |
        | **Regulatory** | Tech regulation, antitrust | Industry-specific |
        | **Pandemic** | COVID-19 | Global economic shock |
        """)
        
        st.markdown("""
        ### 📊 Historical Market Reactions
        
        | Event | Initial Drop | Recovery Time |
        |-------|--------------|---------------|
        | 9/11 Attacks (2001) | -12% | 1 month |
        | Iraq War Start (2003) | -5% | 1 month |
        | COVID Crash (2020) | -34% | 5 months |
        | Russia-Ukraine (2022) | -13% | 3 months |
        
        **Key Pattern**: Markets initially panic, then recover as uncertainty resolves.
        """)
        
        st.success("""
        💡 **Historical Perspective from Vanguard**: Since 1945, there have been dozens of 
        major geopolitical crises. In almost every case, investors who stayed invested 
        were rewarded. Panic selling during crises is usually the wrong move.
        """)
        
        st.markdown("""
        ### 🛡️ How to Handle Geopolitical Risk
        
        1. **Stay diversified** - Don't concentrate in regions/sectors exposed to specific risks
        2. **Maintain perspective** - Markets have recovered from every crisis in history
        3. **Avoid panic selling** - Selling into a crisis locks in losses
        4. **Consider rebalancing** - Crises create opportunities to buy quality assets cheap
        5. **Keep cash reserves** - Emergency fund prevents forced selling
        
        ### ⚠️ What NOT to Do
        
        - Don't try to predict geopolitical events (experts can't either)
        - Don't make portfolio changes based on news headlines
        - Don't assume "this time is different" - it rarely is
        """)


# ==================== MODULE 3: TECHNICAL ANALYSIS ====================
def render_technical_analysis_content(lesson_id):
    """Render Technical Analysis module content."""
    
    if lesson_id == "candlesticks":
        st.markdown("*📖 Source: Investopedia, TradingView, CMT Association*")
        
        st.markdown("""
        ## Candlestick Patterns
        
        Candlesticks originated in 18th century Japan for rice trading. They show price action 
        over a time period (day, hour, minute) in a visually intuitive way.
        
        ### 🕯️ Anatomy of a Candlestick
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ```
                 │ ← High (upper wick/shadow)
                 │
            ┌────┴────┐
            │  GREEN  │ ← Body (Open to Close)
            │  (UP)   │   Close is ABOVE Open
            └────┬────┘
                 │
                 │ ← Low (lower wick/shadow)
            ```
            **Green/White** = Bullish (price went UP)
            """)
        
        with col2:
            st.markdown("""
            ```
                 │ ← High (upper wick/shadow)
                 │
            ┌────┴────┐
            │   RED   │ ← Body (Open to Close)
            │ (DOWN)  │   Close is BELOW Open
            └────┬────┘
                 │
                 │ ← Low (lower wick/shadow)
            ```
            **Red/Black** = Bearish (price went DOWN)
            """)
        
        # Generate sample candlestick data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30)
        opens = 100 + np.cumsum(np.random.randn(30) * 1.5)
        closes = opens + np.random.randn(30) * 2
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(30)) * 0.5
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(30)) * 0.5
        
        fig = go.Figure(data=[go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            increasing_line_color='#4ade80',
            decreasing_line_color='#ef4444'
        )])
        fig.update_layout(
            title="Sample Candlestick Chart",
            height=400,
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📊 Key Single-Candle Patterns
        
        | Pattern | Shape | Signal | Reliability |
        |---------|-------|--------|-------------|
        | **Doji** | + shape, tiny body | Indecision, potential reversal | Medium |
        | **Hammer** | Small body, long lower wick | Bullish reversal | High |
        | **Shooting Star** | Small body, long upper wick | Bearish reversal | High |
        | **Marubozu** | Full body, no wicks | Strong trend continuation | High |
        | **Spinning Top** | Small body, equal wicks | Indecision | Low |
        
        ### 📈 Key Multi-Candle Patterns
        
        | Pattern | Description | Signal |
        |---------|-------------|--------|
        | **Bullish Engulfing** | Green candle fully engulfs prior red | Strong bullish reversal |
        | **Bearish Engulfing** | Red candle fully engulfs prior green | Strong bearish reversal |
        | **Morning Star** | Down, small, up (3 candles) | Bullish reversal |
        | **Evening Star** | Up, small, down (3 candles) | Bearish reversal |
        | **Three White Soldiers** | Three consecutive green candles | Strong bullish |
        | **Three Black Crows** | Three consecutive red candles | Strong bearish |
        """)
        
        st.warning("""
        ⚠️ **Important**: Candlestick patterns work best when:
        1. They occur at key support/resistance levels
        2. They're confirmed by volume
        3. They're used with other indicators
        
        No pattern works 100% of the time!
        """)
        
    elif lesson_id == "support_resistance":
        st.markdown("*📖 Source: Investopedia, TradingView, Technical Analysis of Stock Trends*")
        
        st.markdown("""
        ## Support & Resistance
        
        **Support** and **Resistance** are price levels where buying or selling pressure 
        tends to concentrate, causing price to pause or reverse.
        
        ### 📉 Support
        - A price level where **buying interest** is strong enough to overcome selling pressure
        - Price tends to "bounce" off support
        - Think of it as a "floor" under the price
        - If broken, often becomes new resistance
        
        ### 📈 Resistance
        - A price level where **selling interest** is strong enough to overcome buying pressure
        - Price tends to "reject" at resistance
        - Think of it as a "ceiling" above the price
        - If broken, often becomes new support
        """)
        
        # Create support/resistance visualization
        np.random.seed(123)
        x = np.arange(100)
        y = 100 + np.cumsum(np.random.randn(100) * 0.5)
        # Add artificial S/R bounces
        y = np.clip(y, 97, 105)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price', line=dict(color='#94a3b8')))
        fig.add_hline(y=105, line_dash="dash", line_color="#ef4444", 
                      annotation_text="Resistance $105")
        fig.add_hline(y=97, line_dash="dash", line_color="#4ade80", 
                      annotation_text="Support $97")
        fig.update_layout(
            title="Support and Resistance Example",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 How to Identify S/R Levels
        
        | Method | Description |
        |--------|-------------|
        | **Historical highs/lows** | Previous peaks and troughs |
        | **Round numbers** | $100, $50, etc. (psychological) |
        | **Moving averages** | 50-day, 200-day MA act as dynamic S/R |
        | **Fibonacci levels** | 38.2%, 50%, 61.8% retracements |
        | **Volume profile** | High-volume price levels |
        | **Trendlines** | Diagonal S/R connecting highs or lows |
        
        ### 📊 Trading Support & Resistance
        
        | Scenario | Strategy |
        |----------|----------|
        | Price approaches support | Consider buying (with stop below support) |
        | Price approaches resistance | Consider selling/taking profits |
        | Support breaks | May signal more downside (stop loss triggered) |
        | Resistance breaks | May signal breakout (potential entry) |
        """)
        
        st.info("""
        💡 **Pro Tip**: The more times a level is tested, the more significant it becomes.
        However, each test also weakens it slightly. A level tested 4-5 times may eventually break.
        """)
        
    elif lesson_id == "moving_averages":
        st.markdown("*📖 Source: Investopedia, TradingView, Fidelity*")
        
        st.markdown("""
        ## Moving Averages (SMA & EMA)
        
        Moving averages smooth price data to reveal the underlying trend direction.
        They're among the most widely used technical indicators.
        
        ### 📊 Types of Moving Averages
        
        | Type | Calculation | Characteristics |
        |------|-------------|-----------------|
        | **SMA** (Simple) | Average of last N prices | Equal weight to all prices |
        | **EMA** (Exponential) | Weighted toward recent prices | Faster reaction to changes |
        | **WMA** (Weighted) | Linear weighting | Between SMA and EMA |
        """)
        
        # Generate MA chart
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
        ma_20 = pd.Series(prices).rolling(20).mean()
        ma_50 = pd.Series(prices).rolling(50).mean()
        ma_200 = pd.Series(prices).rolling(200).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, name='Price', line=dict(color='#94a3b8', width=1)))
        fig.add_trace(go.Scatter(y=ma_20, name='20 MA', line=dict(color='#a78bfa', width=2)))
        fig.add_trace(go.Scatter(y=ma_50, name='50 MA', line=dict(color='#22c55e', width=2)))
        fig.add_trace(go.Scatter(y=ma_200, name='200 MA', line=dict(color='#ef4444', width=2)))
        fig.update_layout(
            title="Price with Multiple Moving Averages",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Common Moving Averages
        
        | MA Period | Use Case |
        |-----------|----------|
        | **10-20 day** | Short-term trend, active trading |
        | **50 day** | Medium-term trend, swing trading |
        | **200 day** | Long-term trend, investing |
        
        ### ⚔️ Classic Trading Signals
        
        | Signal | What It Means | Reliability |
        |--------|---------------|-------------|
        | **Golden Cross** | 50 MA crosses ABOVE 200 MA | Bullish (moderate) |
        | **Death Cross** | 50 MA crosses BELOW 200 MA | Bearish (moderate) |
        | **Price above 200 MA** | Long-term uptrend | Bullish context |
        | **Price below 200 MA** | Long-term downtrend | Bearish context |
        """)
        
        st.warning("""
        ⚠️ **Limitations of Moving Averages**:
        - They **lag** - by the time you see a signal, much of the move has happened
        - They generate **false signals** in choppy, sideways markets
        - They work best in **trending markets**
        
        Use them as part of a broader analysis, not as standalone signals.
        """)
        
    elif lesson_id == "indicators":
        st.markdown("*📖 Source: Investopedia, TradingView, CMT Association*")
        
        st.markdown("""
        ## RSI, MACD & Momentum Indicators
        
        Momentum indicators measure the speed and strength of price movements.
        They help identify overbought/oversold conditions and trend strength.
        
        ### 📊 RSI (Relative Strength Index)
        
        RSI measures the speed and magnitude of recent price changes to evaluate 
        overbought or oversold conditions. It oscillates between 0 and 100.
        """)
        
        # Generate RSI example
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 1)
        
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rsi, name='RSI', line=dict(color='#6366f1', width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="RSI (14-period)",
            height=250,
            yaxis=dict(range=[0, 100]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        | RSI Level | Interpretation | Potential Action |
        |-----------|----------------|------------------|
        | **Above 70** | Overbought | Watch for reversal, don't buy |
        | **Below 30** | Oversold | Watch for bounce, potential buy |
        | **50** | Neutral | Trend direction unclear |
        
        ### 📈 MACD (Moving Average Convergence Divergence)
        
        MACD shows the relationship between two moving averages and helps identify 
        trend changes and momentum.
        
        **Components:**
        - **MACD Line** = 12-period EMA - 26-period EMA
        - **Signal Line** = 9-period EMA of MACD Line
        - **Histogram** = MACD Line - Signal Line
        
        **Signals:**
        - MACD crosses ABOVE signal line = **Bullish**
        - MACD crosses BELOW signal line = **Bearish**
        - Histogram growing = Momentum increasing
        - Histogram shrinking = Momentum decreasing
        """)
        
        st.markdown("""
        ### 🔄 Divergence
        
        **Divergence** occurs when price and indicator move in opposite directions - 
        a powerful warning signal.
        
        | Type | What Happens | Signal |
        |------|--------------|--------|
        | **Bullish Divergence** | Price makes lower low, RSI makes higher low | Potential bottom |
        | **Bearish Divergence** | Price makes higher high, RSI makes lower high | Potential top |
        """)
        
        st.info("""
        💡 **Pro Tip**: Don't rely on any single indicator. The best traders use multiple 
        indicators together with price action and volume for confirmation.
        """)
        
    elif lesson_id == "volume":
        st.markdown("*📖 Source: Investopedia, CMT Association, Technical Analysis of Stock Trends*")
        
        st.markdown("""
        ## Volume Analysis
        
        **Volume** = The number of shares traded in a given period. It shows the 
        strength of conviction behind price moves.
        
        ### 📊 Volume Principles
        
        | Scenario | Interpretation |
        |----------|----------------|
        | Price UP + Volume UP | Strong bullish move (confirmed) |
        | Price UP + Volume DOWN | Weak rally (potential reversal) |
        | Price DOWN + Volume UP | Strong selling pressure (confirmed) |
        | Price DOWN + Volume DOWN | Weak decline (potential bounce) |
        
        ### 🔑 Key Insight
        
        **"Volume precedes price"** - Often, volume will spike before a major price move.
        Watch for unusual volume as a warning sign of impending movement.
        """)
        
        # Generate volume chart
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=60)
        prices = 100 + np.cumsum(np.random.randn(60) * 1)
        volume = np.random.randint(1000000, 5000000, 60)
        volume[40:45] = volume[40:45] * 3  # Volume spike
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=volume, name='Volume', marker_color='#6366f1', opacity=0.7))
        fig.update_layout(
            title="Volume with Spike Example",
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Volume Indicators
        
        | Indicator | What It Shows |
        |-----------|---------------|
        | **OBV** (On-Balance Volume) | Cumulative volume flow |
        | **Volume MA** | Average volume to spot unusual activity |
        | **VWAP** | Volume-weighted average price (institutional benchmark) |
        | **A/D Line** | Accumulation/Distribution |
        
        ### ⚠️ Volume Red Flags
        
        - **Breakout on low volume** = Likely false breakout
        - **New highs on declining volume** = Weakening trend
        - **Huge volume spike** = Potential climax (reversal coming)
        """)
        
    elif lesson_id == "chart_patterns":
        st.markdown("*📖 Source: Edwards & Magee, Investopedia, CMT Association*")
        
        st.markdown("""
        ## Chart Patterns
        
        Chart patterns are formations that appear on price charts and suggest future 
        price movements. They're based on the idea that market psychology creates 
        recurring patterns.
        
        ### 📊 Reversal Patterns
        
        These patterns signal a potential trend change:
        
        | Pattern | Signal | Characteristics |
        |---------|--------|-----------------|
        | **Head & Shoulders** | Bearish reversal | Three peaks, middle highest |
        | **Inverse H&S** | Bullish reversal | Three troughs, middle lowest |
        | **Double Top** | Bearish reversal | Two peaks at similar level |
        | **Double Bottom** | Bullish reversal | Two troughs at similar level |
        | **Triple Top/Bottom** | Strong reversal | Three tests of level |
        
        ### 📈 Continuation Patterns
        
        These patterns suggest the trend will continue:
        
        | Pattern | Signal | Characteristics |
        |---------|--------|-----------------|
        | **Flag** | Continuation | Small rectangle against trend |
        | **Pennant** | Continuation | Small triangle after sharp move |
        | **Wedge** | Continuation/Reversal | Converging trendlines |
        | **Triangle** | Breakout | Symmetrical, ascending, or descending |
        | **Cup & Handle** | Bullish continuation | U-shape with small pullback |
        
        ### 🎯 Trading Patterns
        
        1. **Identify the pattern** - Wait for it to fully form
        2. **Wait for confirmation** - Breakout with volume
        3. **Set entry** - After breakout confirmation
        4. **Set stop loss** - Below pattern support/above resistance
        5. **Set target** - Often equals the height of the pattern
        """)
        
        st.warning("""
        ⚠️ **Pattern Trading Challenges**:
        - Patterns are **subjective** - different traders see different things
        - Many patterns **fail** - not all breakouts follow through
        - **Hindsight bias** - patterns are easier to see after the fact
        
        Always use stops and manage risk when trading patterns!
        """)


# ==================== MODULE 3.5: FUNDAMENTAL ANALYSIS ====================
def render_fundamental_analysis_content(lesson_id):
    """Render Fundamental Analysis module content."""
    
    if lesson_id == "financial_statements":
        st.markdown("*📖 Source: SEC EDGAR, Fidelity, Morningstar*")
        
        st.markdown("""
        ## Reading Financial Statements
        
        Fundamental analysis starts with understanding a company's financial statements.
        Public companies file these with the SEC and they're available for free.
        
        ### 📋 The Three Core Statements
        
        | Statement | What It Shows | Time Period |
        |-----------|---------------|-------------|
        | **Income Statement** | Revenue, expenses, profit | Quarter or year |
        | **Balance Sheet** | Assets, liabilities, equity | Point in time |
        | **Cash Flow Statement** | Where cash came from/went | Quarter or year |
        
        ### 📊 Income Statement (P&L)
        
        Shows profitability over a period:
        
        ```
        Revenue (Sales)                    $100,000,000
        - Cost of Goods Sold (COGS)        - $40,000,000
        ─────────────────────────────────────────────────
        = Gross Profit                      $60,000,000
        - Operating Expenses               - $30,000,000
        ─────────────────────────────────────────────────
        = Operating Income (EBIT)           $30,000,000
        - Interest Expense                  - $5,000,000
        ─────────────────────────────────────────────────
        = Pre-Tax Income                    $25,000,000
        - Taxes                             - $5,000,000
        ─────────────────────────────────────────────────
        = Net Income                        $20,000,000
        ```
        
        ### 🏦 Balance Sheet
        
        Shows financial position at a moment:
        
        **Assets = Liabilities + Shareholders' Equity**
        
        | Assets (What company owns) | Liabilities (What company owes) |
        |----------------------------|--------------------------------|
        | Cash & equivalents | Accounts payable |
        | Accounts receivable | Short-term debt |
        | Inventory | Long-term debt |
        | Property & equipment | Other liabilities |
        | Intangible assets | **Shareholders' Equity** |
        
        ### 💵 Cash Flow Statement
        
        Shows actual cash movements:
        
        | Section | What It Includes |
        |---------|------------------|
        | **Operating** | Cash from business operations |
        | **Investing** | Capital expenditures, acquisitions |
        | **Financing** | Debt, dividends, stock buybacks |
        """)
        
        st.success("""
        💡 **Key Insight**: Net income can be manipulated with accounting tricks, 
        but cash flow is harder to fake. Always compare net income to operating cash flow.
        If cash flow is consistently lower than net income, investigate why.
        """)
        
    elif lesson_id == "pe_ratio":
        st.markdown("*📖 Source: Morningstar, Fidelity, Investopedia*")
        
        st.markdown("""
        ## P/E Ratio & Valuation Metrics
        
        Valuation metrics help you determine if a stock is cheap, fairly valued, or expensive
        relative to its earnings, assets, or growth.
        
        ### 📊 Price-to-Earnings (P/E) Ratio
        
        **P/E = Stock Price / Earnings Per Share (EPS)**
        
        | P/E Range | Interpretation |
        |-----------|----------------|
        | **Below 10** | Possibly undervalued or troubled |
        | **10-20** | Reasonable for mature companies |
        | **20-30** | Premium valuation, growth expected |
        | **Above 30** | High growth expectations or overvalued |
        """)
        
        # P/E comparison chart
        companies = ['Value Stock', 'S&P 500 Avg', 'Growth Stock', 'High-Growth Tech']
        pe_ratios = [12, 22, 35, 80]
        colors = ['#4ade80', '#fbbf24', '#f97316', '#ef4444']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=companies, y=pe_ratios, marker_color=colors))
        fig.update_layout(
            title="P/E Ratio Comparison",
            height=300,
            yaxis_title="P/E Ratio",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Other Key Valuation Metrics
        
        | Metric | Formula | Best For |
        |--------|---------|----------|
        | **P/E** | Price / EPS | Most companies |
        | **Forward P/E** | Price / Next Year's EPS | Growth companies |
        | **PEG** | P/E / Growth Rate | Comparing growth stocks |
        | **P/S** | Price / Sales | Unprofitable companies |
        | **P/B** | Price / Book Value | Banks, asset-heavy companies |
        | **EV/EBITDA** | Enterprise Value / EBITDA | Comparing across capital structures |
        
        ### 🎯 PEG Ratio (P/E to Growth)
        
        **PEG = P/E / Annual EPS Growth Rate**
        
        | PEG | Interpretation |
        |-----|----------------|
        | **Below 1** | Potentially undervalued |
        | **Around 1** | Fairly valued |
        | **Above 2** | Potentially overvalued |
        
        Example: Company with P/E of 30 and 30% growth rate → PEG = 1.0 (fair)
        """)
        
        st.warning("""
        ⚠️ **Valuation Caveats**:
        - Compare within industries (tech vs utilities have different norms)
        - Consider growth rates (high P/E may be justified)
        - Look at trends over time, not just current values
        - Cheap stocks can stay cheap (value traps)
        """)
        
    elif lesson_id == "growth":
        st.markdown("*📖 Source: Morningstar, Fidelity, Company Filings*")
        
        st.markdown("""
        ## Revenue & Earnings Growth
        
        Growth is the engine of stock returns. Understanding how to analyze 
        growth helps you identify winning companies.
        
        ### 📈 Revenue Growth
        
        Revenue (sales) is the "top line" - all growth starts here.
        
        | Growth Rate | Assessment |
        |-------------|------------|
        | **< 5%** | Slow growth, mature company |
        | **5-15%** | Moderate growth |
        | **15-25%** | Strong growth |
        | **> 25%** | High growth (often tech, disruptors) |
        
        ### 💰 Earnings Growth
        
        Earnings (profits) are the "bottom line" - what shareholders actually own.
        
        **Quality of earnings matters:**
        - Is growth from revenue increase or cost cutting?
        - Is it sustainable or one-time?
        - Does cash flow support reported earnings?
        """)
        
        # Growth comparison
        years = ['2020', '2021', '2022', '2023', '2024']
        revenue = [100, 115, 135, 160, 190]
        earnings = [10, 12, 15, 20, 25]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=revenue, name='Revenue', 
                                 line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=years, y=earnings, name='Earnings', 
                                 line=dict(color='#4ade80', width=3)))
        fig.update_layout(
            title="Revenue vs Earnings Growth Example",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 Key Growth Metrics
        
        | Metric | What to Look For |
        |--------|------------------|
        | **Revenue CAGR** | Compound annual growth rate (3-5 year) |
        | **EPS Growth** | Earnings per share trend |
        | **Margin Expansion** | Growing profits faster than revenue |
        | **TAM Growth** | Is the total addressable market growing? |
        | **Market Share** | Is the company gaining vs competitors? |
        
        ### ⚠️ Growth Red Flags
        
        - Slowing growth rates quarter over quarter
        - Revenue growth without profit growth
        - Growth only from acquisitions
        - Unsustainable customer acquisition costs
        - Growth dependent on one product/customer
        """)
        
    elif lesson_id == "balance_sheet":
        st.markdown("*📖 Source: Fidelity, Morningstar, Warren Buffett's Letters*")
        
        st.markdown("""
        ## Balance Sheet Analysis
        
        The balance sheet shows financial health at a point in time.
        Strong balance sheets protect companies during downturns.
        
        ### 🏦 Key Balance Sheet Metrics
        
        | Metric | Formula | What It Shows |
        |--------|---------|---------------|
        | **Current Ratio** | Current Assets / Current Liabilities | Short-term liquidity |
        | **Quick Ratio** | (Current Assets - Inventory) / Current Liabilities | Immediate liquidity |
        | **Debt-to-Equity** | Total Debt / Shareholders' Equity | Leverage level |
        | **Book Value** | Assets - Liabilities | Net worth |
        | **Working Capital** | Current Assets - Current Liabilities | Operating cushion |
        
        ### 📊 Healthy Ranges
        
        | Metric | Healthy Range | Warning Sign |
        |--------|---------------|--------------|
        | **Current Ratio** | > 1.5 | < 1.0 |
        | **Quick Ratio** | > 1.0 | < 0.5 |
        | **Debt/Equity** | < 1.0 | > 2.0 |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Signs of Strength
            - Growing cash position
            - Decreasing debt levels
            - Increasing book value
            - No goodwill impairments
            - Consistent inventory turnover
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Warning Signs
            - Declining cash, rising debt
            - Large goodwill/intangibles
            - Rising accounts receivable
            - Inventory buildup
            - Off-balance sheet liabilities
            """)
        
        st.info("""
        💡 **Buffett's Rule**: "When a management team with a reputation for brilliance 
        tackles a business with a reputation for bad economics, it is the reputation 
        of the business that remains intact." Look for strong balance sheets first.
        """)
        
    elif lesson_id == "cash_flow":
        st.markdown("*📖 Source: Fidelity, Morningstar, SEC*")
        
        st.markdown("""
        ## Cash Flow Analysis
        
        **"Cash is king."** The cash flow statement shows actual cash movements, 
        which is harder to manipulate than earnings.
        
        ### 💵 Three Types of Cash Flow
        
        | Type | What It Includes | What to Look For |
        |------|------------------|------------------|
        | **Operating (CFO)** | Cash from business | Positive, growing |
        | **Investing (CFI)** | CapEx, acquisitions | Usually negative (investing in growth) |
        | **Financing (CFF)** | Debt, dividends, buybacks | Varies by strategy |
        
        ### 📊 Free Cash Flow (FCF)
        
        **FCF = Operating Cash Flow - Capital Expenditures**
        
        This is the cash available to:
        - Pay dividends
        - Buy back stock
        - Pay down debt
        - Make acquisitions
        - Invest in growth
        
        **Free Cash Flow Yield = FCF / Market Cap**
        - Above 5% = potentially undervalued
        - Below 2% = expensive or reinvesting heavily
        """)
        
        # Cash flow visualization
        categories = ['Net Income', 'Operating CF', 'Free Cash Flow']
        values = [100, 120, 80]
        colors = ['#6366f1', '#4ade80', '#fbbf24']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors))
        fig.update_layout(
            title="Cash Flow Comparison (Healthy Company)",
            height=300,
            yaxis_title="$ Millions",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔍 Cash Flow Quality Checks
        
        | Check | Healthy Sign | Warning Sign |
        |-------|--------------|--------------|
        | **CFO vs Net Income** | CFO > Net Income | CFO << Net Income |
        | **FCF Trend** | Growing over time | Declining or negative |
        | **CapEx/Depreciation** | Roughly equal | CapEx much higher (playing catch-up) |
        | **Working Capital** | Stable or improving | Deteriorating rapidly |
        """)
        
        st.warning("""
        ⚠️ **Red Flag**: If a company reports profits but burns cash, investigate. 
        Common causes: aggressive revenue recognition, inventory issues, or 
        unsustainable working capital practices.
        """)
        
    elif lesson_id == "moats":
        st.markdown("*📖 Source: Morningstar, Warren Buffett's Letters, Pat Dorsey*")
        
        st.markdown("""
        ## Competitive Moats
        
        A **moat** is a sustainable competitive advantage that protects a company's 
        profits from competitors - like the moat around a castle.
        
        ### 🏰 Types of Economic Moats
        
        | Moat Type | Description | Examples |
        |-----------|-------------|----------|
        | **Network Effects** | Product gets better with more users | Visa, Facebook, eBay |
        | **Switching Costs** | Painful for customers to switch | Microsoft, Adobe, Salesforce |
        | **Intangible Assets** | Brands, patents, licenses | Coca-Cola, Disney, Pfizer |
        | **Cost Advantage** | Produce cheaper than competitors | Walmart, Amazon, Costco |
        | **Efficient Scale** | Market only supports few players | Railroads, utilities |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Signs of a Moat
            
            - High and stable profit margins
            - Returns on capital above 15%
            - Pricing power (can raise prices)
            - Market share stability or gains
            - Customer retention above 90%
            - High barriers to entry
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Moat Erosion Signs
            
            - Declining margins over time
            - Losing market share
            - Needing to compete on price
            - Technology disruption
            - Regulatory changes
            - New well-funded competitors
            """)
        
        st.markdown("""
        ### 📊 Moat Metrics
        
        | Metric | Moat Indication | No Moat |
        |--------|-----------------|---------|
        | **ROIC** | > 15% sustained | < 10% or declining |
        | **Gross Margin** | > 40% stable | < 20% or falling |
        | **Customer Retention** | > 90% | < 80% |
        | **Market Share** | Growing or stable | Declining |
        """)
        
        st.success("""
        💡 **Buffett's Advice**: "The key to investing is not assessing how much an industry 
        is going to affect society, but rather determining the competitive advantage of 
        any given company and, above all, the durability of that advantage."
        """)


# ==================== MODULE 4: QUANT STRATEGIES ====================
def render_quant_strategies_content(lesson_id):
    """Render Quant Strategies module content."""
    
    if lesson_id == "factor_investing":
        st.markdown("*📖 Source: AQR Capital, Fama-French Research, MSCI*")
        
        st.markdown("""
        ## Factor Investing
        
        **Factor investing** identifies characteristics (factors) that explain 
        differences in returns across assets. It's the foundation of modern 
        quantitative investing.
        
        ### 📊 The Classic Factors
        
        | Factor | Description | Academic Support |
        |--------|-------------|------------------|
        | **Market (Beta)** | Exposure to overall market | CAPM (1960s) |
        | **Size** | Small caps outperform large caps | Fama-French (1992) |
        | **Value** | Cheap stocks beat expensive | Fama-French (1992) |
        | **Momentum** | Recent winners keep winning | Jegadeesh-Titman (1993) |
        | **Quality** | Profitable companies outperform | Novy-Marx (2013) |
        | **Low Volatility** | Less risky stocks outperform | Black (1972) |
        """)
        
        # Factor returns visualization
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low Vol']
        premiums = [7.0, 2.0, 3.5, 6.0, 3.0, 2.5]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=factors, y=premiums, marker_color='#6366f1'))
        fig.update_layout(
            title="Historical Factor Premiums (Annual %)",
            height=300,
            yaxis_title="Premium (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Factor ETFs for Individual Investors
        
        | Factor | ETF Examples | Expense Ratio |
        |--------|--------------|---------------|
        | **Value** | VTV, IWD, VLUE | 0.04-0.15% |
        | **Size (Small)** | VB, IJR, IWM | 0.05-0.20% |
        | **Momentum** | MTUM, QMOM | 0.15-0.25% |
        | **Quality** | QUAL, SPHQ | 0.15-0.30% |
        | **Low Vol** | USMV, SPLV | 0.15-0.25% |
        | **Multi-Factor** | LRGF, VFMF | 0.10-0.20% |
        
        ### ⚠️ Factor Investing Caveats
        
        1. **Factors can underperform for years** - Value was "dead" 2010-2020
        2. **Crowding risk** - Popular factors may be arbitraged away
        3. **Implementation costs** - Transaction costs erode premiums
        4. **Factor timing is hard** - Don't try to time factor rotations
        """)
        
        st.info("""
        💡 **Practical Advice**: For most investors, a simple value tilt (like VTV) or 
        multi-factor fund provides factor exposure without complexity. Don't over-engineer it.
        """)
        
    elif lesson_id == "mean_reversion":
        st.markdown("*📖 Source: Academic Research, AQR Capital, Two Sigma*")
        
        st.markdown("""
        ## Mean Reversion Strategies
        
        **Mean reversion** is the theory that prices tend to return to their average 
        over time. When something moves too far from normal, it tends to snap back.
        
        ### 📊 The Concept
        
        If a stock is:
        - **Far below average** → May be oversold → Potential buy
        - **Far above average** → May be overbought → Potential sell
        
        ### 🎯 Mean Reversion Indicators
        
        | Indicator | Mean Reversion Signal |
        |-----------|----------------------|
        | **RSI** | Below 30 (oversold) or above 70 (overbought) |
        | **Bollinger Bands** | Price at lower/upper band |
        | **Z-Score** | Standard deviations from mean |
        | **Pairs Spread** | Spread between correlated assets widens |
        """)
        
        # Mean reversion example
        np.random.seed(42)
        x = np.arange(200)
        mean = 100
        prices = mean + np.cumsum(np.random.randn(200)) * 0.5
        # Add mean-reverting component
        prices = prices - 0.1 * (prices - mean) + np.random.randn(200) * 0.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, name='Price', line=dict(color='#94a3b8')))
        fig.add_hline(y=mean, line_dash="dash", line_color="#6366f1", annotation_text="Mean")
        fig.add_hline(y=mean+5, line_dash="dot", line_color="#ef4444", annotation_text="+2σ")
        fig.add_hline(y=mean-5, line_dash="dot", line_color="#4ade80", annotation_text="-2σ")
        fig.update_layout(
            title="Mean Reversion Example",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Example Strategy: RSI Mean Reversion
        
        1. **Buy signal**: RSI drops below 30 (oversold)
        2. **Sell signal**: RSI rises above 70 (overbought)
        3. **Risk management**: Stop loss if price continues against you
        
        ### ⚠️ The Catch: Trending Markets
        
        Mean reversion **fails** when markets trend:
        - In 2008, "cheap" stocks kept getting cheaper
        - Catching a falling knife = buying into a crash
        
        **"The market can stay irrational longer than you can stay solvent."** - Keynes
        """)
        
        st.warning("""
        ⚠️ **Key Risk**: Mean reversion assumes prices will return to normal. 
        But sometimes the "mean" itself has shifted. A stock down 50% might 
        deserve to be down 50% due to fundamental changes.
        """)
        
    elif lesson_id == "momentum":
        st.markdown("*📖 Source: AQR Capital, Cliff Asness Research, Academic Papers*")
        
        st.markdown("""
        ## Momentum Strategies
        
        **Momentum** is the tendency for recent winners to keep winning and recent 
        losers to keep losing. It's one of the most robust anomalies in finance.
        
        ### 📊 The Evidence
        
        - Documented across stocks, bonds, currencies, and commodities
        - Works in markets globally
        - Persisted for 200+ years of data
        - ~6% annual premium historically
        
        ### 🎯 Types of Momentum
        
        | Type | Lookback Period | Holding Period |
        |------|-----------------|----------------|
        | **Short-term** | 1 week - 1 month | Days to weeks |
        | **Intermediate** | 3-6 months | 1-3 months |
        | **Long-term** | 6-12 months | 3-6 months |
        
        **The classic strategy**: Buy stocks with best 12-month returns (excluding last month),
        hold for 3-6 months.
        """)
        
        st.markdown("""
        ### 📈 Simple Momentum Rules
        
        1. **Trend Following**: Buy when price > 200-day moving average
        2. **Relative Momentum**: Buy top 10% performers, sell bottom 10%
        3. **Dual Momentum**: Combine absolute and relative momentum
        
        ### ⚠️ Momentum Risks
        
        | Risk | Description |
        |------|-------------|
        | **Momentum Crashes** | Sharp reversals (2009, 2020) |
        | **High Turnover** | Frequent trading = high costs |
        | **Crowding** | Too many momentum traders |
        | **Crash Risk** | Momentum does worst when market recovers |
        
        ### 💡 Practical Implementation
        
        For individual investors, consider:
        - **MTUM** - iShares Momentum ETF
        - **QMOM** - Alpha Architect Momentum
        - **Simple rule**: Only hold stocks trading above their 200-day MA
        """)
        
    elif lesson_id == "stat_arb":
        st.markdown("*📖 Source: Quantitative Trading, Two Sigma, Renaissance Technologies*")
        
        st.markdown("""
        ## Statistical Arbitrage
        
        **Statistical arbitrage** (stat arb) exploits pricing inefficiencies between 
        related securities using mathematical models. It's a mainstay of hedge fund strategies.
        
        ### 📊 Core Concepts
        
        **Pairs Trading**: The classic stat arb strategy
        
        1. Find two stocks that move together (high correlation)
        2. When their relationship diverges (spread widens)
        3. Short the outperformer, long the underperformer
        4. Wait for relationship to normalize
        5. Close both positions for profit
        """)
        
        # Pairs trading visualization
        np.random.seed(42)
        x = np.arange(100)
        common_factor = np.cumsum(np.random.randn(100)) * 0.5
        stock_a = 100 + common_factor + np.cumsum(np.random.randn(100)) * 0.2
        stock_b = 100 + common_factor + np.cumsum(np.random.randn(100)) * 0.2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=stock_a, name='Stock A', line=dict(color='#6366f1')))
        fig.add_trace(go.Scatter(y=stock_b, name='Stock B', line=dict(color='#4ade80')))
        fig.update_layout(
            title="Pairs Trading: Two Correlated Stocks",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Common Stat Arb Approaches
        
        | Strategy | Description | Risk Level |
        |----------|-------------|------------|
        | **Pairs Trading** | Two correlated stocks | Moderate |
        | **Index Arbitrage** | ETF vs underlying stocks | Low |
        | **Sector Neutral** | Long/short within sectors | Moderate |
        | **Market Neutral** | Zero beta overall | Low market risk |
        
        ### ⚠️ Why It's Hard
        
        1. **Requires sophisticated tools** - Math, programming, data
        2. **Alpha decay** - Strategies stop working as they get crowded
        3. **Model risk** - Relationships can permanently break
        4. **Execution matters** - Need low-cost, fast trading
        5. **Competition** - You're competing with PhDs and supercomputers
        """)
        
        st.info("""
        💡 **Reality Check**: Successful stat arb requires institutional-grade infrastructure, 
        data, and execution. For individual investors, factor-based ETFs capture similar 
        concepts with much less complexity.
        """)
        
    elif lesson_id == "backtesting":
        st.markdown("*📖 Source: Quantitative Trading Research, Systematic Trading*")
        
        st.markdown("""
        ## Backtesting & Validation
        
        **Backtesting** tests a trading strategy on historical data to see how it 
        would have performed. It's essential but full of traps.
        
        ### 📊 The Backtesting Process
        
        1. **Define strategy rules** precisely
        2. **Gather clean historical data**
        3. **Run simulation** on past data
        4. **Analyze results** (returns, drawdowns, Sharpe)
        5. **Validate** with out-of-sample testing
        6. **Paper trade** before real money
        
        ### ⚠️ Backtesting Pitfalls
        
        | Pitfall | Description | Solution |
        |---------|-------------|----------|
        | **Overfitting** | Curve-fitting to historical noise | Keep rules simple |
        | **Survivorship Bias** | Only testing stocks that survived | Use delisting data |
        | **Look-Ahead Bias** | Using future data in decisions | Careful data handling |
        | **Transaction Costs** | Ignoring fees, slippage | Realistic cost modeling |
        | **Data Snooping** | Testing many strategies until one works | Out-of-sample testing |
        """)
        
        st.markdown("""
        ### 📈 Validation Framework
        
        ```
        Total Data
        ├── In-Sample (60%) - Develop strategy
        ├── Validation (20%) - Tune parameters  
        └── Out-of-Sample (20%) - Final test (ONLY ONCE!)
        ```
        
        ### 🎯 Key Metrics to Evaluate
        
        | Metric | Good Value | What It Measures |
        |--------|------------|------------------|
        | **Sharpe Ratio** | > 1.0 | Risk-adjusted return |
        | **Max Drawdown** | < 20% | Worst peak-to-trough |
        | **Win Rate** | > 50% | Percentage of winning trades |
        | **Profit Factor** | > 1.5 | Gross profit / gross loss |
        | **Calmar Ratio** | > 1.0 | Return / max drawdown |
        """)
        
        st.error("""
        ⚠️ **Golden Rule**: A backtest that looks too good is probably wrong. 
        The more impressive the results, the more skeptical you should be. 
        Real-world performance is almost always worse than backtests.
        """)
        
    elif lesson_id == "risk_systems":
        st.markdown("*📖 Source: Risk Management, AQR Capital, Professional Trading*")
        
        st.markdown("""
        ## Risk Management Systems
        
        **"Risk management is more important than return management."**
        
        Professional traders spend more time on risk than on finding trades.
        
        ### 📊 Position Sizing
        
        Never risk too much on any single trade:
        
        | Method | Rule | Example |
        |--------|------|---------|
        | **Fixed Percentage** | Risk 1-2% per trade | $100K account → max $2K risk |
        | **Kelly Criterion** | Optimal f = (p*b - q) / b | Math-based sizing |
        | **Volatility-Based** | Size inversely to volatility | Smaller positions in volatile markets |
        
        ### 🛡️ Risk Limits
        
        | Limit Type | Description |
        |------------|-------------|
        | **Position Limit** | Max % in any single stock |
        | **Sector Limit** | Max % in any sector |
        | **Drawdown Limit** | Reduce risk after losses |
        | **Correlation Limit** | Diversify across uncorrelated bets |
        | **Leverage Limit** | Max borrowed amount |
        """)
        
        st.markdown("""
        ### 📈 Stop Loss Strategies
        
        | Type | How It Works |
        |------|--------------|
        | **Fixed Stop** | Sell if down X% |
        | **ATR Stop** | Stop based on volatility (2x ATR) |
        | **Trailing Stop** | Stop moves up with price |
        | **Time Stop** | Exit if no movement in X days |
        
        ### 🎯 The 2% Rule
        
        Never risk more than 2% of your account on any single trade.
        
        Example:
        - Account: $50,000
        - Max risk per trade: $1,000 (2%)
        - If stop loss is 10% below entry → Position size: $10,000
        """)
        
        st.success("""
        💡 **Professional Wisdom**: "Amateurs focus on returns. Professionals focus on risk."
        The best traders are those who survive long enough for their edge to play out.
        Preservation of capital always comes first.
        """)


# ==================== MODULE 5: ADVANCED OPTIONS ====================
def render_advanced_options_content(lesson_id):
    """Render Advanced Options module content."""
    
    if lesson_id == "greeks":
        st.markdown("*📖 Source: CBOE, Options Industry Council, Natenberg*")
        
        st.markdown("""
        ## Options Greeks Deep Dive
        
        The **Greeks** measure how an option's price changes in response to various 
        factors. Understanding Greeks is essential for options trading.
        
        ### 📊 The Five Greeks
        
        | Greek | Measures | Range | Call | Put |
        |-------|----------|-------|------|-----|
        | **Delta (Δ)** | Price sensitivity to stock move | 0 to 1 / -1 to 0 | Positive | Negative |
        | **Gamma (Γ)** | Rate of change of delta | Always positive | Same | Same |
        | **Theta (Θ)** | Time decay per day | Usually negative | Negative | Negative |
        | **Vega (ν)** | Sensitivity to volatility | Always positive | Same | Same |
        | **Rho (ρ)** | Sensitivity to interest rates | Varies | Positive | Negative |
        """)
        
        st.markdown("""
        ### 🎯 Delta Explained
        
        Delta tells you how much the option price moves for a $1 stock move.
        
        | Delta | Interpretation |
        |-------|----------------|
        | **0.50** | At-the-money (50% chance of expiring ITM) |
        | **0.80** | Deep in-the-money |
        | **0.20** | Out-of-the-money |
        | **1.00** | So deep ITM it acts like stock |
        
        **Delta as hedge ratio**: To delta-hedge 1 call with 0.50 delta, short 50 shares.
        
        ### ⏰ Theta Explained
        
        Theta is the daily cost of holding an option. It accelerates near expiration.
        
        - **Long options**: Theta works against you
        - **Short options**: Theta works for you
        - **At expiration**: All time value = 0
        """)
        
        # Theta decay visualization
        days = np.arange(60, 0, -1)
        theta_decay = np.sqrt(days) / np.sqrt(60) * 5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=60-days, y=theta_decay, fill='tozeroy',
                                 line=dict(color='#ef4444'),
                                 fillcolor='rgba(239, 68, 68, 0.3)'))
        fig.update_layout(
            title="Option Time Value Decay",
            height=300,
            xaxis_title="Days Until Expiration",
            yaxis_title="Time Value ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📈 Vega Explained
        
        Vega measures sensitivity to implied volatility (IV).
        
        - High IV = Options are expensive
        - Low IV = Options are cheap
        - Vega is highest for ATM options with longer expirations
        
        **Trading Tip**: Buy options when IV is low, sell when IV is high.
        """)
        
    elif lesson_id == "vertical_spreads":
        st.markdown("*📖 Source: CBOE, Options Playbook, Tastytrade*")
        
        st.markdown("""
        ## Vertical Spreads
        
        A **vertical spread** involves buying and selling options of the same type 
        (calls or puts) with the same expiration but different strikes.
        
        ### 📊 The Four Vertical Spreads
        
        | Spread | Construction | Outlook | Max Profit | Max Loss |
        |--------|--------------|---------|------------|----------|
        | **Bull Call** | Buy lower call, sell higher call | Bullish | Strike diff - premium | Premium paid |
        | **Bear Put** | Buy higher put, sell lower put | Bearish | Strike diff - premium | Premium paid |
        | **Bull Put** | Sell higher put, buy lower put | Bullish | Premium received | Strike diff - premium |
        | **Bear Call** | Sell lower call, buy higher call | Bearish | Premium received | Strike diff - premium |
        
        ### 🎯 Bull Call Spread Example
        
        Stock at $100, moderately bullish:
        
        - Buy 100 call @ $3.00
        - Sell 105 call @ $1.50
        - Net cost: $1.50
        - Max profit: $5.00 - $1.50 = **$3.50** (if stock > $105)
        - Max loss: **$1.50** (if stock < $100)
        - Breakeven: $101.50
        """)
        
        # P/L diagram
        stock_prices = np.arange(95, 115, 0.5)
        bull_call_pl = np.minimum(np.maximum(stock_prices - 100, 0), 5) - 1.5 - \
                       np.minimum(np.maximum(stock_prices - 105, 0), 0)
        bull_call_pl = np.minimum(np.maximum(stock_prices - 100, 0) - 
                                  np.maximum(stock_prices - 105, 0), 5) - 1.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=bull_call_pl * 100,
                                 fill='tozeroy', line=dict(color='#4ade80'),
                                 fillcolor='rgba(74, 222, 128, 0.3)'))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Bull Call Spread P/L Diagram",
            height=300,
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### ✅ Advantages of Spreads
        
        - **Defined risk** - Know max loss upfront
        - **Lower cost** - Cheaper than single options
        - **Theta benefit** - Short leg offsets some decay
        - **Lower breakeven** - Than buying calls outright
        
        ### ⚠️ Disadvantages
        
        - **Capped profit** - Can't make unlimited gains
        - **Complexity** - Two legs to manage
        - **Assignment risk** - Short leg can be assigned
        """)
        
    elif lesson_id == "iron_condor":
        st.markdown("*📖 Source: CBOE, Tastytrade, Options Alpha*")
        
        st.markdown("""
        ## Iron Condors & Butterflies
        
        These are **neutral strategies** that profit when the stock stays in a range.
        They're popular for income generation.
        
        ### 🦅 Iron Condor
        
        Combines a bull put spread and bear call spread:
        
        ```
        Sell put  @ 95  │
        Buy put   @ 90  │ Bull Put Spread
        ─────────────────
        Sell call @ 105 │
        Buy call  @ 110 │ Bear Call Spread
        ```
        
        **Profit zone**: Stock stays between 95 and 105
        **Max profit**: Premium collected
        **Max loss**: Width of spread - premium
        """)
        
        # Iron condor P/L
        stock_prices = np.arange(85, 120, 0.5)
        ic_pl = np.zeros_like(stock_prices)
        premium = 2.0
        
        for i, price in enumerate(stock_prices):
            put_spread = max(95 - price, 0) - max(90 - price, 0)
            call_spread = max(price - 105, 0) - max(price - 110, 0)
            ic_pl[i] = premium - put_spread - call_spread
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=ic_pl * 100,
                                 fill='tozeroy', line=dict(color='#6366f1'),
                                 fillcolor='rgba(99, 102, 241, 0.3)'))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Iron Condor P/L Diagram",
            height=300,
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🦋 Butterfly Spread
        
        Similar to iron condor but with overlapping strikes:
        
        ```
        Buy 1  put/call @ 95
        Sell 2 put/call @ 100
        Buy 1  put/call @ 105
        ```
        
        **Max profit**: At exactly $100 (middle strike)
        **Max loss**: Premium paid
        
        ### 📊 When to Use Each
        
        | Strategy | Best When | Win Rate | Profit Potential |
        |----------|-----------|----------|------------------|
        | **Iron Condor** | Low volatility expected | Higher | Lower per trade |
        | **Butterfly** | Price pin expected | Lower | Higher per trade |
        """)
        
    elif lesson_id == "calendar_spreads":
        st.markdown("*📖 Source: CBOE, Options Playbook, Professional Trading*")
        
        st.markdown("""
        ## Calendar & Diagonal Spreads
        
        These spreads use different expiration dates to profit from time decay 
        and volatility differences.
        
        ### 📅 Calendar Spread (Time Spread)
        
        Same strike, different expirations:
        
        - **Sell** near-term option (faster theta decay)
        - **Buy** longer-term option (slower theta decay)
        
        **Profit from**: Time decay differential
        **Best when**: Stock stays near strike, IV increases
        
        ### 📐 Diagonal Spread
        
        Different strikes AND different expirations:
        
        - **Sell** near-term, out-of-money option
        - **Buy** longer-term, in-the-money option
        
        **Combines**: Directional bias + time decay
        """)
        
        st.markdown("""
        ### 📊 Calendar Spread Example
        
        Stock at $100:
        - Sell 30-day $100 call @ $2.00
        - Buy 60-day $100 call @ $3.50
        - Net cost: $1.50
        
        **Scenarios:**
        
        | Stock Price | 30 Days Later | Result |
        |-------------|---------------|--------|
        | $100 | Near-term expires worthless | Keep $2, long call has value |
        | $95 | Both decline | Small loss |
        | $105 | Near-term ITM | May need to roll |
        
        ### ⚠️ Key Risks
        
        1. **Movement away from strike** - Both calendars and diagonals lose if stock moves too far
        2. **IV collapse** - Hurts long option more than short
        3. **Early assignment** - Short option can be assigned
        4. **Management required** - Not set-and-forget
        """)
        
    elif lesson_id == "vol_trading":
        st.markdown("*📖 Source: Volatility Trading by Sinclair, CBOE VIX, Tastytrade*")
        
        st.markdown("""
        ## Volatility Trading
        
        Instead of betting on direction, you can bet on **volatility** itself.
        
        ### 📊 Implied vs Realized Volatility
        
        | Type | Definition | Trading Implication |
        |------|------------|---------------------|
        | **Implied (IV)** | Market's expected future volatility | What you're paying for |
        | **Realized (HV)** | Actual historical volatility | What actually happens |
        
        **The edge**: IV is often higher than realized volatility 
        (volatility risk premium). Sellers have a long-term advantage.
        """)
        
        # VIX chart example
        np.random.seed(42)
        vix = 20 + np.cumsum(np.random.randn(100) * 0.5)
        vix = np.clip(vix, 12, 40)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vix, name='VIX', line=dict(color='#f97316', width=2)))
        fig.add_hline(y=20, line_dash="dash", line_color="#4ade80", annotation_text="Historical Average ~20")
        fig.update_layout(
            title="VIX (Volatility Index) Example",
            height=300,
            yaxis_title="VIX Level",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Volatility Strategies
        
        | Strategy | Position | Profits When |
        |----------|----------|--------------|
        | **Long Straddle** | Buy call + put same strike | Big move either direction |
        | **Short Straddle** | Sell call + put same strike | Stock doesn't move |
        | **Long Strangle** | Buy OTM call + put | Big move, cheaper than straddle |
        | **Short Strangle** | Sell OTM call + put | Stock stays in range |
        
        ### 📈 VIX as Fear Gauge
        
        | VIX Level | Market Condition |
        |-----------|------------------|
        | **Below 15** | Complacent, low fear |
        | **15-20** | Normal |
        | **20-30** | Elevated concern |
        | **Above 30** | Fear/panic |
        | **Above 40** | Crisis levels |
        """)
        
        st.warning("""
        ⚠️ **Risk Warning**: Selling volatility (short straddles/strangles) has 
        unlimited risk. A single market crash can wipe out years of gains. 
        Always use defined-risk structures or strict position sizing.
        """)
        
    elif lesson_id == "hedging":
        st.markdown("*📖 Source: Options as a Strategic Investment, CBOE, Professional Risk Management*")
        
        st.markdown("""
        ## Portfolio Hedging
        
        Options can protect your portfolio from downside risk while maintaining 
        upside potential. Here are the key hedging strategies.
        
        ### 🛡️ Protective Put (Portfolio Insurance)
        
        Buy puts against your stock holdings:
        
        - **Own 100 shares** of SPY at $450
        - **Buy 1 put** at $430 strike for $5
        - **Max loss**: $450 - $430 + $5 = $25 (5.5%)
        - **Upside**: Unlimited (minus put cost)
        
        **Cost**: 1-3% of portfolio value per year
        
        ### 📊 Collar Strategy
        
        Protective put + covered call to reduce cost:
        
        ```
        Own stock      @ $100
        Buy put        @ $95  ($2)
        Sell call      @ $105 ($2)
        Net cost: $0 (zero-cost collar)
        ```
        
        **Tradeoff**: Give up upside above $105 to pay for downside protection below $95.
        """)
        
        # Collar P/L
        stock_prices = np.arange(85, 120, 0.5)
        stock_pl = stock_prices - 100
        collar_pl = np.minimum(np.maximum(stock_prices, 95), 105) - 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=stock_pl, name='Stock Only',
                                 line=dict(color='#94a3b8', dash='dash')))
        fig.add_trace(go.Scatter(x=stock_prices, y=collar_pl, name='Collar',
                                 line=dict(color='#4ade80', width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
        fig.update_layout(
            title="Stock vs Collar P/L",
            height=300,
            xaxis_title="Stock Price",
            yaxis_title="Profit/Loss ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Hedging Strategies Compared
        
        | Strategy | Cost | Protection | Upside |
        |----------|------|------------|--------|
        | **Protective Put** | High | Full below strike | Unlimited |
        | **Collar** | Low/Zero | Full below put strike | Capped at call strike |
        | **Put Spread** | Medium | Partial | Unlimited |
        | **VIX Calls** | Variable | Portfolio-wide | N/A (hedge only) |
        
        ### 📅 When to Hedge
        
        | Scenario | Hedge? |
        |----------|--------|
        | Long time horizon | Usually no - ride out volatility |
        | Near retirement | Consider collars |
        | Concentrated position | Yes - reduce single-stock risk |
        | Expecting volatility | Evaluate cost/benefit |
        | Market at all-time highs | Hedging is cheap when IV is low |
        """)
        
        st.info("""
        💡 **Professional Perspective**: Most long-term investors don't need to hedge.
        Time and diversification provide natural protection. Hedging makes sense for 
        concentrated positions, short time horizons, or when you can't afford a drawdown.
        """)


if __name__ == "__main__":
    main()
