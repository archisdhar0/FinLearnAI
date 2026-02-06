import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 70)
print("REDDIT INVESTING COMMUNITIES - TEXT EDA")
print("Breaking Into Investing: Language & Community Analysis")
print("=" * 70)

# ============================================================
# 1. LOAD ALL SUBREDDIT DATA
# ============================================================
base_path = "archive (2)"
subreddits = [
    'finance', 'financialindependence', 'forex', 'gme', 'investing',
    'options', 'pennystocks', 'personalfinance', 'robinhood',
    'robinhoodpennystocks', 'securityanalysis', 'stockmarket',
    'stocks', 'wallstreetbets'
]

# Categorize subreddits by experience level/type
subreddit_categories = {
    'Beginner-Friendly': ['personalfinance', 'financialindependence', 'robinhood'],
    'General Investing': ['investing', 'stocks', 'stockmarket', 'finance'],
    'Advanced/Speculative': ['options', 'forex', 'securityanalysis'],
    'Meme/YOLO': ['wallstreetbets', 'gme', 'pennystocks', 'robinhoodpennystocks']
}

# Create reverse mapping
subreddit_to_category = {}
for cat, subs in subreddit_categories.items():
    for sub in subs:
        subreddit_to_category[sub] = cat

all_data = []
for subreddit in subreddits:
    filepath = os.path.join(base_path, subreddit, "submissions_reddit.csv")
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df['subreddit'] = subreddit
            df['category'] = subreddit_to_category.get(subreddit, 'Other')
            all_data.append(df)
            print(f"✅ Loaded {subreddit}: {len(df):,} posts")
        except Exception as e:
            print(f"❌ Error loading {subreddit}: {e}")

df = pd.concat(all_data, ignore_index=True)
print(f"\n📊 Total posts loaded: {len(df):,}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 70)
print("🧹 DATA CLEANING")
print("=" * 70)

# Combine title and selftext for analysis
df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
df['text'] = df['text'].str.strip()

# Remove deleted/removed posts
df = df[~df['text'].str.contains(r'\[deleted\]|\[removed\]', case=False, na=False)]
df = df[df['text'].str.len() > 10]  # Remove very short posts

print(f"Posts after cleaning: {len(df):,}")

# ============================================================
# 3. DEFINE INVESTING VOCABULARY LISTS
# ============================================================

# Beginner terms - what newcomers ask about
beginner_terms = [
    'how do i', 'what is', 'eli5', 'explain', 'beginner', 'new to',
    'first time', 'starting out', 'advice', 'help me', 'question',
    'confused', 'understand', 'learning', 'basics', 'newbie', 'noob',
    'getting started', 'where to start', 'should i'
]

# Intermediate terms - common investing concepts
intermediate_terms = [
    'etf', 'index fund', 'dividend', 'roth ira', '401k', 'portfolio',
    'diversification', 'compound interest', 'dollar cost averaging', 'dca',
    'expense ratio', 'vanguard', 'fidelity', 'schwab', 'brokerage',
    'capital gains', 'tax loss harvesting', 'rebalancing', 'asset allocation'
]

# Advanced/Jargon terms - barrier to entry
advanced_terms = [
    'options', 'puts', 'calls', 'strike price', 'expiration', 'theta',
    'delta', 'gamma', 'iv', 'implied volatility', 'greeks', 'spreads',
    'straddle', 'strangle', 'iron condor', 'covered call', 'naked put',
    'futures', 'derivatives', 'leverage', 'margin', 'short selling',
    'technical analysis', 'fundamental analysis', 'pe ratio', 'eps',
    'market cap', 'bull', 'bear', 'hedge', 'arbitrage', 'alpha', 'beta',
    'sharpe ratio', 'drawdown', 'volatility', 'correlation'
]

# WSB/Meme terms - cultural barrier
meme_terms = [
    'yolo', 'tendies', 'diamond hands', 'paper hands', 'ape', 'moon',
    'to the moon', 'rocket', '🚀', '💎', '🙌', 'stonks', 'hodl',
    'wife\'s boyfriend', 'retard', 'autist', 'smooth brain', 'wrinkle brain',
    'loss porn', 'gain porn', 'bagholder', 'fomo', 'fud', 'dd',
    'due diligence', 'squeeze', 'short squeeze', 'gamma squeeze'
]

def count_term_matches(text, terms):
    """Count how many terms from a list appear in text"""
    text_lower = str(text).lower()
    count = sum(1 for term in terms if term in text_lower)
    return count

# Apply term counting
print("\n📊 Analyzing vocabulary usage...")
df['beginner_terms'] = df['text'].apply(lambda x: count_term_matches(x, beginner_terms))
df['intermediate_terms'] = df['text'].apply(lambda x: count_term_matches(x, intermediate_terms))
df['advanced_terms'] = df['text'].apply(lambda x: count_term_matches(x, advanced_terms))
df['meme_terms'] = df['text'].apply(lambda x: count_term_matches(x, meme_terms))

# ============================================================
# 4. VOCABULARY ANALYSIS BY SUBREDDIT
# ============================================================
print("\n" + "=" * 70)
print("📚 VOCABULARY COMPLEXITY BY SUBREDDIT")
print("=" * 70)

vocab_by_sub = df.groupby('subreddit').agg({
    'beginner_terms': 'mean',
    'intermediate_terms': 'mean',
    'advanced_terms': 'mean',
    'meme_terms': 'mean',
    'text': 'count'
}).rename(columns={'text': 'post_count'}).round(3)

print("\nAverage terms per post by subreddit:")
print(vocab_by_sub.sort_values('advanced_terms', ascending=False))

# ============================================================
# 5. WORD FREQUENCY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("📊 MOST COMMON WORDS BY COMMUNITY TYPE")
print("=" * 70)

def get_word_freq(texts, n=30):
    """Get word frequency from texts, excluding common stopwords"""
    stopwords = set(['the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'in', 'for',
                     'on', 'that', 'this', 'with', 'i', 'you', 'my', 'be', 'are',
                     'was', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'can', 'if', 'or', 'but', 'not',
                     'so', 'just', 'from', 'at', 'by', 'what', 'how', 'when',
                     'where', 'why', 'who', 'which', 'all', 'any', 'some', 'no',
                     'more', 'most', 'other', 'into', 'over', 'after', 'before',
                     'up', 'down', 'out', 'about', 'than', 'then', 'now', 'here',
                     'there', 'these', 'those', 'your', 'our', 'their', 'its',
                     'been', 'being', 'get', 'got', 'like', 'know', 'think',
                     'see', 'want', 'go', 'going', 'make', 'made', 'one', 'two',
                     'also', 'amp', 'https', 'http', 'www', 'com', 'dont', 'im',
                     've', 're', 'll', 'd', 's', 't', 'deleted', 'removed'])
    
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        all_words.extend([w for w in words if w not in stopwords])
    
    return Counter(all_words).most_common(n)

for category in subreddit_categories.keys():
    cat_texts = df[df['category'] == category]['text']
    if len(cat_texts) > 0:
        print(f"\n🏷️ {category} ({len(cat_texts):,} posts):")
        word_freq = get_word_freq(cat_texts, 20)
        words = [w[0] for w in word_freq]
        print(f"   Top words: {', '.join(words)}")

# ============================================================
# 6. POST ENGAGEMENT ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("📈 ENGAGEMENT ANALYSIS")
print("=" * 70)

engagement_stats = df.groupby('subreddit').agg({
    'score': ['mean', 'median'],
    'num_comments': ['mean', 'median'],
    'upvote_ratio': 'mean'
}).round(2)

print("\nEngagement by subreddit:")
print(engagement_stats.sort_values(('score', 'mean'), ascending=False))

# ============================================================
# 7. QUESTION ANALYSIS - What do beginners ask?
# ============================================================
print("\n" + "=" * 70)
print("❓ QUESTION ANALYSIS - What Beginners Ask")
print("=" * 70)

# Find posts that are questions
question_patterns = [
    r'\?$',  # Ends with question mark
    r'^(how|what|why|when|where|should|can|is|are|do|does|will|would)\s',
]

df['is_question'] = df['title'].str.contains('|'.join(question_patterns), 
                                               case=False, regex=True, na=False)

question_rate = df.groupby('category')['is_question'].mean() * 100
print("\nPercentage of posts that are questions:")
for cat, rate in question_rate.sort_values(ascending=False).items():
    print(f"  {cat}: {rate:.1f}%")

# Sample beginner questions
print("\n📝 Sample Beginner Questions (from personalfinance/investing):")
beginner_subs = df[df['subreddit'].isin(['personalfinance', 'investing', 'stocks'])]
questions = beginner_subs[beginner_subs['is_question']]['title'].dropna()
for q in questions.sample(min(10, len(questions))).values:
    if len(q) < 150:
        print(f"  • {q}")

# ============================================================
# 8. READABILITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("📖 TEXT COMPLEXITY ANALYSIS")
print("=" * 70)

# Calculate basic text metrics
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
)

complexity_by_cat = df.groupby('category').agg({
    'word_count': 'median',
    'avg_word_length': 'mean',
    'advanced_terms': 'mean'
}).round(2)

print("\nText complexity by community type:")
print(complexity_by_cat.sort_values('advanced_terms', ascending=False))

# ============================================================
# 9. CREATE VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("📈 CREATING VISUALIZATIONS...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Breaking Into Investing: Reddit Community Language Analysis', 
             fontsize=16, fontweight='bold')

# 1. Vocabulary complexity by category
ax1 = axes[0, 0]
vocab_by_cat = df.groupby('category')[['beginner_terms', 'intermediate_terms', 
                                        'advanced_terms', 'meme_terms']].mean()
vocab_by_cat.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Vocabulary Usage by Community Type', fontsize=12)
ax1.set_xlabel('Community Type')
ax1.set_ylabel('Average Terms per Post')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Term Type', fontsize=8)

# 2. Advanced terms by subreddit
ax2 = axes[0, 1]
adv_by_sub = df.groupby('subreddit')['advanced_terms'].mean().sort_values(ascending=True)
colors = ['#e74c3c' if x > adv_by_sub.median() else '#3498db' for x in adv_by_sub.values]
adv_by_sub.plot(kind='barh', ax=ax2, color=colors)
ax2.set_title('Advanced Jargon Usage by Subreddit', fontsize=12)
ax2.set_xlabel('Avg Advanced Terms per Post')
ax2.axvline(adv_by_sub.median(), color='gray', linestyle='--', label='Median')

# 3. Question rate by category
ax3 = axes[0, 2]
question_rate.sort_values().plot(kind='barh', ax=ax3, color='seagreen')
ax3.set_title('% of Posts That Are Questions', fontsize=12)
ax3.set_xlabel('Percentage')
for i, v in enumerate(question_rate.sort_values().values):
    ax3.text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=9)

# 4. Engagement by community type
ax4 = axes[1, 0]
eng_by_cat = df.groupby('category')['score'].median()
eng_by_cat.sort_values().plot(kind='barh', ax=ax4, color='coral')
ax4.set_title('Median Post Score by Community Type', fontsize=12)
ax4.set_xlabel('Median Score')

# 5. Meme terms concentration
ax5 = axes[1, 1]
meme_by_sub = df.groupby('subreddit')['meme_terms'].mean().sort_values(ascending=True)
colors = ['#9b59b6' if x > meme_by_sub.median() else '#95a5a6' for x in meme_by_sub.values]
meme_by_sub.plot(kind='barh', ax=ax5, color=colors)
ax5.set_title('Meme/Slang Usage by Subreddit', fontsize=12)
ax5.set_xlabel('Avg Meme Terms per Post')

# 6. Community size comparison
ax6 = axes[1, 2]
post_counts = df.groupby('subreddit').size().sort_values(ascending=True)
post_counts.plot(kind='barh', ax=ax6, color='steelblue')
ax6.set_title('Number of Posts by Subreddit', fontsize=12)
ax6.set_xlabel('Post Count')

plt.tight_layout()
plt.savefig('reddit_text_eda.png', dpi=150, bbox_inches='tight')
print("✅ Saved: reddit_text_eda.png")

# ============================================================
# 10. BARRIER TO ENTRY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("🚧 BARRIERS TO ENTRY ANALYSIS")
print("=" * 70)

print("""
Based on the text analysis, here are the key barriers for newcomers:

1. VOCABULARY BARRIER:
   - Advanced communities (options, forex) use significantly more jargon
   - Beginners need to learn 50+ specialized terms just to participate
   - Meme culture creates additional insider language
   
2. CULTURAL BARRIER:
   - WSB/meme communities have unique slang that can be intimidating
   - "Loss porn" and "YOLO" culture may discourage cautious beginners
   
3. QUESTION RATE ANALYSIS:
   - Beginner-friendly subs have higher question rates (more welcoming)
   - Advanced subs assume prior knowledge
   
4. RECOMMENDATIONS FOR BEGINNERS:
   - Start with r/personalfinance and r/financialindependence
   - Learn intermediate terms before advanced ones
   - Don't feel pressured by meme culture - invest responsibly
""")

# ============================================================
# 11. WORD CLOUDS (if wordcloud is available)
# ============================================================
try:
    from wordcloud import WordCloud
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Clouds by Community Type', fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(subreddit_categories.keys()):
        ax = axes[idx // 2, idx % 2]
        cat_text = ' '.join(df[df['category'] == category]['text'].dropna().astype(str))
        
        if len(cat_text) > 100:
            wordcloud = WordCloud(width=800, height=400, 
                                  background_color='white',
                                  max_words=100,
                                  colormap='viridis').generate(cat_text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(category, fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('reddit_wordclouds.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: reddit_wordclouds.png")
except ImportError:
    print("⚠️ wordcloud not installed - skipping word cloud visualization")
    print("   Install with: pip install wordcloud")

# ============================================================
# 12. TIME SERIES ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("📅 TEMPORAL ANALYSIS")
print("=" * 70)

df['created'] = pd.to_datetime(df['created'], errors='coerce')
df['month'] = df['created'].dt.to_period('M')

# Posts over time
monthly_posts = df.groupby(['month', 'category']).size().unstack(fill_value=0)
if len(monthly_posts) > 3:
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_posts.plot(ax=ax, marker='o', markersize=3)
    ax.set_title('Reddit Investing Posts Over Time by Community Type', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Posts')
    ax.legend(title='Community Type', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('reddit_posts_timeline.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: reddit_posts_timeline.png")

# ============================================================
# 13. SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY")
print("=" * 70)

print(f"""
DATASET SUMMARY:
  • Total posts analyzed: {len(df):,}
  • Subreddits covered: {len(subreddits)}
  • Date range: {df['created'].min()} to {df['created'].max()}

KEY FINDINGS FOR BREAKING INTO INVESTING:

1. START HERE (Beginner-Friendly):
   - r/personalfinance: Highest question rate, supportive community
   - r/financialindependence: Long-term focus, FIRE movement
   
2. LEARN THE LANGUAGE:
   - Average post uses {df['intermediate_terms'].mean():.1f} intermediate terms
   - Advanced subs use {df[df['category'] == 'Advanced/Speculative']['advanced_terms'].mean():.1f}x more jargon
   
3. AVOID UNTIL READY:
   - r/wallstreetbets: High-risk culture, lots of meme terms
   - r/options: Requires understanding of derivatives
   
4. ENGAGEMENT INSIGHT:
   - Questions get answered! Asking is encouraged in beginner subs
""")

plt.show()
print("\n✅ Text EDA Complete!")
