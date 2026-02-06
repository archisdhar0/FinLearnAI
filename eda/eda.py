import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the SCF 2022 data
df = pd.read_csv('SCFP2022.csv')

print("=" * 60)
print("SURVEY OF CONSUMER FINANCES 2022 - INVESTING EDA")
print("=" * 60)

# ============================================================
# 1. BASIC DATA OVERVIEW
# ============================================================
print("\n📊 DATASET OVERVIEW")
print("-" * 40)
print(f"Total observations: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Key investing-related columns
invest_cols = {
    'EQUITY': 'Total equity holdings (stocks + mutual funds)',
    'HEQUITY': 'Has any equity (1=Yes)',
    'STOCKS': 'Direct stock holdings',
    'HSTOCKS': 'Has stocks (1=Yes)',
    'NMMF': 'Non-money market mutual funds',
    'BOND': 'Bond holdings',
    'HBOND': 'Has bonds (1=Yes)',
    'RETQLIQ': 'Retirement account liquid assets',
    'FIN': 'Total financial assets',
    'NETWORTH': 'Net worth',
    'INCOME': 'Total household income',
    'AGE': 'Age of respondent',
    'AGECL': 'Age class (1=<35, 2=35-44, 3=45-54, 4=55-64, 5=65-74, 6=75+)',
    'EDCL': 'Education class (1=No HS, 2=HS, 3=Some college, 4=College+)',
    'RACECL4': 'Race (1=White, 2=Black, 3=Hispanic, 4=Other)',
    'YESFINRISK': 'Willing to take financial risk (1=Yes)',
    'KNOWL': 'Self-rated financial knowledge (1-10)',
    'WGT': 'Survey weight'
}

print("\n📋 KEY INVESTING COLUMNS:")
for col, desc in invest_cols.items():
    if col in df.columns:
        print(f"  • {col}: {desc}")

# ============================================================
# 2. EQUITY OWNERSHIP ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("💰 EQUITY OWNERSHIP ANALYSIS")
print("=" * 60)

# Weighted statistics (SCF requires using weights for population estimates)
total_weight = df['WGT'].sum()

# Percentage who own equity
equity_owners = df[df['HEQUITY'] == 1]['WGT'].sum()
pct_equity = (equity_owners / total_weight) * 100
print(f"\n📈 Population owning equity: {pct_equity:.1f}%")

# Stock ownership
stock_owners = df[df['HSTOCKS'] == 1]['WGT'].sum()
pct_stocks = (stock_owners / total_weight) * 100
print(f"📈 Population owning stocks directly: {pct_stocks:.1f}%")

# Bond ownership
bond_owners = df[df['HBOND'] == 1]['WGT'].sum()
pct_bonds = (bond_owners / total_weight) * 100
print(f"📈 Population owning bonds: {pct_bonds:.1f}%")

# ============================================================
# 3. EQUITY BY DEMOGRAPHICS
# ============================================================
print("\n" + "=" * 60)
print("👥 EQUITY OWNERSHIP BY DEMOGRAPHICS")
print("=" * 60)

# By Age Group
age_labels = {1: '<35', 2: '35-44', 3: '45-54', 4: '55-64', 5: '65-74', 6: '75+'}
print("\n📊 By Age Group:")
for age_cl, label in age_labels.items():
    age_df = df[df['AGECL'] == age_cl]
    owners = age_df[age_df['HEQUITY'] == 1]['WGT'].sum()
    total = age_df['WGT'].sum()
    pct = (owners / total) * 100 if total > 0 else 0
    median_equity = age_df[age_df['EQUITY'] > 0]['EQUITY'].median()
    print(f"  {label:>6}: {pct:5.1f}% own equity | Median (if own): ${median_equity:,.0f}")

# By Education
edu_labels = {1: 'No HS', 2: 'HS Diploma', 3: 'Some College', 4: 'College+'}
print("\n📊 By Education Level:")
for edu_cl, label in edu_labels.items():
    edu_df = df[df['EDCL'] == edu_cl]
    owners = edu_df[edu_df['HEQUITY'] == 1]['WGT'].sum()
    total = edu_df['WGT'].sum()
    pct = (owners / total) * 100 if total > 0 else 0
    median_equity = edu_df[edu_df['EQUITY'] > 0]['EQUITY'].median()
    print(f"  {label:>12}: {pct:5.1f}% own equity | Median (if own): ${median_equity:,.0f}")

# By Race
race_labels = {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Other'}
print("\n📊 By Race/Ethnicity:")
for race_cl, label in race_labels.items():
    race_df = df[df['RACECL4'] == race_cl]
    owners = race_df[race_df['HEQUITY'] == 1]['WGT'].sum()
    total = race_df['WGT'].sum()
    pct = (owners / total) * 100 if total > 0 else 0
    median_equity = race_df[race_df['EQUITY'] > 0]['EQUITY'].median()
    print(f"  {label:>8}: {pct:5.1f}% own equity | Median (if own): ${median_equity:,.0f}")

# ============================================================
# 4. INCOME & WEALTH DISTRIBUTION
# ============================================================
print("\n" + "=" * 60)
print("💵 INCOME & WEALTH DISTRIBUTION")
print("=" * 60)

# Income percentiles
income_percentiles = df['INCOME'].quantile([0.25, 0.5, 0.75, 0.9, 0.99])
print("\n📊 Income Distribution:")
print(f"  25th percentile: ${income_percentiles[0.25]:,.0f}")
print(f"  50th percentile (Median): ${income_percentiles[0.5]:,.0f}")
print(f"  75th percentile: ${income_percentiles[0.75]:,.0f}")
print(f"  90th percentile: ${income_percentiles[0.9]:,.0f}")
print(f"  99th percentile: ${income_percentiles[0.99]:,.0f}")

# Net worth percentiles
nw_percentiles = df['NETWORTH'].quantile([0.25, 0.5, 0.75, 0.9, 0.99])
print("\n📊 Net Worth Distribution:")
print(f"  25th percentile: ${nw_percentiles[0.25]:,.0f}")
print(f"  50th percentile (Median): ${nw_percentiles[0.5]:,.0f}")
print(f"  75th percentile: ${nw_percentiles[0.75]:,.0f}")
print(f"  90th percentile: ${nw_percentiles[0.9]:,.0f}")
print(f"  99th percentile: ${nw_percentiles[0.99]:,.0f}")

# Equity as % of net worth for equity holders
equity_holders = df[df['EQUITY'] > 0].copy()
equity_holders['equity_pct'] = (equity_holders['EQUITY'] / equity_holders['NETWORTH'].replace(0, np.nan)) * 100
print(f"\n📊 Equity as % of Net Worth (for equity holders):")
print(f"  Median: {equity_holders['equity_pct'].median():.1f}%")
print(f"  Mean: {equity_holders['equity_pct'].mean():.1f}%")

# ============================================================
# 5. INVESTMENT PARTICIPATION BY INCOME QUARTILE
# ============================================================
print("\n" + "=" * 60)
print("📈 INVESTMENT PARTICIPATION BY INCOME QUARTILE")
print("=" * 60)

df['income_quartile'] = pd.qcut(df['INCOME'], q=4, labels=['Bottom 25%', '25-50%', '50-75%', 'Top 25%'])

for quartile in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']:
    q_df = df[df['income_quartile'] == quartile]
    equity_pct = (q_df[q_df['HEQUITY'] == 1]['WGT'].sum() / q_df['WGT'].sum()) * 100
    stock_pct = (q_df[q_df['HSTOCKS'] == 1]['WGT'].sum() / q_df['WGT'].sum()) * 100
    bond_pct = (q_df[q_df['HBOND'] == 1]['WGT'].sum() / q_df['WGT'].sum()) * 100
    median_nw = q_df['NETWORTH'].median()
    print(f"\n{quartile}:")
    print(f"  Equity ownership: {equity_pct:.1f}%")
    print(f"  Direct stock ownership: {stock_pct:.1f}%")
    print(f"  Bond ownership: {bond_pct:.1f}%")
    print(f"  Median net worth: ${median_nw:,.0f}")

# ============================================================
# 6. RISK TOLERANCE & FINANCIAL KNOWLEDGE
# ============================================================
print("\n" + "=" * 60)
print("🎯 RISK TOLERANCE & FINANCIAL KNOWLEDGE")
print("=" * 60)

# Risk tolerance and equity ownership
risk_takers = df[df['YESFINRISK'] == 1]
non_risk = df[df['YESFINRISK'] == 0]

risk_equity = (risk_takers[risk_takers['HEQUITY'] == 1]['WGT'].sum() / risk_takers['WGT'].sum()) * 100
nonrisk_equity = (non_risk[non_risk['HEQUITY'] == 1]['WGT'].sum() / non_risk['WGT'].sum()) * 100

print(f"\n📊 Risk Tolerance & Equity Ownership:")
print(f"  Risk-takers who own equity: {risk_equity:.1f}%")
print(f"  Risk-averse who own equity: {nonrisk_equity:.1f}%")

# Financial knowledge and equity ownership
print(f"\n📊 Financial Knowledge (1-10 scale) & Equity Ownership:")
for knowl_level in sorted(df['KNOWL'].unique()):
    knowl_df = df[df['KNOWL'] == knowl_level]
    if len(knowl_df) > 0:
        equity_pct = (knowl_df[knowl_df['HEQUITY'] == 1]['WGT'].sum() / knowl_df['WGT'].sum()) * 100
        count = len(knowl_df)
        print(f"  Knowledge level {knowl_level}: {equity_pct:.1f}% own equity (n={count:,})")

# ============================================================
# 7. PORTFOLIO COMPOSITION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("📊 PORTFOLIO COMPOSITION (for investors)")
print("=" * 60)

investors = df[df['FIN'] > 0].copy()

# Calculate asset allocation for those with financial assets
investors['pct_stocks'] = (investors['STOCKS'] / investors['FIN']) * 100
investors['pct_bonds'] = (investors['BOND'] / investors['FIN']) * 100
investors['pct_retirement'] = (investors['RETQLIQ'] / investors['FIN']) * 100
investors['pct_equity'] = (investors['EQUITY'] / investors['FIN']) * 100

print("\n📊 Median Portfolio Allocation (% of financial assets):")
print(f"  Direct stocks: {investors['pct_stocks'].median():.1f}%")
print(f"  Bonds: {investors['pct_bonds'].median():.1f}%")
print(f"  Total equity: {investors['pct_equity'].median():.1f}%")
print(f"  Retirement accounts: {investors['pct_retirement'].median():.1f}%")

# ============================================================
# 8. CREATE VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("📈 CREATING VISUALIZATIONS...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Survey of Consumer Finances 2022 - Investing Analysis', fontsize=14, fontweight='bold')

# 1. Equity ownership by age
ax1 = axes[0, 0]
age_equity = []
for age_cl, label in age_labels.items():
    age_df = df[df['AGECL'] == age_cl]
    owners = age_df[age_df['HEQUITY'] == 1]['WGT'].sum()
    total = age_df['WGT'].sum()
    age_equity.append((label, (owners / total) * 100 if total > 0 else 0))
age_equity_df = pd.DataFrame(age_equity, columns=['Age', 'Pct'])
ax1.bar(age_equity_df['Age'], age_equity_df['Pct'], color='steelblue', edgecolor='navy')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('% Owning Equity')
ax1.set_title('Equity Ownership by Age')
ax1.set_ylim(0, 100)
for i, v in enumerate(age_equity_df['Pct']):
    ax1.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

# 2. Equity ownership by education
ax2 = axes[0, 1]
edu_equity = []
for edu_cl, label in edu_labels.items():
    edu_df = df[df['EDCL'] == edu_cl]
    owners = edu_df[edu_df['HEQUITY'] == 1]['WGT'].sum()
    total = edu_df['WGT'].sum()
    edu_equity.append((label, (owners / total) * 100 if total > 0 else 0))
edu_equity_df = pd.DataFrame(edu_equity, columns=['Education', 'Pct'])
ax2.bar(edu_equity_df['Education'], edu_equity_df['Pct'], color='seagreen', edgecolor='darkgreen')
ax2.set_xlabel('Education Level')
ax2.set_ylabel('% Owning Equity')
ax2.set_title('Equity Ownership by Education')
ax2.set_ylim(0, 100)
ax2.tick_params(axis='x', rotation=15)
for i, v in enumerate(edu_equity_df['Pct']):
    ax2.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

# 3. Equity ownership by income quartile
ax3 = axes[0, 2]
income_equity = []
for quartile in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']:
    q_df = df[df['income_quartile'] == quartile]
    owners = q_df[q_df['HEQUITY'] == 1]['WGT'].sum()
    total = q_df['WGT'].sum()
    income_equity.append((quartile, (owners / total) * 100 if total > 0 else 0))
income_equity_df = pd.DataFrame(income_equity, columns=['Income', 'Pct'])
ax3.bar(income_equity_df['Income'], income_equity_df['Pct'], color='coral', edgecolor='darkred')
ax3.set_xlabel('Income Quartile')
ax3.set_ylabel('% Owning Equity')
ax3.set_title('Equity Ownership by Income')
ax3.set_ylim(0, 100)
ax3.tick_params(axis='x', rotation=15)
for i, v in enumerate(income_equity_df['Pct']):
    ax3.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

# 4. Net worth distribution (log scale)
ax4 = axes[1, 0]
positive_nw = df[df['NETWORTH'] > 0]['NETWORTH']
ax4.hist(np.log10(positive_nw), bins=50, color='purple', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Net Worth (log10 scale)')
ax4.set_ylabel('Count')
ax4.set_title('Net Worth Distribution (Positive NW)')
ax4.axvline(np.log10(df['NETWORTH'].median()), color='red', linestyle='--', label=f'Median: ${df["NETWORTH"].median():,.0f}')
ax4.legend()

# 5. Equity vs Income scatter
ax5 = axes[1, 1]
sample = df[(df['EQUITY'] > 0) & (df['INCOME'] > 0)].sample(min(2000, len(df)))
ax5.scatter(np.log10(sample['INCOME']), np.log10(sample['EQUITY']), alpha=0.3, s=10, c='teal')
ax5.set_xlabel('Income (log10)')
ax5.set_ylabel('Equity Holdings (log10)')
ax5.set_title('Income vs Equity Holdings')

# 6. Equity ownership by race
ax6 = axes[1, 2]
race_equity = []
for race_cl, label in race_labels.items():
    race_df = df[df['RACECL4'] == race_cl]
    owners = race_df[race_df['HEQUITY'] == 1]['WGT'].sum()
    total = race_df['WGT'].sum()
    race_equity.append((label, (owners / total) * 100 if total > 0 else 0))
race_equity_df = pd.DataFrame(race_equity, columns=['Race', 'Pct'])
ax6.bar(race_equity_df['Race'], race_equity_df['Pct'], color='goldenrod', edgecolor='darkgoldenrod')
ax6.set_xlabel('Race/Ethnicity')
ax6.set_ylabel('% Owning Equity')
ax6.set_title('Equity Ownership by Race')
ax6.set_ylim(0, 100)
for i, v in enumerate(race_equity_df['Pct']):
    ax6.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('investing_eda_charts.png', dpi=150, bbox_inches='tight')
print("✅ Saved: investing_eda_charts.png")

# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("🔗 CORRELATION ANALYSIS")
print("=" * 60)

corr_vars = ['INCOME', 'NETWORTH', 'EQUITY', 'STOCKS', 'BOND', 'AGE', 'EDCL', 'YESFINRISK', 'KNOWL']
corr_df = df[corr_vars].corr()

print("\n📊 Key Correlations with Equity Holdings:")
equity_corr = corr_df['EQUITY'].drop('EQUITY').sort_values(ascending=False)
for var, corr in equity_corr.items():
    print(f"  {var}: {corr:.3f}")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix: Financial Variables', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
print("✅ Saved: correlation_heatmap.png")

# ============================================================
# 10. KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("💡 KEY INSIGHTS FOR BREAKING INTO INVESTING")
print("=" * 60)

print("""
1. EQUITY OWNERSHIP GAP: 
   - Only about {:.0f}% of Americans own equity investments
   - Large disparities exist by income, education, and race

2. EDUCATION MATTERS:
   - College graduates are significantly more likely to invest
   - Financial knowledge correlates with investment participation

3. INCOME & INVESTING:
   - Top 25% income earners have much higher equity ownership
   - Even at lower incomes, some people invest - it's possible!

4. RISK TOLERANCE:
   - Risk-takers are more likely to own equity
   - Understanding your risk tolerance is key before investing

5. BARRIERS TO ENTRY:
   - Knowledge gap is a significant barrier
   - Starting small is possible - not everyone has large portfolios

6. RECOMMENDATIONS FOR BEGINNERS:
   - Start with retirement accounts (401k, IRA)
   - Learn about low-cost index funds
   - Understand your risk tolerance
   - Start early - time in market matters
   - Education and financial literacy pay off
""".format(pct_equity))

plt.show()
print("\n✅ EDA Complete!")