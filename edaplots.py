import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
df = pd.read_csv(r'c:\Users\pooja\OneDrive\Desktop\ML FINAL\bitcoin_sentiments_21_24.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create Benchmark_Label column based on sentiment thresholds
def label_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Benchmark_Label'] = df['Accurate Sentiments'].apply(label_sentiment)

# Create Desc_Length column
df['Desc_Length'] = df['Short Description'].str.len()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bitcoin News Sentiment — Exploratory Data Analysis (2021–2024)',
             fontsize=14, fontweight='bold')

# Plot A: Sentiment distribution
ax = axes[0, 0]
ax.hist(df['Accurate Sentiments'], bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(0.05, color='green', linestyle='--', linewidth=1, label='Pos threshold')
ax.axvline(-0.05, color='red', linestyle='--', linewidth=1, label='Neg threshold')
ax.set_title('A — Benchmark sentiment distribution')
ax.set_xlabel('Score')
ax.set_ylabel('Count')
ax.legend(fontsize=9)

# Plot B: Label breakdown
ax = axes[0, 1]
label_counts = df['Benchmark_Label'].value_counts()
colors = {'Positive': '#4CAF50', 'Neutral': '#9E9E9E', 'Negative': '#F44336'}
bars = ax.bar(label_counts.index, label_counts.values,
              color=[colors[l] for l in label_counts.index], edgecolor='white')
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)
ax.set_title('B — Article count by sentiment label')
ax.set_ylabel('Count')
ax.set_ylim(0, label_counts.max() * 1.2)

# Plot C: Monthly sentiment timeline
ax = axes[1, 0]
monthly = df.set_index('Date').resample('ME')['Accurate Sentiments'].mean()
ax.plot(monthly.index, monthly.values, color='steelblue', linewidth=1.8, marker='o', markersize=3)
ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
ax.fill_between(monthly.index, monthly.values, 0,
                where=(monthly.values >= 0), color='#4CAF50', alpha=0.15, label='Net positive')
ax.fill_between(monthly.index, monthly.values, 0,
                where=(monthly.values < 0), color='#F44336', alpha=0.15, label='Net negative')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.set_title('C — Monthly average sentiment')
ax.set_ylabel('Mean score')
ax.legend(fontsize=9)

# Plot D: Description length
ax = axes[1, 1]
ax.hist(df['Desc_Length'], bins=50, color='coral', edgecolor='white', linewidth=0.4)
ax.axvline(df['Desc_Length'].mean(), color='darkred', linestyle='--', linewidth=1.5,
           label=f'Mean = {df["Desc_Length"].mean():.0f} chars')
ax.set_title('D — Description length distribution')
ax.set_xlabel('Characters')
ax.set_ylabel('Count')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Saved: plot_eda.png')