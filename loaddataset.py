import pandas as pd

df = pd.read_csv(r'c:\Users\pooja\OneDrive\Desktop\ML FINAL\bitcoin_sentiments_21_24.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print('='*55)
print('DATASET OVERVIEW')
print('='*55)
print(f'Total articles : {len(df):,}')
print(f'Date range     : {df["Date"].min().date()}  to  {df["Date"].max().date()}')
print(f'Columns        : {df.columns.tolist()}')
print(f'Missing values : {df.isnull().sum().to_dict()}')
print()

# Flag zero-score rows
zero_count = (df['Accurate Sentiments'] == 0.0).sum()
print(f'Zero-score rows: {zero_count:,} ({zero_count/len(df)*100:.1f}%) — may be unscored entries')
print()
print('Accurate Sentiments stats:')
print(df['Accurate Sentiments'].describe().round(4))

# Add helper columns
def sentiment_label(score):
    if score > 0.05: return 'Positive'
    elif score < -0.05: return 'Negative'
    else: return 'Neutral'

df['Benchmark_Label'] = df['Accurate Sentiments'].apply(sentiment_label)
df['Desc_Length'] = df['Short Description'].str.len()

print()
print('Label breakdown (benchmark):')
for label, count in df['Benchmark_Label'].value_counts().items():
    print(f'  {label:<10}: {count:,}  ({count/len(df)*100:.1f}%)')
