import pandas as pd
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv(r'c:\Users\pooja\OneDrive\Desktop\ML FINAL\bitcoin_sentiments_21_24.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

analyzer = SentimentIntensityAnalyzer()

print('Running VADER on 11,295 headlines...')
df['vader_score'] = df['Short Description'].apply(
    lambda text: analyzer.polarity_scores(str(text))['compound']
)
print('✓ VADER complete')
print()
print('VADER score stats:')
print(df['vader_score'].describe().round(4))
print()
print('Sample results:')
print(df[['Short Description', 'vader_score', 'Accurate Sentiments']].head(5).to_string())
