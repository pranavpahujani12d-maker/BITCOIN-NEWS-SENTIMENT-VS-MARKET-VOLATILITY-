import pandas as pd
import sys
import os

print("Step 1: Importing libraries...", flush=True)

try:
    from transformers import pipeline
    import torch
    print("Step 2: Transformers and torch imported successfully", flush=True)
except Exception as e:
    print(f"ERROR importing transformers/torch: {e}", flush=True)
    sys.exit(1)

print("Step 3: Loading the dataset...", flush=True)

try:
    # Load the dataset
    df = pd.read_csv(r'c:\Users\pooja\OneDrive\Desktop\ML FINAL\bitcoin_sentiments_21_24.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Step 4: Dataset loaded with {len(df)} rows", flush=True)
except Exception as e:
    print(f"ERROR loading dataset: {e}", flush=True)
    sys.exit(1)

try:
    device = 0 if torch.cuda.is_available() else -1
    print(f'Using: {"GPU ✓" if device == 0 else "CPU (slower)"}', flush=True)
    print('Loading DistilBERT model...', flush=True)

    classifier = pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        device=device,
        truncation=True,
        max_length=512
    )
    print('✓ Model loaded — running inference...', flush=True)
except Exception as e:
    print(f"ERROR loading model: {e}", flush=True)
    sys.exit(1)

texts = df['Short Description'].tolist()
batch_size = 32
results = []

print(f"Step 5: Starting inference on {len(texts)} texts...", flush=True)

try:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)
        if (i // batch_size) % 20 == 0:
            pct = min(100, i / len(texts) * 100)
            print(f'  {pct:.0f}% complete ({i:,}/{len(texts):,} rows)', flush=True)
except KeyboardInterrupt:
    print('\n⚠ Analysis interrupted by user', flush=True)
    if results:
        print(f'Processed {len(results)} rows before interruption', flush=True)
except Exception as e:
    print(f"ERROR during inference: {e}", flush=True)
    print(f"Processed {len(results)} rows before error", flush=True)

print(f"Step 6: Converting results to signed scores...", flush=True)

# Convert to signed score: POSITIVE = +confidence, NEGATIVE = -confidence
def to_signed_score(result):
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

if results:
    # Trim dataframe to match number of results first
    df = df.iloc[:len(results)].reset_index(drop=True)
    
    # Then assign the scores
    df['bert_score'] = [to_signed_score(r) for r in results]
    
    print('\n✓ DistilBERT complete', flush=True)
    print(f'\nProcessed {len(results)} rows', flush=True)
    print('\nDistilBERT score stats:', flush=True)
    print(df['bert_score'].describe().round(4))
    sys.stdout.flush()
else:
    print('\n✗ No results to process', flush=True)
