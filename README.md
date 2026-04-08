# BITCOIN-NEWS-SENTIMENT-VS-MARKET-VOLATILITY-(2021–2024)
MSc Business Analytics - Text as Data (Problem Set #2)
Research Question: How effectively can sentiment extracted from 11,000+ Bitcoin news headlines predict short-term market volatility ($|Returns|$) between 2021 and 2024?

Project Overview-

This repository investigates the intersection of Natural Language Processing (NLP) and Financial Analytics. In a market often driven by "FUD" (Fear, Uncertainty, and Doubt), this study evaluates whether automated sentiment signals can reliably predict Bitcoin price instability.
We compare two distinct analytical approaches:

VADER: A fast, rule-based lexicon baseline.
DistilBERT: A state-of-the-art transformer model (Sanh et al., 2019) used to capture deep contextual nuances in financial reporting.

Model:Correlation (r)Significance (p)Benchmark Agreement

VADER−              0.0789           0.0187    40.2%
DistilBERT−         0.1154           0.0006    42.3%

Conclusion: DistilBERT proved superior, demonstrating that context-aware deep learning captures market-moving signals more effectively than simple word-counts.

How to Reproduce-

This project is optimized for Google Colab to leverage T4 GPU acceleration for the DistilBERT model.
Dataset: Download the Bitcoin News Dataset from Kaggle.
Environment: Open the .ipynb in Colab and ensure the runtime is set to GPU.
Execution:  Upload bitcoin_sentiments_21_24.csv.
Run all cells to fetch live market data from Yahoo Finance and generate the sentiment scores.
Note: DistilBERT inference takes ~3 minutes on GPU.

The Analytical Pipeline-

Sentiment Extraction: Processing 11,295 headlines through VADER and DistilBERT
Market Alignment: Merging sentiment scores with daily Returns from yfinance.
Event Analysis: Mapping sentiment downturns to major historical catalysts (Terra/LUNA, FTX Bankruptcy, ETF Approvals).
Dynamic Correlation: Generating rolling 30-day Pearson $r$ coefficients to see how the news-price relationship evolves over time.

Key Insights-

Sensitivity: DistilBERT tracked negative market events (like the FTX collapse) much more sharply than VADER, explaining its higher mean negativity (−0.4151).
Time-Variance: The relationship between news and volatility is strongest during crisis periods, where media narratives actively drive investor behavior.
Validation: Both models were validated against the Accurate Sentiments benchmark to ensure high data integrity.

