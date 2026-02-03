#!/usr/bin/env python3
"""Check columns in live_candles_cache.pkl"""

import pickle
import polars as pl

# Load cache
with open('live_candles_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

print(f'Total pairs in cache: {len(cache):,}')

# Get first pair's data to inspect columns
first_pair = list(cache.keys())[0]
first_df = cache[first_pair]

print(f'\nSample pair: {first_pair}')
print(f'Type: {type(first_df)}')
print(f'Shape: {first_df.shape}')
print(f'\nColumns ({len(first_df.columns)}):')
for i, col in enumerate(first_df.columns, 1):
    print(f'  {i}. {col}')

print(f'\nSample data (first 3 rows):')
print(first_df.head(3))

print(f'\nData types:')
print(first_df.schema)
