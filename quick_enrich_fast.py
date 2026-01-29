#!/usr/bin/env python3
"""Super fast signal enrichment using polars"""

import polars as pl
import sys
from pathlib import Path

input_file = sys.argv[1] if len(sys.argv) > 1 else 'signals_live.csv'
output_file = input_file.replace('.csv', '_readable.csv')

print(f"Loading {input_file}...")
signals = pl.read_csv(input_file)
print(f"Found {len(signals)} signals")

print("Loading pair universe...")
universe = pl.read_parquet("Files/pair-universe")
print(f"Loaded {len(universe)} pairs")

# Select only needed columns from universe
universe_slim = universe.select(['address', 'token0_symbol', 'token1_symbol'])

# Join
print("Joining...")
enriched = signals.join(universe_slim, left_on='pair_address', right_on='address', how='left')

# Create pair_name
enriched = enriched.with_columns([
    (pl.col('token0_symbol').fill_null('UNKNOWN') + '/' + pl.col('token1_symbol').fill_null('UNKNOWN')).alias('pair_name')
])

# Sort by confidence
enriched = enriched.sort('pred_proba', descending=True)

# Save
enriched.write_csv(output_file)
print(f"✅ Saved to {output_file}")

# Stats
matched = enriched.filter(pl.col('token0_symbol').is_not_null()).height
print(f"\n📊 Matched {matched}/{len(enriched)} pairs ({matched/len(enriched)*100:.1f}%)")
print(f"Avg Confidence: {enriched['pred_proba'].mean():.2%}")
print(f">80%: {enriched.filter(pl.col('pred_proba') > 0.8).height}")
print(f">85%: {enriched.filter(pl.col('pred_proba') > 0.85).height}")

# Top 10
print("\n" + "="*80)
print("TOP 10 SIGNALS")
print("="*80)
for i, row in enumerate(enriched.head(10).iter_rows(named=True), 1):
    print(f"\n#{i} {row['pair_name']}")
    print(f"   Confidence: {row['pred_proba']:.2%} | Price: ${row['close']:.6f}")
    print(f"   Returns: 15m={row['return_1c']:.2f}% | 1h={row['return_4c']:.2f}% | 4h={row['return_16c']:.2f}%")
    print(f"   Pair: {row['pair_address']}")
print("="*80)
