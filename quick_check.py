#!/usr/bin/env python3
"""Quick data inspection to understand filter columns"""

import polars as pl
from pathlib import Path

# Load pair universe
pair_universe_path = Path("/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe")
print(f"Loading pair universe from: {pair_universe_path}")

df = pl.scan_parquet(pair_universe_path / "*.parquet")

# Show schema
print("\n" + "="*80)
print("SCHEMA:")
print("="*80)
schema = df.collect_schema()
for name, dtype in zip(schema.names(), schema.dtypes()):
    print(f"  {name:40s} {dtype}")

# Filter for Base chain
print(f"\n" + "="*80)
print("BASE CHAIN FILTERING:")
print("="*80)
df_base = df.filter(pl.col("chain_id") == 8453)
count_base = df_base.select(pl.count()).collect().item()
print(f"Pairs on Base (chain_id=8453): {count_base:,}")

# Check flag columns
print(f"\n" + "="*80)
print("FLAG COLUMNS ANALYSIS:")
print("="*80)

flag_cols = [
    "flag_inactive",
    "flag_blacklisted_manually",
    "flag_unsupported_quote_token",
    "flag_unknown_exchange"
]

available_flags = []
for col in flag_cols:
    if col in schema.names():
        available_flags.append(col)
        print(f"\n{col}:")
        value_counts = df_base.group_by(col).agg(pl.count().alias("count")).collect().sort("count", descending=True)
        print(value_counts)

# Check tax columns
print(f"\n" + "="*80)
print("TAX COLUMNS ANALYSIS:")
print("="*80)

tax_cols = ["buy_tax", "sell_tax", "transfer_tax"]
for col in tax_cols:
    if col in schema.names():
        print(f"\n{col}:")
        stats = df_base.select([
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).median().alias("median"),
            (pl.col(col).is_null().sum()).alias("nulls")
        ]).collect()
        print(stats)
        
        # Show distribution
        over_5pct = df_base.filter(pl.col(col) > 5.0).select(pl.count()).collect().item()
        print(f"  Pairs with {col} > 5%: {over_5pct:,}")

print(f"\n" + "="*80)
print("DONE")
print("="*80)
