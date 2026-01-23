#!/usr/bin/env python3
"""Quick data check script"""

import os
from pathlib import Path

print("Checking data paths...")

# Check paths
pair_universe = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe"
candles = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/candles-1d.parquet"

print(f"\nPair universe path: {pair_universe}")
print(f"  Exists: {Path(pair_universe).exists()}")
if Path(pair_universe).exists():
    print(f"  Is directory: {Path(pair_universe).is_dir()}")
    if Path(pair_universe).is_dir():
        files = list(Path(pair_universe).glob("*.parquet"))
        print(f"  Parquet files: {len(files)}")
        if files:
            print(f"  First file: {files[0]}")

print(f"\nCandles path: {candles}")
print(f"  Exists: {Path(candles).exists()}")
if Path(candles).exists():
    print(f"  Size: {Path(candles).stat().st_size / 1024 / 1024:.2f} MB")

# Try to load a sample
try:
    import polars as pl
    
    print("\n" + "="*80)
    print("LOADING SAMPLE DATA...")
    print("="*80)
    
    # Try pair universe
    if Path(pair_universe).is_dir():
        files = list(Path(pair_universe).glob("*.parquet"))
        if files:
            print(f"\nLoading pair universe from: {files[0]}")
            df_pairs = pl.read_parquet(files[0], n_rows=5)
            print(f"  Shape: {df_pairs.shape}")
            print(f"  Columns: {df_pairs.columns}")
            print(f"\nFirst few rows:")
            print(df_pairs.head())
    
    # Try candles
    if Path(candles).exists():
        print(f"\n\nLoading candles from: {candles}")
        df_candles = pl.read_parquet(candles, n_rows=5)
        print(f"  Shape: {df_candles.shape}")
        print(f"  Columns: {df_candles.columns}")
        print(f"\nFirst few rows:")
        print(df_candles.head())
        
except Exception as e:
    print(f"\nError loading data: {e}")
    import traceback
    traceback.print_exc()
