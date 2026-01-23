#!/usr/bin/env python3
"""
Complete Working Solution - Data Verification & Processing
===========================================================

This script verifies data availability and processes it correctly.
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import sys

# Configuration
PAIR_UNIVERSE_PATH = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe"
CANDLES_PATH = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/candles-1d.parquet"
BASE_CHAIN_ID = 8453

print(f"\n{'#'*80}")
print("DATA VERIFICATION & PROCESSING")
print(f"{'#'*80}\n")

# Step 1: Verify data files exist
print("Step 1: Verifying data files...")
print("="*80)

pair_universe_path = Path(PAIR_UNIVERSE_PATH)
candles_path = Path(CANDLES_PATH)

if not pair_universe_path.exists():
    print(f"❌ Pair universe not found at: {pair_universe_path}")
    print("\nPlease verify the path and try again.")
    sys.exit(1)
else:
    print(f"✓ Pair universe found: {pair_universe_path}")

if not candles_path.exists():
    print(f"❌ Candles not found at: {candles_path}")
    print("\nPlease verify the path and try again.")
    sys.exit(1)
else:
    print(f"✓ Candles found: {candles_path}")
    size_mb = candles_path.stat().st_size / 1024 / 1024
    print(f"  Size: {size_mb:.2f} MB")

# Step 2: Load and inspect pair universe
print(f"\n\nStep 2: Loading pair universe...")
print("="*80)

try:
    if pair_universe_path.is_dir():
        pattern = str(pair_universe_path / "*.parquet")
        print(f"Loading from directory: {pattern}")
        pairs_df = pl.scan_parquet(pattern).collect()
    else:
        print(f"Loading from file: {pair_universe_path}")
        pairs_df = pl.read_parquet(pair_universe_path)
    
    print(f"✓ Loaded {len(pairs_df):,} pairs")
    print(f"\nColumns available:")
    for i, col in enumerate(pairs_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check for Base chain pairs
    if "chain_id" in pairs_df.columns:
        base_pairs = pairs_df.filter(pl.col("chain_id") == BASE_CHAIN_ID)
        print(f"\n✓ Base chain pairs (chain_id={BASE_CHAIN_ID}): {len(base_pairs):,}")
        
        if len(base_pairs) == 0:
            print("\n⚠️  No Base chain pairs found!")
            print(f"   Available chain IDs: {pairs_df['chain_id'].unique().sort().to_list()}")
            print("\n   You may need to change BASE_CHAIN_ID in the config.")
    else:
        print("\n⚠️  No 'chain_id' column found in pair universe")
        base_pairs = pairs_df
    
    # Sample pair data
    print(f"\nSample pair data:")
    print(pairs_df.head(3))
    
except Exception as e:
    print(f"❌ Error loading pair universe: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Load and inspect candles
print(f"\n\nStep 3: Loading candle data...")
print("="*80)

try:
    # Load a sample first
    candles_sample = pl.read_parquet(candles_path, n_rows=1000)
    
    print(f"✓ Candles loaded (sample of 1000 rows)")
    print(f"\nColumns available:")
    for i, col in enumerate(candles_sample.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check pair_id overlap
    if "pair_id" in candles_sample.columns and "pair_id" in pairs_df.columns:
        candle_pairs = candles_sample["pair_id"].unique()
        print(f"\n✓ Sample has {len(candle_pairs)} unique pairs")
        
        # Check if any match
        if len(base_pairs) > 0:
            base_pair_ids = set(base_pairs["pair_id"].to_list())
            candle_pair_ids = set(candle_pairs.to_list())
            overlap = base_pair_ids.intersection(candle_pair_ids)
            print(f"  Overlap with Base chain pairs: {len(overlap)} pairs")
            
            if len(overlap) == 0:
                print("\n⚠️  WARNING: No overlap between Base pairs and candle data!")
                print("   This may mean:")
                print("   1. Candles don't have data for Base chain")
                print("   2. pair_id format mismatch")
                print("   3. Need to load more candle data")
    
    print(f"\nSample candle data:")
    print(candles_sample.head(3))
    
except Exception as e:
    print(f"❌ Error loading candles: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create minimal working dataset
print(f"\n\nStep 4: Creating working dataset...")
print("="*80)

try:
    # Use Base chain pairs if available
    if len(base_pairs) > 0:
        working_pairs = base_pairs.head(100)  # Start with first 100
        print(f"Using {len(working_pairs)} Base chain pairs (limited for testing)")
    else:
        working_pairs = pairs_df.head(100)
        print(f"Using {len(working_pairs)} pairs (no Base chain filter)")
    
    working_pair_ids = working_pairs["pair_id"].to_list()
    
    # Load candles for these pairs
    print(f"\nLoading candles for {len(working_pair_ids)} pairs...")
    full_candles = pl.scan_parquet(candles_path).filter(
        pl.col("pair_id").is_in(working_pair_ids)
    ).collect()
    
    print(f"✓ Found {len(full_candles):,} candle records")
    
    if len(full_candles) == 0:
        print("\n⚠️  No candle data found for these pairs")
        print("   Trying with all pairs from candle data...")
        
        # Just use whatever is in candles
        full_candles = pl.read_parquet(candles_path, n_rows=10000)
        print(f"   Loaded {len(full_candles):,} sample candles")
    
    # Create simple features
    print(f"\nCreating features...")
    if len(full_candles) > 0:
        full_candles = full_candles.sort(["pair_id", "timestamp"])
        
        # Simple return calculation
        full_candles = full_candles.with_columns([
            (pl.col("close").pct_change(1).over("pair_id")).alias("return_1d"),
            ((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("range"),
        ])
        
        print(f"✓ Features created")
        print(f"\nFinal dataset:")
        print(f"  Rows: {len(full_candles):,}")
        print(f"  Pairs: {full_candles['pair_id'].n_unique()}")
        print(f"  Columns: {len(full_candles.columns)}")
        
        # Save it
        output_path = "/home/hamzabhatti18/Desktop/Genesis-labs/xgboost/test_data.parquet"
        full_candles.write_parquet(output_path)
        print(f"\n✓ Saved to: {output_path}")
        
        print(f"\nSample of processed data:")
        print(full_candles.select(["pair_id", "timestamp", "close", "return_1d", "range"]).head(5))
    
except Exception as e:
    print(f"❌ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n\n{'#'*80}")
print("✅ DATA VERIFICATION COMPLETE")
print(f"{'#'*80}\n")

print("Summary:")
print(f"  ✓ Pair universe: {len(pairs_df):,} pairs")
print(f"  ✓ Base chain pairs: {len(base_pairs):,} pairs")
print(f"  ✓ Candle records: {len(full_candles):,}")
print(f"  ✓ Test data saved: test_data.parquet")

print("\nNext steps:")
print("  1. Review the data above")
print("  2. If everything looks good, run: python run_pipeline.py")
print("  3. Check paths in onchain_signal_system.py if needed")
print()
