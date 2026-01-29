#!/usr/bin/env python3
"""Quick signal enrichment using only pair universe (no blockchain queries)"""

import pandas as pd
import polars as pl
from pathlib import Path
import sys

def load_pair_universe():
    """Load existing pair universe with symbols"""
    pair_universe_path = Path("Files/pair-universe")
    
    if pair_universe_path.exists():
        df = pl.read_parquet(pair_universe_path)
        print(f"✓ Loaded pair universe with {len(df)} pairs")
        return df
    return None

def quick_enrich(input_csv, output_csv):
    """Enrich signals using only pair universe (fast, no RPC calls)"""
    
    print(f"Loading signals from {input_csv}...")
    signals_df = pl.read_csv(input_csv)
    print(f"Found {len(signals_df)} signals")
    
    # Load pair universe
    pair_universe = load_pair_universe()
    
    if pair_universe is None:
        print("❌ No pair universe found!")
        return signals_df
    
    # Prepare pair universe columns - check what columns actually exist
    available_cols = pair_universe.columns
    print(f"Available columns: {', '.join(available_cols[:10])}...")  # Show first 10
    
    # Build mapping based on available columns
    rename_map = {}
    select_cols = ['address']
    
    if 'token0_symbol' in available_cols:
        select_cols.append('token0_symbol')
    elif 'symbol0' in available_cols:
        rename_map['symbol0'] = 'token0_symbol'
        select_cols.append('symbol0')
        
    if 'token1_symbol' in available_cols:
        select_cols.append('token1_symbol')
    elif 'symbol1' in available_cols:
        rename_map['symbol1'] = 'token1_symbol'
        select_cols.append('symbol1')
    
    # Create simplified universe for joining
    universe_simple = pair_universe.select([col for col in select_cols if col in available_cols])
    if rename_map:
        universe_simple = universe_simple.rename(rename_map)
    
    # Ensure we have minimum columns
    if 'token0_symbol' not in universe_simple.columns:
        universe_simple = universe_simple.with_columns(pl.lit('UNKNOWN').alias('token0_symbol'))
    if 'token1_symbol' not in universe_simple.columns:
        universe_simple = universe_simple.with_columns(pl.lit('UNKNOWN').alias('token1_symbol'))
    
    print(f"Universe columns for join: {universe_simple.columns}")
    
    # Join signals with universe
    enriched = signals_df.join(
        universe_simple,
        left_on='pair_address',
        right_on='address',
        how='left'
    )
    
    # Create pair_name column
    enriched = enriched.with_columns([
        (pl.col('token0_symbol').fill_null('UNKNOWN') + '/' + pl.col('token1_symbol').fill_null('UNKNOWN')).alias('pair_name')
    ])
    
    # Fill nulls
    enriched = enriched.with_columns([
        pl.col('token0_symbol').fill_null('UNKNOWN'),
        pl.col('token1_symbol').fill_null('UNKNOWN')
    ])
    
    matched = enriched.filter(pl.col('token0_symbol') != 'UNKNOWN').height
    print(f"✅ Matched {matched}/{len(enriched)} pairs ({matched/len(enriched)*100:.1f}%)")
    
    # Sort by confidence
    enriched = enriched.sort('pred_proba', descending=True)
    
    # Save
    enriched.write_csv(output_csv)
    print(f"\n💾 Saved to {output_csv}")
    
    # Convert to pandas for display
    df = enriched.to_pandas()
    
    # Print top 10
    print("\n" + "="*100)
    print("TOP 10 SIGNALS")
    print("="*100)
    
    for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"\n#{i} - {row['pair_name']}")
        print(f"   Confidence: {row['pred_proba']:.2%}")
        print(f"   Price: ${row['close']:.6f}")
        print(f"   Volume 24h: ${row['volume_token1']:.2f}")
        print(f"   Returns: 15m={row['return_1c']:.2f}% | 1h={row['return_4c']:.2f}% | 4h={row['return_16c']:.2f}%")
        print(f"   Volatility: 4h={row['volatility_4h']:.2f} | 24h={row['volatility_24h']:.2f}")
        print(f"   Pair: {row['pair_address']}")
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print(f"Total Signals: {len(df)}")
    print(f"Unique Pairs: {df['pair_name'].nunique()}")
    print(f"Avg Confidence: {df['pred_proba'].mean():.2%}")
    print(f"High Confidence (>80%): {len(df[df['pred_proba'] > 0.8])}")
    print(f"Very High (>85%): {len(df[df['pred_proba'] > 0.85])}")
    print(f"Matched from Universe: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    print("="*100)
    
    return df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_readable.csv')
    else:
        input_file = 'signals_live.csv'
        output_file = 'signals_readable.csv'
    
    quick_enrich(input_file, output_file)
