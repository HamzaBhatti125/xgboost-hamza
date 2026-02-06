#!/usr/bin/env python3
"""
Quick verification script to check if liquidity and VWAP updates are working
"""
import pickle
import polars as pl
from pathlib import Path

# Load the cache
cache_path = Path("live_candles_cache.pkl")

if not cache_path.exists():
    print("❌ Cache file not found")
    exit(1)

print("📂 Loading cache...")
with open(cache_path, 'rb') as f:
    candles_history = pickle.load(f)

print(f"✅ Loaded {len(candles_history)} pairs\n")

# Check a sample pair
if candles_history:
    sample_pair = list(candles_history.keys())[0]
    df = candles_history[sample_pair]
    
    print(f"📊 Sample pair: {sample_pair}")
    print(f"   Candles: {len(df)}")
    print(f"   Columns: {df.columns}\n")
    
    # Check if new columns exist
    has_liquidity = "avg_liquidity" in df.columns
    has_vwap = "vwap" in df.columns
    
    print("🔍 New Fields Check:")
    print(f"   ✅ avg_liquidity: {'YES' if has_liquidity else '❌ NO'}")
    print(f"   ✅ vwap: {'YES' if has_vwap else '❌ NO'}\n")
    
    if has_liquidity and has_vwap:
        # Show some sample data
        print("📈 Sample Data (last 5 candles):")
        sample = df.tail(5).select([
            "timestamp", "close", "volume_token1", "avg_liquidity", "vwap"
        ])
        print(sample)
        
        # Check if values are non-zero (old cache had defaults of 0.0)
        non_zero_liquidity = (df["avg_liquidity"] > 0).sum()
        non_zero_vwap = (df["vwap"] > 0).sum()
        
        print(f"\n💡 Data Quality:")
        print(f"   Liquidity > 0: {non_zero_liquidity}/{len(df)} candles")
        print(f"   VWAP > 0: {non_zero_vwap}/{len(df)} candles")
        
        if non_zero_liquidity > 0 and non_zero_vwap > 0:
            print("\n✅ SUCCESS: New fields are being populated with real data!")
        else:
            print("\n⚠️  WARNING: New fields exist but contain only zeros (waiting for fresh data)")
    else:
        print("❌ FAILED: New fields not found in cache")
        
else:
    print("❌ No pairs in cache")
