#!/usr/bin/env python3
"""
Signal Performance Tracker (Live Cache Version)
Uses live_candles_cache.pkl for historical entry prices and queries current prices.
"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time
import pickle
from typing import Dict, List, Tuple

# Configuration
TARGET_PROFIT_PCT = 2.0
STOP_LOSS_PCT = -1.0
HOLDING_PERIOD_HOURS = 4
CACHE_PATH = "live_candles_cache.pkl"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"
BASE_CHAIN_ID = "base"


def load_candle_cache() -> pl.DataFrame:
    """Load the live candles cache."""
    print(f"\n📦 Loading {CACHE_PATH}...")
    
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    
    if isinstance(cache, dict):
        # New format: {pair_address: DataFrame}
        all_candles = []
        for pair_address, df in cache.items():
            if not df.is_empty():
                df = df.with_columns(pl.lit(pair_address).alias("pair_address"))
                all_candles.append(df)
        
        if all_candles:
            candles_df = pl.concat(all_candles)
            print(f"   Loaded {len(candles_df):,} candles for {len(cache)} pairs")
            return candles_df
    
    print("   ⚠️  Cache format not recognized")
    return pl.DataFrame()


def load_signal_file(filepath: str) -> pl.DataFrame:
    """Load a timestamped signal file."""
    print(f"\n📂 Loading {filepath}...")
    df = pl.read_csv(filepath, try_parse_dates=False)
    
    # Parse dates
    df = df.with_columns([
        pl.col("timestamp").str.to_datetime().alias("entry_time"),
        pl.col("generated_at").str.to_datetime().alias("generated_time")
    ])
    
    print(f"   Found {len(df)} signals")
    return df


def query_dexscreener_price(pair_address: str) -> Dict:
    """Query current price from DEX Screener."""
    try:
        url = f"{DEXSCREENER_API}/pairs/{BASE_CHAIN_ID}/{pair_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("pair"):
                pair_data = data["pair"]
                return {
                    "success": True,
                    "price_native": float(pair_data.get("priceNative", 0)),
                    "price_usd": float(pair_data.get("priceUsd", 0)),
                    "pair_name": f"{pair_data.get('baseToken', {}).get('symbol', 'UNKNOWN')}/{pair_data.get('quoteToken', {}).get('symbol', 'UNKNOWN')}"
                }
        
        return {"success": False}
    
    except Exception as e:
        return {"success": False}


def track_signals(signal_file: str, max_pairs: int = None):
    """Track signal performance using live cache for entry prices."""
    
    print("\n" + "="*80)
    print("🎯 SIGNAL PERFORMANCE TRACKER (Live Cache)")
    print("="*80)
    
    # Load candle cache
    candles_df = load_candle_cache()
    if candles_df.is_empty():
        print("\n❌ No candle cache found! Run live_signal_generator.py first.")
        return
    
    # Parse candle timestamps
    candles_df = candles_df.with_columns([
        pl.col("timestamp").cast(pl.Datetime).alias("candle_time")
    ])
    
    # Load signals
    signals_df = load_signal_file(signal_file)
    
    # Calculate expiry times
    current_time = datetime.now()
    signals_df = signals_df.with_columns([
        (pl.col("entry_time") + pl.duration(hours=HOLDING_PERIOD_HOURS)).alias("expiry_time"),
        ((pl.lit(current_time) - pl.col("entry_time")).dt.total_seconds() / 3600).alias("hours_elapsed")
    ])
    
    # Determine window status
    signals_df = signals_df.with_columns([
        pl.when(pl.col("hours_elapsed") < 0)
        .then(pl.lit("FUTURE"))
        .when(pl.col("hours_elapsed") < HOLDING_PERIOD_HOURS)
        .then(pl.lit("PENDING"))
        .otherwise(pl.lit("EXPIRED"))
        .alias("window_status")
    ])
    
    # Show statistics
    print("\n📊 SIGNAL WINDOWS:")
    for row in signals_df.group_by("window_status").len().sort("window_status").iter_rows(named=True):
        print(f"   {row['window_status']}: {row['len']} signals")
    
    # Filter trackable signals
    trackable = signals_df.filter(pl.col("window_status") != "FUTURE")
    
    if len(trackable) == 0:
        print("\n⚠️  No trackable signals (all are in the future)")
        return
    
    if max_pairs:
        trackable = trackable.head(max_pairs)
        print(f"\n⚠️  Limiting to first {max_pairs} pairs for testing")
    
    print(f"\n🔍 Tracking {len(trackable)} signals...")
    print("   (Matching entry prices from cache and querying current prices...)\n")
    
    # Track each signal
    results = []
    success_count = 0
    
    for i, signal in enumerate(trackable.iter_rows(named=True), 1):
        pair_address = signal["pair_address"]
        entry_time = signal["entry_time"]
        
        # Find entry candle from cache
        entry_candle = candles_df.filter(
            (pl.col("pair_address") == pair_address) &
            (pl.col("candle_time") == entry_time)
        )
        
        if entry_candle.is_empty():
            # No entry price in cache
            results.append({
                "pair_address": pair_address,
                "pair_name": "UNKNOWN",
                "entry_time": entry_time,
                "hours_elapsed": round(signal["hours_elapsed"], 2),
                "pred_proba": signal["pred_proba"],
                "entry_price": 0,
                "current_price": 0,
                "actual_return_pct": 0,
                "outcome": "NO_ENTRY_DATA",
                "window_status": signal["window_status"]
            })
            continue
        
        # Get entry price
        entry_price = entry_candle["close"][0]
        
        # Query current price
        price_data = query_dexscreener_price(pair_address)
        
        if not price_data["success"]:
            results.append({
                "pair_address": pair_address,
                "pair_name": "UNKNOWN",
                "entry_time": entry_time,
                "hours_elapsed": round(signal["hours_elapsed"], 2),
                "pred_proba": signal["pred_proba"],
                "entry_price": entry_price,
                "current_price": 0,
                "actual_return_pct": 0,
                "outcome": "NO_CURRENT_PRICE",
                "window_status": signal["window_status"]
            })
            continue
        
        # Calculate return
        current_price = price_data["price_native"]
        actual_return_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Determine outcome
        if signal["window_status"] == "PENDING":
            if actual_return_pct >= TARGET_PROFIT_PCT:
                outcome = "WIN_ACTIVE"
            elif actual_return_pct <= STOP_LOSS_PCT:
                outcome = "LOSS_ACTIVE"
            else:
                outcome = "PENDING"
        else:  # EXPIRED
            # For expired signals, we can only see current price
            # We'd need historical data to know if +2% was hit during the window
            if actual_return_pct >= TARGET_PROFIT_PCT:
                outcome = "WIN"  # Currently above target
            elif actual_return_pct <= STOP_LOSS_PCT:
                outcome = "LOSS"
            else:
                outcome = "NEUTRAL"  # In between
        
        results.append({
            "pair_address": pair_address,
            "pair_name": price_data["pair_name"],
            "entry_time": entry_time,
            "hours_elapsed": round(signal["hours_elapsed"], 2),
            "pred_proba": signal["pred_proba"],
            "entry_price": entry_price,
            "current_price": current_price,
            "actual_return_pct": round(actual_return_pct, 2),
            "outcome": outcome,
            "window_status": signal["window_status"]
        })
        
        success_count += 1
        
        # Progress
        if i % 10 == 0:
            print(f"   Processed {i}/{len(trackable)} pairs...")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Create results DataFrame
    results_df = pl.DataFrame(results)
    
    # Save
    output_file = signal_file.replace(".csv", "_tracked_live.csv")
    results_df.write_csv(output_file)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n📊 Total Signals: {len(results_df)}")
    print(f"   Successfully Tracked: {success_count}")
    print(f"   Failed: {len(results_df) - success_count}")
    
    # Outcome breakdown
    print("\n🎯 OUTCOMES:")
    for row in results_df.filter(
        ~pl.col("outcome").str.contains("NO_")
    ).group_by("outcome").len().sort("outcome").iter_rows(named=True):
        count = row['len']
        pct = (count / success_count * 100) if success_count > 0 else 0
        print(f"   {row['outcome']:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Calculate win rate for completed/active signals
    completed = results_df.filter(
        pl.col("window_status").is_in(["EXPIRED", "PENDING"]) &
        ~pl.col("outcome").str.contains("NO_")
    )
    
    if len(completed) > 0:
        wins = len(completed.filter(pl.col("outcome").is_in(["WIN", "WIN_ACTIVE"])))
        losses = len(completed.filter(pl.col("outcome").is_in(["LOSS", "LOSS_ACTIVE"])))
        
        win_rate = (wins / len(completed) * 100) if len(completed) > 0 else 0
        
        print(f"\n✅ COMPLETED/ACTIVE SIGNALS ({len(completed)}):")
        print(f"   Wins:   {wins:3d} ({wins/len(completed)*100:5.1f}%)")
        print(f"   Losses: {losses:3d} ({losses/len(completed)*100:5.1f}%)")
        print(f"\n   📊 WIN RATE: {win_rate:.1f}% (Model predicted: 91.7%)")
        
        # Average returns
        win_returns = completed.filter(pl.col("outcome").is_in(["WIN", "WIN_ACTIVE"]))
        if len(win_returns) > 0:
            avg_win = win_returns["actual_return_pct"].mean()
            print(f"   💰 Avg Win Return: +{avg_win:.2f}%")
        
        loss_returns = completed.filter(pl.col("outcome").is_in(["LOSS", "LOSS_ACTIVE"]))
        if len(loss_returns) > 0:
            avg_loss = loss_returns["actual_return_pct"].mean()
            print(f"   📉 Avg Loss Return: {avg_loss:.2f}%")
    
    # Top performers
    print("\n🏆 TOP 10 SIGNALS (by actual return):")
    top_signals = results_df.filter(
        ~pl.col("outcome").str.contains("NO_")
    ).sort("actual_return_pct", descending=True).head(10)
    
    for i, row in enumerate(top_signals.iter_rows(named=True), 1):
        status_emoji = {
            "WIN": "✅", "WIN_ACTIVE": "🟢",
            "LOSS": "❌", "LOSS_ACTIVE": "🔴",
            "PENDING": "⏳", "NEUTRAL": "➖"
        }.get(row["outcome"], "❓")
        
        print(f"   #{i:2d} {status_emoji} {row['pair_name']:20s} "
              f"Return: {row['actual_return_pct']:+7.2f}% | "
              f"Confidence: {row['pred_proba']*100:5.1f}% | "
              f"Elapsed: {row['hours_elapsed']:.1f}h")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python track_signals_live.py <signal_file.csv> [max_pairs]")
        print("\nExample:")
        print("  python track_signals_live.py signals_2026-01-29_01-32-44.csv")
        print("  python track_signals_live.py signals_2026-01-29_01-32-44.csv 50")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    track_signals(signal_file, max_pairs)
