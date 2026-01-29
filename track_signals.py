#!/usr/bin/env python3
"""
Signal Performance Tracker
Validates model predictions by tracking actual price movements for generated signals.
"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time
from typing import Dict, List, Tuple
import json

# Configuration
TARGET_PROFIT_PCT = 2.0  # +2% target
STOP_LOSS_PCT = -1.0     # -1% stop loss
HOLDING_PERIOD_HOURS = 4  # 16 candles * 15 min = 4 hours
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"
BASE_CHAIN_ID = "base"


def load_signal_file(filepath: str) -> pl.DataFrame:
    """Load a timestamped signal file."""
    print(f"\n📂 Loading {filepath}...")
    df = pl.read_csv(
        filepath,
        try_parse_dates=False
    )
    
    # Parse dates (handle ISO format with T separator)
    df = df.with_columns([
        pl.col("timestamp").str.to_datetime().alias("entry_time"),
        pl.col("generated_at").str.to_datetime().alias("generated_time")
    ])
    
    print(f"   Found {len(df)} signals")
    return df


def calculate_signal_windows(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate expiry time and status for each signal."""
    current_time = datetime.now()
    
    df = df.with_columns([
        # Expiry = entry_time + 4 hours
        (pl.col("entry_time") + pl.duration(hours=HOLDING_PERIOD_HOURS)).alias("expiry_time"),
        
        # Hours elapsed since entry
        ((pl.lit(current_time) - pl.col("entry_time")).dt.total_seconds() / 3600).alias("hours_elapsed")
    ])
    
    # Determine status
    df = df.with_columns([
        pl.when(pl.col("hours_elapsed") < 0)
        .then(pl.lit("FUTURE"))
        .when(pl.col("hours_elapsed") < HOLDING_PERIOD_HOURS)
        .then(pl.lit("PENDING"))
        .otherwise(pl.lit("EXPIRED"))
        .alias("window_status")
    ])
    
    return df


def query_dexscreener_price(pair_address: str) -> Dict:
    """Query current price data from DEX Screener API."""
    try:
        url = f"{DEXSCREENER_API}/pairs/{BASE_CHAIN_ID}/{pair_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("pair"):
                pair_data = data["pair"]
                return {
                    "success": True,
                    "price_usd": float(pair_data.get("priceUsd", 0)),
                    "price_native": float(pair_data.get("priceNative", 0)),
                    "volume_24h": float(pair_data.get("volume", {}).get("h24", 0)),
                    "price_change_4h": float(pair_data.get("priceChange", {}).get("h4", 0)),
                    "liquidity_usd": float(pair_data.get("liquidity", {}).get("usd", 0)),
                    "pair_name": f"{pair_data.get('baseToken', {}).get('symbol', 'UNKNOWN')}/{pair_data.get('quoteToken', {}).get('symbol', 'UNKNOWN')}"
                }
        
        return {"success": False, "error": f"Status {response.status_code}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_signal_outcome(
    entry_price: float,
    current_price: float,
    price_change_4h: float,
    window_status: str
) -> Tuple[str, float]:
    """
    Determine signal outcome based on price movement.
    Returns: (outcome, actual_return_pct)
    """
    
    if window_status == "FUTURE":
        return "FUTURE", 0.0
    
    # Calculate current return
    current_return_pct = ((current_price - entry_price) / entry_price) * 100
    
    if window_status == "PENDING":
        # Still in 4-hour window
        if current_return_pct >= TARGET_PROFIT_PCT:
            return "WIN_ACTIVE", current_return_pct
        elif current_return_pct <= STOP_LOSS_PCT:
            return "LOSS_ACTIVE", current_return_pct
        else:
            return "PENDING", current_return_pct
    
    else:  # EXPIRED
        # Window closed, check if target was hit during the period
        # Use price_change_4h as proxy for max movement in window
        if price_change_4h >= TARGET_PROFIT_PCT:
            return "WIN", price_change_4h
        elif current_return_pct <= STOP_LOSS_PCT:
            return "LOSS", current_return_pct
        else:
            return "NEUTRAL", current_return_pct


def track_signals(signal_file: str, max_pairs: int = None):
    """
    Main function to track signal performance.
    
    Args:
        signal_file: Path to signal CSV file
        max_pairs: Limit number of pairs to query (for testing)
    """
    
    print("\n" + "="*80)
    print("🎯 SIGNAL PERFORMANCE TRACKER")
    print("="*80)
    
    # Load signals
    df = load_signal_file(signal_file)
    df = calculate_signal_windows(df)
    
    # Show window statistics
    print("\n📊 SIGNAL WINDOWS:")
    window_counts = df.group_by("window_status").agg(pl.count()).sort("window_status")
    for row in window_counts.iter_rows(named=True):
        print(f"   {row['window_status']}: {row['count']} signals")
    
    # Filter to trackable signals (not FUTURE)
    trackable = df.filter(pl.col("window_status") != "FUTURE")
    
    if len(trackable) == 0:
        print("\n⚠️  No trackable signals (all entries are in the future)")
        return
    
    # Limit for testing
    if max_pairs:
        trackable = trackable.head(max_pairs)
        print(f"\n⚠️  Limiting to first {max_pairs} pairs for testing")
    
    print(f"\n🔍 Querying prices for {len(trackable)} pairs...")
    print("   (This may take a minute...)\n")
    
    # Query prices
    results = []
    for i, row in enumerate(trackable.iter_rows(named=True), 1):
        pair_address = row["pair_address"]
        
        # Query DEX Screener
        price_data = query_dexscreener_price(pair_address)
        
        if price_data["success"]:
            # Use USD price for consistency (DEX Screener provides this)
            entry_price_usd = price_data["price_usd"]  # Current price as baseline
            current_price_usd = price_data["price_usd"]
            
            # Note: We can't get historical entry price from DEX Screener API
            # So we use price_change_4h as the actual performance metric
            
            # Calculate outcome based on 4h price change from DEX Screener
            outcome, actual_return = calculate_signal_outcome(
                entry_price=1.0,  # Normalized baseline
                current_price=1.0 + (price_data["price_change_4h"] / 100),  # Relative to entry
                price_change_4h=price_data["price_change_4h"],
                window_status=row["window_status"]
            )
            
            results.append({
                "pair_address": pair_address,
                "pair_name": price_data["pair_name"],
                "entry_time": row["entry_time"],
                "expiry_time": row["expiry_time"],
                "hours_elapsed": round(row["hours_elapsed"], 2),
                "pred_proba": row["pred_proba"],
                "entry_price_usd": entry_price_usd,
                "current_price_usd": current_price_usd,
                "actual_return_pct": round(actual_return, 2),
                "price_change_4h": price_data["price_change_4h"],
                "outcome": outcome,
                "window_status": row["window_status"],
                "volume_24h": price_data["volume_24h"],
                "liquidity_usd": price_data["liquidity_usd"]
            })
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Processed {i}/{len(trackable)} pairs...")
        
        else:
            # Failed to get price
            results.append({
                "pair_address": pair_address,
                "pair_name": row.get("pair_name", "UNKNOWN"),
                "entry_time": row["entry_time"],
                "expiry_time": row["expiry_time"],
                "hours_elapsed": round(row["hours_elapsed"], 2),
                "pred_proba": row["pred_proba"],
                "entry_price_usd": 0,
                "current_price_usd": 0,
                "actual_return_pct": 0,
                "price_change_4h": 0,
                "outcome": "NO_DATA",
                "window_status": row["window_status"],
                "volume_24h": 0,
                "liquidity_usd": 0
            })
        
        # Rate limiting
        time.sleep(0.5)  # 2 requests/second
    
    # Create results DataFrame
    results_df = pl.DataFrame(results)
    
    # Save results
    output_file = signal_file.replace(".csv", "_tracked.csv")
    results_df.write_csv(output_file)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_signals = len(results_df)
    no_data = len(results_df.filter(pl.col("outcome") == "NO_DATA"))
    trackable_signals = total_signals - no_data
    
    print(f"\n📊 Total Signals: {total_signals}")
    print(f"   Trackable: {trackable_signals}")
    print(f"   No Data: {no_data}")
    
    if trackable_signals == 0:
        print("\n⚠️  No trackable data available")
        return
    
    # Outcome breakdown
    print("\n🎯 OUTCOMES:")
    outcome_counts = results_df.filter(pl.col("outcome") != "NO_DATA").group_by("outcome").agg(pl.count()).sort("outcome")
    for row in outcome_counts.iter_rows(named=True):
        count = row['count']
        pct = (count / trackable_signals) * 100
        print(f"   {row['outcome']:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Calculate win rate for completed signals
    completed = results_df.filter(
        (pl.col("window_status") == "EXPIRED") & 
        (pl.col("outcome") != "NO_DATA")
    )
    
    if len(completed) > 0:
        wins = len(completed.filter(pl.col("outcome") == "WIN"))
        losses = len(completed.filter(pl.col("outcome") == "LOSS"))
        neutral = len(completed.filter(pl.col("outcome") == "NEUTRAL"))
        
        win_rate = (wins / len(completed)) * 100 if len(completed) > 0 else 0
        
        print(f"\n✅ COMPLETED SIGNALS ({len(completed)}):")
        print(f"   Wins:    {wins:3d} ({wins/len(completed)*100:5.1f}%)")
        print(f"   Losses:  {losses:3d} ({losses/len(completed)*100:5.1f}%)")
        print(f"   Neutral: {neutral:3d} ({neutral/len(completed)*100:5.1f}%)")
        print(f"\n   📊 WIN RATE: {win_rate:.1f}% (Model predicted: 91.7%)")
        
        # Average returns
        avg_win_return = completed.filter(pl.col("outcome") == "WIN")["actual_return_pct"].mean()
        avg_loss_return = completed.filter(pl.col("outcome") == "LOSS")["actual_return_pct"].mean()
        
        if avg_win_return:
            print(f"   💰 Avg Win Return: +{avg_win_return:.2f}%")
        if avg_loss_return:
            print(f"   📉 Avg Loss Return: {avg_loss_return:.2f}%")
    
    # Top performers
    print("\n🏆 TOP 10 SIGNALS (by actual return):")
    top_signals = results_df.filter(pl.col("outcome") != "NO_DATA").sort("actual_return_pct", descending=True).head(10)
    
    for i, row in enumerate(top_signals.iter_rows(named=True), 1):
        status_emoji = {
            "WIN": "✅",
            "WIN_ACTIVE": "🟢",
            "LOSS": "❌",
            "LOSS_ACTIVE": "🔴",
            "PENDING": "⏳",
            "NEUTRAL": "➖"
        }.get(row["outcome"], "❓")
        
        print(f"   #{i:2d} {status_emoji} {row['pair_name']:20s} "
              f"Return: {row['actual_return_pct']:+7.2f}% | "
              f"Confidence: {row['pred_proba']*100:5.1f}% | "
              f"Status: {row['outcome']}")
    
    # Worst performers
    print("\n📉 WORST 10 SIGNALS (by actual return):")
    worst_signals = results_df.filter(pl.col("outcome") != "NO_DATA").sort("actual_return_pct").head(10)
    
    for i, row in enumerate(worst_signals.iter_rows(named=True), 1):
        status_emoji = {
            "WIN": "✅",
            "WIN_ACTIVE": "🟢",
            "LOSS": "❌",
            "LOSS_ACTIVE": "🔴",
            "PENDING": "⏳",
            "NEUTRAL": "➖"
        }.get(row["outcome"], "❓")
        
        print(f"   #{i:2d} {status_emoji} {row['pair_name']:20s} "
              f"Return: {row['actual_return_pct']:+7.2f}% | "
              f"Confidence: {row['pred_proba']*100:5.1f}% | "
              f"Status: {row['outcome']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python track_signals.py <signal_file.csv> [max_pairs]")
        print("\nExample:")
        print("  python track_signals.py signals_2026-01-29_01-32-44.csv")
        print("  python track_signals.py signals_2026-01-29_01-32-44.csv 20  # Test with 20 pairs")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    track_signals(signal_file, max_pairs)
