#!/usr/bin/env python3
"""
Signal Performance Tracker - Cache Only Version
Tracks signals using ONLY live_candles_cache.pkl (no external APIs).
Compares entry candle price to latest candle price for each pair.
"""

import polars as pl
from datetime import datetime, timedelta
import pickle

# Configuration
TARGET_PROFIT_PCT = 2.0
STOP_LOSS_PCT = -1.0
HOLDING_PERIOD_HOURS = 4
CACHE_PATH = "live_candles_cache.pkl"


def load_candle_cache() -> pl.DataFrame:
    """Load live candles cache."""
    print(f"\n📦 Loading {CACHE_PATH}...")
    
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    
    if isinstance(cache, dict):
        all_candles = []
        for pair_address, df in cache.items():
            if not df.is_empty():
                df = df.with_columns(pl.lit(pair_address).alias("pair_address"))
                all_candles.append(df)
        
        if all_candles:
            candles_df = pl.concat(all_candles)
            candles_df = candles_df.with_columns([
                pl.col("timestamp").cast(pl.Datetime).alias("candle_time")
            ])
            print(f"   Loaded {len(candles_df):,} candles for {len(cache)} pairs")
            return candles_df
    
    return pl.DataFrame()


def load_signal_file(filepath: str) -> pl.DataFrame:
    """Load signal CSV file."""
    print(f"\n📂 Loading {filepath}...")
    df = pl.read_csv(filepath, try_parse_dates=False)
    
    df = df.with_columns([
        pl.col("timestamp").str.to_datetime().alias("entry_time"),
        pl.col("generated_at").str.to_datetime().alias("generated_time")
    ])
    
    print(f"   Found {len(df)} signals")
    return df


def get_price_at_time(candles_df: pl.DataFrame, pair_address: str, target_time: datetime) -> float:
    """Get the close price for a pair at a specific time."""
    candle = candles_df.filter(
        (pl.col("pair_address") == pair_address) &
        (pl.col("candle_time") == target_time)
    )
    
    if not candle.is_empty():
        return candle["close"][0]
    return None


def get_latest_price(candles_df: pl.DataFrame, pair_address: str) -> tuple:
    """Get the latest price and timestamp for a pair."""
    pair_candles = candles_df.filter(pl.col("pair_address") == pair_address)
    
    if pair_candles.is_empty():
        return None, None
    
    latest = pair_candles.sort("candle_time", descending=True).head(1)
    return latest["close"][0], latest["candle_time"][0]


def track_signals(signal_file: str, max_pairs: int = None):
    """Track signal performance using cache only."""
    
    print("\n" + "="*80)
    print("🎯 SIGNAL PERFORMANCE TRACKER (Cache Only)")
    print("="*80)
    
    # Load cache
    candles_df = load_candle_cache()
    if candles_df.is_empty():
        print("\n❌ No candle cache found!")
        return
    
    # Load signals
    signals_df = load_signal_file(signal_file)
    
    # Calculate windows
    current_time = datetime.now()
    signals_df = signals_df.with_columns([
        (pl.col("entry_time") + pl.duration(hours=HOLDING_PERIOD_HOURS)).alias("expiry_time"),
        ((pl.lit(current_time) - pl.col("entry_time")).dt.total_seconds() / 3600).alias("hours_elapsed")
    ])
    
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
    
    # Filter trackable
    trackable = signals_df.filter(pl.col("window_status") != "FUTURE")
    
    if len(trackable) == 0:
        print("\n⚠️  No trackable signals")
        return
    
    if max_pairs:
        trackable = trackable.head(max_pairs)
        print(f"\n⚠️  Limiting to first {max_pairs} pairs for testing")
    
    print(f"\n🔍 Tracking {len(trackable)} signals using cache data...\n")
    
    # Track each signal
    results = []
    success_count = 0
    
    for i, signal in enumerate(trackable.iter_rows(named=True), 1):
        pair_address = signal["pair_address"]
        entry_time = signal["entry_time"]
        
        # Get entry price
        entry_price = get_price_at_time(candles_df, pair_address, entry_time)
        
        if entry_price is None:
            results.append({
                "pair_address": pair_address,
                "entry_time": entry_time,
                "hours_elapsed": round(signal["hours_elapsed"], 2),
                "pred_proba": signal["pred_proba"],
                "entry_price": 0,
                "current_price": 0,
                "latest_candle_time": None,
                "actual_return_pct": 0,
                "outcome": "NO_ENTRY_DATA",
                "window_status": signal["window_status"]
            })
            continue
        
        # Get latest price
        latest_price, latest_time = get_latest_price(candles_df, pair_address)
        
        if latest_price is None:
            results.append({
                "pair_address": pair_address,
                "entry_time": entry_time,
                "hours_elapsed": round(signal["hours_elapsed"], 2),
                "pred_proba": signal["pred_proba"],
                "entry_price": entry_price,
                "current_price": 0,
                "latest_candle_time": None,
                "actual_return_pct": 0,
                "outcome": "NO_CURRENT_DATA",
                "window_status": signal["window_status"]
            })
            continue
        
        # Calculate return
        actual_return_pct = ((latest_price - entry_price) / entry_price) * 100
        
        # Determine outcome
        if signal["window_status"] == "PENDING":
            if actual_return_pct >= TARGET_PROFIT_PCT:
                outcome = "WIN_ACTIVE"
            elif actual_return_pct <= STOP_LOSS_PCT:
                outcome = "LOSS_ACTIVE"
            else:
                outcome = "PENDING"
        else:  # EXPIRED
            # Check if we have candles within the 4-hour window
            window_candles = candles_df.filter(
                (pl.col("pair_address") == pair_address) &
                (pl.col("candle_time") > entry_time) &
                (pl.col("candle_time") <= signal["expiry_time"])
            )
            
            if not window_candles.is_empty():
                # Check max return during window
                max_price = window_candles["high"].max()
                max_return = ((max_price - entry_price) / entry_price) * 100
                
                min_price = window_candles["low"].min()
                min_return = ((min_price - entry_price) / entry_price) * 100
                
                if max_return >= TARGET_PROFIT_PCT:
                    outcome = "WIN"
                elif min_return <= STOP_LOSS_PCT:
                    outcome = "LOSS"
                else:
                    outcome = "NEUTRAL"
            else:
                # No candles in window, use latest
                if actual_return_pct >= TARGET_PROFIT_PCT:
                    outcome = "WIN"
                elif actual_return_pct <= STOP_LOSS_PCT:
                    outcome = "LOSS"
                else:
                    outcome = "NEUTRAL"
        
        results.append({
            "pair_address": pair_address,
            "entry_time": entry_time,
            "hours_elapsed": round(signal["hours_elapsed"], 2),
            "pred_proba": signal["pred_proba"],
            "entry_price": entry_price,
            "current_price": latest_price,
            "latest_candle_time": latest_time,
            "actual_return_pct": round(actual_return_pct, 2),
            "outcome": outcome,
            "window_status": signal["window_status"]
        })
        
        success_count += 1
        
        if i % 50 == 0:
            print(f"   Processed {i}/{len(trackable)} pairs...")
    
    # Create results DataFrame
    results_df = pl.DataFrame(results)
    
    # Save
    output_file = signal_file.replace(".csv", "_tracked_cache.csv")
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
    
    # Calculate win rate
    completed = results_df.filter(
        pl.col("window_status").is_in(["EXPIRED", "PENDING"]) &
        ~pl.col("outcome").str.contains("NO_")
    )
    
    if len(completed) > 0:
        wins = len(completed.filter(pl.col("outcome").is_in(["WIN", "WIN_ACTIVE"])))
        losses = len(completed.filter(pl.col("outcome").is_in(["LOSS", "LOSS_ACTIVE"])))
        pending = len(completed.filter(pl.col("outcome") == "PENDING"))
        neutral = len(completed.filter(pl.col("outcome") == "NEUTRAL"))
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        print(f"\n✅ SIGNAL STATUS ({len(completed)}):")
        print(f"   Wins:    {wins:3d} ({wins/len(completed)*100:5.1f}%)")
        print(f"   Losses:  {losses:3d} ({losses/len(completed)*100:5.1f}%)")
        print(f"   Pending: {pending:3d} ({pending/len(completed)*100:5.1f}%)")
        print(f"   Neutral: {neutral:3d} ({neutral/len(completed)*100:5.1f}%)")
        
        if wins + losses > 0:
            print(f"\n   📊 WIN RATE: {win_rate:.1f}% (Model predicted: 91.7%)")
            print(f"      Based on {wins + losses} completed signals (wins + losses)")
        
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
    print("\n🏆 TOP 20 SIGNALS (by actual return):")
    top_signals = results_df.filter(
        ~pl.col("outcome").str.contains("NO_")
    ).sort("actual_return_pct", descending=True).head(20)
    
    for i, row in enumerate(top_signals.iter_rows(named=True), 1):
        status_emoji = {
            "WIN": "✅", "WIN_ACTIVE": "🟢",
            "LOSS": "❌", "LOSS_ACTIVE": "🔴",
            "PENDING": "⏳", "NEUTRAL": "➖"
        }.get(row["outcome"], "❓")
        
        print(f"   #{i:2d} {status_emoji} {row['pair_address'][:10]}... "
              f"Return: {row['actual_return_pct']:+7.2f}% | "
              f"Confidence: {row['pred_proba']*100:5.1f}% | "
              f"Elapsed: {row['hours_elapsed']:.1f}h | "
              f"Status: {row['outcome']}")
    
    # Worst performers
    print("\n📉 WORST 10 SIGNALS (by actual return):")
    worst_signals = results_df.filter(
        ~pl.col("outcome").str.contains("NO_")
    ).sort("actual_return_pct").head(10)
    
    for i, row in enumerate(worst_signals.iter_rows(named=True), 1):
        status_emoji = {
            "WIN": "✅", "WIN_ACTIVE": "🟢",
            "LOSS": "❌", "LOSS_ACTIVE": "🔴",
            "PENDING": "⏳", "NEUTRAL": "➖"
        }.get(row["outcome"], "❓")
        
        print(f"   #{i:2d} {status_emoji} {row['pair_address'][:10]}... "
              f"Return: {row['actual_return_pct']:+7.2f}% | "
              f"Confidence: {row['pred_proba']*100:5.1f}% | "
              f"Elapsed: {row['hours_elapsed']:.1f}h")
    
    print("\n" + "="*80)
    
    # Additional analysis
    print("\n💡 INSIGHTS:")
    
    # Confidence analysis
    high_conf = results_df.filter((pl.col("pred_proba") >= 0.85) & ~pl.col("outcome").str.contains("NO_"))
    if len(high_conf) > 0:
        high_conf_wins = len(high_conf.filter(pl.col("outcome").is_in(["WIN", "WIN_ACTIVE"])))
        high_conf_losses = len(high_conf.filter(pl.col("outcome").is_in(["LOSS", "LOSS_ACTIVE"])))
        if high_conf_wins + high_conf_losses > 0:
            high_conf_wr = (high_conf_wins / (high_conf_wins + high_conf_losses) * 100)
            print(f"   High Confidence (≥85%): {high_conf_wr:.1f}% win rate ({high_conf_wins}W / {high_conf_losses}L)")
    
    # Time analysis
    early_signals = results_df.filter((pl.col("hours_elapsed") < 2) & ~pl.col("outcome").str.contains("NO_"))
    if len(early_signals) > 0:
        early_wins = len(early_signals.filter(pl.col("outcome").is_in(["WIN", "WIN_ACTIVE"])))
        early_losses = len(early_signals.filter(pl.col("outcome").is_in(["LOSS", "LOSS_ACTIVE"])))
        if early_wins + early_losses > 0:
            early_wr = (early_wins / (early_wins + early_losses) * 100)
            print(f"   Early Signals (<2h): {early_wr:.1f}% win rate ({early_wins}W / {early_losses}L)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python track_signals_cache.py <signal_file.csv> [max_pairs]")
        print("\nExample:")
        print("  python track_signals_cache.py signals_2026-01-29_01-32-44.csv")
        print("  python track_signals_cache.py signals_2026-01-29_01-32-44.csv 100")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    track_signals(signal_file, max_pairs)
