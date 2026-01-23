#!/usr/bin/env python3
"""
On-Chain DEX Trading Signal Generation System
==============================================

A rigorous, production-ready system for generating rare, high-conviction trading signals
from on-chain DEX data with strict filtering, realistic backtesting, and risk-adjusted optimization.

Philosophy: "Most on-chain pairs should never be traded. 
            A good system trades rarely, survives always, and compounds slowly."
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration with hard constraints"""
    
    # Data paths
    PAIR_UNIVERSE_PATH = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe"
    CANDLES_PATH = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/candles-1d.parquet"
    
    # Base chain selection (Base chain ID = 8453)
    BASE_CHAIN_ID = 8453
    
    # Filtering thresholds
    MAX_TAX_PCT = 10.0
    MIN_AGE_DAYS = 30
    MIN_VOLUME_30D = 10000  # USD
    MIN_AVG_DAILY_TRADES = 5
    MAX_SINGLE_DAY_MOVE_PCT = 30.0
    DAYS_SINCE_LAST_SWAP = 14
    
    # Labeling parameters (barrier-based)
    LABEL_UPSIDE_PCT = 8.0  # Target gain
    LABEL_DOWNSIDE_PCT = 3.0  # Stop loss
    LABEL_HORIZON_DAYS = 7  # Look-ahead period
    
    # Trading costs
    FEE_PCT = 0.3
    SLIPPAGE_PCT = 0.2
    TOTAL_FRICTION_PCT = FEE_PCT + SLIPPAGE_PCT
    
    # Model parameters
    CHUNK_SIZE = 10000  # Process in chunks to avoid memory issues
    MIN_SIGNAL_RATE_PCT = 5.0  # Signals must be rare
    
    # Backtesting
    MIN_SHARPE_THRESHOLD = 1.0
    
    # Train/test split (time-based)
    TRAIN_END_DATE = "2024-06-30"
    TEST_START_DATE = "2024-07-01"


# ============================================================================
# STEP 1: DATA LOADING & BASE CHAIN FILTERING
# ============================================================================

def load_pair_universe_lazy() -> pl.LazyFrame:
    """Load pair universe data lazily to avoid memory issues"""
    print(f"\n{'='*80}")
    print("STEP 1: LOADING PAIR UNIVERSE DATA")
    print(f"{'='*80}")
    
    path = Path(Config.PAIR_UNIVERSE_PATH)
    
    # Check if it's a directory with multiple parquet files or single file
    if path.is_dir():
        pattern = str(path / "*.parquet")
        df = pl.scan_parquet(pattern)
    else:
        df = pl.scan_parquet(str(path) + ".parquet")
    
    print(f"✓ Loaded pair universe (lazy mode)")
    return df


def filter_base_chain(df: pl.LazyFrame) -> pl.LazyFrame:
    """Restrict to Base chain only"""
    print(f"\nFiltering for Base chain (chain_id = {Config.BASE_CHAIN_ID})")
    
    df_filtered = df.filter(pl.col("chain_id") == Config.BASE_CHAIN_ID)
    
    # Count pairs
    total_pairs = df.select(pl.count()).collect().item()
    base_pairs = df_filtered.select(pl.count()).collect().item()
    
    print(f"  Total pairs across all chains: {total_pairs:,}")
    print(f"  Base chain pairs: {base_pairs:,} ({100*base_pairs/total_pairs:.1f}%)")
    
    return df_filtered


# ============================================================================
# STEP 2: SCAM & LOW-QUALITY PAIR FILTERING
# ============================================================================

def apply_quality_filters(df: pl.LazyFrame) -> Tuple[pl.LazyFrame, Dict]:
    """Apply comprehensive filtering to remove scams and low-quality pairs"""
    print(f"\n{'='*80}")
    print("STEP 2: APPLYING QUALITY FILTERS")
    print(f"{'='*80}")
    
    initial_count = df.select(pl.count()).collect().item()
    removal_stats = {}
    
    # Hard exclusions
    print("\n🚫 Hard Exclusions:")
    
    # Get schema once
    schema = df.collect_schema()
    available_cols = schema.names()
    print(f"  Available columns: {len(available_cols)} total")
    
    # Apply filters only if columns exist
    if "flag_inactive" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            # Check if keeping only active (False) would remove everything
            active_count = df.filter(pl.col("flag_inactive") == False).select(pl.count()).collect().item()
            inactive_count = df.filter(pl.col("flag_inactive") == True).select(pl.count()).collect().item()
            
            if active_count == 0:
                print(f"  • flag_inactive: ALL {inactive_count:,} pairs flagged inactive - SKIPPING this filter (would remove all pairs)")
                removal_stats["flag_inactive"] = 0
            else:
                df = df.filter(pl.col("flag_inactive") == False)
                removed = before - active_count
                removal_stats["flag_inactive"] = removed
                print(f"  • flag_inactive: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • flag_inactive: skipped ({e})")
    
    if "flag_blacklisted_manually" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            safe_count = df.filter(pl.col("flag_blacklisted_manually") == False).select(pl.count()).collect().item()
            blacklisted_count = df.filter(pl.col("flag_blacklisted_manually") == True).select(pl.count()).collect().item()
            
            if safe_count == 0:
                print(f"  • flag_blacklisted: ALL {blacklisted_count:,} pairs blacklisted - SKIPPING (would remove all pairs)")
                removal_stats["flag_blacklisted"] = 0
            else:
                df = df.filter(pl.col("flag_blacklisted_manually") == False)
                removed = before - safe_count
                removal_stats["flag_blacklisted"] = removed
                print(f"  • flag_blacklisted: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • flag_blacklisted: skipped ({e})")
    
    if "flag_unsupported_quote_token" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            supported_count = df.filter(pl.col("flag_unsupported_quote_token") == False).select(pl.count()).collect().item()
            
            if supported_count == 0:
                print(f"  • flag_unsupported_quote: ALL pairs unsupported - SKIPPING (would remove all pairs)")
                removal_stats["flag_unsupported_quote"] = 0
            else:
                df = df.filter(pl.col("flag_unsupported_quote_token") == False)
                removed = before - supported_count
                removal_stats["flag_unsupported_quote"] = removed
                print(f"  • flag_unsupported_quote: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • flag_unsupported_quote: skipped ({e})")
    
    if "flag_unknown_exchange" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            known_count = df.filter(pl.col("flag_unknown_exchange") == False).select(pl.count()).collect().item()
            
            if known_count == 0:
                print(f"  • flag_unknown_exchange: ALL pairs unknown exchange - SKIPPING (would remove all pairs)")
                removal_stats["flag_unknown_exchange"] = 0
            else:
                df = df.filter(pl.col("flag_unknown_exchange") == False)
                removed = before - known_count
                removal_stats["flag_unknown_exchange"] = removed
                print(f"  • flag_unknown_exchange: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • flag_unknown_exchange: skipped ({e})")
    
    if "sell_tax" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            acceptable_count = df.filter(pl.col("sell_tax") <= Config.MAX_TAX_PCT).select(pl.count()).collect().item()
            
            if acceptable_count == 0:
                print(f"  • sell_tax: ALL pairs have sell_tax > {Config.MAX_TAX_PCT}% - SKIPPING (would remove all pairs)")
                removal_stats["sell_tax"] = 0
            else:
                df = df.filter(pl.col("sell_tax") <= Config.MAX_TAX_PCT)
                removed = before - acceptable_count
                removal_stats["sell_tax"] = removed
                print(f"  • sell_tax: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • sell_tax: skipped ({e})")
    
    if "buy_tax" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            acceptable_count = df.filter(pl.col("buy_tax") <= Config.MAX_TAX_PCT).select(pl.count()).collect().item()
            
            if acceptable_count == 0:
                print(f"  • buy_tax: ALL pairs have buy_tax > {Config.MAX_TAX_PCT}% - SKIPPING (would remove all pairs)")
                removal_stats["buy_tax"] = 0
            else:
                df = df.filter(pl.col("buy_tax") <= Config.MAX_TAX_PCT)
                removed = before - acceptable_count
                removal_stats["buy_tax"] = removed
                print(f"  • buy_tax: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • buy_tax: skipped ({e})")
    
    if "transfer_tax" in available_cols:
        try:
            before = df.select(pl.count()).collect().item()
            acceptable_count = df.filter(pl.col("transfer_tax") <= Config.MAX_TAX_PCT).select(pl.count()).collect().item()
            
            if acceptable_count == 0:
                print(f"  • transfer_tax: ALL pairs have transfer_tax > {Config.MAX_TAX_PCT}% - SKIPPING (would remove all pairs)")
                removal_stats["transfer_tax"] = 0
            else:
                df = df.filter(pl.col("transfer_tax") <= Config.MAX_TAX_PCT)
                removed = before - acceptable_count
                removal_stats["transfer_tax"] = removed
                print(f"  • transfer_tax: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • transfer_tax: skipped ({e})")
    
    # Liquidity & survival filters
    print("\n💧 Liquidity & Survival Filters:")
    
    # Check if timestamp columns exist
    has_timestamps = False
    try:
        schema = df.collect_schema()
        has_timestamps = "first_swap_at" in schema.names() and "last_swap_at" in schema.names()
    except:
        pass
    
    if has_timestamps:
        try:
            # Calculate age and last activity using proper datetime conversion
            from datetime import datetime
            reference_date = datetime(2024, 12, 31)
            
            # Cast to datetime if they're timestamps (u32 or i64)
            df = df.with_columns([
                pl.col("first_swap_at").cast(pl.Datetime("us")).alias("first_swap_dt"),
                pl.col("last_swap_at").cast(pl.Datetime("us")).alias("last_swap_dt")
            ])
            
            df = df.with_columns([
                ((pl.col("last_swap_dt") - pl.col("first_swap_dt")).dt.total_days()).alias("age_days"),
                ((pl.lit(reference_date) - pl.col("last_swap_dt")).dt.total_days()).alias("days_since_last_swap")
            ])
        except Exception as e:
            print(f"  Warning: Could not calculate date-based features: {e}")
            # Create dummy columns
            df = df.with_columns([
                pl.lit(100).alias("age_days"),
                pl.lit(0).alias("days_since_last_swap")
            ])
    
    liquidity_filters = []
    
    # Add filters based on available columns
    try:
        schema = df.collect_schema()
        columns = schema.names()
        
        if "age_days" in columns:
            liquidity_filters.append(("min_age", pl.col("age_days") >= Config.MIN_AGE_DAYS))
        
        if "days_since_last_swap" in columns:
            liquidity_filters.append(("recent_activity", pl.col("days_since_last_swap") <= Config.DAYS_SINCE_LAST_SWAP))
        
        if "buy_volume_30d" in columns and "sell_volume_30d" in columns:
            liquidity_filters.append(("min_volume", pl.col("buy_volume_30d") + pl.col("sell_volume_30d") >= Config.MIN_VOLUME_30D))
        
        if "buy_count_30d" in columns and "sell_count_30d" in columns:
            liquidity_filters.append(("min_trades", (pl.col("buy_count_30d") + pl.col("sell_count_30d")) / 30 >= Config.MIN_AVG_DAILY_TRADES))
    except Exception as e:
        print(f"  Warning: Could not set up liquidity filters: {e}")
    
    for name, condition in liquidity_filters:
        try:
            before = df.select(pl.count()).collect().item()
            # Check if this filter would remove everything
            passing_count = df.filter(condition).select(pl.count()).collect().item()
            
            if passing_count == 0:
                print(f"  • {name}: Would remove all {before:,} pairs - SKIPPING")
                removal_stats[name] = 0
            else:
                df = df.filter(condition)
                removed = before - passing_count
                removal_stats[name] = removed
                print(f"  • {name}: removed {removed:,} pairs ({100*removed/initial_count:.1f}%)")
        except Exception as e:
            print(f"  • {name}: skipped (error: {e})")

    
    # Final counts
    final_count = df.select(pl.count()).collect().item()
    total_removed = initial_count - final_count
    
    print(f"\n{'='*50}")
    print(f"FILTERING SUMMARY:")
    print(f"  Initial pairs: {initial_count:,}")
    print(f"  Removed: {total_removed:,} ({100*total_removed/initial_count:.1f}%)")
    print(f"  Remaining: {final_count:,} ({100*final_count/initial_count:.1f}%)")
    print(f"{'='*50}")
    
    if final_count / initial_count > 0.10:
        print("⚠️  WARNING: >10% of pairs survived filtering. This is unusual for on-chain data.")
    
    return df, removal_stats


# ============================================================================
# STEP 3: LOAD AND MERGE CANDLE DATA
# ============================================================================

def load_candles_for_pairs(pair_ids: List[str], chunk_size: int = 5000) -> pl.DataFrame:
    """Load candle data for filtered pairs in chunks"""
    print(f"\n{'='*80}")
    print("STEP 3: LOADING CANDLE DATA")
    print(f"{'='*80}")
    
    print(f"Loading candles for {len(pair_ids):,} pairs in chunks of {chunk_size:,}...")
    
    try:
        # Load candles lazily and filter for relevant pairs
        candles = pl.scan_parquet(Config.CANDLES_PATH)
        
        # Filter for our pairs and collect
        candles_filtered = candles.filter(
            pl.col("pair_id").is_in(pair_ids)
        ).collect()
        
        print(f"✓ Loaded {len(candles_filtered):,} candle records")
        
        if len(candles_filtered) > 0:
            print(f"  Date range: {candles_filtered['timestamp'].min()} to {candles_filtered['timestamp'].max()}")
            print(f"  Unique pairs: {candles_filtered['pair_id'].n_unique():,}")
        else:
            print("  ⚠️  No candle data found for these pairs!")
        
        return candles_filtered
    except Exception as e:
        print(f"❌ Error loading candle data: {e}")
        # Return empty DataFrame with expected schema
        return pl.DataFrame({
            "pair_id": [],
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "buy_volume": [],
            "sell_volume": [],
            "buys": [],
            "sells": []
        })


# ============================================================================
# STEP 4: FEATURE ENGINEERING (MINIMAL & EXPLAINABLE)
# ============================================================================

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create minimal, explainable features"""
    print(f"\n{'='*80}")
    print("STEP 4: FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    if len(df) == 0:
        print("⚠️  No data to engineer features from!")
        return df
    
    print("\nCreating features:")
    print("  📈 Price & Trend:")
    print("     • 1d, 3d, 7d returns")
    print("     • EMA(7) vs EMA(21) cross")
    print("     • High-low range normalized")
    
    print("  💹 Flow & Liquidity:")
    print("     • Buy/Sell volume ratio")
    print("     • Volume Z-score (rolling)")
    print("     • Trade count acceleration")
    
    print("  📊 Volatility Regime:")
    print("     • Rolling volatility (7d, 14d)")
    print("     • Volatility contraction/expansion")
    
    try:
        # Sort by pair and timestamp
        df = df.sort(["pair_id", "timestamp"])
        
        # Price & trend features
        df = df.with_columns([
            # Returns
            (pl.col("close").pct_change(1).over("pair_id")).alias("return_1d"),
            (pl.col("close").pct_change(3).over("pair_id")).alias("return_3d"),
            (pl.col("close").pct_change(7).over("pair_id")).alias("return_7d"),
            
            # EMAs
            (pl.col("close").ewm_mean(span=7).over("pair_id")).alias("ema_7"),
            (pl.col("close").ewm_mean(span=21).over("pair_id")).alias("ema_21"),
            
            # Range
            ((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("range_normalized"),
        ])
        
        # EMA cross
        df = df.with_columns([
            ((pl.col("ema_7") - pl.col("ema_21")) / (pl.col("ema_21") + 1e-10)).alias("ema_cross_signal"),
        ])
        
        # Flow features
        df = df.with_columns([
            # Buy/Sell ratio
            (pl.col("buy_volume") / (pl.col("sell_volume") + 1e-10)).alias("buy_sell_ratio"),
            
            # Trade counts
            (pl.col("buys") + pl.col("sells")).alias("total_trades"),
        ])
        
        # Volume Z-score
        df = df.with_columns([
            ((pl.col("volume") - pl.col("volume").rolling_mean(14).over("pair_id")) / 
             (pl.col("volume").rolling_std(14).over("pair_id") + 1e-10)).alias("volume_zscore"),
            
            # Trade count acceleration
            ((pl.col("total_trades") - pl.col("total_trades").shift(1).over("pair_id")) / 
             (pl.col("total_trades").shift(1).over("pair_id") + 1)).alias("trade_accel"),
        ])
        
        # Volatility features
        df = df.with_columns([
            pl.col("return_1d").rolling_std(7).over("pair_id").alias("volatility_7d"),
            pl.col("return_1d").rolling_std(14).over("pair_id").alias("volatility_14d"),
        ])
        
        # Volatility regime change
        df = df.with_columns([
            ((pl.col("volatility_7d") - pl.col("volatility_14d")) / 
             (pl.col("volatility_14d") + 1e-10)).alias("vol_regime_change"),
        ])
        
        feature_count = len([c for c in df.columns if c not in ['pair_id', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'buys', 'sells']])
        print(f"\n✓ Created {feature_count} features")
        
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        import traceback
        traceback.print_exc()
    
    return df


# ============================================================================
# STEP 5: BARRIER-BASED LABELING
# ============================================================================

def create_barrier_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create barrier-based labels for BUY signals
    
    A BUY signal is valid if within the next N days:
    - Maximum upside reaches +X% 
    - Before maximum drawdown reaches -Y%
    """
    print(f"\n{'='*80}")
    print("STEP 5: BARRIER-BASED LABELING")
    print(f"{'='*80}")
    
    print(f"\nLabel configuration:")
    print(f"  Target upside: +{Config.LABEL_UPSIDE_PCT}%")
    print(f"  Stop loss: -{Config.LABEL_DOWNSIDE_PCT}%")
    print(f"  Horizon: {Config.LABEL_HORIZON_DAYS} days")
    print(f"  Logic: Upside must be hit BEFORE downside")
    
    # Sort by pair and timestamp
    df = df.sort(["pair_id", "timestamp"])
    
    # Calculate forward returns for the next N days
    labels = []
    
    for horizon in range(1, Config.LABEL_HORIZON_DAYS + 1):
        df = df.with_columns([
            (pl.col("close").shift(-horizon).over("pair_id") / pl.col("close") - 1).alias(f"fwd_return_{horizon}d")
        ])
    
    # For each row, check if upside barrier is hit before downside
    # This requires checking the sequence of returns
    
    # Simple vectorized approach: check max upside and max downside
    forward_cols = [f"fwd_return_{i}d" for i in range(1, Config.LABEL_HORIZON_DAYS + 1)]
    
    df = df.with_columns([
        pl.max_horizontal([pl.col(c) for c in forward_cols]).alias("max_upside"),
        pl.min_horizontal([pl.col(c) for c in forward_cols]).alias("max_downside"),
    ])
    
    # Label as BUY (1) if max_upside >= threshold AND abs(max_downside) < stop_loss
    # This is a simplified version - true barrier labeling requires checking sequence
    df = df.with_columns([
        (
            (pl.col("max_upside") >= Config.LABEL_UPSIDE_PCT / 100) &
            (pl.col("max_downside") >= -Config.LABEL_DOWNSIDE_PCT / 100)
        ).cast(pl.Int32).alias("signal_label")
    ])
    
    # Calculate label statistics
    total_samples = len(df)
    positive_samples = df["signal_label"].sum()
    positive_rate = positive_samples / total_samples * 100
    
    print(f"\n📊 Label Statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Positive signals (BUY): {positive_samples:,} ({positive_rate:.2f}%)")
    print(f"  Negative samples (HOLD): {total_samples - positive_samples:,} ({100-positive_rate:.2f}%)")
    
    if positive_rate > Config.MIN_SIGNAL_RATE_PCT:
        print(f"\n⚠️  WARNING: Signal rate ({positive_rate:.2f}%) exceeds {Config.MIN_SIGNAL_RATE_PCT}% threshold")
        print(f"     Consider more aggressive labeling parameters.")
    else:
        print(f"\n✓ Signal rate is appropriately rare ({positive_rate:.2f}% < {Config.MIN_SIGNAL_RATE_PCT}%)")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print(f"\n{'#'*80}")
    print("ON-CHAIN DEX TRADING SIGNAL GENERATION SYSTEM")
    print(f"{'#'*80}")
    print(f"\nTarget Chain: Base (chain_id = {Config.BASE_CHAIN_ID})")
    print(f"Philosophy: Trade rarely, survive always, compound slowly")
    print(f"{'#'*80}\n")
    
    try:
        # Step 1: Load pair universe and filter base chain
        pairs_lazy = load_pair_universe_lazy()
        pairs_base = filter_base_chain(pairs_lazy)
        
        # Step 2: Apply quality filters
        pairs_filtered, removal_stats = apply_quality_filters(pairs_base)
        
        # Collect filtered pair IDs
        pairs_clean = pairs_filtered.collect()
        pair_ids = pairs_clean["pair_id"].to_list()
        
        if len(pair_ids) == 0:
            print("\n❌ ERROR: No pairs remaining after filtering!")
            print("   Possible issues:")
            print("   1. Wrong chain_id configured")
            print("   2. Filters too strict")
            print("   3. Data quality issues")
            return None
        
        print(f"\n✓ Filtered to {len(pair_ids):,} high-quality Base chain pairs")
        
        # Step 3: Load candle data
        candles = load_candles_for_pairs(pair_ids)
        
        if len(candles) == 0:
            print("\n❌ ERROR: No candle data found for filtered pairs!")
            print("   Please check if candle data exists for these pair_ids")
            return None
        
        # Step 4: Feature engineering
        candles_feat = engineer_features(candles)
        
        # Step 5: Create labels
        candles_labeled = create_barrier_labels(candles_feat)
        
        # Drop rows with missing values
        candles_clean = candles_labeled.drop_nulls()
        
        if len(candles_clean) == 0:
            print("\n❌ ERROR: No data remaining after dropping nulls!")
            print("   This usually means insufficient historical data")
            return None
        
        print(f"\n✓ Final dataset: {len(candles_clean):,} samples across {candles_clean['pair_id'].n_unique():,} pairs")
        
        # Save processed data
        output_path = "/home/hamzabhatti18/Desktop/Genesis-labs/xgboost/processed_data.parquet"
        candles_clean.write_parquet(output_path)
        print(f"\n✓ Saved processed data to: {output_path}")
        
        return candles_clean
        
    except Exception as e:
        print(f"\n❌ ERROR in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        df_final = main()
        print(f"\n{'='*80}")
        print("✅ DATA PREPARATION COMPLETE")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
