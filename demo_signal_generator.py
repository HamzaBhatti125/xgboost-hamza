#!/usr/bin/env python3
"""
Demo Signal Generator - Uses existing data to simulate real-time signals
=======================================================================

This script simulates the real-time signal generation by:
1. Loading recent data from parquet files
2. Processing it as if it were streamed
3. Generating signals (with mock model if needed)
"""

import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import feature engineering
try:
    from onchain_signal_system import engineer_features
except ImportError:
    def engineer_features(df):
        return df


def create_mock_model():
    """Create a simple mock XGBoost model for demonstration"""
    # Create a simple binary classifier
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 10,
        'random_state': 42
    }
    
    # Train on dummy data
    X_dummy = np.random.rand(100, 11)
    y_dummy = np.random.randint(0, 2, 100)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_dummy, y_dummy)
    
    return model


def load_recent_candles(sample_size: int = 500) -> pl.DataFrame:
    """Load recent candles from parquet file using lazy loading"""
    candles_path = Path("Files/candles-1d.parquet")
    
    if not candles_path.exists():
        print(f"❌ Candles file not found: {candles_path}")
        return pl.DataFrame()
    
    print(f"📂 Loading candles from {candles_path} (lazy mode)")
    
    try:
        # Use lazy loading to avoid memory issues
        df_lazy = pl.scan_parquet(candles_path)
        
        # Get a sample - take from the end of the dataset
        # First, get total count (lazy)
        total_count = df_lazy.select(pl.count()).collect().item()
        print(f"   Total candles in file: {total_count:,}")
        
        # Take a sample from the end (most recent)
        if "timestamp" in df_lazy.columns:
            # Sort descending and take top N
            df = df_lazy.sort("timestamp", descending=True).head(sample_size).collect()
            # Sort back to ascending
            df = df.sort("timestamp")
        else:
            # Just take a sample
            df = df_lazy.head(sample_size).collect()
        
        print(f"✓ Loaded {len(df):,} candles (sample from file)")
        if len(df) > 0 and "timestamp" in df.columns:
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading candles: {e}")
        import traceback
        traceback.print_exc()
        return pl.DataFrame()


def prepare_features_for_prediction(candles: pl.DataFrame) -> pl.DataFrame:
    """Prepare features for model prediction"""
    if len(candles) == 0:
        return pl.DataFrame()
    
    # Sort by pair and timestamp
    df = candles.sort(["pair_id", "timestamp"])
    
    # Ensure required columns exist
    required_cols = {
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0,
        "buys": 0, "sells": 0
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            if isinstance(default_val, float):
                df = df.with_columns(pl.lit(default_val).alias(col))
            else:
                df = df.with_columns(pl.lit(default_val).alias(col))
    
    # Create features
    feature_names = [
        "return_1d", "return_3d", "return_7d",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_7d", "volatility_14d", "vol_regime_change"
    ]
    
    try:
        # Use feature engineering if available
        df_features = engineer_features(df)
    except:
        # Fallback: create basic features
        df_features = df.with_columns([
            (pl.col("close").pct_change(1).over("pair_id")).fill_null(0.0).alias("return_1d"),
            (pl.col("close").pct_change(3).over("pair_id")).fill_null(0.0).alias("return_3d"),
            (pl.col("close").pct_change(7).over("pair_id")).fill_null(0.0).alias("return_7d"),
            ((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("range_normalized"),
            (pl.col("buy_volume") / (pl.col("sell_volume") + 1e-10)).alias("buy_sell_ratio"),
            pl.lit(0.0).alias("volume_zscore"),
            pl.lit(0.0).alias("trade_accel"),
            pl.col("return_1d").abs().alias("volatility_7d"),
            pl.col("return_1d").abs().alias("volatility_14d"),
            pl.lit(0.0).alias("vol_regime_change"),
        ])
        
        # EMA cross
        df_features = df_features.with_columns([
            (pl.col("close").ewm_mean(span=7).over("pair_id") - 
             pl.col("close").ewm_mean(span=21).over("pair_id")) / 
            (pl.col("close").ewm_mean(span=21).over("pair_id") + 1e-10)
            .alias("ema_cross_signal")
        ])
    
    # Fill missing features
    for feat in feature_names:
        if feat not in df_features.columns:
            df_features = df_features.with_columns(pl.lit(0.0).alias(feat))
    
    # Select only feature columns
    feature_df = df_features.select(feature_names)
    feature_df = feature_df.fill_null(0.0)
    
    return feature_df


def generate_signals(candles: pl.DataFrame, model_path: str = None, top_n: int = 5) -> list:
    """Generate trading signals"""
    print(f"\n🔮 Generating signals from {len(candles)} candles...")
    
    # Prepare features
    feature_df = prepare_features_for_prediction(candles)
    
    if len(feature_df) == 0:
        print("⚠️  No features generated")
        return []
    
    # Load or create model
    model = None
    if model_path and Path(model_path).exists():
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
        except:
            print(f"⚠️  Could not load model, using mock model")
            model = None
    
    if model is None:
        print("⚠️  Using mock model for demonstration")
        mock_model = create_mock_model()
        # Convert to booster format
        X = feature_df.to_numpy()
        pred_proba = mock_model.predict_proba(X)[:, 1]
    else:
        # Use real model
        X = feature_df.to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=feature_df.columns)
        pred_proba = model.predict(dmatrix)
    
    # Add predictions to candles
    candles_with_pred = candles.with_columns([
        pl.Series("pred_proba", pred_proba),
        pl.Series("signal", (pred_proba >= 0.7).astype(int))
    ])
    
    # Get top signals
    signals_df = candles_with_pred.sort("pred_proba", descending=True).head(top_n)
    
    # Convert to list of dicts
    signals = []
    for row in signals_df.iter_rows(named=True):
        def safe_int(val, default=0):
            try:
                return int(val) if val is not None else default
            except:
                return default
        
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except:
                return default
        
        signal = {
            "timestamp": str(row.get("timestamp", datetime.now())),
            "pair_id": safe_int(row.get("pair_id"), 0),
            "pred_proba": safe_float(row.get("pred_proba"), 0.0),
            "signal": safe_int(row.get("signal"), 0),
            "return_1d": safe_float(row.get("return_1d"), 0.0),
            "return_7d": safe_float(row.get("return_7d"), 0.0),
            "volatility_7d": safe_float(row.get("volatility_7d"), 0.0),
            "volume": safe_float(row.get("volume"), 0.0),
            "buys": safe_int(row.get("buys"), 0),
            "sells": safe_int(row.get("sells"), 0),
        }
        signals.append(signal)
    
    return signals


def main():
    """Main execution"""
    print(f"\n{'#'*80}")
    print("DEMO SIGNAL GENERATOR")
    print(f"{'#'*80}\n")
    
    # Load recent candles
    candles = load_recent_candles(sample_size=1000)
    
    if len(candles) == 0:
        print("❌ No candles loaded. Cannot generate signals.")
        return 1
    
    # Try to load model, fallback to mock
    model_path = "xgb_model.json"
    
    # Generate signals
    signals = generate_signals(candles, model_path=model_path, top_n=5)
    
    if not signals:
        print("⚠️  No signals generated")
        return 1
    
    # Print signals
    print(f"\n{'='*80}")
    print(f"🎯 GENERATED {len(signals)} SIGNALS")
    print(f"{'='*80}\n")
    
    for i, signal in enumerate(signals, 1):
        signal_icon = "🟢 BUY" if signal["signal"] == 1 else "⚪ HOLD"
        print(f"{i}. {signal_icon}")
        print(f"   Timestamp: {signal['timestamp']}")
        print(f"   Pair ID: {signal['pair_id']}")
        print(f"   Probability: {signal['pred_proba']:.3f}")
        print(f"   Volume: {signal['volume']:,.0f}")
        print(f"   Trades: {signal['buys']} buys, {signal['sells']} sells")
        if signal.get("return_1d") is not None:
            print(f"   1d Return: {signal['return_1d']:+.2f}%")
        if signal.get("volatility_7d") is not None:
            print(f"   7d Volatility: {signal['volatility_7d']:.2f}%")
        print()
    
    # Save to JSON
    output_path = Path("realtime_signals.json")
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2)
    
    print(f"💾 Saved signals to: {output_path}")
    print(f"\n{'='*80}")
    print("✅ Signal generation complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
