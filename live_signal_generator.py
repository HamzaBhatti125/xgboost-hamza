#!/usr/bin/env python3
"""
Live Signal Generator - HyperSync Stream
=========================================

Streams live data from Envio HyperSync and generates trading signals.
Uses mock model if trained model not available.
"""

import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from hypersync_streamer import HyperSyncStreamer
from onchain_signal_system import engineer_features, Config as DataConfig


def create_mock_model():
    """Create a simple mock XGBoost model for demonstration"""
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


def prepare_features(candles: pl.DataFrame) -> pl.DataFrame:
    """Prepare features for model prediction"""
    if len(candles) == 0:
        return pl.DataFrame()
    
    # Sort by pair and timestamp
    df = candles.sort(["pair_id", "timestamp"])
    
    # Ensure required columns
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
    
    # Create basic features
    feature_names = [
        "return_1d", "return_3d", "return_7d",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_7d", "volatility_14d", "vol_regime_change"
    ]
    
    try:
        # Try to use feature engineering
        df_features = engineer_features(df)
    except:
        # Fallback: create simplified features
        df_features = df.with_columns([
            pl.when(pl.col("close").shift(1).over("pair_id").is_not_null())
            .then((pl.col("close") / pl.col("close").shift(1).over("pair_id") - 1))
            .otherwise(0.0)
            .alias("return_1d"),
            pl.lit(0.0).alias("return_3d"),
            pl.lit(0.0).alias("return_7d"),
            pl.col("close").alias("ema_7"),
            pl.col("close").alias("ema_21"),
            ((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("range_normalized"),
        ])
        
        df_features = df_features.with_columns([
            ((pl.col("ema_7") - pl.col("ema_21")) / (pl.col("ema_21") + 1e-10)).alias("ema_cross_signal"),
            (pl.col("buy_volume") / (pl.col("sell_volume") + 1e-10)).alias("buy_sell_ratio"),
            pl.lit(0.0).alias("volume_zscore"),
            pl.lit(0.0).alias("trade_accel"),
            pl.col("return_1d").abs().alias("volatility_7d"),
            pl.col("return_1d").abs().alias("volatility_14d"),
            pl.lit(0.0).alias("vol_regime_change"),
        ])
    
    # Fill missing features
    for feat in feature_names:
        if feat not in df_features.columns:
            df_features = df_features.with_columns(pl.lit(0.0).alias(feat))
    
    # Select only feature columns
    feature_df = df_features.select(feature_names)
    feature_df = feature_df.fill_null(0.0)
    
    return feature_df


def generate_signals(candles: pl.DataFrame, model_path: str = None, threshold: float = 0.6, top_n: int = 5) -> list:
    """Generate trading signals from candles"""
    if len(candles) == 0:
        return []
    
    print(f"\n🔮 Generating signals from {len(candles)} candles...")
    
    # Prepare features
    feature_df = prepare_features(candles)
    
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
            print("⚠️  Could not load model, using mock model")
            model = None
    
    if model is None:
        print("⚠️  Using mock model for demonstration")
        mock_model = create_mock_model()
        X = feature_df.to_numpy()
        pred_proba = mock_model.predict_proba(X)[:, 1]
    else:
        X = feature_df.to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=feature_df.columns)
        pred_proba = model.predict(dmatrix)
    
    # Add predictions to candles
    candles_with_pred = candles.with_columns([
        pl.Series("pred_proba", pred_proba),
        pl.Series("signal", (pred_proba >= threshold).astype(int))
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
        
        # Get current price (use close price if available, otherwise use open)
        current_price = safe_float(row.get("close"), safe_float(row.get("open"), 1.0))
        
        # Calculate take profit and stop loss prices
        take_profit_pct = DataConfig.LABEL_UPSIDE_PCT  # 8.0%
        stop_loss_pct = DataConfig.LABEL_DOWNSIDE_PCT  # 3.0%
        take_profit_price = current_price * (1 + take_profit_pct / 100)
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        
        signal = {
            "timestamp": str(row.get("timestamp", datetime.now())),
            "pair_id": safe_int(row.get("pair_id"), 0),
            "pool_address": str(row.get("pool_address", ""))[:42],
            "pred_proba": safe_float(row.get("pred_proba"), 0.0),
            "signal": safe_int(row.get("signal"), 0),
            "current_price": current_price,
            "take_profit_pct": take_profit_pct,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": stop_loss_pct,
            "stop_loss_price": stop_loss_price,
            "holding_period_days": DataConfig.LABEL_HORIZON_DAYS,
            "return_1d": safe_float(row.get("return_1d"), 0.0),
            "return_7d": safe_float(row.get("return_7d"), 0.0),
            "volatility_7d": safe_float(row.get("volatility_7d"), 0.0),
            "volume": safe_float(row.get("volume"), 0.0),
            "buy_volume": safe_float(row.get("buy_volume"), 0.0),
            "sell_volume": safe_float(row.get("sell_volume"), 0.0),
            "buys": safe_int(row.get("buys"), 0),
            "sells": safe_int(row.get("sells"), 0),
        }
        signals.append(signal)
    
    return signals


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate live trading signals from HyperSync")
    parser.add_argument("--duration", type=int, default=60, help="Stream duration in seconds")
    parser.add_argument("--signals", type=int, default=5, help="Number of signals to generate")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold")
    parser.add_argument("--token", type=str, default="1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8", help="Envio API token")
    parser.add_argument("--model", type=str, default="xgb_model.json", help="Model path")
    parser.add_argument("--output", type=str, default="live_signals.json", help="Output file")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print("LIVE SIGNAL GENERATOR - ENVIO HYPERSYNC")
    print(f"{'#'*80}")
    print(f"Duration: {args.duration} seconds")
    print(f"Target signals: {args.signals}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print(f"{'#'*80}\n")
    
    # Collect candles from stream
    collected_candles = []
    
    def signal_callback(candles: pl.DataFrame):
        """Callback when data is collected"""
        nonlocal collected_candles
        collected_candles.append(candles)
        print(f"  📊 Collected {len(candles)} candles (total: {sum(len(c) for c in collected_candles)})")
    
    # Start streaming
    streamer = HyperSyncStreamer(
        api_token=args.token,
        chain="base",
        signal_callback=signal_callback
    )
    
    try:
        print("🚀 Starting HyperSync stream...")
        candles = streamer.start_streaming(duration_seconds=args.duration)
        
        # Use collected candles or streamed candles
        if collected_candles:
            candles = pl.concat(collected_candles) if len(collected_candles) > 1 else collected_candles[0]
        
        if len(candles) == 0:
            print("\n⚠️  No candles collected. Cannot generate signals.")
            return 1
        
        # Generate signals
        signals = generate_signals(candles, model_path=args.model, threshold=args.threshold, top_n=args.signals)
        
        if not signals:
            print("\n⚠️  No signals generated")
            return 1
        
        # Print signals
        print(f"\n{'='*80}")
        print(f"🎯 GENERATED {len(signals)} LIVE SIGNALS")
        print(f"{'='*80}\n")
        
        for i, signal in enumerate(signals, 1):
            signal_icon = "🟢 BUY" if signal["signal"] == 1 else "⚪ HOLD"
            print(f"{i}. {signal_icon}")
            print(f"   Timestamp: {signal['timestamp']}")
            print(f"   Pair ID: {signal['pair_id']}")
            print(f"   Pool: {signal['pool_address']}")
            print(f"   Probability: {signal['pred_proba']:.3f}")
            print(f"   Current Price: {signal.get('current_price', 0):.6f}")
            print(f"   Take Profit: +{signal.get('take_profit_pct', 0):.1f}% → {signal.get('take_profit_price', 0):.6f}")
            print(f"   Stop Loss: -{signal.get('stop_loss_pct', 0):.1f}% → {signal.get('stop_loss_price', 0):.6f}")
            print(f"   Max Holding: {signal.get('holding_period_days', 7)} days")
            print(f"   Volume: {signal['volume']:,.0f}")
            print(f"   Trades: {signal['buys']} buys, {signal['sells']} sells")
            if signal.get("return_1d") is not None:
                print(f"   1d Return: {signal['return_1d']:+.2f}%")
            print()
        
        # Save to file
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(signals, f, indent=2)
        
        print(f"💾 Saved {len(signals)} signals to: {output_path}")
        print(f"\n{'='*80}")
        print("✅ Signal generation complete!")
        print(f"{'='*80}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        streamer.stop()
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
