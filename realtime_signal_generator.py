#!/usr/bin/env python3
"""
Real-Time Signal Generator using Envio HyperSync
=================================================

Streams swap data from Envio HyperSync, processes it into features,
and generates trading signals using the trained XGBoost model.

Generates 5 signals after 1 minute of data collection.
"""

import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from hypersync_streamer import HyperSyncStreamer, SwapToCandleConverter
try:
    from onchain_signal_system import engineer_features
except ImportError:
    # Fallback if import fails
    def engineer_features(df):
        return df


class RealTimeSignalGenerator:
    """Generate trading signals from real-time HyperSync data"""
    
    def __init__(self, model_path: str = "xgb_model.json", api_token: str = None):
        self.model_path = Path(model_path)
        self.api_token = api_token or "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
        self.model = None
        self.feature_names = [
            "return_1d", "return_3d", "return_7d",
            "ema_cross_signal", "range_normalized",
            "buy_sell_ratio", "volume_zscore", "trade_accel",
            "volatility_7d", "volatility_14d", "vol_regime_change"
        ]
        self.threshold = 0.7  # Signal threshold
        
    def load_model(self):
        """Load trained XGBoost model"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Please train the model first using: python model_training.py"
            )
        
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        print(f"✓ Loaded XGBoost model from {self.model_path}")
    
    def prepare_features(self, candles: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Convert candles to features for model prediction"""
        if len(candles) == 0:
            return None
        
        try:
            # For real-time data with limited history, create simplified features
            # Sort by pair and timestamp
            df = candles.sort(["pair_id", "timestamp"])
            
            # Ensure we have required columns with defaults
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
            
            # Create basic features (simplified for real-time)
            df = df.with_columns([
                # Returns (use 0 if not enough history)
                pl.when(pl.col("close").shift(1).over("pair_id").is_not_null())
                .then((pl.col("close") / pl.col("close").shift(1).over("pair_id") - 1))
                .otherwise(0.0)
                .alias("return_1d"),
                
                pl.lit(0.0).alias("return_3d"),  # Need 3 days history
                pl.lit(0.0).alias("return_7d"),  # Need 7 days history
                
                # EMAs (simplified)
                pl.col("close").alias("ema_7"),  # Use current price as proxy
                pl.col("close").alias("ema_21"),
                
                # Range
                ((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("range_normalized"),
            ])
            
            # EMA cross signal
            df = df.with_columns([
                ((pl.col("ema_7") - pl.col("ema_21")) / (pl.col("ema_21") + 1e-10)).alias("ema_cross_signal"),
            ])
            
            # Flow features
            df = df.with_columns([
                (pl.col("buy_volume") / (pl.col("sell_volume") + 1e-10)).alias("buy_sell_ratio"),
                (pl.col("buys") + pl.col("sells")).alias("total_trades"),
            ])
            
            # Volume Z-score (simplified - use current volume)
            df = df.with_columns([
                pl.lit(0.0).alias("volume_zscore"),  # Need history for z-score
                pl.lit(0.0).alias("trade_accel"),  # Need history
            ])
            
            # Volatility (simplified)
            df = df.with_columns([
                pl.col("return_1d").abs().alias("volatility_7d"),  # Use 1d return as proxy
                pl.col("return_1d").abs().alias("volatility_14d"),
            ])
            
            # Volatility regime change
            df = df.with_columns([
                pl.lit(0.0).alias("vol_regime_change"),
            ])
            
            # Check if all required features exist
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                for feat in missing_features:
                    df = df.with_columns(pl.lit(0.0).alias(feat))
            
            # Select only feature columns
            feature_df = df.select(self.feature_names)
            
            # Fill any nulls with 0
            feature_df = feature_df.fill_null(0.0)
            
            return feature_df
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_signals(self, candles: pl.DataFrame, top_n: int = 5) -> List[Dict]:
        """Generate trading signals from candles"""
        if self.model is None:
            self.load_model()
        
        if len(candles) == 0:
            print("⚠️  No candles to generate signals from")
            return []
        
        # Prepare features
        feature_df = self.prepare_features(candles)
        if feature_df is None or len(feature_df) == 0:
            print("⚠️  Could not prepare features")
            return []
        
        # Convert to numpy for XGBoost
        X = feature_df.to_numpy()
        
        # Handle case where we have fewer features than expected
        if X.shape[1] != len(self.feature_names):
            print(f"⚠️  Feature mismatch: expected {len(self.feature_names)}, got {X.shape[1]}")
            return []
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Get predictions
        pred_proba = self.model.predict(dmatrix)
        
        # Add predictions to candles
        candles_with_pred = candles.with_columns([
            pl.Series("pred_proba", pred_proba),
            pl.Series("signal", (pred_proba >= self.threshold).astype(int))
        ])
        
        # Filter to signals only and sort by probability
        signals_df = candles_with_pred.filter(pl.col("signal") == 1).sort("pred_proba", descending=True)
        
        if len(signals_df) == 0:
            print(f"⚠️  No signals above threshold {self.threshold}")
            # Return top predictions even if below threshold
            signals_df = candles_with_pred.sort("pred_proba", descending=True).head(top_n)
            print(f"   Showing top {len(signals_df)} predictions (below threshold):")
        else:
            print(f"✓ Found {len(signals_df)} signals above threshold {self.threshold}")
            signals_df = signals_df.head(top_n)
        
        # Convert to list of dicts
        signals = []
        for row in signals_df.iter_rows(named=True):
            signal = {
                "timestamp": row.get("timestamp", datetime.now()),
                "pair_id": row.get("pair_id", 0),
                "pool_address": row.get("pool_address", ""),
                "pred_proba": float(row.get("pred_proba", 0.0)),
                "signal": int(row.get("signal", 0)),
                "return_1d": float(row.get("return_1d", 0.0)) if "return_1d" in row else 0.0,
                "return_7d": float(row.get("return_7d", 0.0)) if "return_7d" in row else 0.0,
                "volatility_7d": float(row.get("volatility_7d", 0.0)) if "volatility_7d" in row else 0.0,
                "volume": float(row.get("volume", 0.0)),
                "buys": int(row.get("buys", 0)),
                "sells": int(row.get("sells", 0)),
            }
            signals.append(signal)
        
        return signals
    
    def print_signals(self, signals: List[Dict]):
        """Print formatted signals"""
        if not signals:
            print("\n⚠️  No signals to display")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 GENERATED {len(signals)} SIGNALS")
        print(f"{'='*80}\n")
        
        for i, signal in enumerate(signals, 1):
            signal_icon = "🟢 BUY" if signal["signal"] == 1 else "⚪ HOLD"
            timestamp = signal["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            print(f"{i}. {signal_icon}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Pair ID: {signal['pair_id']}")
            print(f"   Pool: {signal['pool_address'][:20]}...")
            print(f"   Probability: {signal['pred_proba']:.3f}")
            print(f"   Volume: {signal['volume']:,.0f}")
            print(f"   Trades: {signal['buys']} buys, {signal['sells']} sells")
            if signal.get("return_1d") is not None:
                print(f"   1d Return: {signal['return_1d']:+.2f}%")
            if signal.get("volatility_7d") is not None:
                print(f"   7d Volatility: {signal['volatility_7d']:.2f}%")
            print()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate real-time trading signals from HyperSync")
    parser.add_argument(
        "--model",
        type=str,
        default="xgb_model.json",
        help="Path to trained XGBoost model (default: xgb_model.json)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8",
        help="Envio API token"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Data collection duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--signals",
        type=int,
        default=5,
        help="Number of signals to generate (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Probability threshold for signals (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print("REAL-TIME SIGNAL GENERATOR - ENVIO HYPERSYNC")
    print(f"{'#'*80}")
    print(f"Model: {args.model}")
    print(f"Duration: {args.duration} seconds")
    print(f"Target signals: {args.signals}")
    print(f"Threshold: {args.threshold}")
    print(f"{'#'*80}\n")
    
    # Initialize generator
    generator = RealTimeSignalGenerator(model_path=args.model, api_token=args.token)
    generator.threshold = args.threshold
    
    # Load model
    try:
        generator.load_model()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    
    # Collect candles from stream
    collected_candles = []
    
    def signal_callback(candles: pl.DataFrame):
        """Callback when data is collected"""
        nonlocal collected_candles
        collected_candles = candles
        print(f"\n📊 Collected {len(candles)} candles")
    
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
        if len(collected_candles) > 0:
            candles = collected_candles
        
        if len(candles) == 0:
            print("\n⚠️  No candles collected. Cannot generate signals.")
            return 1
        
        # Generate signals
        print(f"\n🔮 Generating signals from {len(candles)} candles...")
        signals = generator.generate_signals(candles, top_n=args.signals)
        
        # Print signals
        generator.print_signals(signals)
        
        # Save signals to file
        if signals:
            output_path = Path("realtime_signals.json")
            import json
            
            # Convert datetime to string for JSON
            signals_json = []
            for s in signals:
                s_copy = s.copy()
                if isinstance(s_copy["timestamp"], datetime):
                    s_copy["timestamp"] = s_copy["timestamp"].isoformat()
                signals_json.append(s_copy)
            
            with open(output_path, "w") as f:
                json.dump(signals_json, f, indent=2)
            
            print(f"💾 Saved signals to: {output_path}")
        
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
