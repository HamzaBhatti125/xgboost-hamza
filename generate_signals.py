#!/usr/bin/env python3
"""
Signal Generation & Analysis
=============================

View historical signals and generate new predictions using the trained model.

Usage:
    python generate_signals.py --mode history    # View historical test set signals
    python generate_signals.py --mode latest     # Generate signals for latest data
    python generate_signals.py --mode analyze    # Detailed analysis of predictions
"""

import polars as pl
import xgboost as xgb
import argparse
from pathlib import Path
from datetime import datetime, timedelta


class SignalGenerator:
    """Generate and analyze trading signals"""
    
    def __init__(self, model_path="xgb_model.json", data_path="processed_data.parquet"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.feature_names = [
            "return_1d", "return_3d", "return_7d",
            "ema_cross_signal", "range_normalized",
            "buy_sell_ratio", "volume_zscore", "trade_accel",
            "volatility_7d", "volatility_14d", "vol_regime_change"
        ]
        self.threshold = 0.8  # From training results (best threshold)
        
    def load_model(self):
        """Load trained XGBoost model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        print(f"✓ Loaded model from {self.model_path}")
        
    def load_data(self):
        """Load processed data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        df = pl.read_parquet(self.data_path)
        print(f"✓ Loaded {len(df):,} samples from {self.data_path}")
        return df
    
    def predict(self, df):
        """Generate predictions on dataframe"""
        if self.model is None:
            self.load_model()
        
        # Prepare features
        X = df.select(self.feature_names).to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Get predictions (probabilities)
        pred_proba = self.model.predict(dmatrix)
        
        # Add predictions to dataframe
        result = df.with_columns([
            pl.Series("pred_proba", pred_proba),
            pl.Series("signal", (pred_proba >= self.threshold).astype(int))
        ])
        
        return result
    
    def show_historical_signals(self, lookback_days=90):
        """Show signals from test set"""
        print("\n" + "="*80)
        print("HISTORICAL SIGNALS (Test Set)")
        print("="*80)
        
        df = self.load_data()
        
        # Use test set (Oct 2024+)
        test_start = datetime(2024, 10, 1)
        df_test = df.filter(pl.col("timestamp") >= test_start)
        
        # Generate predictions
        df_pred = self.predict(df_test)
        
        # Filter to signals only
        signals = df_pred.filter(pl.col("signal") == 1).sort("timestamp", descending=True)
        
        if len(signals) == 0:
            print("\n⚠️  No signals generated (threshold too high?)")
            return
        
        print(f"\n🎯 Found {len(signals):,} signals in test set")
        print(f"   Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
        print(f"   Signal rate: {100*len(signals)/len(df_test):.2f}%")
        print(f"   Threshold: {self.threshold}")
        
        # Show recent signals
        print(f"\n📊 Most Recent Signals (last {min(50, len(signals))}):")
        print("-" * 80)
        
        recent = signals.head(50).select([
            "timestamp", "pair_id", "pred_proba", "signal_label",
            "return_1d", "return_7d", "volatility_7d"
        ])
        
        for row in recent.iter_rows(named=True):
            outcome = "✅ WIN" if row["signal_label"] == 1 else "❌ LOSS"
            print(f"{row['timestamp']}  |  Pair: {row['pair_id']:6d}  |  "
                  f"Prob: {row['pred_proba']:.3f}  |  {outcome}  |  "
                  f"Ret1d: {row['return_1d']:+.1f}%  Vol: {row['volatility_7d']:.1f}%")
        
        # Performance summary
        signal_performance = signals.select([
            pl.col("signal_label").mean().alias("win_rate"),
            pl.col("signal_label").sum().alias("wins"),
            pl.count().alias("total"),
            pl.col("pred_proba").mean().alias("avg_confidence")
        ]).to_dicts()[0]
        
        print("\n" + "-" * 80)
        print(f"📈 Signal Performance:")
        print(f"   Win Rate: {signal_performance['win_rate']:.1%} "
              f"({signal_performance['wins']}/{signal_performance['total']})")
        print(f"   Avg Confidence: {signal_performance['avg_confidence']:.3f}")
        
        # Save to CSV
        output_path = Path("signals_history.csv")
        signals.select([
            "timestamp", "pair_id", "pred_proba", "signal", "signal_label",
            "return_1d", "return_3d", "return_7d", "volatility_7d"
        ]).write_csv(output_path)
        print(f"\n💾 Saved all {len(signals)} signals to: {output_path}")
        
    def show_latest_signals(self, top_n=20):
        """Show signals for most recent data (for live trading)"""
        print("\n" + "="*80)
        print("LATEST SIGNALS (Most Recent Data)")
        print("="*80)
        
        df = self.load_data()
        
        # Get most recent date for each pair
        latest = df.group_by("pair_id").agg([
            pl.col("timestamp").max().alias("latest_date")
        ])
        
        # Join to get latest samples
        df_latest = df.join(latest, on="pair_id").filter(
            pl.col("timestamp") == pl.col("latest_date")
        )
        
        # Generate predictions
        df_pred = self.predict(df_latest)
        
        # Sort by prediction probability
        df_sorted = df_pred.sort("pred_proba", descending=True)
        
        print(f"\n📅 Latest data date: {df_latest['timestamp'].max()}")
        print(f"   Analyzing {len(df_latest)} pairs")
        print(f"   Threshold for signal: {self.threshold}")
        
        # Show top predictions
        signals = df_sorted.filter(pl.col("signal") == 1)
        
        if len(signals) == 0:
            print(f"\n⚠️  No signals above threshold {self.threshold}")
            print(f"\n🔍 Top {top_n} pairs by confidence (below threshold):")
        else:
            print(f"\n🎯 {len(signals)} ACTIVE SIGNALS:")
        
        print("-" * 80)
        
        top_pairs = df_sorted.head(top_n).select([
            "pair_id", "timestamp", "pred_proba", "signal",
            "return_1d", "return_7d", "volatility_7d", "ema_cross_signal"
        ])
        
        for row in top_pairs.iter_rows(named=True):
            signal_icon = "🟢 BUY" if row["signal"] == 1 else "⚪ HOLD"
            print(f"{signal_icon}  |  Pair: {row['pair_id']:6d}  |  "
                  f"Prob: {row['pred_proba']:.3f}  |  "
                  f"Date: {row['timestamp']}  |  "
                  f"Ret7d: {row['return_7d']:+.1f}%  |  Vol: {row['volatility_7d']:.1f}%")
        
        # Save to CSV
        output_path = Path("signals_latest.csv")
        df_sorted.head(50).select([
            "timestamp", "pair_id", "pred_proba", "signal",
            "return_1d", "return_3d", "return_7d", "volatility_7d",
            "ema_cross_signal", "buy_sell_ratio"
        ]).write_csv(output_path)
        print(f"\n💾 Saved top 50 predictions to: {output_path}")
        
    def analyze_predictions(self):
        """Detailed analysis of model predictions"""
        print("\n" + "="*80)
        print("PREDICTION ANALYSIS")
        print("="*80)
        
        df = self.load_data()
        df_pred = self.predict(df)
        
        # Test set only
        test_start = datetime(2024, 10, 1)
        df_test = df_pred.filter(pl.col("timestamp") >= test_start)
        
        print(f"\nAnalyzing {len(df_test):,} test samples...")
        
        # Probability distribution
        print("\n📊 Prediction Probability Distribution:")
        bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = df_test.filter(
                (pl.col("pred_proba") >= bins[i]) & (pl.col("pred_proba") < bins[i+1])
            ).select(pl.count()).item()
            pct = 100 * count / len(df_test)
            bar = "█" * int(pct)
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:6,} ({pct:5.1f}%) {bar}")
        
        # Performance by threshold
        print("\n🎯 Win Rate by Probability Threshold:")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            subset = df_test.filter(pl.col("pred_proba") >= thresh)
            if len(subset) > 0:
                win_rate = subset.select(pl.col("signal_label").mean()).item()
                signal_count = len(subset)
                signal_rate = 100 * signal_count / len(df_test)
                print(f"  Threshold {thresh:.1f}: Win Rate {win_rate:.1%} | "
                      f"Signals: {signal_count:4,} ({signal_rate:.2f}%)")
            else:
                print(f"  Threshold {thresh:.1f}: No signals")
        
        # Top features for high-confidence signals
        print("\n📈 Characteristics of High-Confidence Signals (prob > 0.8):")
        high_conf = df_test.filter(pl.col("pred_proba") >= 0.8)
        if len(high_conf) > 0:
            stats = high_conf.select([
                pl.col("return_7d").mean().alias("avg_return_7d"),
                pl.col("volatility_7d").mean().alias("avg_vol_7d"),
                pl.col("ema_cross_signal").mean().alias("avg_ema_cross"),
                pl.col("signal_label").mean().alias("win_rate")
            ]).to_dicts()[0]
            
            print(f"   Avg 7d return: {stats['avg_return_7d']:+.2f}%")
            print(f"   Avg volatility: {stats['avg_vol_7d']:.2f}%")
            print(f"   EMA cross signal: {stats['avg_ema_cross']:.2f}")
            print(f"   Actual win rate: {stats['win_rate']:.1%}")
        else:
            print("   No high-confidence signals in test set")


def main():
    parser = argparse.ArgumentParser(description="Generate and analyze trading signals")
    parser.add_argument(
        "--mode",
        choices=["history", "latest", "analyze"],
        default="history",
        help="Mode: view historical signals, latest signals, or analyze predictions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Probability threshold for generating signals (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    generator = SignalGenerator()
    generator.threshold = args.threshold
    
    try:
        if args.mode == "history":
            generator.show_historical_signals()
        elif args.mode == "latest":
            generator.show_latest_signals()
        elif args.mode == "analyze":
            generator.analyze_predictions()
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the pipeline first:")
        print("  python run_pipeline.py")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
