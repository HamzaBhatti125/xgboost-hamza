#!/usr/bin/env python3
"""
Online Learning with Feedback
==============================

Retrains the model using feedback from signal outcomes.
This enables the model to learn from real trading results.

This is NOT full RL, but a practical feedback loop that:
1. Tracks signal outcomes
2. Retrains model with new labeled data
3. Adjusts thresholds based on performance
4. Adapts to changing market conditions
"""

import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from signal_tracker import SignalTracker
from hybrid_training import TrainingConfig, combine_datasets, split_data, train_model, evaluate_model


def load_feedback_data(tracker: SignalTracker) -> pl.DataFrame:
    """Load feedback data from signal tracker"""
    completed = tracker.get_completed_signals()
    
    if len(completed) == 0:
        return pl.DataFrame()
    
    # Convert to training format
    data = []
    for signal in completed:
        # We need the original features - these should be stored with the signal
        # For now, create a simplified version
        data.append({
            "pair_id": signal.get("pair_id", 0),
            "timestamp": signal.get("timestamp", ""),
            "pred_proba": signal.get("pred_proba", 0.0),
            "signal_label": 1 if signal.get("outcome") == "win" else 0,
            # Features would need to be stored with signal
            # For demonstration, using placeholder
            "return_1d": 0.0,
            "return_3d": 0.0,
            "return_7d": signal.get("return_pct", 0.0) / 100,
            "ema_cross_signal": 0.0,
            "range_normalized": 0.0,
            "buy_sell_ratio": 0.0,
            "volume_zscore": 0.0,
            "trade_accel": 0.0,
            "volatility_7d": 0.0,
            "volatility_14d": 0.0,
            "vol_regime_change": 0.0,
        })
    
    return pl.DataFrame(data)


def retrain_with_feedback(historical_data_path: str = "processed_data.parquet",
                         feedback_db: str = "signal_history.jsonl",
                         output_model: str = "xgb_model_retrained.json"):
    """Retrain model with historical + feedback data"""
    print(f"\n{'='*80}")
    print("ONLINE LEARNING: RETRAINING WITH FEEDBACK")
    print(f"{'='*80}\n")
    
    # Load historical data
    historical = None
    if Path(historical_data_path).exists():
        historical = pl.read_parquet(historical_data_path)
        print(f"✓ Loaded {len(historical):,} historical samples")
    else:
        print(f"⚠️  Historical data not found: {historical_data_path}")
    
    # Load feedback data
    tracker = SignalTracker(feedback_db)
    feedback = load_feedback_data(tracker)
    
    if len(feedback) == 0:
        print("⚠️  No feedback data available. Need completed signals with outcomes.")
        print("   Mark some signal outcomes first using:")
        print("   python signal_tracker.py --outcome SIGNAL_ID --result win --exit-price 1.08")
        return None
    
    print(f"✓ Loaded {len(feedback):,} feedback samples")
    
    # Combine datasets
    combined = combine_datasets(historical, feedback)
    
    # Split data
    df_train, df_val, df_test = split_data(combined)
    
    if len(df_train) == 0:
        print("❌ No training data available")
        return None
    
    # Retrain model
    model = train_model(df_train, df_val)
    
    # Evaluate
    if len(df_val) > 0:
        evaluate_model(model, df_val, "VALIDATION")
    
    # Save retrained model
    model.save_model(output_model)
    print(f"\n✓ Saved retrained model to: {output_model}")
    
    return model


def adjust_threshold_from_performance(tracker: SignalTracker, current_threshold: float = 0.5) -> float:
    """Adjust signal threshold based on recent performance"""
    stats = tracker.get_performance_stats()
    
    if stats["completed"] < 10:
        print("⚠️  Not enough completed signals to adjust threshold")
        return current_threshold
    
    win_rate = stats["win_rate"]
    avg_return = stats["avg_return"]
    
    # If win rate is low, increase threshold (be more selective)
    if win_rate < 0.5:
        new_threshold = min(current_threshold + 0.1, 0.9)
        print(f"📉 Low win rate ({win_rate:.1%}), increasing threshold: {current_threshold:.2f} → {new_threshold:.2f}")
        return new_threshold
    
    # If win rate is high but returns are low, might be too conservative
    if win_rate > 0.7 and avg_return < 2.0:
        new_threshold = max(current_threshold - 0.05, 0.3)
        print(f"📈 High win rate but low returns, decreasing threshold: {current_threshold:.2f} → {new_threshold:.2f}")
        return new_threshold
    
    return current_threshold


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Online learning with feedback")
    parser.add_argument("--retrain", action="store_true", help="Retrain model with feedback")
    parser.add_argument("--adjust-threshold", action="store_true", help="Adjust threshold from performance")
    parser.add_argument("--feedback-db", default="signal_history.jsonl", help="Feedback database path")
    
    args = parser.parse_args()
    
    if args.retrain:
        model = retrain_with_feedback(feedback_db=args.feedback_db)
        if model:
            print("\n✅ Model retrained successfully!")
        else:
            print("\n❌ Retraining failed")
            return 1
    
    elif args.adjust_threshold:
        tracker = SignalTracker(args.feedback_db)
        new_threshold = adjust_threshold_from_performance(tracker)
        print(f"\n✓ Recommended threshold: {new_threshold:.2f}")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
