#!/usr/bin/env python3
"""
Test Calibration Impact
Compare raw vs calibrated probabilities on recent signals.
"""

import polars as pl
import xgboost as xgb
import numpy as np
from pathlib import Path
from calibrate_model import XGBoostCalibrator

print("="*80)
print("TESTING CALIBRATION IMPACT")
print("="*80)

# Load model
print("\n1. Loading model...")
model = xgb.Booster()
model.load_model("xgb_model.json")
print("   ✓ Model loaded")

# Load calibrator
print("\n2. Loading calibrator...")
calibrator = XGBoostCalibrator.load("xgb_calibrator.pkl")
print("   ✓ Calibrator loaded")

# Load recent signal data
print("\n3. Loading recent signals...")
signals_file = "signals_2026-01-29_01-32-44.csv"
df = pl.read_csv(signals_file, try_parse_dates=False)
print(f"   ✓ Loaded {len(df)} signals")

# Features
feature_cols = [
    "return_1c", "return_4c", "return_16c",
    "ema_cross_signal", "range_normalized",
    "buy_sell_ratio", "volume_zscore", "trade_accel",
    "volatility_4h", "volatility_24h", "vol_regime_change"
]

# Get predictions
print("\n4. Computing predictions...")
X = df.select(feature_cols).to_numpy()
dmatrix = xgb.DMatrix(X, feature_names=feature_cols)

# Raw probabilities
pred_proba_raw = model.predict(dmatrix)
print(f"   Raw probabilities: min={pred_proba_raw.min():.3f}, max={pred_proba_raw.max():.3f}, mean={pred_proba_raw.mean():.3f}")

# Calibrated probabilities
pred_proba_calibrated = calibrator.transform(pred_proba_raw)
print(f"   Calibrated probabilities: min={pred_proba_calibrated.min():.3f}, max={pred_proba_calibrated.max():.3f}, mean={pred_proba_calibrated.mean():.3f}")

# Compare signal counts at different thresholds
print("\n5. Signal counts at different thresholds:")
print(f"   Threshold | Raw Signals | Calibrated Signals | Change")
print(f"   " + "-"*60)

for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    raw_count = (pred_proba_raw >= thresh).sum()
    cal_count = (pred_proba_calibrated >= thresh).sum()
    change = cal_count - raw_count
    change_pct = (change / raw_count * 100) if raw_count > 0 else 0
    print(f"   {thresh:.2f}      | {raw_count:11d} | {cal_count:18d} | {change:+6d} ({change_pct:+6.1f}%)")

# Show examples
print("\n6. Example predictions (top 10 raw confidence):")
print(f"   {'Pair':<12} {'Raw Prob':<12} {'Calibrated':<12} {'Change':<12}")
print(f"   " + "-"*60)

# Get top 10 by raw probability
top_indices = np.argsort(pred_proba_raw)[-10:][::-1]
pair_addresses = df["pair_address"].to_list()
for idx in top_indices:
    idx = int(idx)  # Convert numpy int to Python int
    pair = pair_addresses[idx][:12]
    raw = pred_proba_raw[idx]
    cal = pred_proba_calibrated[idx]
    change = cal - raw
    print(f"   {pair:<12} {raw:>6.1f}%       {cal:>6.1f}%       {change:+6.1f}%")

print("\n" + "="*80)
print("SUMMARY:")
print("-"*80)
print(f"Calibration reduces overconfident predictions:")
print(f"  • Raw mean: {pred_proba_raw.mean()*100:.1f}%")
print(f"  • Calibrated mean: {pred_proba_calibrated.mean()*100:.1f}%")
print(f"  • Reduction: {(pred_proba_raw.mean() - pred_proba_calibrated.mean())*100:.1f}%")
print()
print(f"At 0.80 threshold (current setting):")
print(f"  • Raw signals: {(pred_proba_raw >= 0.80).sum()}")
print(f"  • Calibrated signals: {(pred_proba_calibrated >= 0.80).sum()}")
print(f"  • Change: {(pred_proba_calibrated >= 0.80).sum() - (pred_proba_raw >= 0.80).sum():+d}")
print("="*80)
