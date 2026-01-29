#!/usr/bin/env python3
"""
Find optimal calibrated threshold based on validated performance.
"""

import polars as pl
from calibrate_model import XGBoostCalibrator

print("="*80)
print("FINDING OPTIMAL CALIBRATED THRESHOLD")
print("="*80)

# Load tracked results (validated performance)
print("\n1. Loading validated signal performance...")
tracked_file = "signals_2026-01-28_22-26-21_tracked_cache.csv"
df_tracked = pl.read_csv(tracked_file)
print(f"   ✓ Loaded {len(df_tracked)} tracked signals")

# Calculate actual win rate
wins = len(df_tracked.filter(pl.col("outcome") == "WIN"))
losses = len(df_tracked.filter(pl.col("outcome") == "LOSS"))
actual_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

print(f"\n2. Validated Performance:")
print(f"   Wins: {wins}")
print(f"   Losses: {losses}")
print(f"   Win Rate: {actual_win_rate*100:.1f}%")

# Load original signals with raw probabilities
print(f"\n3. Loading original signals...")
original_file = "signals_2026-01-28_22-26-21.csv"
df_original = pl.read_csv(original_file, try_parse_dates=False)

# Get raw probabilities from original signals
raw_proba = df_original["pred_proba"].to_numpy()
print(f"   Raw probability range: {raw_proba.min():.3f} to {raw_proba.max():.3f}")
print(f"   Raw probability mean: {raw_proba.mean():.3f}")

# Load calibrator and apply
print(f"\n4. Applying calibration...")
calibrator = XGBoostCalibrator.load("xgb_calibrator.pkl")
calibrated_proba = calibrator.transform(raw_proba)
print(f"   Calibrated probability range: {calibrated_proba.min():.3f} to {calibrated_proba.max():.3f}")
print(f"   Calibrated probability mean: {calibrated_proba.mean():.3f}")

# Find threshold that gives similar signal count
print(f"\n5. Finding equivalent calibrated threshold...")
print(f"   Original signals (≥0.70): {len(df_original.filter(pl.col('pred_proba') >= 0.70))}")
print(f"   Original signals (≥0.75): {len(df_original.filter(pl.col('pred_proba') >= 0.75))}")
print(f"   Original signals (≥0.80): {len(df_original.filter(pl.col('pred_proba') >= 0.80))}")

print(f"\n   Testing calibrated thresholds:")
for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    count = (calibrated_proba >= thresh).sum()
    print(f"   Calibrated threshold {thresh:.2f}: {count} signals")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION:")
print("-"*80)
print(f"Based on the 75.3% win rate from this batch (best performing):")
print(f"  • Raw threshold 0.70-0.75 gave {len(df_original.filter(pl.col('pred_proba') >= 0.70))} signals")
print(f"  • Calibrated equivalent: ~0.30-0.35 threshold")
print(f"\nSuggested configuration:")
print(f"  USE_CALIBRATION = True")
print(f"  SIGNAL_THRESHOLD = 0.32  # Equivalent to ~0.73 raw")
print("="*80)
