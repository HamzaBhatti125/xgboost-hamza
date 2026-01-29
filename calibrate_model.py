#!/usr/bin/env python3
"""
Model Probability Calibration
Recalibrates XGBoost model predictions to match actual win rates.

Based on multi-batch analysis:
- Model predicts: 91.7% win rate
- Actual results: 66.8% win rate
- Calibration needed: pred_proba ≥85% actually wins 48.4%
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import polars as pl
import xgboost as xgb
from typing import Tuple, Dict


# ============================================================================
# CALIBRATION METHODS
# ============================================================================

class XGBoostCalibrator:
    """Calibrate XGBoost probability predictions."""
    
    def __init__(self, method='isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit calibration on validation data.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred_proba: Raw model probabilities
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            # For Platt scaling, we'd use logistic regression
            # but sklearn's CalibratedClassifierCV handles this
            raise NotImplementedError("Use isotonic method or implement Platt scaling")
        
        self.calibrator.fit(y_pred_proba, y_true)
        self.is_fitted = True
        
        print(f"✅ Calibrator fitted using {self.method} method")
        
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.
        
        Args:
            y_pred_proba: Raw model probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        calibrated = self.calibrator.predict(y_pred_proba)
        
        # Ensure valid probabilities [0, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        return calibrated
    
    def save(self, path: str):
        """Save calibrator to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Calibrator saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'XGBoostCalibrator':
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"📂 Calibrator loaded from {path}")
        return calibrator


# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================

def analyze_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Analyze how well-calibrated the model is.
    
    Returns calibration curve data showing predicted vs actual probabilities.
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    predicted_probs = []
    actual_probs = []
    counts = []
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        
        if i == n_bins - 1:  # Include upper bound in last bin
            in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i+1])
        
        if in_bin.sum() > 0:
            predicted_probs.append(y_pred_proba[in_bin].mean())
            actual_probs.append(y_true[in_bin].mean())
            counts.append(in_bin.sum())
        else:
            predicted_probs.append(bin_centers[i])
            actual_probs.append(0.0)
            counts.append(0)
    
    return {
        'bin_edges': bin_edges,
        'predicted_probs': np.array(predicted_probs),
        'actual_probs': np.array(actual_probs),
        'counts': np.array(counts)
    }


def print_calibration_report(y_true: np.ndarray, y_pred_proba: np.ndarray):
    """Print detailed calibration analysis."""
    print("\n" + "="*80)
    print("📊 CALIBRATION ANALYSIS")
    print("="*80)
    
    # Overall stats
    print(f"\nOverall Statistics:")
    print(f"   Total samples: {len(y_true)}")
    print(f"   Positive rate: {y_true.mean()*100:.1f}%")
    print(f"   Mean predicted prob: {y_pred_proba.mean()*100:.1f}%")
    
    # Calibration by confidence buckets
    print(f"\nCalibration by Confidence Bucket:")
    print(f"{'Predicted Range':<20} {'Count':<10} {'Actual Win Rate':<20} {'Calibration Gap':<20}")
    print("-" * 80)
    
    buckets = [
        (0.0, 0.5, "Very Low (0-50%)"),
        (0.5, 0.7, "Low (50-70%)"),
        (0.7, 0.8, "Medium (70-80%)"),
        (0.8, 0.85, "High (80-85%)"),
        (0.85, 0.9, "Very High (85-90%)"),
        (0.9, 1.0, "Extreme (90-100%)"),
    ]
    
    for min_prob, max_prob, label in buckets:
        mask = (y_pred_proba >= min_prob) & (y_pred_proba < max_prob)
        if max_prob == 1.0:  # Include upper bound for last bucket
            mask = (y_pred_proba >= min_prob) & (y_pred_proba <= max_prob)
        
        count = mask.sum()
        if count > 0:
            actual_rate = y_true[mask].mean() * 100
            predicted_rate = y_pred_proba[mask].mean() * 100
            gap = predicted_rate - actual_rate
            
            print(f"{label:<20} {count:<10} {actual_rate:>6.1f}% (pred: {predicted_rate:>5.1f}%) {gap:>+6.1f}%")
    
    print("\n" + "="*80)


# ============================================================================
# RECALIBRATION WORKFLOW
# ============================================================================

def recalibrate_model(
    model_path: str = "xgb_model.json",
    data_path: str = "processed_data.parquet",
    output_path: str = "xgb_calibrator.pkl",
    validation_split: float = 0.2
) -> XGBoostCalibrator:
    """
    Main workflow to recalibrate an existing XGBoost model.
    
    Steps:
    1. Load trained model
    2. Load training data
    3. Split into train/validation for calibration
    4. Fit isotonic regression calibrator
    5. Save calibrator
    
    Args:
        model_path: Path to trained XGBoost model
        data_path: Path to processed training data
        output_path: Where to save calibrator
        validation_split: Fraction of data to use for calibration
    """
    
    print("\n" + "="*80)
    print("🎯 MODEL RECALIBRATION WORKFLOW")
    print("="*80)
    
    # 1. Load model
    print(f"\n📂 Loading model from {model_path}...")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # 2. Load data
    print(f"📂 Loading data from {data_path}...")
    df = pl.read_parquet(data_path)
    print(f"   Loaded {len(df):,} samples")
    
    # Define features
    feature_cols = [
        "return_1c", "return_4c", "return_16c",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_4h", "volatility_24h", "vol_regime_change"
    ]
    
    # 3. Split for calibration (use recent data)
    split_idx = int(len(df) * (1 - validation_split))
    df_cal = df[split_idx:]
    
    print(f"\n✂️  Using last {len(df_cal):,} samples for calibration ({validation_split*100:.0f}%)")
    
    # 4. Get predictions
    X_cal = df_cal.select(feature_cols).to_numpy()
    y_cal = df_cal["signal_label"].to_numpy()
    
    dmatrix = xgb.DMatrix(X_cal, feature_names=feature_cols)
    y_pred_proba_raw = model.predict(dmatrix)
    
    print(f"\n🔍 Analyzing calibration BEFORE recalibration...")
    print_calibration_report(y_cal, y_pred_proba_raw)
    
    # 5. Fit calibrator
    print(f"\n🔧 Fitting isotonic regression calibrator...")
    calibrator = XGBoostCalibrator(method='isotonic')
    calibrator.fit(y_cal, y_pred_proba_raw)
    
    # 6. Check calibration AFTER
    y_pred_proba_calibrated = calibrator.transform(y_pred_proba_raw)
    
    print(f"\n🔍 Analyzing calibration AFTER recalibration...")
    print_calibration_report(y_cal, y_pred_proba_calibrated)
    
    # 7. Save
    calibrator.save(output_path)
    
    print(f"\n✅ Recalibration complete!")
    print(f"\nUsage in live system:")
    print(f"   1. Load calibrator: calibrator = XGBoostCalibrator.load('{output_path}')")
    print(f"   2. Get raw predictions: raw_proba = model.predict(data)")
    print(f"   3. Calibrate: calibrated_proba = calibrator.transform(raw_proba)")
    print(f"   4. Use calibrated probabilities for decisions")
    
    return calibrator


# ============================================================================
# INTEGRATION WITH LIVE SYSTEM
# ============================================================================

def apply_calibration_to_signals(
    signals_df: pl.DataFrame,
    calibrator_path: str = "xgb_calibrator.pkl"
) -> pl.DataFrame:
    """
    Apply calibration to existing signal predictions.
    
    Args:
        signals_df: DataFrame with 'pred_proba' column
        calibrator_path: Path to saved calibrator
        
    Returns:
        DataFrame with additional 'pred_proba_calibrated' column
    """
    # Load calibrator
    calibrator = XGBoostCalibrator.load(calibrator_path)
    
    # Get raw probabilities
    raw_proba = signals_df['pred_proba'].to_numpy()
    
    # Calibrate
    calibrated_proba = calibrator.transform(raw_proba)
    
    # Add to dataframe
    result = signals_df.with_columns([
        pl.Series("pred_proba_calibrated", calibrated_proba)
    ])
    
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "recalibrate":
        # Run recalibration
        calibrator = recalibrate_model(
            model_path="xgb_model.json",
            data_path="processed_data.parquet",
            output_path="xgb_calibrator.pkl",
            validation_split=0.2
        )
    else:
        print("="*80)
        print("MODEL CALIBRATION TOOLKIT")
        print("="*80)
        print("\nUsage:")
        print("  python calibrate_model.py recalibrate")
        print("\nThis will:")
        print("  1. Load your trained XGBoost model")
        print("  2. Analyze calibration issues")
        print("  3. Fit isotonic regression calibrator")
        print("  4. Save calibrator for use in live system")
        print("\nBased on multi-batch analysis:")
        print("  • Model predicts: 91.7% win rate")
        print("  • Actual results: 66.8% win rate")
        print("  • High confidence (≥85%) actually wins: 48.4%")
        print("\nCalibration will fix these prediction errors!")
        print("="*80)
