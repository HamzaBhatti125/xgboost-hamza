#!/usr/bin/env python3
"""
Hybrid Training: Historical Data + Envio HyperSync Stream
========================================================

Trains XGBoost model on a combination of:
1. Historical data from parquet files
2. Real-time data streamed from Envio HyperSync

This approach ensures the model learns from both historical patterns
and current market conditions.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

from hypersync_streamer import HyperSyncStreamer, SwapToCandleConverter
from onchain_signal_system import (
    Config as DataConfig,
    filter_base_chain,
    apply_quality_filters,
    engineer_features,
    create_barrier_labels,
    load_pair_universe_lazy,
    load_candles_for_pairs
)


class TrainingConfig:
    """Training configuration"""
    
    # Historical data paths
    PROCESSED_DATA_PATH = "processed_data.parquet"
    PAIR_UNIVERSE_PATH = "Files/pair-universe"
    CANDLES_PATH = "Files/candles-1d.parquet"
    
    # Envio HyperSync
    ENVIO_API_TOKEN = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
    STREAM_DURATION_SECONDS = 300  # 5 minutes of streaming
    
    # Train/test split (time-based)
    TRAIN_END_DATE = "2024-06-30"
    VAL_END_DATE = "2024-09-30"
    
    # Feature columns
    FEATURE_COLS = [
        "return_1d", "return_3d", "return_7d",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_7d", "volatility_14d", "vol_regime_change"
    ]
    
    TARGET_COL = "signal_label"
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'scale_pos_weight': None,  # Will calculate
        'seed': 42,
        'tree_method': 'hist',
    }
    
    NUM_BOOST_ROUNDS = 500
    EARLY_STOPPING_ROUNDS = 50
    
    # Model output
    MODEL_OUTPUT_PATH = "xgb_model.json"


def load_historical_data() -> Optional[pl.DataFrame]:
    """Load historical processed data if available"""
    processed_path = Path(TrainingConfig.PROCESSED_DATA_PATH)
    
    if processed_path.exists():
        print(f"\n📂 Loading historical data from {processed_path}")
        try:
            df = pl.read_parquet(processed_path)
            print(f"✓ Loaded {len(df):,} historical samples")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            print(f"⚠️  Error loading historical data: {e}")
            return None
    else:
        print(f"⚠️  No historical processed data found at {processed_path}")
        print("   Will use only streamed data for training")
        return None


def stream_envio_data(duration_seconds: int = 300) -> pl.DataFrame:
    """Stream data from Envio HyperSync and convert to candles"""
    print(f"\n{'='*80}")
    print("STREAMING DATA FROM ENVIO HYPERSYNC")
    print(f"{'='*80}")
    print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"API Token: {TrainingConfig.ENVIO_API_TOKEN[:20]}...")
    
    collected_candles = []
    
    def callback(candles: pl.DataFrame):
        """Callback when candles are collected"""
        nonlocal collected_candles
        collected_candles.append(candles)
        print(f"  📊 Collected {len(candles)} candles (total: {sum(len(c) for c in collected_candles)})")
    
    # Start streaming
    streamer = HyperSyncStreamer(
        api_token=TrainingConfig.ENVIO_API_TOKEN,
        chain="base",
        signal_callback=callback
    )
    
    try:
        print("\n🚀 Starting HyperSync stream...")
        candles = streamer.start_streaming(duration_seconds=duration_seconds)
        
        # Combine all collected candles
        if collected_candles:
            candles = pl.concat(collected_candles)
        
        if len(candles) == 0:
            print("⚠️  No candles collected from stream")
            return pl.DataFrame()
        
        print(f"\n✓ Streamed {len(candles):,} candles from HyperSync")
        return candles
        
    except Exception as e:
        print(f"❌ Error streaming data: {e}")
        import traceback
        traceback.print_exc()
        return pl.DataFrame()


def process_streamed_data(candles: pl.DataFrame) -> pl.DataFrame:
    """Process streamed candles through the same pipeline as historical data"""
    if len(candles) == 0:
        return pl.DataFrame()
    
    print(f"\n{'='*80}")
    print("PROCESSING STREAMED DATA")
    print(f"{'='*80}")
    
    # Ensure required columns exist
    required_cols = {
        "pair_id": 0,
        "timestamp": datetime.now(),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0,
        "buys": 0, "sells": 0
    }
    
    for col, default_val in required_cols.items():
        if col not in candles.columns:
            if isinstance(default_val, (int, float)):
                candles = candles.with_columns(pl.lit(default_val).alias(col))
            elif isinstance(default_val, datetime):
                candles = candles.with_columns(pl.lit(datetime.now()).alias(col))
    
    # Engineer features
    print("  🔧 Engineering features...")
    try:
        candles_feat = engineer_features(candles)
    except Exception as e:
        print(f"  ⚠️  Error in feature engineering: {e}")
        # Use simplified features
        candles_feat = candles
    
    # Create labels (barrier-based)
    print("  🏷️  Creating labels...")
    try:
        candles_labeled = create_barrier_labels(candles_feat)
    except Exception as e:
        print(f"  ⚠️  Error creating labels: {e}")
        # Add dummy labels
        candles_labeled = candles_feat.with_columns([
            pl.lit(0).alias(TrainingConfig.TARGET_COL)
        ])
    
    # Drop nulls
    candles_clean = candles_labeled.drop_nulls()
    
    print(f"✓ Processed {len(candles_clean):,} samples from streamed data")
    
    return candles_clean


def combine_datasets(historical: Optional[pl.DataFrame], streamed: pl.DataFrame) -> pl.DataFrame:
    """Combine historical and streamed datasets"""
    print(f"\n{'='*80}")
    print("COMBINING DATASETS")
    print(f"{'='*80}")
    
    datasets = []
    
    if historical is not None and len(historical) > 0:
        print(f"  Historical: {len(historical):,} samples")
        datasets.append(historical)
    
    if len(streamed) > 0:
        print(f"  Streamed: {len(streamed):,} samples")
        datasets.append(streamed)
    
    if not datasets:
        raise ValueError("No data available for training!")
    
    # Combine
    combined = pl.concat(datasets)
    
    # Remove duplicates (if any)
    combined = combined.unique(subset=["pair_id", "timestamp"], keep="first")
    
    # Sort by timestamp
    combined = combined.sort("timestamp")
    
    print(f"\n✓ Combined dataset: {len(combined):,} total samples")
    print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f"  Unique pairs: {combined['pair_id'].n_unique():,}")
    
    # Check label distribution
    if TrainingConfig.TARGET_COL in combined.columns:
        pos_rate = combined[TrainingConfig.TARGET_COL].mean() * 100
        pos_count = combined[TrainingConfig.TARGET_COL].sum()
        print(f"  Positive labels: {pos_count:,} ({pos_rate:.2f}%)")
    
    return combined


def split_data(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Time-based train/val/test split"""
    print(f"\n{'='*80}")
    print("TIME-BASED DATA SPLIT")
    print(f"{'='*80}")
    
    train_end = datetime(2024, 6, 30)
    val_end = datetime(2024, 9, 30)
    
    df_train = df.filter(pl.col("timestamp") <= train_end)
    df_val = df.filter(
        (pl.col("timestamp") > train_end) & 
        (pl.col("timestamp") <= val_end)
    )
    df_test = df.filter(pl.col("timestamp") > val_end)
    
    print(f"\n📊 Split results:")
    print(f"  Train: {len(df_train):,} samples")
    if len(df_train) > 0:
        print(f"    Range: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    
    print(f"  Val:   {len(df_val):,} samples")
    if len(df_val) > 0:
        print(f"    Range: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    
    print(f"  Test:  {len(df_test):,} samples")
    if len(df_test) > 0:
        print(f"    Range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    
    # Label distribution
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        if len(split) > 0 and TrainingConfig.TARGET_COL in split.columns:
            pos_rate = split[TrainingConfig.TARGET_COL].mean() * 100
            print(f"  {name} positive rate: {pos_rate:.2f}%")
    
    return df_train, df_val, df_test


def train_model(df_train: pl.DataFrame, df_val: pl.DataFrame) -> xgb.Booster:
    """Train XGBoost model"""
    print(f"\n{'='*80}")
    print("TRAINING XGBOOST MODEL")
    print(f"{'='*80}")
    
    # Prepare features
    X_train = df_train.select(TrainingConfig.FEATURE_COLS).to_numpy()
    y_train = df_train[TrainingConfig.TARGET_COL].to_numpy()
    
    X_val = df_val.select(TrainingConfig.FEATURE_COLS).to_numpy()
    y_val = df_val[TrainingConfig.TARGET_COL].to_numpy()
    
    print(f"\n  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Features: {len(TrainingConfig.FEATURE_COLS)}")
    
    # Calculate scale_pos_weight for class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
        TrainingConfig.XGB_PARAMS['scale_pos_weight'] = scale_pos_weight
        print(f"  Class imbalance ratio: {scale_pos_weight:.2f}")
    
    # Create DMatrices
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=TrainingConfig.FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=TrainingConfig.FEATURE_COLS)
    
    # Train
    print(f"\n  Training with {TrainingConfig.NUM_BOOST_ROUNDS} rounds...")
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        TrainingConfig.XGB_PARAMS,
        dtrain,
        num_boost_round=TrainingConfig.NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=TrainingConfig.EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )
    
    print(f"\n✓ Model training complete")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    
    return model


def evaluate_model(model: xgb.Booster, df: pl.DataFrame, name: str):
    """Evaluate model performance"""
    print(f"\n{'='*80}")
    print(f"EVALUATION: {name}")
    print(f"{'='*80}")
    
    X = df.select(TrainingConfig.FEATURE_COLS).to_numpy()
    y_true = df[TrainingConfig.TARGET_COL].to_numpy()
    
    dmatrix = xgb.DMatrix(X, feature_names=TrainingConfig.FEATURE_COLS)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n  AUC-ROC: {auc:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}")
    print(f"  Recall: {report['1']['recall']:.4f}")
    print(f"  F1-Score: {report['1']['f1-score']:.4f}")
    
    return {
        'auc': auc,
        'predictions': y_pred_proba,
        'report': report
    }


def main():
    """Main training pipeline"""
    print(f"\n{'#'*80}")
    print("HYBRID TRAINING: HISTORICAL + ENVIO HYPERSYNC")
    print(f"{'#'*80}\n")
    
    try:
        # Step 1: Load historical data
        historical = load_historical_data()
        
        # Step 2: Stream new data from Envio
        streamed_candles = stream_envio_data(duration_seconds=TrainingConfig.STREAM_DURATION_SECONDS)
        
        # Step 3: Process streamed data
        streamed_processed = process_streamed_data(streamed_candles)
        
        # Step 4: Combine datasets
        combined = combine_datasets(historical, streamed_processed)
        
        # Step 5: Split data
        df_train, df_val, df_test = split_data(combined)
        
        if len(df_train) == 0:
            print("\n❌ No training data available!")
            return 1
        
        # Step 6: Train model
        model = train_model(df_train, df_val)
        
        # Step 7: Evaluate
        if len(df_val) > 0:
            evaluate_model(model, df_val, "VALIDATION")
        
        if len(df_test) > 0:
            evaluate_model(model, df_test, "TEST")
        
        # Step 8: Save model
        model_path = Path(TrainingConfig.MODEL_OUTPUT_PATH)
        model.save_model(str(model_path))
        print(f"\n✓ Saved model to: {model_path}")
        
        print(f"\n{'='*80}")
        print("✅ HYBRID TRAINING COMPLETE!")
        print(f"{'='*80}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
