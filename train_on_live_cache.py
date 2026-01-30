#!/usr/bin/env python3
"""
Train XGBoost Model on Live Cache Data
=======================================

Trains a new model using candles collected from the live system (envio cache)
and compares performance with the original model.
"""

import pickle
import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Cache file
    CACHE_FILE = "live_candles_cache.pkl"
    
    # Feature columns (same as original model)
    FEATURE_COLS = [
        "return_1c", "return_4c", "return_16c",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_4h", "volatility_24h", "vol_regime_change"
    ]
    
    TARGET_COL = "signal_label"
    
    # Target definition
    TARGET_GAIN = 0.02  # +2% = win
    TARGET_LOSS = -0.01  # -1% = loss
    HOLDING_PERIOD_CANDLES = 16  # 4 hours (16 * 15min)
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'seed': 42,
        'tree_method': 'hist',
    }
    
    NUM_BOOST_ROUNDS = 500
    EARLY_STOPPING_ROUNDS = 50
    
    # Data split (time-based)
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create ML features from candles"""
    
    print("Creating features...")
    
    # Returns
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1)) - 1).alias('return_1c'),
        ((pl.col('close') / pl.col('close').shift(4)) - 1).alias('return_4c'),
        ((pl.col('close') / pl.col('close').shift(16)) - 1).alias('return_16c'),
    ])
    
    # EMA cross signal
    ema_fast = pl.col('close').ewm_mean(span=4, adjust=False)
    ema_slow = pl.col('close').ewm_mean(span=16, adjust=False)
    df = df.with_columns([
        ((ema_fast - ema_slow) / ema_slow).alias('ema_cross_signal')
    ])
    
    # Range normalized
    df = df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_normalized')
    ])
    
    # Buy/sell ratio (using volume as proxy)
    df = df.with_columns([
        (pl.col('volume_token0') / (pl.col('volume_token1') + 1e-10)).alias('buy_sell_ratio')
    ])
    
    # Volume z-score
    vol_mean = pl.col('volume_token0').rolling_mean(window_size=96)
    vol_std = pl.col('volume_token0').rolling_std(window_size=96)
    df = df.with_columns([
        ((pl.col('volume_token0') - vol_mean) / (vol_std + 1e-10)).alias('volume_zscore')
    ])
    
    # Trade acceleration
    df = df.with_columns([
        (pl.col('num_trades') - pl.col('num_trades').shift(1)).alias('trade_accel')
    ])
    
    # Volatility
    vol_4h = pl.col('close').rolling_std(window_size=16) / pl.col('close')
    vol_24h = pl.col('close').rolling_std(window_size=96) / pl.col('close')
    df = df.with_columns([
        vol_4h.alias('volatility_4h'),
        vol_24h.alias('volatility_24h'),
        ((vol_4h - vol_24h) / (vol_24h + 1e-10)).alias('vol_regime_change')
    ])
    
    return df


def create_target(df: pl.DataFrame) -> pl.DataFrame:
    """Create target variable based on future returns"""
    
    print("Creating target variable...")
    
    # Calculate future max/min returns in holding period
    future_returns = []
    for i in range(1, Config.HOLDING_PERIOD_CANDLES + 1):
        future_returns.append(
            ((pl.col('close').shift(-i) / pl.col('close')) - 1).alias(f'future_return_{i}')
        )
    
    df = df.with_columns(future_returns)
    
    # Max gain and max loss
    return_cols = [f'future_return_{i}' for i in range(1, Config.HOLDING_PERIOD_CANDLES + 1)]
    df = df.with_columns([
        pl.max_horizontal(return_cols).alias('max_gain'),
        pl.min_horizontal(return_cols).alias('max_loss')
    ])
    
    # Label: 1 if hit target gain before target loss, 0 otherwise
    df = df.with_columns([
        pl.when(pl.col('max_gain') >= Config.TARGET_GAIN)
        .then(1)
        .otherwise(0)
        .alias(Config.TARGET_COL)
    ])
    
    # Drop future return columns
    df = df.drop([col for col in df.columns if col.startswith('future_return_')])
    df = df.drop(['max_gain', 'max_loss'])
    
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cache_data() -> pl.DataFrame:
    """Load all candles from cache and combine into single DataFrame"""
    
    print(f"\n{'='*80}")
    print("LOADING LIVE CACHE DATA")
    print(f"{'='*80}\n")
    
    # Load cache
    print(f"Loading cache from {Config.CACHE_FILE}...")
    with open(Config.CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    
    print(f"✓ Loaded {len(cache):,} pairs from cache")
    
    # Combine all pair candles
    all_candles = []
    for pair_address, pair_df in cache.items():
        # Add pair identifier
        pair_df = pair_df.with_columns(pl.lit(pair_address).alias('pair_address'))
        all_candles.append(pair_df)
    
    # Concatenate all
    df = pl.concat(all_candles)
    
    print(f"✓ Combined {len(df):,} total candles")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Pairs: {df['pair_address'].n_unique():,}")
    
    return df


def prepare_training_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load, process, and split data for training"""
    
    # Load cache
    df = load_cache_data()
    
    # Create features per pair
    print("\nProcessing features per pair...")
    processed_pairs = []
    
    for i, (pair_address, pair_df) in enumerate(df.group_by('pair_address')):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1} pairs...")
        
        # Sort by timestamp
        pair_df = pair_df.sort('timestamp')
        
        # Create features
        pair_df = create_features(pair_df)
        
        # Create target
        pair_df = create_target(pair_df)
        
        processed_pairs.append(pair_df)
    
    df = pl.concat(processed_pairs)
    
    # Drop nulls (from rolling windows and future returns)
    df = df.drop_nulls(subset=Config.FEATURE_COLS + [Config.TARGET_COL])
    
    print(f"\n✓ Processed data: {len(df):,} samples")
    
    # Time-based split
    df = df.sort('timestamp')
    n = len(df)
    
    train_end_idx = int(n * Config.TRAIN_SPLIT)
    val_end_idx = int(n * (Config.TRAIN_SPLIT + Config.VAL_SPLIT))
    
    df_train = df[:train_end_idx]
    df_val = df[train_end_idx:val_end_idx]
    df_test = df[val_end_idx:]
    
    print(f"\n📊 Time-based split:")
    print(f"  Train: {len(df_train):,} samples ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
    print(f"  Val:   {len(df_val):,} samples ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
    print(f"  Test:  {len(df_test):,} samples ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
    
    # Class distribution
    print(f"\n📈 Class distribution:")
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        pos_rate = split_df[Config.TARGET_COL].mean()
        print(f"  {split_name}: {pos_rate:.2%} positive class")
    
    return df_train, df_val, df_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(df_train: pl.DataFrame, df_val: pl.DataFrame) -> xgb.Booster:
    """Train XGBoost model"""
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'='*80}\n")
    
    # Prepare data
    X_train = df_train.select(Config.FEATURE_COLS).to_numpy()
    y_train = df_train[Config.TARGET_COL].to_numpy()
    
    X_val = df_val.select(Config.FEATURE_COLS).to_numpy()
    y_val = df_val[Config.TARGET_COL].to_numpy()
    
    # Calculate scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = Config.XGB_PARAMS.copy()
    params['scale_pos_weight'] = pos_weight
    
    print(f"Class imbalance: {pos_weight:.2f}:1 (negative:positive)")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=Config.FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=Config.FEATURE_COLS)
    
    # Train
    print("\nTraining XGBoost...")
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=Config.NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )
    
    print(f"\n✓ Training complete (best iteration: {model.best_iteration})")
    
    return model


def evaluate_model(model: xgb.Booster, df_test: pl.DataFrame) -> Dict:
    """Evaluate model on test set"""
    
    print(f"\n{'='*80}")
    print("MODEL EVALUATION")
    print(f"{'='*80}\n")
    
    # Prepare data
    X_test = df_test.select(Config.FEATURE_COLS).to_numpy()
    y_test = df_test[Config.TARGET_COL].to_numpy()
    
    dtest = xgb.DMatrix(X_test, feature_names=Config.FEATURE_COLS)
    
    # Predict
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test AUC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Feature importance
    print(f"\nTop 10 Features:")
    importance = model.get_score(importance_type='gain')
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features[:10]:
        print(f"  {feat}: {score:.2f}")
    
    return {
        'auc': auc,
        'model': model,
        'predictions': y_pred_proba
    }


# ============================================================================
# BACKTESTING
# ============================================================================

def simple_backtest(model: xgb.Booster, df_test: pl.DataFrame, threshold: float = 0.7) -> Dict:
    """Simple backtest on test set"""
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING (Threshold: {threshold})")
    print(f"{'='*80}\n")
    
    # Prepare data
    X_test = df_test.select(Config.FEATURE_COLS).to_numpy()
    dtest = xgb.DMatrix(X_test, feature_names=Config.FEATURE_COLS)
    
    # Predict
    y_pred_proba = model.predict(dtest)
    
    # Add predictions to dataframe
    df_results = df_test.with_columns([
        pl.Series('pred_proba', y_pred_proba)
    ])
    
    # Filter by threshold
    df_signals = df_results.filter(pl.col('pred_proba') >= threshold)
    
    print(f"Signals generated: {len(df_signals):,} / {len(df_test):,} ({len(df_signals)/len(df_test)*100:.2f}%)")
    
    if len(df_signals) == 0:
        print("⚠️  No signals generated at this threshold")
        return {}
    
    # Calculate metrics
    total_signals = len(df_signals)
    wins = (df_signals[Config.TARGET_COL] == 1).sum()
    losses = total_signals - wins
    win_rate = wins / total_signals
    
    # Simulate returns
    returns = []
    for row in df_signals.iter_rows(named=True):
        if row[Config.TARGET_COL] == 1:
            returns.append(Config.TARGET_GAIN)
        else:
            returns.append(Config.TARGET_LOSS)
    
    returns = np.array(returns)
    total_return = returns.sum()
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 24 * 4)  # 15-min periods
    
    print(f"\n📊 Backtest Results:")
    print(f"  Win Rate: {win_rate:.2%} ({wins}/{total_signals})")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Avg Return: {returns.mean()*100:.4f}%")
    print(f"  Std Dev: {returns.std()*100:.4f}%")
    
    return {
        'total_signals': total_signals,
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe,
        'avg_return': returns.mean(),
        'std_return': returns.std()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    print(f"\n{'='*80}")
    print("XGBoost Training on Live Cache Data")
    print(f"{'='*80}")
    
    # Prepare data
    df_train, df_val, df_test = prepare_training_data()
    
    # Train model
    model = train_model(df_train, df_val)
    
    # Save model
    model_path = 'xgb_live_cache_model.json'
    model.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Evaluate
    eval_results = evaluate_model(model, df_test)
    
    # Backtest at different thresholds
    for threshold in [0.5, 0.6, 0.7, 0.73, 0.8]:
        simple_backtest(model, df_test, threshold=threshold)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("1. Compare this model with original: xgb_model.json")
    print("2. Test with calibration if needed")
    print("3. Deploy if performance is better")


if __name__ == "__main__":
    main()
