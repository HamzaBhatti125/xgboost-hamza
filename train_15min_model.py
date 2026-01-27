#!/usr/bin/env python3
"""
15-Minute Candle Model Training
================================

Trains XGBoost model on 15-minute candle data using chunked processing
to handle the large dataset (370M rows) efficiently.
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration for 15-minute candles"""
    
    # Data path
    DATA_PATH = "Files/candles-15m.parquet"
    MODEL_OUTPUT = "xgboost_15min_model.json"
    
    # Chunk size for processing (adjust based on available memory)
    CHUNK_SIZE = 1_000_000  # 1M rows per chunk
    
    # Sample size for training (to make it manageable)
    # We'll use recent data + stratified sampling
    MAX_TRAINING_SAMPLES = 10_000_000  # 10M rows (~1GB)
    
    # Feature engineering parameters (adjusted for 15-min timeframe)
    FEATURE_WINDOWS = {
        'ma_short': 4,      # 1 hour (4 * 15min)
        'ma_medium': 16,    # 4 hours (16 * 15min)
        'ma_long': 96,      # 24 hours (96 * 15min)
        'volume_window': 24,  # 6 hours
        'volatility_window': 96,  # 24 hours
    }
    
    # Train/test split (time-based)
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'reg:squarederror',  # Predicting price or return
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'seed': 42,
        'tree_method': 'hist',
        'n_estimators': 200,
    }
    
    # Pairs to focus on (optional - set to None for all pairs)
    FOCUS_PAIRS = [1, 2, 3, 4, 5]  # Top 5 pairs by volume


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and features for 15-min candles"""
    print("  Creating features...")
    
    df = df.copy()
    
    # Sort by pair_id and timestamp
    df = df.sort_values(['pair_id', 'timestamp'])
    
    # Basic price features
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_range_pct'] = (df['high'] - df['low']) / df['open']
    
    # Volume features
    df['volume_change'] = df.groupby('pair_id')['volume'].pct_change()
    df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-10)
    
    # Moving averages (grouped by pair_id)
    for window_name, window_size in Config.FEATURE_WINDOWS.items():
        if 'ma_' in window_name:
            df[window_name] = df.groupby('pair_id')['close'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
    
    # Price relative to moving averages
    df['price_vs_ma_short'] = (df['close'] - df['ma_short']) / df['ma_short']
    df['price_vs_ma_medium'] = (df['close'] - df['ma_medium']) / df['ma_medium']
    df['price_vs_ma_long'] = (df['close'] - df['ma_long']) / df['ma_long']
    
    # Volatility (rolling std of returns)
    df['returns'] = df.groupby('pair_id')['close'].pct_change()
    df['volatility'] = df.groupby('pair_id')['returns'].transform(
        lambda x: x.rolling(window=Config.FEATURE_WINDOWS['volatility_window'], min_periods=1).std()
    )
    
    # Volume indicators
    df['volume_ma'] = df.groupby('pair_id')['volume'].transform(
        lambda x: x.rolling(window=Config.FEATURE_WINDOWS['volume_window'], min_periods=1).mean()
    )
    df['volume_vs_ma'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1e-10)
    
    # Lag features (previous candle info)
    for lag in [1, 4, 16]:  # 15min, 1h, 4h ago
        df[f'close_lag_{lag}'] = df.groupby('pair_id')['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df.groupby('pair_id')['volume'].shift(lag)
    
    # Momentum indicators
    df['momentum_4'] = df['close'] - df['close_lag_4']  # 1h momentum
    df['momentum_16'] = df['close'] - df['close_lag_16']  # 4h momentum
    
    # Target: predict next candle's return (or price)
    df['target_return'] = df.groupby('pair_id')['close'].shift(-1) / df['close'] - 1
    df['target_price'] = df.groupby('pair_id')['close'].shift(-1)
    
    return df


# ============================================================================
# DATA LOADING WITH CHUNKED PROCESSING
# ============================================================================

def load_data_smartly() -> pd.DataFrame:
    """Load data intelligently with chunking and sampling"""
    print(f"\n{'='*80}")
    print("STEP 1: SMART DATA LOADING")
    print(f"{'='*80}")
    
    parquet_file = pq.ParquetFile(Config.DATA_PATH)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"\n📊 Dataset info:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Columns: {parquet_file.schema.names}")
    
    # Strategy: Load most recent data + sample older data
    # This ensures we have recent patterns while still learning from history
    
    print(f"\n⚙️  Loading strategy:")
    print(f"  1. Load most recent {Config.MAX_TRAINING_SAMPLES:,} rows")
    print(f"  2. Focus on pairs: {Config.FOCUS_PAIRS if Config.FOCUS_PAIRS else 'All'}")
    
    # Read in chunks from the end (most recent data)
    chunks = []
    rows_loaded = 0
    target_rows = Config.MAX_TRAINING_SAMPLES
    
    # Determine how many row groups to read
    num_row_groups = parquet_file.num_row_groups
    print(f"  Total row groups: {num_row_groups}")
    
    # Read from the last row groups (most recent data)
    for i in range(num_row_groups - 1, -1, -1):
        if rows_loaded >= target_rows:
            break
            
        row_group = parquet_file.read_row_group(i)
        chunk_df = row_group.to_pandas()
        
        # Filter by focus pairs if specified
        if Config.FOCUS_PAIRS:
            chunk_df = chunk_df[chunk_df['pair_id'].isin(Config.FOCUS_PAIRS)]
        
        chunks.append(chunk_df)
        rows_loaded += len(chunk_df)
        
        if (i % 10 == 0):
            print(f"  Loaded {rows_loaded:,} rows from {num_row_groups - i} row groups...")
    
    print(f"\n✓ Loaded {rows_loaded:,} rows")
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(['pair_id', 'timestamp']).reset_index(drop=True)
    
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Pairs: {sorted(df['pair_id'].unique())}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def split_data(df: pd.DataFrame) -> tuple:
    """Time-based train/val/test split (NO SHUFFLE for time series)"""
    print(f"\n{'='*80}")
    print("STEP 2: FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    # Create features
    df = create_features(df)
    
    # Remove rows with NaN in target
    df = df.dropna(subset=['target_return', 'target_price'])
    
    print(f"\n✓ Features created. Shape: {df.shape}")
    
    # Define feature columns
    feature_cols = [
        'open', 'close', 'high', 'low', 'volume',
        'price_change_pct', 'high_low_range_pct',
        'buy_sell_ratio', 'volume_change', 'volume_vs_ma',
        'price_vs_ma_short', 'price_vs_ma_medium', 'price_vs_ma_long',
        'volatility', 'momentum_4', 'momentum_16',
        'close_lag_1', 'close_lag_4', 'close_lag_16',
        'volume_lag_1', 'volume_lag_4', 'volume_lag_16',
        'buys', 'sells',
    ]
    
    # Remove any features with NaN
    df = df.dropna(subset=feature_cols)
    
    print(f"  After removing NaN: {len(df):,} samples")
    
    # Time-based split
    print(f"\n{'='*80}")
    print("STEP 3: TRAIN/VAL/TEST SPLIT")
    print(f"{'='*80}")
    
    n = len(df)
    train_end = int(n * (1 - Config.TEST_SIZE - Config.VAL_SIZE))
    val_end = int(n * (1 - Config.TEST_SIZE))
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"\n📊 Data split (time-based, no shuffle):")
    print(f"  Train: {len(df_train):,} samples ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
    print(f"  Val:   {len(df_val):,} samples ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
    print(f"  Test:  {len(df_test):,} samples ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
    
    # Prepare feature matrices
    X_train = df_train[feature_cols].values
    y_train = df_train['target_return'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['target_return'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['target_return'].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, X_val, y_train, y_val, feature_cols):
    """Train XGBoost model"""
    print(f"\n{'='*80}")
    print("STEP 4: MODEL TRAINING")
    print(f"{'='*80}")
    
    print("\n🎯 Training XGBoost regressor for 15-minute candle prediction")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train):,}")
    
    # Create model
    model = xgb.XGBRegressor(**Config.XGB_PARAMS)
    
    # Train with early stopping on validation set
    print("\n🏋️  Training in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20
    )
    
    print(f"\n✓ Training complete!")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:.<30} {row['importance']:.4f}")
    
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, split_name="Test"):
    """Evaluate model performance"""
    print(f"\n{'='*80}")
    print(f"STEP 5: MODEL EVALUATION ({split_name})")
    print(f"{'='*80}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # R² score (coefficient of determination)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Direction accuracy (for returns prediction)
    direction_correct = np.sum((y_test > 0) == (y_pred > 0))
    direction_accuracy = direction_correct / len(y_test)
    
    print(f"\n📈 Regression Metrics:")
    print(f"  MSE:  {mse:.8f}")
    print(f"  RMSE: {rmse:.8f}")
    print(f"  MAE:  {mae:.8f}")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\n🎯 Direction Prediction:")
    print(f"  Accuracy: {direction_accuracy:.2%}")
    print(f"  (% of times we predicted direction correctly)")
    
    # Distribution analysis
    print(f"\n📊 Prediction Distribution:")
    print(f"  Actual mean return: {y_test.mean():.6f}")
    print(f"  Predicted mean return: {y_pred.mean():.6f}")
    print(f"  Actual std: {y_test.std():.6f}")
    print(f"  Predicted std: {y_pred.std():.6f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'predictions': y_pred
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print(f"\n{'='*80}")
    print("15-MINUTE CANDLE MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now()}")
    
    # Load data
    df = load_data_smartly()
    
    # Split and prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = split_data(df)
    
    # Train model
    model = train_model(X_train, X_val, y_train, y_val, feature_cols)
    
    # Evaluate
    val_results = evaluate_model(model, X_val, y_val, "Validation")
    test_results = evaluate_model(model, X_test, y_test, "Test")
    
    # Save model
    print(f"\n{'='*80}")
    print("STEP 6: SAVING MODEL")
    print(f"{'='*80}")
    
    model.save_model(Config.MODEL_OUTPUT)
    print(f"\n✓ Model saved to: {Config.MODEL_OUTPUT}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now()}")
    print(f"\n📊 Final Performance Summary:")
    print(f"  Validation RMSE: {val_results['rmse']:.8f}")
    print(f"  Test RMSE: {test_results['rmse']:.8f}")
    print(f"  Test Direction Accuracy: {test_results['direction_accuracy']:.2%}")
    print(f"\n✅ Model ready for use with Envio 15-minute candles!")


if __name__ == "__main__":
    main()
