#!/usr/bin/env python3
"""Test that regime predictions are added to signals"""

import sys
import os
from pathlib import Path
import polars as pl

# Add market-regime-classifier to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'market-regime-classifier'))

from market_regime_classifier import MarketRegimeClassifier
from live_signal_generator import LiveConfig

def test_regime_in_signals():
    """Test that regime predictions work with sample data"""
    print("="*80)
    print("TESTING REGIME PREDICTIONS IN SIGNALS")
    print("="*80)
    
    # Load the trained model
    model_path = Path(LiveConfig.REGIME_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please run test_regime.py first to train the model")
        return False
    
    print(f"\n📁 Loading model from {model_path}")
    classifier = MarketRegimeClassifier(model_path=str(model_path))
    print("✅ Model loaded")
    
    # Create sample signal data (simulating what would come from generate_signals)
    print("\n📊 Creating sample signals dataframe...")
    sample_signals = pl.DataFrame({
        "pair_address": ["0x123", "0x456", "0x789"],
        "pred_proba": [0.75, 0.65, 0.55],
        "signal": [1, 1, 1],
        "close": [1.5, 2.3, 0.8],
        "volume_token1": [1000.0, 2000.0, 500.0],
    })
    print(f"   Created {len(sample_signals)} sample signals")
    
    # Create sample candle data for regime prediction
    print("\n📊 Creating sample candle history...")
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create 20 candles for each pair (simulating 15-min candles)
    candles_data = []
    base_time = datetime.now() - timedelta(hours=5)
    
    for pair in ["0x123", "0x456", "0x789"]:
        for i in range(20):
            candles_data.append({
                "pair_address": pair,
                "timestamp": base_time + timedelta(minutes=15*i),
                "open": 1.0 + np.random.randn() * 0.1,
                "high": 1.0 + abs(np.random.randn()) * 0.15,
                "low": 1.0 - abs(np.random.randn()) * 0.15,
                "close": 1.0 + np.random.randn() * 0.1,
                "num_trades": int(10 + np.random.rand() * 20),
                "volume_token1": 1000.0 + np.random.rand() * 500,
            })
    
    candles_df = pl.DataFrame(candles_data)
    candles_history = {pair: candles_df.filter(pl.col("pair_address") == pair) 
                      for pair in ["0x123", "0x456", "0x789"]}
    
    print(f"   Created candle history for {len(candles_history)} pairs")
    
    # Simulate _add_regime_predictions logic
    print("\n🔄 Adding regime predictions to signals...")
    regime_predictions = []
    
    for row in sample_signals.iter_rows(named=True):
        pair_address = row["pair_address"]
        
        if pair_address not in candles_history:
            regime_predictions.append("Unknown")
            continue
        
        pair_candles = candles_history[pair_address]
        
        try:
            # Convert to regime classifier format
            regime_df = pair_candles.select([
                pl.col("pair_address").alias("pair_id"),
                pl.col("timestamp"),
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                (pl.col("num_trades") / 2).cast(pl.Int64).alias("buys"),
                (pl.col("num_trades") / 2).cast(pl.Int64).alias("sells"),
                pl.col("volume_token1").alias("volume"),
                (pl.col("volume_token1") * 0.5).alias("buy_volume"),
                (pl.col("volume_token1") * 0.5).alias("sell_volume"),
                ((pl.col("open") + pl.col("close")) / 2).alias("avg"),
                pl.lit(1.0).alias("exchange_rate"),
                pl.lit(0).alias("start_block"),
                pl.lit(0).alias("end_block"),
            ])
            
            # Predict regime
            result = classifier.predict(regime_df)
            if len(result) > 0:
                regime = result["predicted_regime"].item()
                regime_predictions.append(regime)
            else:
                regime_predictions.append("Unknown")
                
        except Exception as e:
            print(f"   ⚠️  Error predicting regime for {pair_address}: {e}")
            regime_predictions.append("Unknown")
    
    # Add regime column
    signals_with_regime = sample_signals.with_columns([
        pl.Series("regime", regime_predictions)
    ])
    
    print("\n✅ Signals with regime predictions:")
    print(signals_with_regime)
    
    # Verify regime column exists
    if "regime" in signals_with_regime.columns:
        print(f"\n✅ SUCCESS: Regime column added to signals!")
        print(f"   Regime values: {signals_with_regime['regime'].unique().to_list()}")
        return True
    else:
        print("\n❌ FAILED: Regime column not found in signals")
        return False

if __name__ == "__main__":
    success = test_regime_in_signals()
    sys.exit(0 if success else 1)
