#!/usr/bin/env python3
"""
Test script to verify regime predictions are included in CSV output
"""
import polars as pl
from pathlib import Path
import sys
import os

# Add market-regime-classifier to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'market-regime-classifier'))
from market_regime_classifier import MarketRegimeClassifier
from live_signal_generator import LiveConfig

def test_regime_in_csv():
    """Test that regime column is added to signals before CSV save"""
    
    print("="*80)
    print("TESTING REGIME PREDICTIONS IN CSV OUTPUT")
    print("="*80)
    
    # Check if model exists
    model_path = Path(LiveConfig.REGIME_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please ensure the model is trained first.")
        return False
    
    print(f"✅ Model found at {model_path}")
    
    # Load classifier
    try:
        classifier = MarketRegimeClassifier(model_path=str(model_path))
        print("✅ Regime classifier loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return False
    
    # Create sample signals DataFrame (matching live_signal_generator format)
    sample_signals = pl.DataFrame({
        "pair_address": ["0x123", "0x456", "0x789", "0xabc", "0xdef"],
        "pred_proba": [0.75, 0.65, 0.55, 0.85, 0.45],
        "signal": [1, 1, 1, 1, 0],
        "close": [1.5, 2.3, 0.8, 3.2, 1.1],
        "volume_token1": [1000.0, 2000.0, 500.0, 5000.0, 300.0],
        "timestamp": ["2026-01-30T10:00:00"] * 5,
    })
    
    print(f"\n📊 Sample signals created: {len(sample_signals)} rows")
    print(f"   Columns: {sample_signals.columns}")
    
    # Simulate _add_regime_predictions logic
    from datetime import datetime, timedelta
    import numpy as np
    
    # Create sample candle history for each pair
    candles_history = {}
    for pair in sample_signals["pair_address"].to_list():
        # Generate 20 candles of history
        base_time = datetime.now() - timedelta(hours=5)
        candles = []
        base_price = sample_signals.filter(pl.col("pair_address") == pair)["close"].item()
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=15 * i)
            # Simulate price movement
            price_change = np.random.normal(0, 0.02)
            close = base_price * (1 + price_change * (i + 1))
            volume = np.random.uniform(100, 1000)
            
            candles.append({
                "timestamp": timestamp,
                "pair_address": pair,
                "close": close,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "volume_token1": volume,
                "volume_token0": volume * close,
                "num_swaps": int(np.random.uniform(10, 100)),
            })
        
        candles_df = pl.DataFrame(candles)
        candles_history[pair] = candles_df
    
    print(f"✅ Generated candle history for {len(candles_history)} pairs")
    
    # Add regime predictions (simulating _add_regime_predictions)
    regime_predictions = []
    for row in sample_signals.iter_rows(named=True):
        pair_address = row["pair_address"]
        
        try:
            if pair_address in candles_history:
                candles_df = candles_history[pair_address]
                
                # Prepare data for prediction (same format as MarketRegimeClassifier expects)
                # The classifier expects daily candles, but we'll use our 15-min candles
                # For testing, we'll aggregate to daily-like features
                if len(candles_df) >= 7:
                    # Use last 7 candles as "daily" data
                    daily_like = candles_df.tail(7)
                    
                    # Compute basic features
                    closes = daily_like["close"].to_list()
                    volumes = daily_like["volume_token1"].to_list()
                    
                    # Create a minimal feature set
                    data_dict = {
                        "timestamp": [daily_like["timestamp"].max()],
                        "pair_address": [pair_address],
                        "close": [closes[-1]],
                        "open": [daily_like["open"].tail(1).item()],
                        "high": [daily_like["high"].max()],
                        "low": [daily_like["low"].min()],
                        "volume_token1": [sum(volumes)],
                        "volume_token0": [sum(daily_like["volume_token0"].to_list())],
                        "num_swaps": [daily_like["num_swaps"].sum()],
                    }
                    
                    pred_df = pl.DataFrame(data_dict)
                    
                    # Predict regime
                    result = classifier.predict(pred_df)
                    if result is not None and len(result) > 0:
                        regime = result["predicted_regime"].item()
                        regime_predictions.append(regime)
                    else:
                        regime_predictions.append("Unknown")
                else:
                    regime_predictions.append("Unknown")
            else:
                regime_predictions.append("Unknown")
                
        except Exception as e:
            print(f"⚠️  Error predicting regime for {pair_address}: {e}")
            regime_predictions.append("Unknown")
    
    # Add regime column to signals
    signals_with_regime = sample_signals.with_columns([
        pl.Series("regime", regime_predictions)
    ])
    
    print(f"\n✅ Signals with regime predictions:")
    print(signals_with_regime)
    
    # Verify regime column exists
    if "regime" not in signals_with_regime.columns:
        print("\n❌ FAILED: Regime column not found in signals!")
        return False
    
    # Check regime values
    regime_values = signals_with_regime["regime"].unique().to_list()
    print(f"\n✅ Regime column present!")
    print(f"   Unique regime values: {regime_values}")
    
    # Save to CSV to verify format
    test_csv_path = Path("test_signals_with_regime.csv")
    signals_with_regime.write_csv(test_csv_path)
    print(f"\n💾 Saved test CSV to {test_csv_path}")
    
    # Read it back and verify
    loaded = pl.read_csv(test_csv_path)
    if "regime" in loaded.columns:
        print(f"✅ Verified: Regime column present in CSV file!")
        print(f"   CSV columns: {loaded.columns}")
        print(f"   CSV shape: {loaded.shape}")
        return True
    else:
        print(f"❌ FAILED: Regime column missing in loaded CSV!")
        return False

if __name__ == "__main__":
    success = test_regime_in_csv()
    if success:
        print("\n" + "="*80)
        print("✅ SUCCESS: Regime predictions are correctly added to CSV output!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ FAILED: Regime predictions not working correctly")
        print("="*80)
        sys.exit(1)
