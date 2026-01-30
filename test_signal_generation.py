#!/usr/bin/env python3
"""
Quick test script to generate signals with current data
This temporarily lowers the candle requirement to test signal generation
"""

import pickle
import polars as pl
from pathlib import Path
import sys
import os

# Set API token
os.environ["ENVIO_API_TOKEN"] = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"

from live_signal_generator import LiveConfig, LiveSignalGenerator, FeatureEngineer
import asyncio

async def test_signal_generation():
    """Test signal generation with current data"""
    
    # Temporarily lower requirement
    original_min = LiveConfig.MIN_HISTORY_CANDLES
    LiveConfig.MIN_HISTORY_CANDLES = 15  # Use current data
    
    print("="*80)
    print("TEST SIGNAL GENERATION")
    print("="*80)
    print(f"Using MIN_HISTORY_CANDLES = {LiveConfig.MIN_HISTORY_CANDLES} (normally {original_min})")
    print()
    
    generator = LiveSignalGenerator()
    
    # Load model
    try:
        generator.load_model()
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Get candles
    all_candles = generator.get_all_candles()
    print(f"📊 Total candles: {len(all_candles)}")
    print(f"📊 Total pairs: {len(generator.candles_history)}")
    
    if len(all_candles) == 0:
        print("❌ No candles available")
        return
    
    # Check for BB data
    has_bb = all(col in all_candles.columns for col in ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position"])
    print(f"📊 Has BB data: {has_bb}")
    
    if not has_bb:
        print("⚠️  No BB data in candles - will only generate OHLCV signals")
        print("   (BB data comes from live streaming with BB enabled)")
    
    # Generate OHLCV signals
    print("\n" + "="*80)
    print("GENERATING OHLCV SIGNALS (XGBoost)")
    print("="*80)
    ohlcv_signals = generator.generate_signals()
    
    if len(ohlcv_signals) > 0:
        print(f"✅ Generated {len(ohlcv_signals)} OHLCV signals")
        print("\nTop 5 signals:")
        for i, row in enumerate(ohlcv_signals.head(5).iter_rows(named=True), 1):
            print(f"  {i}. Pair: {row['pair_address'][:20]}...")
            print(f"     Confidence: {row.get('pred_proba', 0):.3f}")
            print(f"     Price: ${row['close']:.6f}")
            print(f"     TP: ${row.get('take_profit_price', 0):.6f} | SL: ${row.get('stop_loss_price', 0):.6f}")
    else:
        print("⚠️  No OHLCV signals generated (threshold not met or insufficient data)")
    
    # Generate BB signals if data available
    if has_bb:
        print("\n" + "="*80)
        print("GENERATING BOLLINGER BAND SIGNALS")
        print("="*80)
        bb_signals = generator.generate_bb_signals()
        
        if len(bb_signals) > 0:
            print(f"✅ Generated {len(bb_signals)} BB signals")
            print("\nTop 5 signals:")
            for i, row in enumerate(bb_signals.head(5).iter_rows(named=True), 1):
                print(f"  {i}. Pair: {row['pair_address'][:20]}...")
                print(f"     Reason: {row.get('bb_signal_reason', 'unknown')}")
                print(f"     BB Position: {row.get('bb_position', 0):.3f}")
                print(f"     Strength: {row.get('bb_signal_strength', 0):.3f}")
                print(f"     Price: ${row['close']:.6f}")
                print(f"     TP: ${row.get('take_profit_price', 0):.6f} | SL: ${row.get('stop_loss_price', 0):.6f}")
        else:
            print("⚠️  No BB signals generated (no pairs meet BB criteria)")
    
    # Restore original requirement
    LiveConfig.MIN_HISTORY_CANDLES = original_min
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nNote: The live signal generator requires {original_min}+ candles per pair")
    print("      and runs every 15 minutes. It's currently running in the background.")
    print("      Check 'signal_generator.log' for progress.")

if __name__ == "__main__":
    asyncio.run(test_signal_generation())
