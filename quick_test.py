#!/usr/bin/env python3
"""
Simple Quick Test - Verify Basic Setup
=======================================
A minimal test to check if everything is ready to run.
"""

print("Testing basic setup...\n")

# Test 1: Import packages
print("1. Testing package imports...")
try:
    import polars as pl
    import xgboost as xgb
    import numpy as np
    import hypersync
    from dotenv import load_dotenv
    print("   ✅ All packages imported successfully")
except ImportError as e:
    print(f"   ❌ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
    exit(1)

# Test 2: Check .env file
print("\n2. Checking .env file...")
import os
load_dotenv()
if not Path(".env").exists():
    print("   ⚠️  .env file not found")
    print("   Copy .env.example and add your ENVIO_API_TOKEN")
else:
    token = os.getenv("ENVIO_API_TOKEN")
    if not token:
        print("   ❌ ENVIO_API_TOKEN not set in .env")
        print("   Add your token to .env file")
    else:
        print(f"   ✅ .env file configured (token length: {len(token)})")

# Test 3: Check if model exists
print("\n3. Checking for trained model...")
from pathlib import Path
model_path = Path("xgb_model.json")
if model_path.exists():
    print(f"   ✅ Model found: {model_path}")
else:
    print(f"   ⚠️  Model not found: {model_path}")
    print("   Run: python run_pipeline.py")

# Test 3: Try importing custom modules
print("\n4. Testing custom modules...")
try:
    from envio_hypersync import EnvioConfig, LiveSwapStreamer
    print("   ✅ envio_hypersync.py imports OK")
except Exception as e:
    print(f"   ❌ Error importing envio_hypersync: {e}")
    exit(1)

try:
    from live_signal_generator import LiveConfig, LiveSignalGenerator
    print("   ✅ live_signal_generator.py imports OK")
except Exception as e:
    print(f"   ❌ Error importing live_signal_generator: {e}")
    exit(1)

# Test 4: Test feature engineering on sample data
print("\n5. Testing feature computation...")
try:
    from live_signal_generator import FeatureEngineer
    from datetime import datetime, timedelta
    
    # Create minimal sample
    dates = [datetime.now() - timedelta(days=i) for i in range(15, 0, -1)]
    sample = pl.DataFrame({
        "pair_address": ["0xtest"] * 15,
        "timestamp": dates,
        "open": [1.0] * 15,
        "high": [1.1] * 15,
        "low": [0.9] * 15,
        "close": [1.0] * 15,
        "volume_token0": [1000.0] * 15,
        "volume_token1": [1000.0] * 15,
        "num_trades": [100] * 15,
    })
    
    features = FeatureEngineer.compute_all_features(sample)
    print(f"   ✅ Features computed: {features.shape}")
except Exception as e:
    print(f"   ❌ Feature engineering failed: {e}")
    exit(1)

# Summary
print("\n" + "="*50)
print("✅ BASIC SETUP OK")
print("="*50)
print("\nYou're ready to run the live system!")
print("\nNext steps:")
print("  1. python live_signal_generator.py")
print("  2. tail -f signals_live.csv")
print("\nFor comprehensive tests, run:")
print("  python test_live_system.py")
