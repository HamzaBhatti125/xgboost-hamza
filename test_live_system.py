#!/usr/bin/env python3
"""
Test Live System Setup
======================

Validates that all components are properly configured and can run.
Tests each component individually before running the full system.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_imports():
    """Test that all required packages are installed"""
    print_section("TEST 1: Package Imports")
    
    required_packages = [
        ("polars", "polars"),
        ("xgboost", "xgboost"),
        ("numpy", "numpy"),
        ("hypersync", "hypersync"),
        ("dotenv", "python-dotenv"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
    ]
    
    all_good = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name:20s} - installed")
        except ImportError:
            print(f"❌ {package_name:20s} - MISSING")
            all_good = False
            
    return all_good


def test_imports():
    """Test that all required packages are installed"""
    print_section("TEST 3: Package Imports")
    
    required_packages = [
        ("polars", "polars"),
        ("xgboost", "xgboost"),
        ("numpy", "numpy"),
        ("hypersync", "hypersync"),
        ("dotenv", "python-dotenv"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
    ]
    
    all_good = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name:20s} - installed")
        except ImportError:
            print(f"❌ {package_name:20s} - MISSING")
            all_good = False
            
    return all_good


def test_env_file():
    """Check if .env file exists with API token"""
    print_section("TEST 2: Environment Configuration")
    
    import os
    from dotenv import load_dotenv
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print(f"❌ .env file not found")
        print(f"   Create .env file from .env.example:")
        print(f"   cp .env.example .env")
        print(f"   Then add your ENVIO_API_TOKEN")
        return False
    
    print(f"✅ .env file exists")
    
    # Load env
    load_dotenv()
    token = os.getenv("ENVIO_API_TOKEN")
    
    if not token:
        print(f"❌ ENVIO_API_TOKEN not set in .env file")
        print(f"   Add your token to .env file:")
        print(f"   ENVIO_API_TOKEN=your_token_here")
        return False
    
    print(f"✅ ENVIO_API_TOKEN is set")
    print(f"   Token length: {len(token)} characters")
    
    return True


def test_model_exists():
    """Check if trained model exists"""
    print_section("TEST 2: Model File")
    
    model_path = Path("xgb_model.json")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model NOT found: {model_path}")
        print("   Run: python run_pipeline.py")
        return False


def test_envio_module():
    """Test Envio Hypersync module"""
    print_section("TEST 4: Envio Hypersync Module")
    
    try:
        from envio_hypersync import (
            EnvioConfig, 
            EnvioHypersyncClient, 
            SwapEventProcessor,
            CandleAggregator,
            LiveSwapStreamer
        )
        
        print("✅ Module imports successful")
        print(f"   Chain ID: {EnvioConfig.CHAIN_ID}")
        print(f"   Hypersync URL: {EnvioConfig.HYPERSYNC_URL}")
        print(f"   Candle Interval: {EnvioConfig.CANDLE_INTERVAL_SECONDS}s")
        
        # Test instantiation
        processor = SwapEventProcessor()
        aggregator = CandleAggregator()
        print("✅ Component instantiation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_signal_module():
    """Test live signal generator module"""
    print_section("TEST 5: Live Signal Generator Module")
    
    try:
        from live_signal_generator import (
            LiveConfig,
            FeatureEngineer,
            LiveSignalGenerator
        )
        
        print("✅ Module imports successful")
        print(f"   Model Path: {LiveConfig.MODEL_PATH}")
        print(f"   Signal Threshold: {LiveConfig.SIGNAL_THRESHOLD}")
        print(f"   Update Interval: {LiveConfig.UPDATE_INTERVAL_SECONDS}s")
        print(f"   Feature Count: {len(LiveConfig.FEATURE_COLS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature computation on sample data"""
    print_section("TEST 6: Feature Engineering")
    
    try:
        from live_signal_generator import FeatureEngineer, LiveConfig
        
        # Create sample candle data
        dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        
        sample_data = pl.DataFrame({
            "pair_address": ["0xtest123"] * 20,
            "timestamp": dates,
            "open": np.random.uniform(0.9, 1.1, 20).tolist(),
            "high": np.random.uniform(1.0, 1.2, 20).tolist(),
            "low": np.random.uniform(0.8, 1.0, 20).tolist(),
            "close": np.random.uniform(0.9, 1.1, 20).tolist(),
            "volume_token0": np.random.uniform(1000, 5000, 20).tolist(),
            "volume_token1": np.random.uniform(1000, 5000, 20).tolist(),
            "num_trades": np.random.randint(50, 200, 20).tolist(),
        })
        
        print(f"Created sample data: {len(sample_data)} rows")
        
        # Compute features
        features_df = FeatureEngineer.compute_all_features(sample_data)
        
        print("✅ Feature computation successful")
        print(f"   Output shape: {features_df.shape}")
        
        # Check for required features
        missing = []
        for col in LiveConfig.FEATURE_COLS:
            if col not in features_df.columns:
                missing.append(col)
                
        if missing:
            print(f"❌ Missing features: {missing}")
            return False
        else:
            print(f"✅ All {len(LiveConfig.FEATURE_COLS)} features present")
            
        # Check for NaN in latest row
        latest = features_df.tail(1)
        nan_cols = []
        for col in LiveConfig.FEATURE_COLS:
            if latest[col].is_null().any():
                nan_cols.append(col)
                
        if nan_cols:
            print(f"⚠️  Some features have NaN (expected for small sample): {nan_cols}")
        else:
            print("✅ No NaN values in latest row")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_envio_connection():
    """Test connection to Envio Hypersync"""
    print_section("TEST 7: Envio Hypersync Connection")
    
    try:
        from envio_hypersync import EnvioHypersyncClient
        
        print("Connecting to Envio Hypersync...")
        
        async with EnvioHypersyncClient() as client:
            # Try to get current block
            current_block = await client.get_current_block()
            
            if current_block > 0:
                print(f"✅ Connection successful")
                print(f"   Current block: {current_block:,}")
                
                # Try a small query (last 100 blocks)
                logs = await client.query_logs(
                    from_block=current_block - 100,
                    to_block=current_block
                )
                
                print(f"✅ Query successful")
                print(f"   Logs returned: {len(logs)}")
                
                return True
            else:
                print("❌ Could not get current block")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading the XGBoost model"""
    print_section("TEST 8: Model Loading")
    
    try:
        import xgboost as xgb
        from live_signal_generator import LiveConfig
        
        model_path = Path(LiveConfig.MODEL_PATH)
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
            
        model = xgb.Booster()
        model.load_model(str(model_path))
        
        print(f"✅ Model loaded successfully")
        
        # Test prediction on dummy data
        from live_signal_generator import FeatureEngineer
        
        dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        sample_data = pl.DataFrame({
            "pair_address": ["0xtest"] * 20,
            "timestamp": dates,
            "open": [1.0] * 20,
            "high": [1.1] * 20,
            "low": [0.9] * 20,
            "close": [1.0] * 20,
            "volume_token0": [1000] * 20,
            "volume_token1": [1000] * 20,
            "num_trades": [100] * 20,
        })
        
        features_df = FeatureEngineer.compute_all_features(sample_data)
        latest = features_df.tail(1)
        
        X = latest.select(LiveConfig.FEATURE_COLS).to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=LiveConfig.FEATURE_COLS)
        
        pred = model.predict(dmatrix)
        
        print(f"✅ Prediction successful")
        print(f"   Predicted probability: {pred[0]:.4f}")
        
        if pred[0] >= LiveConfig.SIGNAL_THRESHOLD:
            print(f"   Signal: BUY (above threshold {LiveConfig.SIGNAL_THRESHOLD})")
        else:
            print(f"   Signal: HOLD (below threshold {LiveConfig.SIGNAL_THRESHOLD})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_paths():
    """Check output directories are writable"""
    print_section("TEST 9: Output Paths")
    
    from live_signal_generator import LiveConfig
    
    paths = [
        ("Signals CSV", LiveConfig.SIGNALS_OUTPUT_PATH),
        ("Candles Parquet", LiveConfig.HISTORICAL_CANDLES_PATH),
    ]
    
    all_good = True
    for name, path in paths:
        path_obj = Path(path)
        parent = path_obj.parent
        
        if parent.exists() and parent.is_dir():
            print(f"✅ {name:20s} - directory exists: {parent}")
        else:
            print(f"⚠️  {name:20s} - directory will be created: {parent}")
            
        # Try to create a test file
        try:
            test_file = parent / f".test_{datetime.now().timestamp()}"
            test_file.touch()
            test_file.unlink()
            print(f"   Write permission: OK")
        except Exception as e:
            print(f"❌ Write permission: FAILED - {e}")
            all_good = False
            
    return all_good


async def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  LIVE TRADING SYSTEM - SETUP VALIDATION".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    results = {}
    
    # Sequential tests
    results["imports"] = test_imports()
    results["env_file"] = test_env_file()
    results["model_exists"] = test_model_exists()
    results["envio_module"] = test_envio_module()
    results["live_signal_module"] = test_live_signal_module()
    results["feature_engineering"] = test_feature_engineering()
    results["output_paths"] = test_output_paths()
    
    # Model loading (depends on model existing)
    if results["model_exists"]:
        results["model_loading"] = test_model_loading()
    else:
        results["model_loading"] = None
        
    # Async tests
    results["envio_connection"] = await test_envio_connection()
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    print(f"Tests Passed:  {passed}/{total}")
    print(f"Tests Failed:  {failed}/{total}")
    print(f"Tests Skipped: {skipped}/{total}")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"  {status}  {test_name}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready to run.")
        print("\nNext steps:")
        print("  1. If model doesn't exist: python run_pipeline.py")
        print("  2. Start live system: python live_signal_generator.py")
        print("  3. Monitor signals: tail -f signals_live.csv")
        print("\n" + "="*80)
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n" + "="*80)
        return 1


def main():
    """Main entry point"""
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
