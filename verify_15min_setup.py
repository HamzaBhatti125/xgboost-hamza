#!/usr/bin/env python3
"""
Verify 15-Minute Candle Setup
==============================
Quick verification script to ensure all components are configured correctly.
"""

import os
from pathlib import Path
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_config_value(filepath, config_class, attr_name, expected_value):
    """Check configuration value by parsing file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Look for the attribute assignment
            search_pattern = f"{attr_name} = {expected_value}"
            if search_pattern in content:
                print(f"✅ {filepath}: {attr_name} = {expected_value}")
                return True
            else:
                # Try to find what the actual value is
                import re
                pattern = rf"{attr_name}\s*=\s*(\d+)"
                match = re.search(pattern, content)
                if match:
                    actual_value = match.group(1)
                    print(f"❌ {filepath}: {attr_name} = {actual_value} (expected: {expected_value})")
                else:
                    print(f"⚠️ {filepath}: Could not find {attr_name}")
                return False
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    print("="*80)
    print("15-MINUTE CANDLE SETUP VERIFICATION")
    print("="*80)
    
    all_checks_passed = True
    
    # Check data files
    print("\n📁 DATA FILES:")
    all_checks_passed &= check_file_exists("Files/candles-15m.parquet", "15-min candle data")
    all_checks_passed &= check_file_exists("Files/candles-1d.parquet", "Daily candle data")
    
    # Check model files
    print("\n🤖 MODEL FILES:")
    all_checks_passed &= check_file_exists("xgboost_15min_model.json", "15-min trained model")
    
    # Check code files
    print("\n📝 CODE FILES:")
    all_checks_passed &= check_file_exists("train_15min_model.py", "Training script")
    all_checks_passed &= check_file_exists("envio_hypersync.py", "Envio client")
    all_checks_passed &= check_file_exists("live_signal_generator.py", "Live signal generator")
    
    # Check configurations
    print("\n⚙️  CONFIGURATION VALUES:")
    all_checks_passed &= check_config_value("envio_hypersync.py", "EnvioConfig", "CANDLE_INTERVAL_SECONDS", 900)
    all_checks_passed &= check_config_value("envio_hypersync.py", "EnvioConfig", "CANDLE_INTERVAL_MINUTES", 15)
    all_checks_passed &= check_config_value("live_signal_generator.py", "LiveConfig", "UPDATE_INTERVAL_SECONDS", 900)
    all_checks_passed &= check_config_value("live_signal_generator.py", "LiveConfig", "MIN_HISTORY_CANDLES", 200)
    
    # Check model path
    print("\n🎯 MODEL CONFIGURATION:")
    try:
        with open("live_signal_generator.py", 'r') as f:
            content = f.read()
            expected_model = "xgboost_15min_model.json"
            if f'MODEL_PATH = "{expected_model}"' in content:
                print(f"✅ LiveConfig.MODEL_PATH = {expected_model}")
            else:
                print(f"❌ LiveConfig.MODEL_PATH not set to {expected_model}")
                all_checks_passed = False
    except Exception as e:
        print(f"❌ Error checking model path: {e}")
        all_checks_passed = False
    
    # Environment check
    print("\n🔐 ENVIRONMENT:")
    env_token = os.getenv("ENVIO_API_TOKEN")
    has_token = bool(env_token)
    status = "✅" if has_token else "⚠️"
    print(f"{status} ENVIO_API_TOKEN {'set' if has_token else 'NOT set (needed for live streaming)'}")
    
    # Summary
    print("\n" + "="*80)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - System is ready for 15-minute candles!")
        print("\nNext steps:")
        print("  1. Test live streaming: python test_live_system.py")
        print("  2. Run signal generation: python live_signal_generator.py")
    else:
        print("❌ SOME CHECKS FAILED - Please review the errors above")
        return 1
    
    print("="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
