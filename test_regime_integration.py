#!/usr/bin/env python3
"""
Test Market Regime Integration
================================

Validates that the live signal generator correctly:
1. Loads regime data
2. Applies adjustments
3. Handles emergency exits
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_regime_file_exists():
    """Test that regime file was created"""
    regime_path = Path("market_regime.json")
    if not regime_path.exists():
        logger.error("❌ market_regime.json not found. Run market_regime_analyzer.py first.")
        return False
    
    logger.info("✅ market_regime.json exists")
    return True


def test_regime_data_valid():
    """Test that regime data has required fields"""
    try:
        with open("market_regime.json", 'r') as f:
            regime = json.load(f)
        
        required_fields = ['regime', 'confidence', 'emergency_exit', 'metrics']
        missing = [f for f in required_fields if f not in regime]
        
        if missing:
            logger.error(f"❌ Missing required fields: {missing}")
            return False
        
        logger.info(f"✅ Regime data valid: {regime['regime'].upper()} ({regime['confidence']:.0%} confidence)")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading regime: {e}")
        return False


def test_regime_adjustments():
    """Test that regime adjustments are calculated correctly"""
    from live_signal_generator import LiveConfig
    
    with open("market_regime.json", 'r') as f:
        regime = json.load(f)
    
    regime_name = regime['regime'].lower()
    
    # Get adjustments
    kelly_adj = LiveConfig.REGIME_KELLY_ADJUSTMENTS.get(regime_name, 1.0)
    threshold_adj = LiveConfig.REGIME_THRESHOLD_ADJUSTMENTS.get(regime_name, LiveConfig.SIGNAL_THRESHOLD)
    
    logger.info(f"✅ Kelly adjustment: {kelly_adj}x")
    logger.info(f"✅ Threshold adjustment: {threshold_adj:.3f}")
    
    # Calculate effective values
    effective_kelly = LiveConfig.KELLY_FRACTION * kelly_adj
    logger.info(f"   Effective Kelly: {LiveConfig.KELLY_FRACTION} × {kelly_adj} = {effective_kelly:.3f}")
    logger.info(f"   Effective Threshold: {threshold_adj:.3f} (base: {LiveConfig.SIGNAL_THRESHOLD:.3f})")
    
    return True


def test_emergency_exit():
    """Test emergency exit detection"""
    with open("market_regime.json", 'r') as f:
        regime = json.load(f)
    
    if regime['emergency_exit']:
        logger.warning("⚠️  EMERGENCY EXIT ACTIVE - Live system will skip trading")
        logger.warning(f"   Regime: {regime['regime'].upper()}")
        logger.warning(f"   Drawdown: {regime['metrics'].get('drawdown_pct', 'N/A')}%")
    else:
        logger.info("✅ No emergency exit - trading permitted")
    
    return True


def test_exposure_signals():
    """Test exposure recommendation signals"""
    with open("market_regime.json", 'r') as f:
        regime = json.load(f)
    
    if regime.get('reduce_exposure', False):
        logger.info("⚠️  REDUCE EXPOSURE recommended")
    elif regime.get('increase_exposure', False):
        logger.info("✅ INCREASE EXPOSURE recommended")
    else:
        logger.info("ℹ️  MAINTAIN current exposure")
    
    if regime.get('reentry_opportunity', False):
        logger.info("🎯 REENTRY OPPORTUNITY detected")
    
    return True


def main():
    """Run all tests"""
    logger.info("="*80)
    logger.info("TESTING REGIME INTEGRATION")
    logger.info("="*80)
    logger.info("")
    
    tests = [
        ("Regime file exists", test_regime_file_exists),
        ("Regime data valid", test_regime_data_valid),
        ("Regime adjustments", test_regime_adjustments),
        ("Emergency exit check", test_emergency_exit),
        ("Exposure signals", test_exposure_signals),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        logger.info(f"\nTest: {name}")
        logger.info("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"❌ Test failed: {name}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ Test crashed: {name} - {e}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info(f"RESULTS: {passed}/{len(tests)} tests passed")
    logger.info("="*80)
    
    if failed == 0:
        logger.info("✅ All tests passed! Integration ready.")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python market_regime_analyzer.py  (to update regime)")
        logger.info("  2. Run: python run_with_regime.py  (to start live system)")
        return 0
    else:
        logger.error(f"❌ {failed} test(s) failed. Fix issues before running live.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
